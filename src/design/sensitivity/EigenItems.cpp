#include "EigenItems.hpp"

#include "FormulationShape.hpp"
#include "oos/assembly/assemble/DifferentialStiffnessAssembler.hpp"
#include "oos/assembly/assemble/MassAssembler.hpp"
#include "oos/assembly/assemble/NonStructuralMassShares.hpp"

#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/InternalError.hpp"
#include "oos/base/errors/OptimisationError.hpp"
#include "oos/design/sensitivity/SubcaseSolve.hpp"
#include "oos/solutions/sol101/StaticSubcases.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"
#include "oos/elements/core/ElementVectorBlock.hpp"
#include "oos/linalg/dense/SmallEigenSolver.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>

namespace oos::design::detail
{

namespace
{

// df/dλ of f = √|λ| / 2π: sign(λ) / (4π·√|λ|).
constexpr double FOUR_PI = 4.0 * std::numbers::pi;
// How far the item's value may differ from the solve's root it names: the same number copied (the eigenvalue table is
// filled from the solve), so only rounding of the cycles of FREQ.
constexpr double ROOT_CHECK_TOLERANCE = 1.0e-12;

[[nodiscard]] std::string itemName(const ResponseItem& label, ResponseType type)
{
    return base::concatenate({std::string(responseTypeName(type)), " item of mode ", std::to_string(label.layer),
                              " in subcase ", std::to_string(label.subcase)});
}

} // namespace

EigenItems::EigenItems(const Inputs& inputs) : m_in(inputs) {}

std::size_t EigenItems::repeatedClusterCount() const noexcept
{
    return static_cast<std::size_t>(
        std::count_if(m_clusters.begin(), m_clusters.end(), [](const Cluster& c) { return c.modes.size() > 1; }));
}

std::optional<std::int32_t> EigenItems::preloadOf(std::int32_t bucklingSubcase) const
{
    if (m_in.buckling == nullptr)
        return std::nullopt;
    for (const solutions::BucklingSolve& solve : m_in.buckling->solves)
        if (solve.subcase == bucklingSubcase)
            return solve.preloadSubcase;
    return std::nullopt;
}

void EigenItems::add(const ItemKey& key, ResponseType type, const ResponseItem& label, double value)
{
    const bool isBuckling = type == ResponseType::BucklingFactor;
    // The solve of the item's subcase, registered once.
    std::optional<std::size_t> found;
    for (std::size_t s = 0; s < m_solves.size(); ++s)
        if (m_solves[s].isBuckling == isBuckling && m_solves[s].subcase == label.subcase)
            found = s;
    if (!found)
    {
        Solve solve;
        solve.isBuckling = isBuckling;
        solve.subcase = label.subcase;
        if (isBuckling && m_in.buckling != nullptr)
        {
            for (const solutions::BucklingSolve& candidate : m_in.buckling->solves)
                if (candidate.subcase == label.subcase)
                {
                    solve.preloadSubcase = candidate.preloadSubcase;
                    for (const output::EigenvalueRow& row : candidate.eigenvalues.rows)
                        solve.roots.push_back(row.eigenvalue);
                    solve.shapes = &candidate.eigenvectors;
                    solve.scales = &candidate.scales;
                }
        }
        else if (!isBuckling && m_in.modes != nullptr)
        {
            for (const solutions::ModesSubcaseResult& result : m_in.modes->subcases)
                if (result.subcase == label.subcase)
                {
                    const solutions::ModesSolve& candidate = m_in.modes->solves[result.solve];
                    solve.roots = candidate.eigenvalues;
                    solve.shapes = &candidate.eigenvectors;
                    solve.scales = &candidate.scales;
                }
            // The mass of the subcase as the modal solution assembles it (PARAM CoupledMass, WeightToMass of the
            // subcase).
            base::DiagnosticSink ignored; // the analysis reported the problems of its PARAMs
            const assembly::MassAssemblyOptions options =
                assembly::MassAssemblyOptions::fromParams(m_in.params, label.subcase, ignored);
            solve.massOptions.formulation = options.formulation;
            solve.weightToMass = options.weightToMass;
        }
        if (solve.shapes == nullptr)
            throw base::OptimisationError(
                base::concatenate({"the sensitivity of a ", itemName(label, type), ": the analysis has no ",
                                   isBuckling ? "buckling" : "modes",
                                   " solve of that subcase; analyse the subcase (DesignProblem(analyses=...))"}));
        found = m_solves.size();
        m_solves.push_back(std::move(solve));
    }
    const Solve& solve = m_solves[*found];
    if (label.layer < 1 || static_cast<std::size_t>(label.layer) > solve.roots.size() ||
        static_cast<std::size_t>(label.layer) > solve.shapes->columns())
        throw base::OptimisationError(base::concatenate({"the sensitivity of a ", itemName(label, type),
                                                         ": the solve found ", std::to_string(solve.roots.size()),
                                                         " roots; the item names another mode: ask a mode the method "
                                                         "finds (mode=, LanczosEigenMethod.desiredModeCount of "
                                                         "LanczosEigenMethod)"}));
    Item item;
    item.key = key;
    item.type = type;
    item.solve = *found;
    item.mode = static_cast<std::size_t>(label.layer - 1);
    const double root = solve.roots[item.mode];
    double expected = root;
    if (type == ResponseType::Frequency)
    {
        // f = √|λ| / 2π (the cycles of the eigenvalue table).
        expected = std::sqrt(std::abs(root)) / (2.0 * std::numbers::pi);
        if (root == 0.0)
            throw base::OptimisationError(base::concatenate(
                {"the sensitivity of a ", itemName(label, type),
                 ": the root is zero, where the frequency √λ has no derivative; constrain EIGN instead"}));
        item.scale = (root > 0.0 ? 1.0 : -1.0) / (FOUR_PI * std::sqrt(std::abs(root)));
    }
    if (std::abs(expected - value) > ROOT_CHECK_TOLERANCE * std::max(std::abs(expected), std::abs(value)))
        throw base::OptimisationError(
            base::concatenate({"the sensitivity of a ", itemName(label, type), ": the item is ", std::to_string(value),
                               ", the root of the solve ", std::to_string(expected),
                               "; the derivative would be that of another root, so it is not returned: evaluate and "
                               "differentiate at the same design point"}));
    m_hasModal = m_hasModal || !isBuckling;
    m_hasBuckling = m_hasBuckling || isBuckling;
    m_items.push_back(item);
}

void EigenItems::formClusters()
{
    for (Item& item : m_items)
    {
        const Solve& solve = m_solves[item.solve];
        const std::vector<double>& roots = solve.roots;
        double largest = 0.0;
        for (const double root : roots)
            largest = std::max(largest, std::abs(root));
        const auto isSameRoot = [&](std::size_t a, std::size_t b)
        {
            const double scale = std::max({std::abs(roots[a]), std::abs(roots[b]), ZERO_ROOT_FRACTION * largest});
            return std::abs(roots[a] - roots[b]) <= REPEATED_ROOT_TOLERANCE * scale;
        };
        // The neighbours of the item's root, in the solve's order (equal roots are next to each other: the vibration
        // roots increase, the buckling ones increase in magnitude).
        std::size_t first = item.mode;
        while (first > 0 && isSameRoot(first - 1, first))
            --first;
        std::size_t last = item.mode;
        while (last + 1 < roots.size() && last + 1 < solve.shapes->columns() && isSameRoot(last, last + 1))
            ++last;
        std::optional<std::size_t> found;
        for (std::size_t c = 0; c < m_clusters.size(); ++c)
            if (m_clusters[c].solve == item.solve && m_clusters[c].modes.front() == first)
                found = c;
        if (!found)
        {
            Cluster cluster;
            cluster.solve = item.solve;
            double sum = 0.0;
            for (std::size_t mode = first; mode <= last; ++mode)
            {
                cluster.modes.push_back(mode);
                sum += roots[mode];
            }
            cluster.root = sum / static_cast<double>(cluster.modes.size());
            found = m_clusters.size();
            m_clusters.push_back(std::move(cluster));
        }
        item.cluster = *found;
        item.rank = item.mode - first;
    }
}

void EigenItems::setPreload(std::int32_t subcase, const elements::NodalDisplacements& preload,
                            std::span<const double> displacementsG)
{
    m_preloads[subcase] = preload;
    m_preloadVectors[subcase] = displacementsG;
}

std::vector<double> EigenItems::normalisedShape(const Solve& solve, std::size_t mode) const
{
    const linalg::DenseMatrix& shapes = *solve.shapes;
    std::vector<double> shape(shapes.rows());
    for (std::size_t i = 0; i < shapes.rows(); ++i)
        shape[i] = shapes(i, mode) / (*solve.scales)[mode];
    return shape;
}

EigenItems::SolveConstraints& EigenItems::constraintsOf(std::size_t solveIndex)
{
    const auto found = m_solveConstraints.find(solveIndex);
    if (found != m_solveConstraints.end())
        return found->second;
    SolveConstraints& made = m_solveConstraints[solveIndex];
    const Solve& solve = m_solves[solveIndex];
    if (m_in.snapshot == nullptr)
        throw std::logic_error("EigenItems: the snapshot of the design point is not given (a bug of O_o_Solver)");
    base::MemoryBudget& budget = *m_in.budget;
    base::DiagnosticSink sink; // the analysis reported the problems of the model
    made.model = std::make_unique<solutions::AnalysisModel>(
        *m_in.snapshot, solve.isBuckling ? solutions::AnalysisKind::Buckling : solutions::AnalysisKind::Modes, budget,
        m_in.pool, nullptr, sink);
    const std::vector<solutions::StaticSubcase> subcases =
        solutions::staticSubcasesOf(made.model->snapshot(), {solve.subcase}, sink);
    if (subcases.empty())
        throw base::OptimisationError(base::concatenate(
            {"the sensitivities of the eigenvalues of subcase ", std::to_string(solve.subcase),
             " under a shape: the subcase is not found again in the model"}));
    const solutions::StaticSubcase& subcase = subcases.front();
    made.system = std::make_unique<solutions::ConstrainedSystem>(
        *made.model,
        solutions::ConstraintSelection{subcase.label, subcase.spc, subcase.mpc, std::nullopt, subcase.contact}, budget,
        m_in.pool, nullptr, sink);
    const assembly::FormulationConstraints* formulation = made.system->formulationConstraints();
    if (formulation == nullptr || formulation->relations().empty())
    {
        made.system.reset();
        return made;
    }
    const assembly::DofMap& dofMap = made.model->dofMap();
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    if (solve.shapes->rows() != gSize)
        throw std::logic_error("EigenItems: the shapes of a solve are not on the g set of its analysis model (a bug "
                               "of O_o_Solver: report it)");
    // The second matrix: the mass of the vibration, the differential stiffness of the preload.
    linalg::BlockSparseMatrix second(made.system->pattern(), budget, "eigenvalue shape second matrix");
    if (solve.isBuckling)
    {
        const auto preload = m_preloadVectors.find(solve.preloadSubcase);
        if (preload == m_preloadVectors.end())
            throw std::logic_error("EigenItems: the preload of a buckling solve was not read (a bug of O_o_Solver)");
        const auto nodal = m_preloads.find(solve.preloadSubcase);
        assembly::DifferentialStiffnessAssembler(made.model->elementSet(), dofMap)
            .assemble(preload->second, nodal != m_preloads.end() ? nodal->second.temperatures : nullptr, second,
                      m_in.pool);
    }
    else
    {
        const assembly::MassAssemblyOptions options =
            assembly::MassAssemblyOptions::fromParams(m_in.params, solve.subcase, sink);
        const assembly::NonStructuralMassShares shares(made.model->elementSet());
        assembly::MassAssembler(made.model->elementSet(), dofMap, shares, options).assemble(second, m_in.pool);
    }
    for (const Cluster& cluster : m_clusters)
    {
        if (cluster.solve != solveIndex)
            continue;
        for (const std::size_t mode : cluster.modes)
        {
            const std::vector<double> shape = normalisedShape(solve, mode);
            std::vector<double> stiffness(gSize, 0.0);
            std::vector<double> other(gSize, 0.0);
            made.system->kgg().multiply(shape, stiffness);
            second.multiply(shape, other);
            made.products[mode] = {std::move(stiffness), std::move(other)};
        }
    }
    return made;
}

void EigenItems::addFormulationShapeTerms(
    std::size_t quantity, const FieldPerturbation& perturbation,
    const std::array<const elements::FormulationModel*, FIELD_SIDE_COUNT>& models)
{
    for (Cluster& cluster : m_clusters)
    {
        SolveConstraints& made = constraintsOf(cluster.solve);
        if (!made.system)
            continue;
        const Solve& solve = m_solves[cluster.solve];
        const assembly::FormulationConstraints& formulation = *made.system->formulationConstraints();
        base::DiagnosticSink sink;
        const std::array<assembly::FormulationRelations, FIELD_SIDE_COUNT> sides =
            sideRelations(formulation, models, sink);
        const std::size_t m = cluster.modes.size();
        std::vector<std::vector<double>> derivatives; // D_a, one per mode of the cluster
        std::vector<std::vector<double>> residuals;   // R_a
        for (const std::size_t mode : cluster.modes)
        {
            derivatives.push_back(dependentDerivative(made.system->gm(), formulation.relations(), sides, perturbation,
                                                      normalisedShape(solve, mode)));
            const auto& [stiffness, other] = made.products.at(mode);
            // (K − λ·M)·φ or (K + λ·K_d)·φ.
            const double factor = solve.isBuckling ? cluster.root : -cluster.root;
            std::vector<double> residual(stiffness.size());
            for (std::size_t i = 0; i < residual.size(); ++i)
                residual[i] = stiffness[i] + factor * other[i];
            residuals.push_back(std::move(residual));
        }
        const auto dot = [](const std::vector<double>& a, const std::vector<double>& b)
        {
            double sum = 0.0;
            for (std::size_t i = 0; i < a.size(); ++i)
                sum += a[i] * b[i];
            return sum;
        };
        std::vector<double>& target = cluster.byQuantity[quantity];
        if (target.empty())
            target.assign(m * m, 0.0);
        for (std::size_t a = 0; a < m; ++a)
            for (std::size_t b = 0; b < m; ++b)
                target[a * m + b] += dot(derivatives[a], residuals[b]) + dot(derivatives[b], residuals[a]);
        if (!solve.isBuckling && m == 1)
        {
            const auto& [stiffness, mass] = made.products.at(cluster.modes.front());
            auto& [strain, kinetic] = cluster.energiesByQuantity[quantity];
            strain += 2.0 * dot(derivatives.front(), stiffness);
            kinetic += 2.0 * dot(derivatives.front(), mass);
        }
    }
}

double EigenItems::shapeValue(const Solve& solve, std::size_t mode, base::DofIndex dof) const
{
    return (*solve.shapes)(static_cast<std::size_t>(dof), mode) / (*solve.scales)[mode];
}

std::vector<base::DofIndex> EigenItems::dofsOf(std::span<const elements::ElementDof> dofs) const
{
    std::vector<base::DofIndex> result; // bounded by the element's DOFs
    result.reserve(dofs.size());
    for (const elements::ElementDof& dof : dofs)
        result.push_back(m_in.dofMap->dofOf(dof));
    return result;
}

void EigenItems::addProjection(const Cluster& cluster, std::span<const base::DofIndex> dofs,
                               std::span<const double> matrix, double factor, std::vector<double>& target) const
{
    const Solve& solve = m_solves[cluster.solve];
    const std::size_t m = cluster.modes.size();
    const std::size_t n = dofs.size();
    std::vector<double> shapes(m * n); // the cluster's normalised shapes on the element's DOFs, bounded by the element
    for (std::size_t a = 0; a < m; ++a)
        for (std::size_t r = 0; r < n; ++r)
            shapes[a * n + r] = shapeValue(solve, cluster.modes[a], dofs[r]);
    std::vector<double> product(n); // A·φ_b
    for (std::size_t b = 0; b < m; ++b)
    {
        for (std::size_t r = 0; r < n; ++r)
        {
            double sum = 0.0;
            for (std::size_t c = 0; c < n; ++c)
                sum += matrix[r * n + c] * shapes[b * n + c];
            product[r] = sum;
        }
        for (std::size_t a = 0; a < m; ++a)
        {
            double sum = 0.0;
            for (std::size_t r = 0; r < n; ++r)
                sum += shapes[a * n + r] * product[r];
            target[a * m + b] += factor * sum;
        }
    }
}

bool EigenItems::addProductWeights(const elements::ElementFamily& family, const elements::ElementBatchView& view,
                                   std::int64_t row, std::span<const Item* const> items,
                                   std::span<const elements::NodalDisplacements> shapes,
                                   std::vector<std::vector<std::pair<base::DofIndex, double>>>& weights) const
{
    for (std::size_t i = 0; i < items.size(); ++i)
    {
        elements::ElementVectorBlock block;
        if (!family.differentialStiffnessPreloadGradient(view, row, shapes[i], block))
            return false; // the family has no product: none of the items takes one
        // φ_normalised = column / scale: the product is quadratic in φ.
        const double scale = (*m_solves[items[i]->solve].scales)[items[i]->mode];
        for (std::size_t e = 0; e < block.size(); ++e)
        {
            const std::span<const elements::ElementDof> dofs = block.dofs(e);
            const std::span<const double> values = block.values(e);
            for (std::size_t k = 0; k < dofs.size(); ++k)
                if (values[k] != 0.0)
                    weights[i].emplace_back(m_in.dofMap->dofOf(dofs[k]), values[k] / (scale * scale));
        }
    }
    return true;
}

std::vector<std::pair<ItemKey, SparseVector>> EigenItems::bucklingWeights() const
{
    // The LAMA items whose cluster has one mode, and the modes they read.
    std::vector<const Item*> items; // bounded by the items
    for (const Item& item : m_items)
        if (item.type == ResponseType::BucklingFactor && m_clusters[item.cluster].modes.size() == 1)
            items.push_back(&item);
    std::vector<std::pair<ItemKey, SparseVector>> result;
    if (items.empty())
        return result;
    const assembly::DofMap& dofMap = *m_in.dofMap;
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    // ∂(φᵀ·K_d(u)·φ)/∂u_k = φᵀ·K_d(e_k)·φ: K_d is linear in u (the preload's internal forces are). A family that has
    // the product gives it in one pass per element and shape (differentialStiffnessPreloadGradient); for the others the
    // differential stiffness of a unit displacement of each DOF of an element gives the element's share exactly. The
    // temperatures are left out: they add a term of K_d that does not depend on u.
    const auto gridCount = static_cast<std::size_t>(dofMap.gridCount());
    const auto scalarCount = static_cast<std::size_t>(dofMap.scalarCount());
    base::BudgetedBuffer<double> unit(*m_in.budget, gSize, "LAMA weights unit displacements");
    std::fill(unit.data(), unit.data() + gSize, 0.0);
    const elements::NodalDisplacements displacements =
        elements::NodalDisplacements::ofGSet(unit.span(), gridCount, scalarCount, &dofMap.entities());
    // The shape of each item as displacements (the solve's columns are whole g-set vectors, not normalised).
    std::vector<elements::NodalDisplacements> shapes; // bounded by the items
    shapes.reserve(items.size());
    for (const Item* item : items)
    {
        const Solve& solve = m_solves[item->solve];
        OOS_ASSERT(solve.shapes->rows() == gSize);
        const std::span<const double> column(solve.shapes->data() + item->mode * solve.shapes->leadingDimension(),
                                             gSize);
        shapes.push_back(elements::NodalDisplacements::ofGSet(column, gridCount, scalarCount, &dofMap.entities()));
    }
    std::vector<std::vector<std::pair<base::DofIndex, double>>> weights(items.size());
    for (const assembly::AssemblyBatch& batch : m_in.set->batches())
    {
        const elements::ElementFamily& family = *batch.family;
        if (!family.facets().has(elements::Facet::Differential) || family.isDifferentiallyInert())
            continue;
        for (std::int64_t row = 0; row < batch.view.rowCount(); ++row)
        {
            if (addProductWeights(family, batch.view, row, items, shapes, weights))
                continue;
            elements::ElementMatrixBlock layout;
            family.computeDifferentialStiffness(batch.view, {row, row + 1}, displacements, layout);
            if (layout.empty())
                continue;
            const std::vector<elements::ElementDof> elementDofs(layout.dofs(0).begin(), layout.dofs(0).end());
            const std::vector<base::DofIndex> dofs = dofsOf(elementDofs);
            const std::size_t n = dofs.size();
            for (std::size_t k = 0; k < n; ++k)
            {
                // The DOF's place in the g set (a grid, scalar or entity DOF alike).
                const auto place = static_cast<std::size_t>(dofs[k]);
                unit[place] = 1.0;
                elements::ElementMatrixBlock block;
                family.computeDifferentialStiffness(batch.view, {row, row + 1}, displacements, block);
                unit[place] = 0.0;
                const std::span<const double> matrix = block.matrix(0);
                for (std::size_t i = 0; i < items.size(); ++i)
                {
                    const Solve& solve = m_solves[items[i]->solve];
                    double sum = 0.0;
                    for (std::size_t r = 0; r < n; ++r)
                    {
                        double product = 0.0;
                        for (std::size_t c = 0; c < n; ++c)
                            product += matrix[r * n + c] * shapeValue(solve, items[i]->mode, dofs[c]);
                        sum += shapeValue(solve, items[i]->mode, dofs[r]) * product;
                    }
                    if (sum != 0.0)
                        weights[i].emplace_back(dofs[k], sum);
                }
            }
        }
    }
    for (std::size_t i = 0; i < items.size(); ++i)
    {
        const double root = m_solves[items[i]->solve].roots[items[i]->mode];
        SparseVector vector = std::move(weights[i]);
        for (auto& entry : vector)
            entry.second *= root * root; // c = λ²·∇_u(φᵀ·K_d·φ)
        compressSparse(vector);
        result.emplace_back(items[i]->key, std::move(vector));
    }
    return result;
}

void EigenItems::projectElement(const elements::ElementFamily& family, const elements::FieldUse& use,
                                const ElementMatrixDerivative& stiffness, const MassOf& massOf,
                                const DifferentialOf& differentialOf, const TargetOf& targetOf,
                                const EnergiesOf& energiesOf)
{
    // The mass derivative per modal solve (its formulation), the differential stiffness at fixed u per preload.
    std::map<std::size_t, ElementMatrixDerivative> mass;
    std::map<std::int32_t, ElementMatrixDerivative> differential;
    const bool hasMass = use.changesMass && family.facets().has(elements::Facet::Mass);
    const bool hasDifferential = (use.changesStiffness || use.changesLoads) &&
                                 family.facets().has(elements::Facet::Differential) && !family.isDifferentiallyInert();
    std::vector<double> energy(1, 0.0);
    for (std::size_t c = 0; c < m_clusters.size(); ++c)
    {
        const Cluster& cluster = m_clusters[c];
        const Solve& solve = m_solves[cluster.solve];
        // A vibration root of one mode keeps its strain and kinetic energy rates apart (the Rayleigh quotient).
        const bool keepsEnergies = !solve.isBuckling && cluster.modes.size() == 1 && energiesOf;
        if (!stiffness.isEmpty())
        {
            addProjection(cluster, dofsOf(stiffness.dofs), stiffness.values, 1.0, targetOf(c));
            if (keepsEnergies)
            {
                energy[0] = 0.0;
                addProjection(cluster, dofsOf(stiffness.dofs), stiffness.values, 1.0, energy);
                energiesOf(c, energy[0], 0.0);
            }
        }
        if (!solve.isBuckling && hasMass)
        {
            auto [found, isNew] = mass.try_emplace(cluster.solve);
            if (isNew)
                found->second = massOf(solve.massOptions);
            if (!found->second.isEmpty())
            {
                addProjection(cluster, dofsOf(found->second.dofs), found->second.values,
                              -cluster.root * solve.weightToMass, targetOf(c));
                if (keepsEnergies)
                {
                    energy[0] = 0.0;
                    addProjection(cluster, dofsOf(found->second.dofs), found->second.values, solve.weightToMass,
                                  energy);
                    energiesOf(c, 0.0, energy[0]);
                }
            }
        }
        if (solve.isBuckling && hasDifferential)
        {
            const auto preload = m_preloads.find(solve.preloadSubcase);
            if (preload == m_preloads.end())
                // no remedy: the analysis reads the preload of every buckling solve; a missing one is a bug.
                throw base::OptimisationError(
                    base::concatenate({"the sensitivity of the load factors of subcase ", std::to_string(solve.subcase),
                                       ": the displacements of its preload subcase ",
                                       std::to_string(solve.preloadSubcase), " were not read"}));
            auto [found, isNew] = differential.try_emplace(solve.preloadSubcase);
            if (isNew)
                found->second = differentialOf(preload->second);
            if (!found->second.isEmpty())
                addProjection(cluster, dofsOf(found->second.dofs), found->second.values, cluster.root, targetOf(c));
        }
    }
}

void EigenItems::addElement(std::size_t quantity, const elements::ElementFamily& family, const elements::FieldUse& use,
                            const ElementMatrixDerivative& stiffness, const MassOf& massOf,
                            const DifferentialOf& differentialOf)
{
    if (m_items.empty())
        return;
    projectElement(
        family, use, stiffness, massOf, differentialOf,
        [&](std::size_t c) -> std::vector<double>&
        {
            const std::size_t m = m_clusters[c].modes.size();
            std::vector<double>& entry = m_clusters[c].byQuantity[quantity];
            entry.resize(m * m, 0.0);
            return entry;
        },
        [&](std::size_t c, double strain, double kinetic)
        {
            std::pair<double, double>& energies = m_clusters[c].energiesByQuantity[quantity];
            energies.first += strain;
            energies.second += kinetic;
        });
}

std::vector<std::pair<ItemKey, double>> EigenItems::elementDerivatives(const elements::ElementFamily& family,
                                                                       const elements::FieldUse& use,
                                                                       const ElementMatrixDerivative& stiffness,
                                                                       const MassOf& massOf,
                                                                       const DifferentialOf& differentialOf)
{
    std::vector<std::vector<double>> matrices(m_clusters.size());
    projectElement(
        family, use, stiffness, massOf, differentialOf,
        [&](std::size_t c) -> std::vector<double>&
        {
            const std::size_t m = m_clusters[c].modes.size();
            matrices[c].resize(m * m, 0.0);
            return matrices[c];
        },
        EnergiesOf{});
    std::vector<std::pair<ItemKey, double>> result;
    for (const Item& item : m_items)
    {
        const Cluster& cluster = m_clusters[item.cluster];
        const Solve& solve = m_solves[cluster.solve];
        const std::vector<double>& matrix = matrices[item.cluster];
        if (matrix.empty())
            continue;
        const double factor = (solve.isBuckling ? cluster.root : 1.0) * item.scale;
        const std::size_t m = cluster.modes.size();
        if (m == 1)
        {
            result.emplace_back(item.key, factor * matrix[0]);
            continue;
        }
        if (solve.isBuckling)
            throw base::OptimisationError(base::concatenate(
                {"the sensitivity map of the load factors of subcase ", std::to_string(solve.subcase),
                 ": the root of mode ", std::to_string(item.mode + 1), " is repeated (", std::to_string(m),
                 " modes), whose derivative along one element's value needs the preload derivatives of the "
                 "pairs of its modes, which this version does not form; map a single root"}));
        // One element's value is one direction: the derivatives of the repeated root are the eigenvalues of D_e.
        linalg::DenseMatrix dense(*m_in.budget, m, m, "repeated root map derivatives");
        for (std::size_t a = 0; a < m; ++a)
            for (std::size_t b = 0; b < m; ++b)
                dense(a, b) = 0.5 * (matrix[a * m + b] + matrix[b * m + a]);
        const linalg::SymmetricEigenDecomposition split =
            linalg::SmallEigenSolver::decompose(dense.view(), *m_in.budget);
        result.emplace_back(item.key, factor * split.eigenvalues[item.rank]);
    }
    return result;
}

std::int64_t EigenItems::addPreloadTerms(std::int32_t preloadSubcase, const SensitivitySystem& system,
                                         const PseudoLoads& loads, std::size_t subcase)
{
    std::vector<std::size_t> clusters; // the buckling clusters of several modes on this preload
    for (std::size_t c = 0; c < m_clusters.size(); ++c)
    {
        const Solve& solve = m_solves[m_clusters[c].solve];
        if (solve.isBuckling && solve.preloadSubcase == preloadSubcase && m_clusters[c].modes.size() > 1)
            clusters.push_back(c);
    }
    if (clusters.empty())
        return 0;
    std::vector<std::size_t> variables; // bounded by the design variables
    for (std::size_t j = 0; j < loads.variableCount(); ++j)
        if (loads.hasLoad(j, subcase))
            variables.push_back(j);
    const assembly::DofMap& dofMap = *m_in.dofMap;
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    const auto equations = static_cast<std::size_t>(system.equationCount());
    base::BudgetedBuffer<double> vectorG(*m_in.budget, gSize, "LAMA preload derivative");
    base::BudgetedBuffer<double> column(*m_in.budget, equations, "LAMA preload equations");
    std::int64_t solves = 0;
    for (std::size_t first = 0; first < variables.size(); first += SENSITIVITY_BLOCK_COLUMNS)
    {
        const std::size_t count = std::min(SENSITIVITY_BLOCK_COLUMNS, variables.size() - first);
        linalg::DenseMatrix rightHandSides(*m_in.budget, equations, count, "LAMA preload right-hand sides");
        for (std::size_t b = 0; b < count; ++b)
        {
            loads.assemble(variables[first + b], subcase, vectorG.span());
            system.toEquations(vectorG.span(), column.span());
            for (std::size_t e = 0; e < equations; ++e)
                rightHandSides(e, b) = column[e];
        }
        // du_j in every unit (the condensed path recovers them all: every element's K_d reads its displacements).
        if (equations > 0)
            system.solve(rightHandSides.view());
        solves += static_cast<std::int64_t>(count);
        for (std::size_t b = 0; b < count; ++b)
        {
            for (std::size_t e = 0; e < equations; ++e)
                column[e] = rightHandSides(e, b);
            system.fromEquations(column.span(), vectorG.span());
            // K_d(du_j), linear in the displacements, without the temperatures (they do not move with x here).
            const elements::NodalDisplacements derivative = elements::NodalDisplacements::ofGSet(
                vectorG.span(), static_cast<std::size_t>(dofMap.gridCount()),
                static_cast<std::size_t>(dofMap.scalarCount()), &dofMap.entities());
            const std::size_t j = variables[first + b];
            for (const assembly::AssemblyBatch& batch : m_in.set->batches())
            {
                const elements::ElementFamily& family = *batch.family;
                if (!family.facets().has(elements::Facet::Differential) || family.isDifferentiallyInert())
                    continue;
                elements::ElementMatrixBlock block;
                family.computeDifferentialStiffness(batch.view, elements::wholeBatch(batch.view), derivative, block);
                for (std::size_t e = 0; e < block.size(); ++e)
                {
                    const std::vector<base::DofIndex> dofs = dofsOf(block.dofs(e));
                    for (const std::size_t c : clusters)
                    {
                        Cluster& cluster = m_clusters[c];
                        std::vector<double>& target = cluster.preloadByVariable[j];
                        target.resize(cluster.modes.size() * cluster.modes.size(), 0.0);
                        addProjection(cluster, dofs, block.matrix(e), 1.0, target);
                    }
                }
            }
        }
    }
    return solves;
}

void EigenItems::addExplicitParts(const std::vector<FieldValue>& quantities, LinearisedItems& linearised) const
{
    for (const Item& item : m_items)
    {
        const Cluster& cluster = m_clusters[item.cluster];
        const Solve& solve = m_solves[cluster.solve];
        // dλ = D (vibration), λ·D (buckling, D = Φᵀ·(∂K + λ·dK_d)·Φ): the factor of the cluster's matrices.
        const double factor = (solve.isBuckling ? cluster.root : 1.0) * item.scale;
        VariableRow& row = linearised.at(item.key).explicitPart;
        const std::size_t m = cluster.modes.size();
        if (m == 1)
        {
            for (const auto& [quantity, matrix] : cluster.byQuantity)
                for (const auto& [column, gradient] : quantities[quantity].gradient)
                    row.emplace_back(column, factor * matrix[0] * gradient);
            continue;
        }
        // A repeated root: D_j for each variable, its eigenvalues sorted, the rank-th to the item.
        for (std::size_t j = 0; j < m_in.variableCount; ++j)
        {
            std::vector<double> matrix(m * m, 0.0);
            bool isNonZero = false;
            for (const auto& [quantity, byQuantity] : cluster.byQuantity)
                for (const auto& [column, gradient] : quantities[quantity].gradient)
                    if (column == j)
                    {
                        for (std::size_t k = 0; k < m * m; ++k)
                            matrix[k] += gradient * byQuantity[k];
                        isNonZero = true;
                    }
            if (const auto preload = cluster.preloadByVariable.find(j); preload != cluster.preloadByVariable.end())
            {
                for (std::size_t k = 0; k < m * m; ++k)
                    matrix[k] += cluster.root * preload->second[k];
                isNonZero = true;
            }
            if (!isNonZero)
                continue;
            linalg::DenseMatrix dense(*m_in.budget, m, m, "repeated root derivatives");
            for (std::size_t a = 0; a < m; ++a)
                for (std::size_t b = 0; b < m; ++b)
                    dense(a, b) = 0.5 * (matrix[a * m + b] + matrix[b * m + a]);
            const linalg::SymmetricEigenDecomposition split =
                linalg::SmallEigenSolver::decompose(dense.view(), *m_in.budget);
            // The eigenvalues increase; λ·μ keeps their order for a positive λ and reverses it for a negative one, as
            // the modes are ordered by the magnitude of their roots.
            const double value = factor * split.eigenvalues[item.rank];
            if (value != 0.0)
                row.emplace_back(j, value);
        }
    }
}

std::vector<RayleighGradient> EigenItems::rayleighParts(
    const std::vector<FieldValue>& quantities,
    const std::map<ItemKey, std::pair<std::int32_t, std::size_t>>& wanted) const
{
    std::vector<RayleighGradient> parts;
    for (const Item& item : m_items)
    {
        const Cluster& cluster = m_clusters[item.cluster];
        const Solve& solve = m_solves[cluster.solve];
        const auto found = wanted.find(item.key);
        if (solve.isBuckling || cluster.modes.size() != 1 || found == wanted.end())
            continue;
        RayleighGradient part;
        part.responseId = found->second.first;
        part.item = found->second.second;
        part.root = cluster.root;
        part.numerator.assign(m_in.variableCount, 0.0);
        part.denominator.assign(m_in.variableCount, 0.0);
        for (const auto& [quantity, energies] : cluster.energiesByQuantity)
            for (const auto& [column, gradient] : quantities[quantity].gradient)
            {
                part.numerator[column] += energies.first * gradient;
                part.denominator[column] += energies.second * gradient;
            }
        parts.push_back(std::move(part));
    }
    return parts;
}

} // namespace oos::design::detail
