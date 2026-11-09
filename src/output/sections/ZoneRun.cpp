#include "oos/output/sections/ZoneRun.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::output
{

namespace
{

// The components of a grid point in ZoneContour's per-point layout (elementForces): three forces, three moments.
constexpr std::size_t COMPONENTS_PER_POINT = 6;

// The four row blocks of a zone's data.
enum Block : std::size_t
{
    Displacements = 0,
    Forces = 1,
    Loads = 2,
    Equilibrium = 3
};

// out (rows × k) = Σ of the products of `parts` (each a rows × g SparseRows with its g × k vectors; a vectors view of
// no column, or rows of no row, adds nothing).
void applySum(std::initializer_list<std::pair<const SparseRows*, linalg::ConstDenseMatrixView>> parts,
              linalg::DenseMatrixView out, base::MemoryBudget& budget, base::ThreadPool* pool)
{
    for (std::size_t j = 0; j < out.columns; ++j)
        for (std::size_t r = 0; r < out.rows; ++r)
            out(r, j) = 0.0;
    linalg::DenseMatrix part(budget, out.rows, out.columns, "zone contour data part");
    for (const auto& [rows, vectors] : parts)
    {
        if (rows->rowCount() == 0 || vectors.columns == 0)
            continue;
        rows->apply(vectors, part.view(), pool);
        for (std::size_t j = 0; j < out.columns; ++j)
            for (std::size_t r = 0; r < out.rows; ++r)
                out(r, j) += part(r, j);
    }
}

} // namespace

double ZoneTable::equilibriumResidual(std::size_t c) const
{
    if (!hasValues)
        throw std::invalid_argument(
            "ZoneTable::equilibriumResidual: the values of the zone were not kept; run with keep_zone_values");
    const double* values = equilibrium.data() + c * EQUILIBRIUM_ROWS;
    // The gross magnitudes Σ|F| of the contour forces and Σ|P| of the interior loads, translations and rotations.
    std::array<double, 2> gross{};
    const auto addGross = [&](const base::BudgetedBuffer<double>& data, const std::vector<std::int32_t>& components)
    {
        const std::size_t rows = components.size();
        for (std::size_t r = 0; r < rows; ++r)
            gross[components[r] <= 3 ? 0 : 1] += std::abs(data[c * rows + r]);
    };
    addGross(forces, contourComponents);
    addGross(loads, interiorComponents);
    const std::array<double, 2> scales{gross[0], gross[1] + referenceLength * gross[0]};
    double largest = 0.0;
    for (const std::size_t kind : {std::size_t{0}, std::size_t{1}})
    {
        const std::size_t first = 3 * kind;
        double contour = 0.0;
        double interior = 0.0;
        double residual = 0.0;
        for (std::size_t k = 0; k < 3; ++k)
        {
            contour += values[first + k] * values[first + k];
            interior += values[6 + first + k] * values[6 + first + k];
            const double sum = values[first + k] + values[6 + first + k];
            residual += sum * sum;
        }
        const double scale = std::max({std::sqrt(contour), std::sqrt(interior), scales[kind]});
        if (scale > 0.0)
            largest = std::max(largest, std::sqrt(residual) / scale);
    }
    return largest;
}

ZoneRun::ZoneRun(const ZoneRequest& request, std::span<const ZoneSubcaseConstraint> analysisSubcases,
                 const input::ModelSnapshot& snapshot, const assembly::ElementSet& set, const assembly::DofMap& dofMap,
                 std::size_t caseCount, base::MemoryBudget& budget, base::DiagnosticSink& sink)
    : m_dofMap(&dofMap),
      m_caseCount(caseCount),
      m_keepsValues(request.keepsValues),
      m_keepsBasis(request.keepsBasis),
      m_maximumBatchCases(request.maximumBatchCases)
{
    const std::span<const ZoneSubcaseConstraint> subcases =
        request.contourSubcases.empty() ? analysisSubcases
                                        : std::span<const ZoneSubcaseConstraint>(request.contourSubcases);
    base::DiagnosticSink problems;
    m_contour = std::make_unique<ZoneContour>(request.zones, subcases, snapshot, set, dofMap, budget, problems);
    for (const base::Diagnostic& diagnostic : problems.diagnostics())
        sink.report(diagnostic);
    if (problems.hasErrors())
        return;
    for (std::size_t z = 0; z < m_contour->zones().size(); ++z)
    {
        const ZoneContour::Zone& zone = m_contour->zones()[z];
        const ZoneDefinition& definition = m_contour->definitions()[z];
        (void)m_contour->rows(z, budget);
        ZoneTable table;
        table.id = definition.id;
        table.name = definition.name;
        table.group = definition.group;
        table.frameId = definition.frameId;
        table.report = zone.report;
        for (const ZoneContour::ZoneDof& dof : zone.contourDofs)
        {
            table.contourGrids.push_back(dof.grid);
            table.contourComponents.push_back(dof.component);
        }
        for (const ZoneContour::ZoneDof& dof : zone.interiorDofs)
        {
            table.interiorGrids.push_back(dof.grid);
            table.interiorComponents.push_back(dof.component);
        }
        for (const std::vector<ZoneContour::ZonePoint>* points : {&zone.contour, &zone.interior})
            for (const ZoneContour::ZonePoint& point : *points)
                table.referenceLength = std::max(table.referenceLength, base::norm(point.arm));
        if (m_keepsValues)
        {
            table.displacements =
                base::BudgetedBuffer<double>(budget, table.contourDofCount() * caseCount, "zone contour displacements");
            table.forces =
                base::BudgetedBuffer<double>(budget, table.contourDofCount() * caseCount, "zone contour forces");
            table.loads = base::BudgetedBuffer<double>(budget, table.interiorDofCount() * caseCount, "zone loads");
            table.equilibrium = base::BudgetedBuffer<double>(budget, EQUILIBRIUM_ROWS * caseCount, "zone equilibrium");
            table.hasValues = true;
        }
        m_tables.push_back(std::move(table));
    }
    m_offered.assign(m_tables.size(), 0);
}

ZoneRun::BasisValues ZoneRun::basisValues(const ZoneVectors& vectors, base::MemoryBudget& budget,
                                          base::ThreadPool* pool) const
{
    BasisValues basis;
    const std::size_t k = vectors.displacements.columns;
    for (std::size_t z = 0; z < m_tables.size(); ++z)
    {
        const ZoneContour::ZoneRows& rows = m_contour->builtRows(z);
        std::array<linalg::DenseMatrix, 4> values{
            linalg::DenseMatrix(budget, rows.displacements.rowCount(), k, "zone contour displacements of the basis"),
            linalg::DenseMatrix(budget, rows.forces.rowCount(), k, "zone contour forces of the basis"),
            linalg::DenseMatrix(budget, rows.loads.rowCount(), k, "zone loads of the basis"),
            linalg::DenseMatrix(budget, EQUILIBRIUM_ROWS, k, "zone equilibrium of the basis")};
        applySum({{&rows.displacements, vectors.displacements}}, values[Displacements].view(), budget, pool);
        applySum({{&rows.forces, vectors.displacements}, {&rows.forceConstraints, vectors.mpcForces}},
                 values[Forces].view(), budget, pool);
        applySum({{&rows.loads, vectors.applied}}, values[Loads].view(), budget, pool);
        applySum({{&rows.equilibriumDisplacements, vectors.displacements},
                  {&rows.equilibriumConstraints, vectors.mpcForces},
                  {&rows.equilibriumLoads, vectors.applied},
                  {&rows.equilibriumReactions, vectors.spcForces}},
                 values[Equilibrium].view(), budget, pool);
        basis.zones.push_back(std::move(values));
    }
    return basis;
}

void ZoneRun::keepBasis(std::int32_t constraintSet, std::span<const std::int32_t> cases, const BasisValues& basis,
                        const linalg::DenseMatrix& coefficients, bool hasThermalCases, base::MemoryBudget& budget)
{
    if (!m_keepsBasis)
        return;
    const auto copy = [&](const linalg::DenseMatrix& source, const char* label)
    {
        linalg::DenseMatrix target(budget, source.rows(), source.columns(), label);
        for (std::size_t j = 0; j < source.columns(); ++j)
            for (std::size_t i = 0; i < source.rows(); ++i)
                target(i, j) = source(i, j);
        return target;
    };
    for (std::size_t z = 0; z < m_tables.size(); ++z)
    {
        ZoneBasisSet set;
        set.constraintSet = constraintSet;
        set.cases.assign(cases.begin(), cases.end());
        set.displacements = copy(basis.zones[z][Displacements], "zone basis contour displacements");
        set.forces = copy(basis.zones[z][Forces], "zone basis contour forces");
        set.loads = copy(basis.zones[z][Loads], "zone basis loads");
        set.coefficients = copy(coefficients, "zone basis coefficients");
        set.hasThermalCases = hasThermalCases;
        m_tables[z].basis.push_back(std::move(set));
    }
}

void ZoneRun::store(std::int32_t subcase, std::size_t z, std::span<const double> displacements,
                    std::span<const double> forces, std::span<const double> loads, std::span<const double> equilibrium)
{
    ZoneTable& table = m_tables[z];
    const std::size_t c = m_offered[z];
    if (c >= m_caseCount)
        throw std::invalid_argument(
            base::concatenate({"ZoneRun: expected at most ", std::to_string(m_caseCount),
                               " cases, found one more (subcase ", std::to_string(subcase), ")"}));
    table.cases.push_back(subcase);
    ++m_offered[z];
    if (!table.hasValues)
        return;
    std::copy(displacements.begin(), displacements.end(), table.displacements.data() + c * table.contourDofCount());
    std::copy(forces.begin(), forces.end(), table.forces.data() + c * table.contourDofCount());
    std::copy(loads.begin(), loads.end(), table.loads.data() + c * table.interiorDofCount());
    std::copy(equilibrium.begin(), equilibrium.end(), table.equilibrium.data() + c * EQUILIBRIUM_ROWS);
}

void ZoneRun::offerBatch(const BasisValues& basis, const ResultCombination& combination, std::size_t first,
                         std::span<const std::int32_t> cases, const TemperaturesOf& temperaturesOf,
                         base::MemoryBudget& budget)
{
    if (m_tables.empty() || cases.empty())
        return;
    const std::size_t size = m_maximumBatchCases > 0 ? std::min(m_maximumBatchCases, cases.size()) : cases.size();
    for (std::size_t start = 0; start < cases.size(); start += size)
    {
        const std::size_t count = std::min(size, cases.size() - start);
        // The temperatures of the cases of the batch, made once for every zone.
        std::vector<std::unique_ptr<assembly::TemperatureField>> temperatures(count);
        if (temperaturesOf)
            for (std::size_t j = 0; j < count; ++j)
                temperatures[j] = temperaturesOf(start + j);
        for (std::size_t z = 0; z < m_tables.size(); ++z)
        {
            std::array<linalg::DenseMatrix, 4> values;
            for (std::size_t b = 0; b < 4; ++b)
            {
                values[b] = linalg::DenseMatrix(budget, basis.zones[z][b].rows(), count, "zone data of a batch");
                if (values[b].rows() > 0)
                    combination.combine(basis.zones[z][b].view(), first + start, count, values[b].view(), budget);
            }
            for (std::size_t j = 0; j < count; ++j)
            {
                if (temperatures[j])
                {
                    const ZoneContour::ThermalPart part = m_contour->thermalPart(z, temperatures[j]->temperatures());
                    for (std::size_t r = 0; r < part.forces.size(); ++r)
                        values[Forces](r, j) += part.forces[r];
                    for (std::size_t r = 0; r < EQUILIBRIUM_ROWS; ++r)
                        values[Equilibrium](r, j) += part.equilibrium[r];
                }
                const auto column = [&](std::size_t b) -> std::span<const double>
                {
                    if (values[b].rows() == 0)
                        return {};
                    return {&values[b](0, j), values[b].rows()};
                };
                store(cases[start + j], z, column(Displacements), column(Forces), column(Loads), column(Equilibrium));
            }
        }
    }
}

void ZoneRun::offerCase(std::int32_t subcase, const ZoneVectors& vectors,
                        const elements::ElementTemperatures* temperatures, base::MemoryBudget& budget,
                        std::span<const ZoneGivenForces> given)
{
    if (m_tables.empty())
        return;
    BasisValues values = basisValues(vectors, budget, nullptr);
    for (std::size_t z = 0; z < m_tables.size(); ++z)
    {
        std::array<linalg::DenseMatrix, 4>& zone = values.zones[z];
        const ZoneGivenForces* givenForces = nullptr;
        for (const ZoneGivenForces& entry : given)
            if (entry.zoneId == m_tables[z].id)
                givenForces = &entry;
        if (givenForces != nullptr)
        {
            // The unit's boundary forces at the contour points, six per point in CD (ZoneContour::elementForces'
            // layout), then turned to the contour DOFs and their resultant as the element part is.
            const std::vector<ZoneContour::ZonePoint>& contour = m_contour->zones()[z].contour;
            std::vector<double> perPoint(COMPONENTS_PER_POINT * contour.size(), 0.0); // bounded by the contour
            for (std::size_t k = 0; k < givenForces->dofs.size(); ++k)
            {
                const linalg::NodeDof nodeDof = m_dofMap->nodeDofOf(givenForces->dofs[k]);
                for (std::size_t c = 0; c < contour.size(); ++c)
                    if (contour[c].gridRow == nodeDof.node)
                    {
                        perPoint[COMPONENTS_PER_POINT * c + static_cast<std::size_t>(nodeDof.component)] +=
                            givenForces->forces[k];
                        break;
                    }
            }
            const std::vector<double> forces = m_contour->toContourDofs(z, perPoint);
            for (std::size_t r = 0; r < forces.size(); ++r)
                zone[Forces](r, 0) = forces[r];
            const std::array<double, 6> resultant = m_contour->contourResultant(z, perPoint);
            for (std::size_t r = 0; r < resultant.size(); ++r)
                zone[Equilibrium](r, 0) = resultant[r];
        }
        if (temperatures != nullptr)
        {
            const ZoneContour::ThermalPart part = m_contour->thermalPart(z, *temperatures);
            for (std::size_t r = 0; r < part.forces.size(); ++r)
                zone[Forces](r, 0) += part.forces[r];
            for (std::size_t r = 0; r < EQUILIBRIUM_ROWS; ++r)
                zone[Equilibrium](r, 0) += part.equilibrium[r];
        }
        if (givenForces != nullptr && givenForces->hasInteriorReactions)
        {
            // The interior resultant (its loads and the reactions of the constraints it holds) balances the contour's:
            // the unit's equilibrium (the reactions inside a unit that is not recovered are known as totals only).
            // Without interior reactions the interior resultant is that of its loads, known on the g set.
            constexpr std::size_t HALF = EQUILIBRIUM_ROWS / 2;
            for (std::size_t r = 0; r < HALF; ++r)
                zone[Equilibrium](HALF + r, 0) = -zone[Equilibrium](r, 0);
        }
        const auto column = [&](std::size_t b) -> std::span<const double>
        {
            if (zone[b].rows() == 0)
                return {};
            return {&zone[b](0, 0), zone[b].rows()};
        };
        store(subcase, z, column(Displacements), column(Forces), column(Loads), column(Equilibrium));
    }
}

std::vector<ZoneTable> ZoneRun::takeTables()
{
    return std::move(m_tables);
}

} // namespace oos::output
