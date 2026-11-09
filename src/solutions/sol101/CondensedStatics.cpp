#include "oos/solutions/sol101/CondensedStatics.hpp"

#include "oos/assembly/assemble/MassAssembler.hpp"
#include "oos/assembly/assemble/NonStructuralMassShares.hpp"
#include "oos/assembly/assemble/ParamLookup.hpp"
#include "oos/assembly/constraints/FastenerConstraints.hpp"
#include "oos/assembly/constraints/GlueConstraints.hpp"
#include "oos/assembly/constraints/MpcEquations.hpp"
#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/assembly/dofs/DofSets.hpp"
#include "oos/base/errors/CacheError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/InternalError.hpp"
#include "oos/base/errors/MemoryBudgetError.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/base/memory/KeptObject.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"
#include "oos/linalg/eigen/BucklingPencil.hpp"
#include "oos/linalg/eigen/SparsePencil.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"
#include "oos/linalg/sparse/SparsePattern.hpp"
#include "oos/linalg/supernodal/PartialFactor.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"
#include "oos/reduction/absorption/SpiderWeights.hpp"
#include "oos/reduction/cache/CacheHasher.hpp"
#include "oos/reduction/cache/CacheKey.hpp"
#include "oos/reduction/condense/PlannedUnits.hpp"
#include "oos/reduction/condense/StaticCondensation.hpp"
#include "oos/reduction/condense/UnitElementIndex.hpp"
#include "oos/reduction/condense/UnitSystem.hpp"
#include "oos/reduction/interface/BoundaryBasis.hpp"
#include "oos/reduction/interface/InterfaceProblem.hpp"
#include "oos/reduction/patterns/UnitLoadBasis.hpp"
#include "oos/reduction/scheduling/CondensedMemoryPlan.hpp"
#include "oos/reduction/scheduling/FactorRetention.hpp"
#include "oos/reduction/scheduling/InterfaceFootprint.hpp"
#include "oos/reduction/scheduling/UnitFootprint.hpp"
#include "oos/reduction/scheduling/UnitScheduler.hpp"
#include "oos/solutions/common/CommonConstraints.hpp"
#include "oos/solutions/common/RunChecks.hpp"
#include "oos/solutions/sol101/StaticLoadCases.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace oos::solutions
{

namespace
{

using assembly::LoadFamily;

constexpr std::size_t RIGID_BODY_MODE_COUNT = 6;
// The cases recovered together: their vectors (the interface's displacements, loads, right-hand sides, residuals,
// solutions, reactions, applied and all loads: INTERFACE_BATCH_VECTORS; each unit's loads and applied loads:
// UNIT_BATCH_VECTORS; a recovered unit's solution, loads and K·u on its equations, RECOVERED_EQUATION_VECTORS, and its
// enforced values, displacements and reactions) are held for the batch, so the batch is bounded by the budget left
// beside the result tables to come (BATCH_BUDGET_FRACTION of it: the rest for what the recovery of a case allocates
// beside the batch, its tables and their temporaries) and by this count, which keeps the combinations in blocks the
// dense kernels take well.
constexpr std::size_t LARGEST_CASE_BATCH = 64;
constexpr double BATCH_BUDGET_FRACTION = 0.25;
constexpr std::size_t INTERFACE_BATCH_VECTORS = 8;
constexpr std::uint64_t UNIT_BATCH_VECTORS = 2;
constexpr std::uint64_t RECOVERED_EQUATION_VECTORS = 3;
constexpr std::uint64_t RECOVERED_LOCAL_VECTORS = 2; // its enforced values and displacements
// The IDs listed in a diagnostic of the outputs left out (the count says the rest).
constexpr std::size_t LISTED_ID_COUNT = 20;
// The Gm a constraint set makes beside the full path's: the interface's and the outputs' (no larger than it).
constexpr std::uint64_t SET_GM_COPIES = 2;
// The case vectors of a batch on the g set (solveGroup): displacements, applied loads, loads, SPC forces.
constexpr std::uint64_t CASE_BATCH_G_VECTORS = 4;
// The share of the bytes free a wave of units of the recovery may take beside a batch (solveGroup): the rest
// for the result tables of the batch's cases and their temporaries.
constexpr double WAVE_BUDGET_FRACTION = 0.5;

// Whether two output plans select the same points and elements for every kind (what the outputs left out of a unit
// not recovered depend on).
bool isSameSelection(const output::OutputPlan& a, const output::OutputPlan& b)
{
    for (std::size_t k = 0; k < output::OUTPUT_KIND_COUNT; ++k)
    {
        const auto kind = static_cast<output::OutputKind>(k);
        const output::OutputRequestPlan& first = a.of(kind);
        const output::OutputRequestPlan& second = b.of(kind);
        if (first.isRequested != second.isRequested)
            return false;
        if (first.isRequested && (first.isAll != second.isAll || (!first.isAll && first.members != second.members)))
            return false;
    }
    return true;
}

// The seconds of the phases of the record (CondensedRunRecord, CondensedSetRecord): each lap the time since the last.
class PhaseClock
{
public:
    double lap() noexcept
    {
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - m_last).count();
        m_last = now;
        return seconds;
    }

private:
    std::chrono::steady_clock::time_point m_last = std::chrono::steady_clock::now();
};

base::Diagnostic problem(std::string code, std::string card, base::EntityId id, std::string message,
                         base::Severity severity = base::Severity::Error)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = severity;
    diagnostic.code = std::move(code);
    diagnostic.card = std::move(card);
    diagnostic.id = id;
    diagnostic.message = std::move(message);
    return diagnostic;
}

std::string listOf(const std::vector<std::int32_t>& ids)
{
    std::string text;
    for (std::size_t k = 0; k < ids.size() && k < LISTED_ID_COUNT; ++k)
        text.append(k == 0 ? "" : ", ").append(std::to_string(ids[k]));
    if (ids.size() > LISTED_ID_COUNT)
        text.append(" and ").append(std::to_string(ids.size() - LISTED_ID_COUNT)).append(" more");
    return text;
}

// The coefficients of some cases (columns) of a coefficient matrix.
reduction::BasisCoefficients columnsOf(const reduction::BasisCoefficients& coefficients,
                                       std::span<const std::size_t> cases, base::MemoryBudget& budget)
{
    linalg::DenseMatrix alpha(budget, coefficients.size(), cases.size(), "condensed group coefficients");
    for (std::size_t c = 0; c < cases.size(); ++c)
        for (std::size_t i = 0; i < coefficients.size(); ++i)
            alpha(i, c) = coefficients(i, cases[c]);
    return reduction::BasisCoefficients(std::move(alpha));
}

// x = A⁻¹·b for a small symmetric positive definite A (n ≤ 6), by Cholesky (Golub and Van Loan, Algorithm 4.2.1);
// false when A is not positive definite to round-off.
bool solveSmallSpd(std::vector<double> a, std::size_t n, std::vector<double>& b)
{
    constexpr double PIVOT_RATIO = 1.0e-14;
    double largest = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        largest = std::max(largest, a[i * n + i]);
    for (std::size_t j = 0; j < n; ++j)
    {
        double pivot = a[j * n + j];
        for (std::size_t k = 0; k < j; ++k)
            pivot -= a[j * n + k] * a[j * n + k];
        if (!(pivot > PIVOT_RATIO * largest))
            return false;
        const double root = std::sqrt(pivot);
        a[j * n + j] = root;
        for (std::size_t i = j + 1; i < n; ++i)
        {
            double value = a[i * n + j];
            for (std::size_t k = 0; k < j; ++k)
                value -= a[i * n + k] * a[j * n + k];
            a[i * n + j] = value / root;
        }
    }
    const std::size_t columns = b.size() / n;
    for (std::size_t c = 0; c < columns; ++c)
    {
        double* x = b.data() + c * n;
        for (std::size_t i = 0; i < n; ++i)
        {
            for (std::size_t k = 0; k < i; ++k)
                x[i] -= a[i * n + k] * x[k];
            x[i] /= a[i * n + i];
        }
        for (std::size_t i = n; i-- > 0;)
        {
            for (std::size_t k = i + 1; k < n; ++k)
                x[i] -= a[k * n + i] * x[k];
            x[i] /= a[i * n + i];
        }
    }
    return true;
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// The condensed units

namespace
{

// One condensed unit of the run.
struct CondensedUnit
{
    std::size_t planIndex = 0;
    std::int64_t unitId = 0;
    reduction::UnitDescription description;
    std::shared_ptr<const reduction::UnitSystem> system;
    reduction::UnitLoadBasis basis;
    std::size_t loadPatternCount = 0; // the basis; the inertial patterns follow
    std::size_t patternCount = 0;     // with the six inertial patterns of inertia relief
    bool hasEnforcedPatterns = false;
    // The patterns on the local DOFs: the load patterns on the interior rows (the boundary rows are the interface's),
    // then, with inertia relief, the six inertial patterns −M·Φ_r on every row.
    linalg::DenseMatrix patternLoads;
    linalg::DenseMatrix patternEnforced;                         // no column when no pattern enforces a displacement
    std::vector<std::uint8_t> isThermalPattern;                  // per pattern (bounded by the unit's basis)
    std::shared_ptr<reduction::StaticCondensation> condensation; // shared with the kept sets
    // What the cache keeps of the condensation (the reduced unit, shared with `condensation`, and the pattern work
    // G = F_oᵀ·K_oo⁻¹·F_o over every pattern): the cached entry on a hit, this run's on a miss.
    std::shared_ptr<const reduction::CachedReduction> reduction;
    bool isRecovered = false;
    // Its key in the condensation cache (with a cache), and its factorisation's options (the namer of its DOFs).
    std::optional<reduction::CacheKey> key;
    linalg::FactorOptions factorOptions;
    // The hash of its element IDs (reduction::unitIdentityOf, the unit across analyses), and the interior factor an
    // earlier analysis kept for its equations, taken for its condensation (until it is made).
    std::uint64_t identity = 0;
    std::optional<reduction::KeptInteriorFactor> keptFactor;
    // Its estimated footprint, and whether FactorRetention kept its interior factor for the recovery.
    reduction::UnitFootprint footprint;
    bool isFactorKept = false;
    // The memory mode: its equations released after its condensation (CondensedMemoryPlan), and the system without them
    // it keeps then (its boundary data), which it takes back after each recovery.
    bool releasesEquations = false;
    std::shared_ptr<const reduction::UnitSystem> lightSystem;
    // Per local DOF: its g-set DOF (-1 for a generated point) and whether its point is interior.
    base::BudgetedBuffer<base::DofIndex> modelDofOfLocal;
    base::BudgetedBuffer<std::uint8_t> isInteriorLocal;
    // The held DOFs of the unit at its boundary points with a stiffness (their SPC force needs the unit recovered), and
    // the boundary points its relations act on (their MPC forces need it): left out of the outputs when not recovered.
    std::vector<base::EntityId> pointsNeedingRecovery; // bounded by the unit's boundary
    // Craig-Bampton: its modal part (reduceModes) and its fixed-interface modes while needed.
    std::shared_ptr<const reduction::ModalReduction> modal;
    std::shared_ptr<const reduction::CachedRecoveryData> modes;

    [[nodiscard]] reduction::UnitPatterns patterns() const
    {
        return {patternLoads.view(), patternEnforced.view(), basis.patternStamp()};
    }
    [[nodiscard]] const linalg::DenseMatrix& work() const noexcept { return reduction->patternWork; }
};

// The value of a PARAM as the cache key keeps it ("" when the model does not give it), the Bulk Data or global one (the
// units are condensed once for every subcase).
std::string parameterValue(const input::ParamTable* params, std::string_view name)
{
    if (params == nullptr)
        return {};
    const std::optional<input::ParamValue> value = params->find(name);
    if (!value)
        return {};
    switch (value->kind)
    {
    case input::ParamValue::Kind::Integer:
        return std::to_string(value->integer);
    case input::ParamValue::Kind::Real:
    {
        // Every bit of the value: 17 significant digits round-trip a double.
        std::array<char, 32> text{};
        std::snprintf(text.data(), text.size(), "%.17g", value->real);
        return std::string(text.data());
    }
    case input::ParamValue::Kind::Text:
        return std::string(value->text);
    }
    return {};
}

// The design stamp of a unit of the plan: those of the units O_o_Reader gives that it is made of (its member units),
// mixed in their order; empty when no design variable drives any of them.
std::uint64_t designStampOf(const reduction::UnitPlan& plan, std::span<const UnitDesignStamp> stamps)
{
    reduction::CacheHasher hasher;
    bool isDriven = false;
    for (const std::int64_t member :
         plan.memberUnits.empty() ? std::vector<std::int64_t>{plan.unitId} : plan.memberUnits)
        for (const UnitDesignStamp& stamp : stamps)
            if (stamp.unitId == member && stamp.stamp != 0)
            {
                isDriven = true;
                hasher.mix(member);
                hasher.mix(stamp.stamp);
            }
    if (!isDriven)
        return 0;
    // A driven unit's stamp is never the empty one.
    return hasher.value() == 0 ? 1 : hasher.value();
}

// Whether a unit's system numbers its local DOFs as `modelDofOfLocal` does (the numbering its patterns were made in:
// reduction::unitLocalModelDofs before its system, UnitSystem::numberPoints).
bool isSameNumbering(std::span<const base::DofIndex> modelDofOfLocal, const reduction::UnitSystem& system)
{
    if (static_cast<std::size_t>(system.localDofCount()) != modelDofOfLocal.size())
        return false;
    for (std::size_t local = 0; local < modelDofOfLocal.size(); ++local)
        if (system.modelDofOf(static_cast<std::int64_t>(local)).value_or(-1) != modelDofOfLocal[local])
            return false;
    return true;
}

// The names of a unit's equations for the diagnostics of its factorisations (FactorOptions::equationNamer).
linalg::EquationNamer equationNamerOf(const reduction::UnitSystem* system)
{
    return [system](std::int32_t equation)
    {
        const reduction::UnitDofLabel label = system->labelOf(system->localDofOfEquation(equation));
        return base::GridComponent{label.id, label.component};
    };
}

// The families of a component (a unit, or the interface) with the rows that load it.
struct FamilyFlags
{
    LoadFamily family;
    std::vector<std::uint8_t> flags; // bounded by the family's compact rows
};

std::vector<FamilyFlags> familiesFor(const StaticLoadCases& cases,
                                     const std::function<bool(LoadFamily, std::int64_t)>& isRow)
{
    std::vector<FamilyFlags> families;
    for (const LoadFamily family : assembly::ALL_LOAD_FAMILIES)
    {
        const StaticLoadCases::FamilyDescription* description = cases.familyDescription(family);
        if (description == nullptr)
            continue;
        FamilyFlags entry{family, {}};
        if (family != LoadFamily::Inertial && family != LoadFamily::Thermal)
        {
            entry.flags.resize(static_cast<std::size_t>(description->rowCount));
            for (std::int64_t row = 0; row < description->rowCount; ++row)
                entry.flags[static_cast<std::size_t>(row)] = isRow(family, row) ? 1 : 0;
        }
        families.push_back(std::move(entry));
    }
    return families;
}

reduction::UnitLoadBasis chooseBasis(const StaticLoadCases& cases, const std::vector<FamilyFlags>& families,
                                     std::size_t caseCount, base::MemoryBudget& budget, base::ThreadPool* pool)
{
    std::vector<reduction::UnitFamilyLoads> described; // bounded by the families
    for (const FamilyFlags& entry : families)
    {
        const StaticLoadCases::FamilyDescription& description = *cases.familyDescription(entry.family);
        reduction::UnitFamilyLoads familyLoads;
        familyLoads.family = entry.family;
        familyLoads.cases = description.isCombinationOfSets ? nullptr : &description.cases;
        familyLoads.isCombinationOfSets = description.isCombinationOfSets;
        familyLoads.loadSets = description.loadSets ? &*description.loadSets : nullptr;
        familyLoads.setCoefficients = description.setFactors ? &*description.setFactors : nullptr;
        familyLoads.offersLoadedDofs = description.offersLoadedDofs;
        familyLoads.dofOfRow = description.offersLoadedDofs ? description.dofOfRow : std::span<const base::DofIndex>{};
        familyLoads.tolerance = description.tolerance;
        familyLoads.isUnitRow = entry.flags;
        described.push_back(familyLoads);
    }
    return reduction::UnitLoadBasis::choose(described, caseCount, budget, pool);
}

// The node of a point of the model (grid rows, then scalar points), or nothing.
std::optional<std::int32_t> nodeOfPoint(const assembly::DofMap& dofMap, base::EntityId nodeId)
{
    const std::optional<input::PointRef> point = dofMap.points().find(nodeId);
    if (!point)
        return std::nullopt;
    return point->kind == input::PointKind::Grid ? point->row : dofMap.gridCount() + point->row;
}

// The DOFs of a node of the model: six for a grid point, one for a scalar point.
std::int64_t dofCountOfNode(const assembly::DofMap& dofMap, std::int32_t node)
{
    constexpr std::int64_t GRID_DOFS = 6;
    return node < dofMap.gridCount() ? GRID_DOFS : 1;
}

// The nonzero g-set entries of one basis vector: its DOFs, loads and enforced values.
struct SparseVector
{
    std::vector<base::DofIndex> dofs; // bounded by the loads of one load set
    std::vector<double> loads;
    std::vector<double> enforced;
};

// The entries of a basis vector whose family makes its vectors columns of its compact matrix
// (StaticLoadCases::FamilyDescription): a concentrated load (a loaded row, a case, a load set as written) and the unit
// enforced displacement of a loaded row; nothing for the others (distributed, inertial, thermal loads, the enforced
// displacements of a case), which StaticLoadCases::familyVector makes on the g set. These are the values familyVector
// writes, without its passes over the g set (a pass per vector cost the patterns of a thousand units times
// the g set): the compact matrix sums the same entries in the same order (StaticLoadCases::addFamily, its
// ColumnBuilder) and leaves out the exact zeros, which the g-set vector holds as +0.
std::optional<SparseVector> sparseVectorOf(const StaticLoadCases& cases, LoadFamily family,
                                           const assembly::BasisVector& source)
{
    using Source = assembly::BasisVector::Source;
    const bool isLoadedRow = source.source == Source::LoadedRow;
    if (family != LoadFamily::Concentrated && !(family == LoadFamily::Enforced && isLoadedRow))
        return std::nullopt;
    const StaticLoadCases::FamilyDescription* description = cases.familyDescription(family);
    if (description == nullptr)
        return std::nullopt;
    SparseVector vector;
    const auto add = [&](std::int64_t row, double value)
    {
        vector.dofs.push_back(description->dofOfRow[static_cast<std::size_t>(row)]);
        vector.loads.push_back(family == LoadFamily::Enforced ? 0.0 : value);
        vector.enforced.push_back(family == LoadFamily::Enforced ? value : 0.0);
    };
    if (isLoadedRow)
    {
        add(source.index, 1.0);
        return vector;
    }
    const assembly::CompactLoadMatrix* matrix = nullptr;
    if (source.source == Source::Case)
        matrix = &description->cases;
    else if (description->loadSets)
        matrix = &*description->loadSets;
    if (matrix == nullptr)
        return std::nullopt;
    const std::span<const std::int64_t> rows = matrix->rowsOf(source.index);
    const std::span<const double> values = matrix->valuesOf(source.index);
    for (std::size_t k = 0; k < rows.size(); ++k)
        add(rows[k], values[k]);
    return vector;
}

} // namespace

struct CondensedStatics::Units
{
    std::unique_ptr<CommonConstraints> common;
    std::unique_ptr<reduction::PlannedUnits> planned;
    std::unique_ptr<assembly::MpcEquations> commonMpc;
    std::unique_ptr<assembly::FastenerConstraints> fasteners;
    std::unique_ptr<assembly::GlueConstraints> commonGlue;
    std::shared_ptr<assembly::RigidElimination> unitRelations;
    std::shared_ptr<reduction::UnitElementIndex> elementIndex;
    std::shared_ptr<assembly::NonStructuralMassShares> shares;
    assembly::MassAssemblyOptions massOptions;
    base::BudgetedBuffer<std::uint8_t> isHeldEverywhere;
    // What every unit's equations, factorisation and cache key share.
    reduction::UnitModel unitModel;
    linalg::FactorOptions unitFactorOptions;
    std::vector<reduction::CacheParameter> cacheParameters; // the MATRIX_PARAMETERS and ModelParams OFFDEF, by name
    std::string kernelName;
    std::vector<CondensedUnit> units;                       // the condensed units, by plan order (bounded by the units)
    std::vector<reduction::InterfaceSpider> spiders;        // the absorbed fastener sides of the condensed units
    base::BudgetedBuffer<base::EntityId> interfaceElements; // sorted
    base::BudgetedBuffer<base::EntityId> excludedElements;  // the elements of the units not recovered, sorted
    bool isEveryUnitRecovered = true;
    std::vector<base::EntityId> pointsNeedingRecovery; // of the units not recovered, sorted
    // Per unit of the plan (by plan index): a condensed unit that is recovered (step 7). A lookup, not a search of
    // the units per point or per row (10 000 units times the model's points, per subcase).
    std::vector<std::uint8_t> isRecoveredPlanUnit; // bounded by the units
    // What the outputs may take when some unit is not recovered (outputPlanOf), made once in step 7: the points whose
    // displacements are known, those whose forces are, and the elements of no unit left unrecovered; sorted.
    base::BudgetedBuffer<base::EntityId> knownPoints;
    base::BudgetedBuffer<base::EntityId> forcePoints;
    base::BudgetedBuffer<base::EntityId> knownElements;
};

struct CondensedStatics::Loads
{
    std::vector<const StaticSubcase*> cases; // the solved subcases, in the order of the load cases
    std::vector<std::size_t> caseOfSubcase;  // per subcase of the run: its load case, or SIZE_MAX
    std::unique_ptr<linalg::NodeGraph> massGraph;
    std::unique_ptr<linalg::SparsePattern> massPattern;
    std::unique_ptr<linalg::BlockSparseMatrix> mass;
    std::unique_ptr<StaticLoadCases> loadCases;
    reduction::UnitLoadBasis interfaceBasis;
    // The interface's vectors on the g set, by vector (CSR on the interface points): their loads and enforced values.
    base::BudgetedBuffer<std::int64_t> offsets;
    base::BudgetedBuffer<base::DofIndex> dofs;
    base::BudgetedBuffer<double> loadValues;
    base::BudgetedBuffer<double> enforcedValues;
    bool hasEnforced = false;
    std::vector<std::uint8_t> isThermalVector; // bounded by the interface's basis
    // Two g-set vectors the load vectors of the patterns are made in, one at a time (makeVector), zero elsewhere.
    base::BudgetedBuffer<double> loadsG;
    base::BudgetedBuffer<double> enforcedG;
    // The DOFs the vector in them wrote when it was made from its entries (sorted), or, when isDense, the whole g set
    // (made by StaticLoadCases::familyVector, or not cleared yet).
    std::vector<base::DofIndex> touched; // bounded by the loads of one vector
    bool isDense = true;

    // Makes `vector` in loadsG and enforcedG, the vector before it cleared: from its entries where its family gives
    // them (sparseVectorOf: no pass over the g set), on the whole g set otherwise.
    void makeVector(const StaticLoadCases& cases, const reduction::UnitBasisVector& vector, base::MemoryBudget& budget,
                    base::DiagnosticSink& sink);
    void clearVector();
    // Visits, in increasing order, the DOFs where the vector made has a load or an enforced value.
    template <typename Visit>
    void forEachNonzero(Visit&& visit) const
    {
        const auto isNonzero = [&](std::size_t dof) { return loadsG[dof] != 0.0 || enforcedG[dof] != 0.0; };
        if (isDense)
        {
            for (std::size_t dof = 0; dof < loadsG.size(); ++dof)
                if (isNonzero(dof))
                    visit(dof);
            return;
        }
        for (const base::DofIndex dof : touched)
            if (isNonzero(static_cast<std::size_t>(dof)))
                visit(static_cast<std::size_t>(dof));
    }
};

void CondensedStatics::Loads::clearVector()
{
    if (isDense)
    {
        std::fill(loadsG.data(), loadsG.data() + loadsG.size(), 0.0);
        std::fill(enforcedG.data(), enforcedG.data() + enforcedG.size(), 0.0);
        isDense = false;
    }
    else
        for (const base::DofIndex dof : touched)
        {
            loadsG[static_cast<std::size_t>(dof)] = 0.0;
            enforcedG[static_cast<std::size_t>(dof)] = 0.0;
        }
    touched.clear();
}

void CondensedStatics::Loads::makeVector(const StaticLoadCases& cases, const reduction::UnitBasisVector& vector,
                                         base::MemoryBudget& budget, base::DiagnosticSink& sink)
{
    clearVector();
    if (const std::optional<SparseVector> sparse = sparseVectorOf(cases, vector.family, vector.source))
    {
        for (std::size_t k = 0; k < sparse->dofs.size(); ++k)
        {
            const auto dof = static_cast<std::size_t>(sparse->dofs[k]);
            loadsG[dof] = sparse->loads[k];
            enforcedG[dof] = sparse->enforced[k];
            touched.push_back(sparse->dofs[k]);
        }
        std::sort(touched.begin(), touched.end());
        return;
    }
    cases.familyVector(vector.family, vector.source, loadsG.span(), enforcedG.span(), budget, sink);
    isDense = true;
}

struct CondensedStatics::Memory
{
    std::unique_ptr<reduction::CondensedMemoryPlan> plan;
    std::unique_ptr<reduction::UnitScheduler> scheduler;
    reduction::FactorRetention retention;
};

CondensedStatics::~CondensedStatics() = default;

void CondensedStatics::offerInteriorFactors(const std::shared_ptr<base::MemoryBudget>& analysisBudget)
{
    if (m_inputs.keptFactors == nullptr || m_inputs.keepsSystems || m_inputs.isCraigBampton || !m_units)
        return;
    for (CondensedUnit& unit : m_units->units)
    {
        if (!unit.condensation || !unit.condensation->isInteriorFactorKept())
            continue;
        std::optional<reduction::KeptInteriorFactor> kept =
            unit.condensation->giveInteriorFactor(unit.identity, analysisBudget);
        if (kept)
            m_inputs.keptFactors->offer(std::move(*kept));
    }
}

CondensedStatics::CondensedStatics(const Inputs& inputs, base::MemoryBudget& budget, base::DiagnosticSink& sink)
    : m_inputs(inputs),
      m_units(std::make_unique<Units>()),
      m_loads(std::make_unique<Loads>()),
      m_memory(std::make_unique<Memory>())
{
    if (inputs.snapshot == nullptr || inputs.model == nullptr || inputs.plan == nullptr || inputs.loads == nullptr ||
        inputs.kernels == nullptr || inputs.subcases.size() != inputs.spcSets.size() ||
        inputs.plans.size() != inputs.subcases.size())
        throw std::invalid_argument(
            "CondensedStatics: the inputs need the snapshot, the analysis model, the plan, the load builder, the "
            "kernels, and one SPC set and output plan per subcase");
    if (inputs.cache != nullptr)
        m_cacheSession.emplace(inputs.cache->openSession(inputs.cacheModelTag));
    for (const StaticSubcase& subcase : inputs.subcases)
        if (subcase.nonStructuralMass)
            throw base::SolverError(
                "selected non-structural mass sets require the full static path; disable condensation");
    condenseUnits(inputs, budget, sink);
}

void CondensedStatics::condenseUnits(const Inputs& inputs, base::MemoryBudget& budget, base::DiagnosticSink& sink)
{
    const input::ModelSnapshot& snapshot = *inputs.snapshot;
    const AnalysisModel& model = *inputs.model;
    const assembly::DofMap& dofMap = model.dofMap();
    const assembly::ElementSet& elementSet = model.elementSet();
    const input::ParamTable* params = model.params();
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    Units& units = *m_units;
    Loads& loads = *m_loads;
    PhaseClock clock;

    // 1. What every subcase shares.
    for (const std::size_t s : inputs.solved)
        loads.cases.push_back(&inputs.subcases[s]);
    loads.caseOfSubcase.assign(inputs.subcases.size(), static_cast<std::size_t>(-1));
    for (std::size_t c = 0; c < inputs.solved.size(); ++c)
        loads.caseOfSubcase[inputs.solved[c]] = c;
    std::vector<StaticSubcase> solvedSubcases; // bounded by the subcases
    for (const std::size_t s : inputs.solved)
        solvedSubcases.push_back(inputs.subcases[s]);
    units.common =
        std::make_unique<CommonConstraints>(CommonConstraints::of(snapshot, dofMap, solvedSubcases, budget, sink));
    const CommonConstraints& common = *units.common;
    units.isHeldEverywhere = base::BudgetedBuffer<std::uint8_t>(budget, gSize, "condensed DOFs held everywhere");
    for (std::size_t dof = 0; dof < gSize; ++dof)
        units.isHeldEverywhere[dof] = common.isHeldEverywhere(static_cast<base::DofIndex>(dof)) ? 1 : 0;

    // 2. The relations of the units and their descriptions (their equations are built in the waves of step 5).
    const input::ConstraintTable* constraints =
        snapshot.hasTable(input::TableRole::Spc) ? &snapshot.constraints() : nullptr;
    const input::RigidTable* rigidElements =
        snapshot.hasTable(input::TableRole::RigidElements) ? &snapshot.rigidElements() : nullptr;
    const input::FastenerPiercingTable* piercings =
        snapshot.hasTable(input::TableRole::FastenerPiercings) ? &snapshot.fastenerPiercings() : nullptr;
    units.commonMpc = std::make_unique<assembly::MpcEquations>(constraints, dofMap, common.commonMpcSet(),
                                                               common.commonMpcCards, budget, sink);
    units.fasteners = std::make_unique<assembly::FastenerConstraints>(piercings, dofMap, budget, sink);
    std::vector<base::DofIndex> heldEverywhere; // bounded by the constrained DOFs
    for (std::size_t dof = 0; dof < gSize; ++dof)
        if (units.isHeldEverywhere[dof] != 0)
            heldEverywhere.push_back(static_cast<base::DofIndex>(dof));
    const assembly::GlueContext glueContext{
        rigidElements, units.commonMpc.get(), units.fasteners.get(), heldEverywhere, {}};
    units.commonGlue = std::make_unique<assembly::GlueConstraints>(
        snapshot, common.commonContact, solvedSubcases.empty() ? 1 : solvedSubcases.front().label, dofMap, glueContext,
        budget, sink);
    stopOnErrors(sink, "the condensation");
    // The points the elements of no condensed unit reach (the DOFs of their matrices): a point of a condensed unit
    // among them is on its boundary even when the plan's boundary misses it (PlannedUnits), reported.
    std::vector<base::EntityId> reachedPoints; // bounded by the points of the interface elements
    {
        std::vector<base::EntityId> condensedElements; // bounded by the plan's lists
        for (const reduction::UnitPlan& unit : inputs.plan->units())
            if (unit.decision == reduction::UnitDecision::Condensed)
                condensedElements.insert(condensedElements.end(), unit.elementIds.begin(), unit.elementIds.end());
        std::sort(condensedElements.begin(), condensedElements.end());
        elements::ElementMatrixBlock block;
        constexpr std::int64_t RANGE_ELEMENTS = 2048;
        for (const assembly::AssemblyBatch& batch : elementSet.batches())
        {
            const elements::ElementFamily& family = *batch.family;
            const bool hasStiffness = family.facets().has(elements::Facet::Stiffness);
            if (!hasStiffness && !family.facets().has(elements::Facet::Mass))
                continue;
            for (std::int64_t first = 0; first < batch.view.rowCount(); first += RANGE_ELEMENTS)
            {
                const elements::ElementRange range{first, std::min(batch.view.rowCount(), first + RANGE_ELEMENTS)};
                bool isAnyInterface = false;
                for (std::int64_t row = range.first; row < range.last && !isAnyInterface; ++row)
                    isAnyInterface = !std::binary_search(condensedElements.begin(), condensedElements.end(),
                                                         batch.view.elementId(row));
                if (!isAnyInterface)
                    continue;
                block.clear();
                if (hasStiffness)
                    family.computeStiffness(batch.view, range, block);
                else
                    family.computeMass(batch.view, range, elements::MassOptions{}, block);
                for (std::size_t e = 0; e < block.size(); ++e)
                {
                    if (std::binary_search(condensedElements.begin(), condensedElements.end(),
                                           batch.view.elementId(block.elementRow(e))))
                        continue;
                    const std::span<const double> matrix = block.matrix(e);
                    const std::span<const elements::ElementDof> dofs = block.dofs(e);
                    for (std::size_t i = 0; i < dofs.size(); ++i)
                    {
                        // A DOF with no term (the rotations of the grid points of a fastener's pierced element)
                        // reaches nothing.
                        const bool isZero =
                            std::all_of(matrix.begin() + static_cast<std::ptrdiff_t>(i * dofs.size()),
                                        matrix.begin() + static_cast<std::ptrdiff_t>((i + 1) * dofs.size()),
                                        [](double value) { return value == 0.0; });
                        if (!isZero)
                            reachedPoints.push_back(dofMap.pointIdOf(dofMap.dofOf(dofs[i])));
                    }
                }
            }
        }
        std::sort(reachedPoints.begin(), reachedPoints.end());
        reachedPoints.erase(std::unique(reachedPoints.begin(), reachedPoints.end()), reachedPoints.end());
    }
    units.planned = std::make_unique<reduction::PlannedUnits>(*inputs.plan, dofMap, budget, reachedPoints);
    const reduction::PlannedUnits& planned = *units.planned;
    // The external units are units of the interface problem: their zones lie in units kept full.
    if (inputs.externalUnits != nullptr)
    {
        const std::vector<base::EntityId>& replaced = inputs.externalUnits->replacedElements();
        for (const reduction::UnitPlan& unit : inputs.plan->units())
        {
            if (unit.decision != reduction::UnitDecision::Condensed)
                continue;
            const auto found =
                std::find_if(unit.elementIds.begin(), unit.elementIds.end(), [&](base::EntityId elementId)
                             { return std::binary_search(replaced.begin(), replaced.end(), elementId); });
            if (found != unit.elementIds.end())
                sink.report(
                    problem("external-unit-condensed", "UNIT", static_cast<base::EntityId>(unit.unitId),
                            base::concatenate({"unit ", std::to_string(unit.unitId), " is condensed and holds element ",
                                               std::to_string(*found),
                                               ", which an external unit replaces: an external unit is a unit of the "
                                               "interface problem of its own; keep the units of its zone full (the "
                                               "automatic path choice does)"})));
        }
        stopOnErrors(sink, "the external units");
    }
    for (const std::size_t index : planned.condensedUnits())
    {
        const std::span<const base::EntityId> promoted = planned.promotedPoints(index);
        if (promoted.empty())
            continue;
        const std::vector<std::int32_t> list(promoted.begin(), promoted.end());
        sink.report(problem("boundary-promoted", "UNIT",
                            static_cast<base::EntityId>(inputs.plan->units()[index].unitId),
                            base::concatenate({"unit ", std::to_string(inputs.plan->units()[index].unitId),
                                               ": the points ", listOf(list),
                                               " are reached by elements of the interface but missing from the plan's "
                                               "boundary (the connection tables do not give them); they are put on "
                                               "the boundary, so that no coupling is lost"}),
                            base::Severity::Warning));
    }
    // The units' Gm: its diagnostics are those of the full path's (a cycle, a DOF dependent twice), reported by the
    // constraint sets; here a problem only stops the condensation early.
    base::DiagnosticSink relationSink;
    units.unitRelations = std::make_shared<assembly::RigidElimination>(rigidElements, units.commonMpc.get(), dofMap,
                                                                       budget, relationSink, units.fasteners.get(),
                                                                       units.commonGlue.get(), planned.unitRowFilter());
    if (relationSink.hasErrors())
    {
        for (const base::Diagnostic& diagnostic : relationSink.diagnostics())
            sink.report(diagnostic);
        stopOnErrors(sink, "the condensation");
    }
    units.elementIndex = std::make_shared<reduction::UnitElementIndex>(elementSet, budget);
    units.shares = std::make_shared<assembly::NonStructuralMassShares>(elementSet);
    units.massOptions = assembly::MassAssemblyOptions::fromParams(params, std::nullopt, sink);
    reduction::UnitModel& unitModel = units.unitModel;
    unitModel.elements = &elementSet;
    unitModel.dofMap = &dofMap;
    unitModel.elementIndex = units.elementIndex.get();
    unitModel.unitRelations = units.unitRelations.get();
    unitModel.isConstrainedEverywhere = units.isHeldEverywhere.span();
    unitModel.nonStructuralMass = units.shares.get();
    unitModel.massOptions = units.massOptions;
    unitModel.singularity = assembly::SingularityOptions::fromParams(params, std::nullopt, sink);
    unitModel.keepsMass = inputs.isCraigBampton;
    const reduction::SpiderWeights weights(*units.unitRelations, dofMap, budget);
    for (const std::size_t index : planned.condensedUnits())
    {
        CondensedUnit unit;
        unit.planIndex = index;
        unit.unitId = inputs.plan->units()[index].unitId;
        unit.description = planned.describe(index, weights, piercings);
        for (const reduction::UnitSpider& spider : unit.description.spiders)
            if (spider.key.side != reduction::FastenerSide::None)
                units.spiders.push_back(
                    reduction::InterfaceSpider{spider.key, spider.relation.weights, spider.relation.legDofs,
                                               spider.relation.referenceDofs, spider.generatedPointPosition});
        units.units.push_back(std::move(unit));
    }
    // The elements of the interface: those of no condensed unit.
    std::size_t interfaceCount = 0;
    for (const assembly::AssemblyBatch& batch : elementSet.batches())
        for (std::int64_t row = 0; row < batch.view.rowCount(); ++row)
            interfaceCount += planned.isCondensedElement(batch.view.elementId(row)) ? 0 : 1;
    units.interfaceElements = base::BudgetedBuffer<base::EntityId>(budget, interfaceCount, "interface elements");
    std::size_t nextElement = 0;
    for (const assembly::AssemblyBatch& batch : elementSet.batches())
        for (std::int64_t row = 0; row < batch.view.rowCount(); ++row)
            if (!planned.isCondensedElement(batch.view.elementId(row)))
                units.interfaceElements[nextElement++] = batch.view.elementId(row);
    std::sort(units.interfaceElements.data(), units.interfaceElements.data() + interfaceCount);

    // 3. The load cases of every solved subcase, with the mass of the inertial loads.
    bool needsMass = false;
    for (const StaticSubcase* subcase : loads.cases)
        needsMass = needsMass || (subcase->load && inputs.loads->needsMass(*subcase->load));
    if (needsMass)
    {
        // The inertial loads are Mgg times an acceleration field (LoadVectorBuilder): the mass of the model on the
        // pattern of its elements. TODO(9.2): the units' own masses (M_U·a) in place of Mgg, so that the condensed path
        // holds no model-sized matrix; acceptable until then: Mgg is the size of the model's stiffness pattern, below
        // the units' factors, and accounted in the budget before the memory plan (its baseline).
        std::vector<linalg::NodeGroupList> groups{model.connectivity().groups()};
        // The contour points of each external unit coupled: its M* is a clique there.
        if (inputs.externalUnits != nullptr && inputs.externalUnits->couplings().groupCount() > 0)
            groups.push_back(inputs.externalUnits->couplings());
        loads.massGraph = std::make_unique<linalg::NodeGraph>(dofMap.nodeCount(), groups, budget, inputs.pool);
        loads.massPattern = std::make_unique<linalg::SparsePattern>(*loads.massGraph, model.nodeDofCounts(), budget);
        loads.mass = std::make_unique<linalg::BlockSparseMatrix>(*loads.massPattern, budget, "condensed mass Mgg");
        assembly::MassAssembler(elementSet, dofMap, *units.shares, units.massOptions)
            .assemble(*loads.mass, inputs.pool, inputs.cancellation);
        // The reduced mass of the external units, as given (unscaled by WTMASS, as on the full path).
        if (inputs.externalUnits != nullptr)
            inputs.externalUnits->addMass(*loads.mass);
    }
    StaticLoadCases::Inputs caseInputs;
    caseInputs.snapshot = &snapshot;
    caseInputs.points = &model.index().points();
    caseInputs.dofMap = &dofMap;
    caseInputs.loads = inputs.loads;
    caseInputs.cases = loads.cases;
    for (const std::size_t s : inputs.solved)
        caseInputs.spcs.push_back(&inputs.spcSets[s]);
    caseInputs.mass = loads.mass.get();
    caseInputs.tolerance = inputs.tolerance;
    caseInputs.pool = inputs.pool;
    caseInputs.kernel = inputs.kernels->kind();
    caseInputs.keepsFamilies = true;
    loads.loadCases = std::make_unique<StaticLoadCases>(std::move(caseInputs), budget, sink);
    stopOnErrors(sink, "the loads");
    const StaticLoadCases& cases = *loads.loadCases;
    const std::size_t caseCount = loads.cases.size();
    loads.loadsG = base::BudgetedBuffer<double>(budget, gSize, "condensed vector loads");
    loads.enforcedG = base::BudgetedBuffer<double>(budget, gSize, "condensed vector enforced values");

    // 3b. Each unit's load basis, on the rows that load its interior, before its equations: the rows are known from the
    // plan (its interior points, its elements), and a DOF it holds is a DOF of one of its points held in every subcase
    // that no relation inside it makes dependent (its role Constrained in UnitSystem::assignRoles).
    const auto nodeOfDof = [&](base::DofIndex dof) { return dofMap.nodeDofOf(dof).node; };
    base::BudgetedBuffer<std::uint8_t> isUnitDependent(budget, gSize, "condensed units' dependent DOFs");
    for (const base::DofIndex dof : units.unitRelations->mDofs())
        isUnitDependent[static_cast<std::size_t>(dof)] = 1;
    for (CondensedUnit& unit : units.units)
    {
        const reduction::UnitPlan& plan = inputs.plan->units()[unit.planIndex];
        const std::span<const base::EntityId> unitElements = plan.elementIds;
        const std::span<const base::EntityId> unitPoints = unit.description.nodeIds;
        const auto isUnitNode = [&](std::int32_t node)
        {
            const std::optional<std::size_t> owner = planned.unitOfInteriorNode(node);
            return owner && *owner == unit.planIndex;
        };
        const std::vector<FamilyFlags> families = familiesFor(
            cases,
            [&](LoadFamily family, std::int64_t row)
            {
                const StaticLoadCases::FamilyDescription& description = *cases.familyDescription(family);
                if (family == LoadFamily::Concentrated)
                    return isUnitNode(nodeOfDof(description.dofOfRow[static_cast<std::size_t>(row)]));
                if (family == LoadFamily::Enforced)
                {
                    const base::DofIndex dof = description.dofOfRow[static_cast<std::size_t>(row)];
                    return units.isHeldEverywhere[static_cast<std::size_t>(dof)] != 0 &&
                           isUnitDependent[static_cast<std::size_t>(dof)] == 0 &&
                           std::find(unitPoints.begin(), unitPoints.end(), dofMap.pointIdOf(dof)) != unitPoints.end();
                }
                // Distributed: the unit's elements, or a PointsPressureLoad face with a point inside it.
                const StaticLoadCases::DistributedRowPlace& place = description.places[static_cast<std::size_t>(row)];
                if (place.elementId != 0)
                    return std::binary_search(unitElements.begin(), unitElements.end(), place.elementId);
                return std::any_of(place.faceGrids.begin(), place.faceGrids.end(),
                                   [&](std::int32_t nodeId)
                                   {
                                       const std::optional<std::int32_t> node =
                                           nodeId != 0 ? nodeOfPoint(dofMap, nodeId) : std::nullopt;
                                       return node && isUnitNode(*node);
                                   });
            });
        unit.basis = chooseBasis(cases, families, caseCount, budget, inputs.pool);
        unit.loadPatternCount = unit.basis.size();
        unit.patternCount = unit.loadPatternCount + (inputs.hasInertiaRelief ? RIGID_BODY_MODE_COUNT : 0);
        for (const reduction::UnitBasisVector& vector : unit.basis.vectors())
            unit.hasEnforcedPatterns = unit.hasEnforcedPatterns || vector.family == LoadFamily::Enforced;
    }
    decideRecoveredUnits(inputs);

    // 4. The memory plan: refused here, before any unit is condensed, when the budget cannot hold the analysis (only
    // recorded at the MemoryPlan stage, which stops here).
    planMemory(inputs, budget);
    for (const reduction::UnitPlan& plan : inputs.plan->units())
        if (plan.decision != reduction::UnitDecision::Condensed)
            m_record.fullUnits.push_back(plan.unitId);
    m_record.preparationSeconds = clock.lap();
    if (inputs.stage == CondensedStage::MemoryPlan)
        return;

    // 5. The units, wave by wave (the Condensation stage stops after them; an eigenvalue analysis goes on to the
    // outputs of the units it does not recover, step 7).
    condenseInWaves(inputs, budget, sink);
    m_record.condensationSeconds = clock.lap();
    if (inputs.stage == CondensedStage::Condensation && !inputs.isCraigBampton)
        return;

    // 6. The interface's own loads: the rows on its points (and every distributed row of an element touching them).
    const auto isInteriorNode = [&](std::int32_t node)
    { return planned.interiorNodes()[static_cast<std::size_t>(node)] != 0; };
    std::vector<base::EntityId> boundaryElements; // the condensed units' elements with a boundary point
    for (const CondensedUnit& unit : units.units)
    {
        const std::span<const base::EntityId> elements = unit.system->boundaryElementIds();
        boundaryElements.insert(boundaryElements.end(), elements.begin(), elements.end());
    }
    std::sort(boundaryElements.begin(), boundaryElements.end());
    const std::vector<FamilyFlags> interfaceFamilies = familiesFor(
        cases,
        [&](LoadFamily family, std::int64_t row)
        {
            const StaticLoadCases::FamilyDescription& description = *cases.familyDescription(family);
            if (family == LoadFamily::Concentrated || family == LoadFamily::Enforced)
                return !isInteriorNode(nodeOfDof(description.dofOfRow[static_cast<std::size_t>(row)]));
            const StaticLoadCases::DistributedRowPlace& place = description.places[static_cast<std::size_t>(row)];
            if (place.elementId != 0)
                return !planned.isCondensedElement(place.elementId) ||
                       std::binary_search(boundaryElements.begin(), boundaryElements.end(), place.elementId);
            return std::any_of(place.faceGrids.begin(), place.faceGrids.end(),
                               [&](std::int32_t nodeId)
                               {
                                   const std::optional<std::int32_t> node =
                                       nodeId != 0 ? nodeOfPoint(dofMap, nodeId) : std::nullopt;
                                   return node && !isInteriorNode(*node);
                               });
        });
    loads.interfaceBasis = chooseBasis(cases, interfaceFamilies, caseCount, budget, inputs.pool);
    m_record.interfaceLoadBases = loads.interfaceBasis.reports();
    const std::size_t vectorCount = loads.interfaceBasis.size();
    // Two passes: the nonzeros of each vector on the interface points, then the values.
    loads.offsets = base::BudgetedBuffer<std::int64_t>(budget, vectorCount + 1, "interface vectors");
    for (const reduction::UnitBasisVector& vector : loads.interfaceBasis.vectors())
        loads.hasEnforced = loads.hasEnforced || vector.family == LoadFamily::Enforced;
    std::size_t nonzeros = 0;
    for (std::size_t k = 0; k < vectorCount; ++k)
    {
        const reduction::UnitBasisVector& vector = loads.interfaceBasis.vectors()[k];
        loads.makeVector(cases, vector, budget, sink);
        loads.offsets[k] = static_cast<std::int64_t>(nonzeros);
        loads.forEachNonzero(
            [&](std::size_t dof)
            {
                if (!isInteriorNode(nodeOfDof(static_cast<base::DofIndex>(dof))))
                    ++nonzeros;
            });
        loads.isThermalVector.push_back(vector.family == LoadFamily::Thermal ? 1 : 0);
    }
    loads.offsets[vectorCount] = static_cast<std::int64_t>(nonzeros);
    loads.dofs = base::BudgetedBuffer<base::DofIndex>(budget, nonzeros, "interface vectors");
    loads.loadValues = base::BudgetedBuffer<double>(budget, nonzeros, "interface vectors");
    loads.enforcedValues = base::BudgetedBuffer<double>(budget, loads.hasEnforced ? nonzeros : 0, "interface vectors");
    std::size_t next = 0;
    for (std::size_t k = 0; k < vectorCount; ++k)
    {
        const reduction::UnitBasisVector& vector = loads.interfaceBasis.vectors()[k];
        loads.makeVector(cases, vector, budget, sink);
        loads.forEachNonzero(
            [&](std::size_t dof)
            {
                if (isInteriorNode(nodeOfDof(static_cast<base::DofIndex>(dof))))
                    return;
                loads.dofs[next] = static_cast<base::DofIndex>(dof);
                loads.loadValues[next] = loads.loadsG[dof];
                if (loads.hasEnforced)
                    loads.enforcedValues[next] = loads.enforcedG[dof];
                ++next;
            });
    }
    m_solvedVectorCount += static_cast<std::int64_t>(vectorCount);
    stopOnErrors(sink, "the loads");
    // The g-set vectors of the patterns are no longer needed: the case vectors of the constraint sets take their room.
    loads.loadsG = base::BudgetedBuffer<double>();
    loads.enforcedG = base::BudgetedBuffer<double>();
    loads.touched.clear();
    loads.isDense = true;

    // 7. The outputs of the units not recovered.
    reportUnrecoveredOutputs(inputs, sink);
    std::size_t excludedCount = 0;
    for (const CondensedUnit& unit : units.units)
        if (!unit.isRecovered)
        {
            excludedCount += unit.description.elementIds.size();
            units.isEveryUnitRecovered = false;
            units.pointsNeedingRecovery.insert(units.pointsNeedingRecovery.end(), unit.pointsNeedingRecovery.begin(),
                                               unit.pointsNeedingRecovery.end());
        }
    std::sort(units.pointsNeedingRecovery.begin(), units.pointsNeedingRecovery.end());
    units.excludedElements = base::BudgetedBuffer<base::EntityId>(budget, excludedCount, "unrecovered elements");
    std::size_t nextExcluded = 0;
    for (const CondensedUnit& unit : units.units)
        if (!unit.isRecovered)
            for (const base::EntityId elementId : unit.description.elementIds)
                units.excludedElements[nextExcluded++] = elementId;
    std::sort(units.excludedElements.data(), units.excludedElements.data() + excludedCount);
    for (std::size_t u = 0; u < units.units.size(); ++u)
        m_record.units[u].isRecovered = units.units[u].isRecovered;
    units.isRecoveredPlanUnit.assign(inputs.plan->units().size(), 0);
    for (const CondensedUnit& unit : units.units)
        if (unit.isRecovered)
            units.isRecoveredPlanUnit[unit.planIndex] = 1;
    if (!units.isEveryUnitRecovered)
        listKnownOutputs(budget);
    m_record.interfaceLoadsSeconds = clock.lap();
}

void CondensedStatics::listKnownOutputs(base::MemoryBudget& budget)
{
    Units& units = *m_units;
    const assembly::DofMap& dofMap = m_inputs.model->dofMap();
    const reduction::PlannedUnits& planned = *units.planned;
    // Two passes over the points (count, then fill): the lists are model-sized, budgeted.
    std::size_t knownCount = 0;
    std::size_t forceCount = 0;
    const auto visitPoints = [&](const auto& visit)
    {
        for (std::int32_t node = 0; node < dofMap.nodeCount(); ++node)
        {
            const std::optional<std::size_t> owner = planned.unitOfInteriorNode(node);
            if (owner && units.isRecoveredPlanUnit[*owner] == 0)
                continue;
            const base::EntityId nodeId =
                node < dofMap.gridCount() ? dofMap.gridId(node) : dofMap.scalarId(node - dofMap.gridCount());
            visit(nodeId,
                  !std::binary_search(units.pointsNeedingRecovery.begin(), units.pointsNeedingRecovery.end(), nodeId));
        }
    };
    visitPoints(
        [&](base::EntityId, bool isForcePoint)
        {
            ++knownCount;
            forceCount += isForcePoint ? 1 : 0;
        });
    units.knownPoints = base::BudgetedBuffer<base::EntityId>(budget, knownCount, "condensed known points");
    units.forcePoints = base::BudgetedBuffer<base::EntityId>(budget, forceCount, "condensed known points");
    std::size_t nextKnown = 0;
    std::size_t nextForce = 0;
    visitPoints(
        [&](base::EntityId nodeId, bool isForcePoint)
        {
            units.knownPoints[nextKnown++] = nodeId;
            if (isForcePoint)
                units.forcePoints[nextForce++] = nodeId;
        });
    std::sort(units.knownPoints.data(), units.knownPoints.data() + knownCount);
    std::sort(units.forcePoints.data(), units.forcePoints.data() + forceCount);
    // The elements of no unit left unrecovered.
    const std::span<const base::EntityId> excluded = units.excludedElements.span();
    std::size_t elementCount = 0;
    for (const assembly::AssemblyBatch& batch : m_inputs.model->elementSet().batches())
        for (std::int64_t row = 0; row < batch.view.rowCount(); ++row)
            elementCount += std::binary_search(excluded.begin(), excluded.end(), batch.view.elementId(row)) ? 0 : 1;
    units.knownElements = base::BudgetedBuffer<base::EntityId>(budget, elementCount, "condensed known elements");
    std::size_t nextElement = 0;
    for (const assembly::AssemblyBatch& batch : m_inputs.model->elementSet().batches())
        for (std::int64_t row = 0; row < batch.view.rowCount(); ++row)
            if (const base::EntityId elementId = batch.view.elementId(row);
                !std::binary_search(excluded.begin(), excluded.end(), elementId))
                units.knownElements[nextElement++] = elementId;
    std::sort(units.knownElements.data(), units.knownElements.data() + elementCount);
}

void CondensedStatics::planMemory(const Inputs& inputs, base::MemoryBudget& budget)
{
    const assembly::DofMap& dofMap = inputs.model->dofMap();
    const reduction::PlannedUnits& planned = *m_units->planned;
    std::vector<CondensedUnit>& units = m_units->units;
    const auto caseCount = static_cast<std::int64_t>(m_loads->cases.size());
    const bool hasRelief = inputs.hasInertiaRelief;

    // The interface's points and DOFs: every model point not interior to a unit, and the generated fastener points.
    std::int64_t interfacePoints = 0;
    std::int64_t interfaceDofs = 0;
    for (std::int32_t node = 0; node < dofMap.nodeCount(); ++node)
        if (planned.interiorNodes()[static_cast<std::size_t>(node)] == 0)
        {
            ++interfacePoints;
            interfaceDofs += dofCountOfNode(dofMap, node);
        }
    constexpr std::int64_t GENERATED_POINT_DOFS = 6;
    for (const reduction::InterfaceSpider& spider : m_units->spiders)
        if (spider.referenceDofs.empty())
        {
            ++interfacePoints;
            interfaceDofs += GENERATED_POINT_DOFS;
        }

    // An upper bound of the interface's own vectors: its basis is chosen after the units' equations (its distributed
    // rows need their boundary elements), and a basis on part of a family's rows has at most min(rows, cases) vectors.
    std::int64_t interfaceVectors = 0;
    for (const LoadFamily family : assembly::ALL_LOAD_FAMILIES)
        if (const StaticLoadCases::FamilyDescription* description = m_loads->loadCases->familyDescription(family))
            interfaceVectors += std::min<std::int64_t>(description->rowCount, caseCount);

    std::vector<reduction::InterfaceUnitSize> sizes; // bounded by the units
    std::int64_t basisColumnBound = interfaceVectors + (hasRelief ? RIGID_BODY_MODE_COUNT : 0);
    for (const CondensedUnit& unit : units)
    {
        const reduction::UnitPlan& plan = inputs.plan->units()[unit.planIndex];
        sizes.push_back({plan.cost.boundaryDofCount, static_cast<std::int64_t>(unit.loadPatternCount)});
        basisColumnBound += static_cast<std::int64_t>(unit.loadPatternCount);
    }
    reduction::InterfaceFootprintInputs interfaceInputs;
    interfaceInputs.estimate = &inputs.plan->interfaceEstimate();
    interfaceInputs.pointCount = interfacePoints;
    interfaceInputs.localDofCount = interfaceDofs;
    interfaceInputs.modelNodeCount = dofMap.nodeCount();
    interfaceInputs.modelDofCount = dofMap.gSetSize();
    interfaceInputs.units = sizes;
    interfaceInputs.interfaceVectorCount = interfaceVectors;
    interfaceInputs.caseCount = caseCount;
    interfaceInputs.hasInertiaRelief = hasRelief;
    // The Condensation stage stops after the units: no interface problem, case or result follows them.
    const bool isCondensationOnly = inputs.stage == CondensedStage::Condensation;
    // The relations a constraint set makes: its MPC equations, glue and Gm made once here for each distinct selection
    // of the solved subcases and measured by the budget, with the interface's and the outputs' Gm, which hold some of
    // its rows (no more than it). The model has an m set when one of them has a row.
    if (!isCondensationOnly)
    {
        const input::ModelSnapshot& snapshot = *inputs.snapshot;
        const input::ConstraintTable* constraints =
            snapshot.hasTable(input::TableRole::Spc) ? &snapshot.constraints() : nullptr;
        const input::RigidTable* rigidElements =
            snapshot.hasTable(input::TableRole::RigidElements) ? &snapshot.rigidElements() : nullptr;
        // The interface's Gm, and the outputs' when a unit is not recovered.
        const bool isEveryUnitRecovered =
            std::all_of(units.begin(), units.end(), [](const CondensedUnit& unit) { return unit.isRecovered; });
        const std::uint64_t copies = isEveryUnitRecovered ? 1 : SET_GM_COPIES;
        std::vector<std::pair<std::optional<base::EntityId>, std::optional<base::EntityId>>> seen; // the selections
        for (const std::size_t s : inputs.solved)
        {
            const StaticSubcase& subcase = inputs.subcases[s];
            if (std::find(seen.begin(), seen.end(), std::make_pair(subcase.mpc, subcase.contact)) != seen.end())
                continue;
            seen.emplace_back(subcase.mpc, subcase.contact);
            base::DiagnosticSink ignored; // the constraint sets report their problems
            const assembly::SpcSet& spc = inputs.spcSets[s];
            const std::uint64_t before = budget.used();
            const assembly::MpcEquations mpc(constraints, dofMap, subcase.mpc, budget, ignored);
            const assembly::GlueContext glueContext{rigidElements, &mpc, m_units->fasteners.get(), spc.sbDofs(),
                                                    spc.sgDofs()};
            const assembly::GlueConstraints glue(snapshot, subcase.contact, subcase.label, dofMap, glueContext, budget,
                                                 ignored);
            const std::uint64_t beforeGm = budget.used();
            const assembly::RigidElimination gm(rigidElements, &mpc, dofMap, budget, ignored, m_units->fasteners.get(),
                                                &glue);
            const std::uint64_t gmBytes = budget.used() - beforeGm;
            interfaceInputs.relationBytes =
                std::max(interfaceInputs.relationBytes, budget.used() - before + copies * gmBytes);
            interfaceInputs.hasMset = interfaceInputs.hasMset || gm.rowCount() > 0;
        }
        for (const std::size_t s : inputs.solved)
        {
            const output::OutputPlan& outputs = inputs.plans[s];
            const bool asksGridPointForces = outputs.of(output::OutputKind::GridPointForces).isRequested;
            interfaceInputs.needsElementForces =
                interfaceInputs.needsElementForces || (!isEveryUnitRecovered && asksGridPointForces) ||
                (interfaceInputs.hasMset &&
                 (asksGridPointForces || outputs.of(output::OutputKind::MpcForces).isRequested));
        }
    }
    const reduction::InterfaceFootprint interface =
        isCondensationOnly ? reduction::InterfaceFootprint{} : reduction::InterfaceFootprint::estimate(interfaceInputs);

    // Each unit's footprint, and what one case of a batch holds (CondensedStatics::solveGroup): the interface's
    // eight case vectors, the batch's case vectors on the g set (displacements, applied loads, loads, SPC forces), and
    // the vectors of one unit of a wave: a unit not recovered its loads, applied loads and reaction resultants, a
    // recovered unit its solution, loads and K·u on its equations, its enforced values, displacements, loads, applied
    // loads and reactions, with the coefficients of its patterns.
    constexpr std::uint64_t INTERFACE_CASE_VECTORS = 8;
    constexpr std::uint64_t UNIT_CASE_VECTORS = 2;
    constexpr std::uint64_t RECOVERED_EQUATION_CASE_VECTORS = 3;
    constexpr std::uint64_t RECOVERED_LOCAL_CASE_VECTORS = 5;
    std::uint64_t caseBytes = isCondensationOnly
                                  ? 0
                                  : (INTERFACE_CASE_VECTORS * static_cast<std::uint64_t>(interfaceDofs) +
                                     CASE_BATCH_G_VECTORS * static_cast<std::uint64_t>(dofMap.gSetSize())) *
                                        sizeof(double);
    std::uint64_t largestUnitCase = 0;
    std::uint64_t largestRead = 0; // the load basis: the largest pass of a block's recovery of one unit
    std::vector<reduction::UnitFootprint> footprints; // bounded by the units
    for (CondensedUnit& unit : units)
    {
        const reduction::UnitPlan& plan = inputs.plan->units()[unit.planIndex];
        std::int64_t localDofs = 0;
        for (const base::EntityId nodeId : unit.description.nodeIds)
            if (const std::optional<std::int32_t> node = nodeOfPoint(dofMap, nodeId))
                localDofs += dofCountOfNode(dofMap, *node);
        std::int64_t points = static_cast<std::int64_t>(unit.description.nodeIds.size());
        std::int64_t spiderBlocks = 0;
        std::int64_t spiderValues = 0;
        for (const reduction::UnitSpider& spider : unit.description.spiders)
        {
            if (spider.relation.referenceDofs.empty())
            {
                ++points;
                localDofs += GENERATED_POINT_DOFS;
            }
            // Its leg points (the relation lists their DOFs, by increasing DOF: a point's DOFs are consecutive), whose
            // node blocks its change of basis couples (counted by DOF: a count by point would be up to six times as
            // large, squared).
            std::int64_t legPoints = 0;
            std::int32_t lastNode = -1;
            for (const base::DofIndex dof : spider.relation.legDofs)
                if (const std::int32_t node = dofMap.nodeDofOf(dof).node; node != lastNode)
                {
                    ++legPoints;
                    lastNode = node;
                }
            spiderBlocks += (legPoints + 1) * (legPoints + 1);
            spiderValues += static_cast<std::int64_t>(spider.relation.weights.legDofCount *
                                                      spider.relation.weights.referenceDofCount);
        }
        reduction::UnitFootprintInputs footprintInputs;
        footprintInputs.cost = &plan.cost;
        footprintInputs.pointCount = points;
        footprintInputs.localDofCount = localDofs;
        footprintInputs.patternCount = static_cast<std::int64_t>(unit.patternCount);
        footprintInputs.hasEnforcedPatterns = unit.hasEnforcedPatterns;
        footprintInputs.spiderBlockCount = spiderBlocks;
        footprintInputs.spiderValueCount = spiderValues;
        footprintInputs.isRecovered = unit.isRecovered;
        footprintInputs.keepsMass = inputs.isCraigBampton;
        footprintInputs.caseCount = caseCount;
        footprintInputs.basisColumnCount = std::min(basisColumnBound, caseCount);
        // The load basis handed over: every unit is one its blocks may read.
        footprintInputs.readColumnCount = inputs.handsOverLoadBasis ? inputs.loadBasisColumnCount : 0;
        unit.footprint = reduction::UnitFootprint::estimate(unit.unitId, footprintInputs);
        largestRead = std::max(largestRead, footprintInputs.readColumnCount > 0 ? unit.footprint.passBytes : 0);
        footprints.push_back(unit.footprint);
        const auto unitDofs = static_cast<std::uint64_t>(localDofs);
        const auto unitEquations = static_cast<std::uint64_t>(
            std::max<std::int64_t>(0, plan.cost.interiorDofCount + plan.cost.boundaryDofCount));
        largestUnitCase =
            std::max(largestUnitCase, (unit.isRecovered ? RECOVERED_EQUATION_CASE_VECTORS * unitEquations +
                                                              RECOVERED_LOCAL_CASE_VECTORS * unitDofs
                                                        : UNIT_CASE_VECTORS * unitDofs) +
                                          unit.patternCount + RIGID_BODY_MODE_COUNT);
    }
    if (!isCondensationOnly)
        caseBytes += largestUnitCase * sizeof(double);
    // The load basis: a block's recovery of one unit beside the factors every recovery at once keeps (in sequence the
    // unit's pass counts it again: an upper bound).
    caseBytes += largestRead;
    const std::size_t threadCount = inputs.pool != nullptr ? inputs.pool->threadCount() : 1;
    // The plan follows the scheduler's order (its condensation phase is the sequence of that order).
    reduction::SchedulerOptions schedulerOptions;
    schedulerOptions.threadCount = threadCount;
    schedulerOptions.largeUnitMinimumFlops = inputs.largeUnitMinimumFlops;
    const reduction::UnitScheduler ordering(footprints, schedulerOptions);
    std::vector<reduction::UnitFootprint> ordered; // bounded by the units
    for (const std::size_t u : ordering.order())
        ordered.push_back(footprints[u]);
    reduction::CondensedMemoryPlan::Inputs planInputs;
    planInputs.units = ordered;
    planInputs.interface = interface;
    planInputs.caseBytes = caseBytes;
    planInputs.resultBytes = inputs.resultBytes;
    // The g-set vectors the patterns' loads are made in (step 3) are released before the interface phase (step 6).
    planInputs.releasedBaselineBytes = isCondensationOnly ? 0 : 2 * m_loads->loadsG.size() * sizeof(double);
    // The recovery in sequence and the release of equations: a complete static run that keeps no unit's system for
    // later (the sensitivities of a kept set, Craig-Bampton), and the load basis handed over, whose blocks recover the
    // units they read in sequence (KeptCondensedUnit::releasesFactor).
    planInputs.allowsSequential = inputs.stage != CondensedStage::Condensation && !inputs.isCraigBampton &&
                                  (!inputs.keepsSystems || inputs.handsOverLoadBasis);
    planInputs.allowsRelease = planInputs.allowsSequential;
    m_memory->plan =
        std::make_unique<reduction::CondensedMemoryPlan>(planInputs, budget.used(), inputs.memoryMode, budget.limit());
    const reduction::CondensedMemoryPlan& memoryPlan = *m_memory->plan;
    for (std::size_t k = 0; k < ordering.order().size(); ++k)
        units[ordering.order()[k]].releasesEquations = memoryPlan.isReleased(k);
    CondensedMemoryRecord& record = m_record.memory;
    record.limitBytes = budget.limit();
    record.baselineBytes = memoryPlan.baselineBytes();
    record.estimatedMinimumBytes = memoryPlan.minimumBudgetBytes();
    record.estimatedCondensationBytes = memoryPlan.condensationPeakBytes();
    record.estimatedLaterBytes = memoryPlan.laterPhaseBytes();
    record.threadCount = static_cast<std::int32_t>(threadCount);
    record.mode = reduction::nameOf(inputs.memoryMode);
    record.recovery = reduction::nameOf(memoryPlan.strategy().recovery);
    record.releasedEquationCount = static_cast<std::int64_t>(memoryPlan.strategy().releasedCount);
    record.modeReason = memoryPlan.reason();
    record.estimatedAllAtOnceBytes = memoryPlan.allAtOnceBudgetBytes();
    record.estimatedSequentialBytes = memoryPlan.sequentialBudgetBytes();
    record.estimatedReleasedBytes = memoryPlan.releasedBudgetBytes();
    record.estimatedWithoutFactorBytes = memoryPlan.withoutFactorBudgetBytes();
    record.estimatedRegionsBytes = memoryPlan.regionsBudgetBytes();
    record.isWithoutFactor = memoryPlan.strategy().isWithoutFactor;
    record.regionBytes = memoryPlan.strategy().regionBytes;
    record.isInterfaceWithoutFactor = memoryPlan.strategy().isInterfaceWithoutFactor;
    record.interfaceRegionBytes = memoryPlan.strategy().interfaceRegionBytes;
    record.interfaceFactorBytes = interface.factorBytes;
    record.estimatedInterfaceRegionsBytes = memoryPlan.interfaceRegionsBudgetBytes();
    record.estimatedInterfaceRecomputedFlops = memoryPlan.interfaceRecomputedFlops();
    std::vector<base::MemoryConsumer> consumers = memoryPlan.consumers();
    constexpr std::size_t LISTED_CONSUMERS = 5;
    if (consumers.size() > LISTED_CONSUMERS)
        consumers.resize(LISTED_CONSUMERS);
    record.largestConsumers = std::move(consumers);
    if (inputs.stage == CondensedStage::MemoryPlan)
        return; // an estimate (a check): the budget is not enforced
    memoryPlan.requireFits(budget);
    schedulerOptions.laterPhaseBytes = memoryPlan.laterPhaseBytes();
    // PERFORMANCE: waves of up to WAVE_UNITS_PER_THREAD small units per thread, within the memory as before: the pool
    // deals them to its threads as they finish (the longest first), and the serial steps of a wave (the keys, the
    // cache stores under one lock) are shared by more units. The schedule changes no result.
    schedulerOptions.unitsPerThread = WAVE_UNITS_PER_THREAD;
    // A unit whose equations are released keeps what is left of it after its wave, and holds its equations and the
    // system without them beside its condensation during it (the plan's figures).
    std::vector<reduction::UnitFootprint> scheduled = footprints; // bounded by the units
    // Without the factors: a unit's condensation holds its fronts, and no factor is left after it.
    if (memoryPlan.strategy().isWithoutFactor)
        for (reduction::UnitFootprint& footprint : scheduled)
        {
            footprint.factorisationBytes = footprint.withoutFactorBytes;
            footprint.factorBytes = 0;
        }
    for (std::size_t u = 0; u < units.size(); ++u)
        if (units[u].releasesEquations)
        {
            reduction::UnitFootprint& footprint = scheduled[u];
            const std::uint64_t light = footprint.keptBytesOf(true);
            footprint.buildBytes += footprint.keptBytes;
            footprint.factorisationBytes += footprint.keptBytes;
            footprint.keptBytes = light;
        }
    m_memory->scheduler = std::make_unique<reduction::UnitScheduler>(scheduled, schedulerOptions);
}

void CondensedStatics::condenseInWaves(const Inputs& inputs, base::MemoryBudget& budget, base::DiagnosticSink& sink)
{
    const assembly::DofMap& dofMap = inputs.model->dofMap();
    const input::ParamTable* params = inputs.model->params();
    Units& units = *m_units;
    Loads& loads = *m_loads;
    const reduction::PlannedUnits& planned = *units.planned;
    const StaticLoadCases& cases = *loads.loadCases;
    reduction::UnitScheduler& scheduler = *m_memory->scheduler;
    reduction::FactorRetention& retention = m_memory->retention;
    const std::size_t threadCount = inputs.pool != nullptr ? inputs.pool->threadCount() : 1;
    // The kernels of a unit condensed beside others: the same family on the calling thread (the same bits).
    linalg::DenseKernelOptions serialOptions = inputs.kernels->options();
    serialOptions.pool = nullptr;
    const linalg::DenseKernels serialKernels(inputs.kernels->kind(), serialOptions);

    linalg::FactorOptions& unitFactorOptions = units.unitFactorOptions;
    unitFactorOptions.pivoting = linalg::PivotingMode::PositiveDefinite;
    if (const std::optional<double> ratio = assembly::findRealParam(params, "MaxDiagonalRatio", std::nullopt, sink);
        ratio && *ratio > 0.0)
        unitFactorOptions.pivotPolicy.maxRatio = *ratio;
    if (const std::optional<std::int32_t> bailout =
            assembly::findIntegerParam(params, "SingularityBailOut", std::nullopt, sink))
        unitFactorOptions.pivotPolicy.bailout = *bailout;
    unitFactorOptions.cancellation = inputs.cancellation;
    // The parts of the cache key the analysis gives (the same for every unit).
    for (const std::string_view name : reduction::MATRIX_PARAMETERS)
        units.cacheParameters.push_back({std::string(name), parameterValue(params, name)});
    units.cacheParameters.push_back({std::string(reduction::OFFSET_DEFINITION_PARAMETER),
                                     std::string(input::offsetDefinitionName(inputs.snapshot->offsetDefinition()))});
    std::sort(units.cacheParameters.begin(), units.cacheParameters.end(),
              [](const reduction::CacheParameter& a, const reduction::CacheParameter& b) { return a.name < b.name; });
    units.kernelName = std::string(linalg::kernelName(inputs.kernels->kind()));
    const auto isInteriorNode = [&](std::int32_t node)
    { return planned.interiorNodes()[static_cast<std::size_t>(node)] != 0; };

    m_record.units.resize(units.units.size());
    // Runs task(position in the wave, pool of its build) for the units of a wave: on every thread for one unit, one
    // thread each for several (the pool runs them at once; each writes only its own unit).
    const auto forEachUnitOf = [&](const reduction::UnitWave& wave, const std::function<void(std::size_t, bool)>& task)
    {
        if (wave.units.size() == 1 || inputs.pool == nullptr)
        {
            for (std::size_t w = 0; w < wave.units.size(); ++w)
                task(w, wave.isShared);
            return;
        }
        inputs.pool->runBatch(wave.units.size(), [&](std::size_t w) { task(w, false); });
    };
    // A unit taken from the cache that the analysis does not recover is built without its equations when the analysis
    // reads none (no Craig-Bampton modes, no inertia relief, no system kept for the sensitivities or the load basis),
    // from the boundary data an earlier build of its key gave (kept by the structure cache; else it is built with them,
    // and its boundary data kept for the next analysis).
    const bool takesHitsWithoutEquations =
        m_cacheSession && inputs.structures != nullptr && inputs.takesHitsWithoutEquations && !inputs.isCraigBampton &&
        !inputs.hasInertiaRelief && !inputs.keepsSystems && !inputs.handsOverLoadBasis;
    const reduction::ReductionKind entryReduction =
        inputs.isCraigBampton ? reduction::ReductionKind::CraigBampton : reduction::ReductionKind::Static;
    // Whether a DOF of a unit is held there in every subcase (role Constrained before the singularity processor: held,
    // not made dependent by a relation inside the unit; an absorbed spider's leg or reference is never held), where an
    // enforced pattern acts.
    const auto isHeldInUnit = [&](base::DofIndex dof)
    {
        return units.isHeldEverywhere[static_cast<std::size_t>(dof)] != 0 &&
               !units.unitRelations->findRow(dof).has_value();
    };

    std::int32_t waveIndex = 0;
    while (!scheduler.isDone())
    {
        const reduction::UnitWave wave = scheduler.nextWave(budget.available());
        m_record.memory.largestWave =
            std::max(m_record.memory.largestWave, static_cast<std::int32_t>(wave.units.size()));

        // a. In order: each unit's local DOFs and load patterns (in the numbering of its UnitSystem, made before it:
        // reduction::unitLocalModelDofs), its key from its inputs and the cache's entry; whether its build takes its
        // equations.
        std::vector<std::shared_ptr<const reduction::CachedReduction>> hits(wave.units.size());
        std::vector<std::shared_ptr<const base::KeptObject<reduction::UnitBoundaryData>>> kept(wave.units.size());
        // A hit built without its equations whose boundary data no build of its key left: made from its stiffness.
        std::vector<std::uint8_t> isWithoutKept(wave.units.size(), 0);
        for (std::size_t w = 0; w < wave.units.size(); ++w)
        {
            const std::size_t u = wave.units[w];
            CondensedUnit& unit = units.units[u];
            const reduction::UnitPlan& plan = inputs.plan->units()[unit.planIndex];
            unit.modelDofOfLocal = reduction::unitLocalModelDofs(dofMap, unit.description, budget);
            unit.identity = reduction::unitIdentityOf(plan.elementIds);
            const std::size_t localDofs = unit.modelDofOfLocal.size();
            unit.isInteriorLocal = base::BudgetedBuffer<std::uint8_t>(budget, localDofs, "condensed unit local DOFs");
            for (std::size_t local = 0; local < localDofs; ++local)
            {
                const base::DofIndex dof = unit.modelDofOfLocal[local];
                unit.isInteriorLocal[local] = dof >= 0 && isInteriorNode(dofMap.nodeDofOf(dof).node) ? 1 : 0;
            }
            unit.patternLoads =
                linalg::DenseMatrix(budget, localDofs, unit.patternCount, "condensed unit pattern loads");
            unit.patternEnforced = linalg::DenseMatrix(
                budget, localDofs, unit.hasEnforcedPatterns ? unit.patternCount : 0, "condensed unit pattern enforced");
            // The load patterns on the interior rows (the boundary rows are the interface's), the enforced values on
            // the DOFs the unit holds (UnitSystem::gatherLocal and its roles give the same values).
            for (std::size_t k = 0; k < unit.loadPatternCount; ++k)
            {
                const reduction::UnitBasisVector& vector = unit.basis.vectors()[k];
                loads.makeVector(cases, vector, budget, sink);
                for (std::size_t local = 0; local < localDofs; ++local)
                {
                    const base::DofIndex dof = unit.modelDofOfLocal[local];
                    if (dof < 0)
                        continue;
                    if (unit.isInteriorLocal[local] != 0)
                        unit.patternLoads(local, k) = loads.loadsG[static_cast<std::size_t>(dof)];
                    if (unit.hasEnforcedPatterns && isHeldInUnit(dof))
                        unit.patternEnforced(local, k) = loads.enforcedG[static_cast<std::size_t>(dof)];
                }
                unit.isThermalPattern.push_back(vector.family == LoadFamily::Thermal ? 1 : 0);
            }
            if (inputs.hasInertiaRelief)
                for (std::size_t mode = 0; mode < RIGID_BODY_MODE_COUNT; ++mode)
                    unit.isThermalPattern.push_back(0);
            stopOnErrors(sink, "the loads");
            CondensedUnitRecord& record = m_record.units[u];
            record.wave = waveIndex;
            // The condensation cache: the unit's key, then its entry (a miss is condensed and stored after).
            if (!m_cacheSession)
                continue;
            reduction::UnitKeySources sources;
            sources.plan = &plan;
            sources.description = &unit.description;
            sources.model = &units.unitModel;
            sources.patterns = {unit.patternLoads.view().block(0, 0, localDofs, unit.loadPatternCount),
                                unit.patternEnforced.view().block(0, 0, localDofs,
                                                                  unit.hasEnforcedPatterns ? unit.loadPatternCount : 0),
                                unit.basis.patternStamp()};
            sources.hasInertialPatterns = inputs.hasInertiaRelief;
            sources.designStamp = designStampOf(plan, inputs.designStamps);
            sources.parameters = units.cacheParameters;
            sources.solverVersion = m_cacheSession->cache().solverVersion();
            sources.kernel = units.kernelName;
            // A Craig-Bampton unit's boundary part (K*, M_aa) is the entry of no mode: it serves every request.
            sources.reduction = entryReduction;
            unit.key = reduction::makeUnitKey(sources, budget);
            const reduction::CacheLookup<reduction::CachedReduction> found =
                m_cacheSession->lookupReduction(*unit.key, budget);
            record.cacheOutcome = std::string(reduction::cacheOutcomeName(found.outcome));
            record.cacheCause = found.cause;
            if (!found.isHit())
                continue;
            hits[w] = found.entry;
            if (takesHitsWithoutEquations && !unit.isRecovered && found.entry->reduced != nullptr)
            {
                kept[w] = inputs.structures->unitBoundary(*unit.key);
                isWithoutKept[w] = kept[w] == nullptr ? 1 : 0;
            }
        }

        // b. The units' systems: with their equations, or without them for a hit the analysis does not recover, from
        // the boundary data a build of its key left (else from its stiffness alone: UnitSystem::boundaryDataOf); with
        // them too in the verification mode, to compare.
        // PERFORMANCE: a hit the analysis does not recover builds no element matrix, no assembly, no ordering when its
        // boundary data were kept (3.6 ms a unit of 744 DOFs on one thread before: the key read the equations; its
        // numbering after), no mass and no ordering otherwise.
        std::vector<std::shared_ptr<const reduction::UnitSystem>> checks(wave.units.size());
        forEachUnitOf(wave,
                      [&](std::size_t w, bool isShared)
                      {
                          CondensedUnit& unit = units.units[wave.units[w]];
                          base::ThreadPool* pool = isShared ? inputs.pool : nullptr;
                          if (isWithoutKept[w] != 0)
                              kept[w] = base::KeptObject<reduction::UnitBoundaryData>::build(
                                  [&](base::MemoryBudget& keptBudget)
                                  {
                                      return reduction::UnitSystem::boundaryDataOf(units.unitModel, unit.description,
                                                                                   *hits[w]->reduced, keptBudget, pool);
                                  });
                          if (kept[w] == nullptr)
                          {
                              unit.system = std::make_shared<const reduction::UnitSystem>(
                                  units.unitModel, unit.description, budget, pool);
                              return;
                          }
                          unit.system = std::make_shared<const reduction::UnitSystem>(units.unitModel, unit.description,
                                                                                      kept[w]->object(), budget);
                          if (inputs.verifiesCacheHits)
                              checks[w] = std::make_shared<const reduction::UnitSystem>(units.unitModel,
                                                                                        unit.description, budget, pool);
                      });

        // c. In order: what each system gives its condensation, the entries of the hits (their equations checked when
        // built), the units to condense.
        std::vector<std::size_t> misses;                          // bounded by the wave
        std::vector<std::uint8_t> isStored(wave.units.size(), 0); // its new condensation is stored
        std::vector<std::uint8_t> isStale(wave.units.size(), 0);  // its entry's key missed an input
        // A unit to condense takes the interior factor an earlier analysis kept for its equations (compared in full by
        // KeptInteriorFactors::take), so that it is not factorised: not a Craig-Bampton unit (its mass is in its
        // reduction), nor under a strategy without the factors (the plan holds none).
        const bool takesKeptFactors =
            inputs.keptFactors != nullptr && !inputs.isCraigBampton && !m_memory->plan->strategy().isWithoutFactor;
        const auto takeKeptFactor = [&](CondensedUnit& unit)
        {
            if (takesKeptFactors && unit.system->hasEquations())
                unit.keptFactor = inputs.keptFactors->take(unit.identity, *unit.system, inputs.kernels->kind(),
                                                           unitFactorOptions.pivotPolicy);
        };
        for (std::size_t w = 0; w < wave.units.size(); ++w)
        {
            const std::size_t u = wave.units[w];
            CondensedUnit& unit = units.units[u];
            const reduction::UnitSystem& system = *unit.system;
            // The numbering the patterns were made in is the system's (UnitSystem::numberPoints).
            if (!isSameNumbering(unit.modelDofOfLocal.span(), system))
                throw base::InternalError(base::concatenate(
                    {"CondensedStatics: unit ", std::to_string(unit.unitId),
                     ": its local DOFs made before its system differ from the system's (an internal error: report "
                     "it)"}));
            if (inputs.hasInertiaRelief)
                for (std::size_t mode = 0; mode < RIGID_BODY_MODE_COUNT; ++mode)
                    for (std::size_t local = 0; local < unit.modelDofOfLocal.size(); ++local)
                        unit.patternLoads(local, unit.loadPatternCount + mode) = system.inertialLoads()(local, mode);
            unit.factorOptions = unitFactorOptions;
            unit.factorOptions.equationNamer = equationNamerOf(unit.system.get());
            CondensedUnitRecord& record = m_record.units[u];
            if (!hits[w])
            {
                misses.push_back(u);
                isStored[w] = m_cacheSession ? 1 : 0;
                takeKeptFactor(unit);
                continue;
            }
            const reduction::CachedReduction& entry = *hits[w];
            // The verification of the key: the equations built compared with those the entry was condensed from; and,
            // in the verification mode, the boundary data kept with those of a build.
            const reduction::UnitSystem* built = system.hasEquations() ? &system : checks[w].get();
            if (built != nullptr && entry.equationStamp != reduction::equationStampOf(*built, entryReduction))
            {
                if (inputs.verifiesCacheHits)
                    throw base::InternalError(base::concatenate(
                        {"CondensedStatics: unit ", std::to_string(unit.unitId),
                         ": the cache entry of its key was condensed from other equations; the key missed an input "
                         "(a bug of O_o_Solver: report it)"}));
                // A key that missed an input (a bug): the entry is not used, the unit condensed again from its
                // equations and not stored (its file would keep the stale entry).
                sink.report(problem(
                    "cache-key-stale", "UNIT", static_cast<base::EntityId>(unit.unitId),
                    base::concatenate({"unit ", std::to_string(unit.unitId),
                                       ": the cache entry of its key was condensed from other equations (the key "
                                       "missed an input: a bug of O_o_Solver, report it); condensed again, not "
                                       "stored; clear the cache to remove the entry"}),
                    base::Severity::Warning));
                record.cacheOutcome = "miss";
                record.cacheCause = "the entry of its key was condensed from other equations (the key missed an input)";
                isStale[w] = 1;
                misses.push_back(u);
                takeKeptFactor(unit);
                continue;
            }
            if (checks[w] != nullptr && !checks[w]->boundaryData(budget).isEqual(kept[w]->object()))
                throw base::InternalError(base::concatenate(
                    {"CondensedStatics: unit ", std::to_string(unit.unitId),
                     ": the boundary data kept for its key differ from those of its build; the key missed an input "
                     "(a bug of O_o_Solver: report it)"}));
            unit.condensation = std::make_shared<reduction::StaticCondensation>(
                unit.system, entry.reduced, entry.largestPivotRatio, unit.factorOptions);
            unit.reduction = hits[w];
            record.isCondensedNow = false;
            record.isCacheStored = true;
            record.isWithoutEquations = !system.hasEquations();
            record.isBoundaryDataReused = !system.hasEquations() && isWithoutKept[w] == 0;
        }

        // d. The condensations of the units the cache did not hold: one thread each beside others, or every thread for
        // one unit when its threaded factorisation fits the budget left (the same bits either way). The units of a wave
        // whose exact factorisations (from their symbolic factors, now known) do not fit at once are condensed in
        // smaller groups.
        // The strategy of the memory plan may condense every unit without its factor (its peak the fronts; the same
        // bits). Craig-Bampton projects its mass with the factor: never without it.
        const bool isWithoutFactor = m_memory->plan->strategy().isWithoutFactor && !inputs.isCraigBampton;
        const auto factorisationBytesOf = [&](std::size_t u, std::size_t threads)
        {
            const CondensedUnit& unit = units.units[u];
            const auto equations = static_cast<std::uint64_t>(unit.system->equationCount());
            const std::uint64_t loadColumns = unit.patternCount + RIGID_BODY_MODE_COUNT;
            if (isWithoutFactor)
                return linalg::PartialFactor::estimateWithoutFactor(unit.system->symbolic(), threads)
                           .bytes(loadColumns) +
                       equations * loadColumns * sizeof(double);
            return linalg::SupernodalFactor::estimateMemoryBytes(unit.system->symbolic(), threads,
                                                                 linalg::PivotingMode::PositiveDefinite) +
                   2 * equations * (loadColumns + unit.patternCount) * sizeof(double);
        };
        const auto condense = [&](std::size_t u, const linalg::DenseKernels& kernels, std::size_t threads)
        {
            CondensedUnit& unit = units.units[u];
            unit.condensation.reset();
            if (unit.keptFactor)
            {
                // The factor of its equations kept by an earlier analysis: no factorisation, the bits of a fresh one.
                // When its bytes do not fit the budget it is freed, and the unit factorised as without it.
                std::optional<reduction::KeptInteriorFactor> kept = std::move(unit.keptFactor);
                unit.keptFactor.reset();
                try
                {
                    unit.condensation = std::make_shared<reduction::StaticCondensation>(
                        unit.system, unit.patterns(), std::move(*kept), kernels, unit.factorOptions, budget);
                    m_record.units[u].isInteriorFactorReused = true;
                }
                catch (const base::MemoryBudgetError&)
                {
                    unit.condensation.reset();
                }
            }
            if (!unit.condensation)
                unit.condensation = isWithoutFactor
                                        ? std::make_shared<reduction::StaticCondensation>(
                                              unit.system, unit.patterns(), kernels, unit.factorOptions, budget,
                                              reduction::StaticCondensation::WithoutFactor{})
                                        : std::make_shared<reduction::StaticCondensation>(
                                              unit.system, unit.patterns(), kernels, unit.factorOptions, budget);
            m_record.units[u].isCondensedWithoutFactor = unit.condensation->isCondensedWithoutFactor();
            auto fresh = std::make_shared<reduction::CachedReduction>();
            fresh->reduced = unit.condensation->sharedReducedUnit();
            fresh->patternWork = unit.condensation->patternWork(unit.patterns(), kernels, budget);
            fresh->largestPivotRatio = unit.condensation->largestPivotRatio();
            fresh->interiorDofCount = unit.system->interiorDofCount();
            // Craig-Bampton: the reduced boundary mass M_aa = Ψ̃ᵀ·M·Ψ̃, from the interior factor just made.
            if (inputs.isCraigBampton)
                fresh->boundaryMass = reduction::CraigBampton::projectOnBoundary(
                    *unit.condensation, unit.system->mass(), kernels, budget, inputs.cancellation);
            // The checksum of its equations, stored with the entry: the verification of a later hit.
            if (unit.key)
            {
                fresh->equationStamp = reduction::equationStampOf(*unit.system, entryReduction);
                unit.key->equationStamp = fresh->equationStamp;
            }
            unit.reduction = fresh;
            m_record.units[u].threadCount = static_cast<std::int32_t>(threads);
        };
        std::size_t done = 0;
        while (done < misses.size())
        {
            std::vector<std::size_t> group{misses[done]};
            if (wave.isShared || misses.size() == 1)
            {
                // One unit: every thread when the threaded factorisation fits.
                const std::size_t u = misses[done];
                const bool isThreaded = threadCount > 1 && factorisationBytesOf(u, threadCount) <= budget.available();
                condense(u, isThreaded ? *inputs.kernels : serialKernels, isThreaded ? threadCount : 1);
                ++done;
                continue;
            }
            std::uint64_t needed = factorisationBytesOf(misses[done], 1);
            const std::uint64_t available = budget.available();
            for (std::size_t k = done + 1; k < misses.size(); ++k)
            {
                const std::uint64_t bytes = factorisationBytesOf(misses[k], 1);
                if (needed + bytes > available)
                    break;
                needed += bytes;
                group.push_back(misses[k]);
            }
            if (group.size() == 1)
                condense(group.front(), serialKernels, 1);
            else
                inputs.pool->runBatch(group.size(), [&](std::size_t k) { condense(group[k], serialKernels, 1); });
            done += group.size();
        }

        // e. In order: the stores (given to the cache session's writer: made off the critical path, their answers in
        // the records at finishCacheStores), the boundary data kept for the next analysis, the records, the points
        // whose forces need a recovery, the factors kept.
        if (m_cacheSession)
        {
            std::vector<std::pair<reduction::CacheKey, std::shared_ptr<const reduction::CachedReduction>>> entries;
            for (std::size_t w = 0; w < wave.units.size(); ++w)
                if (isStored[w] != 0)
                {
                    const CondensedUnit& unit = units.units[wave.units[w]];
                    entries.emplace_back(*unit.key, unit.reduction);
                    m_storedUnits.push_back(wave.units[w]);
                }
            m_cacheSession->storeReductionsLater(std::move(entries));
        }
        for (std::size_t w = 0; w < wave.units.size(); ++w)
        {
            const std::size_t u = wave.units[w];
            CondensedUnit& unit = units.units[u];
            const reduction::UnitSystem& system = *unit.system;
            CondensedUnitRecord& record = m_record.units[u];
            const bool isNew = std::find(misses.begin(), misses.end(), u) != misses.end();
            if (takesHitsWithoutEquations && unit.key && system.hasEquations() && kept[w] == nullptr && isStale[w] == 0)
            {
                inputs.structures->keepUnitBoundary(
                    *unit.key, base::KeptObject<reduction::UnitBoundaryData>::build(
                                   [&](base::MemoryBudget& keptBudget) { return system.boundaryData(keptBudget); }));
                ++m_record.keptUnitBoundaryCount;
            }
            else if (isWithoutKept[w] != 0)
            {
                inputs.structures->keepUnitBoundary(*unit.key, kept[w]);
                ++m_record.keptUnitBoundaryCount;
            }
            m_largestPivotRatio = std::max(m_largestPivotRatio, unit.condensation->largestPivotRatio());
            m_solvedVectorCount += static_cast<std::int64_t>(unit.patternCount);
            // The boundary points whose SPC or MPC forces need the unit recovered: a DOF it holds there with a
            // stiffness, a relation of its own reaching them, the reference of a spider it absorbed.
            std::vector<base::EntityId>& needing = unit.pointsNeedingRecovery;
            const std::span<const std::int64_t> held = system.heldDofs();
            for (std::size_t k = 0; k < held.size(); ++k)
            {
                const auto local = static_cast<std::size_t>(held[k]);
                if (unit.isInteriorLocal[local] == 0 && unit.modelDofOfLocal[local] >= 0 && system.isHeldDofCoupled(k))
                    needing.push_back(dofMap.pointIdOf(unit.modelDofOfLocal[local]));
            }
            for (const reduction::UnitSpider& spider : unit.description.spiders)
                for (const base::DofIndex dof : spider.relation.referenceDofs)
                    needing.push_back(dofMap.pointIdOf(dof));
            // The relation rows of its DOFs, found from its own DOFs (a pass over every unit's rows per unit would
            // cost 10 000 units times the rows of the model).
            for (std::size_t local = 0; local < unit.modelDofOfLocal.size(); ++local)
            {
                const base::DofIndex dependent = unit.modelDofOfLocal[local];
                if (dependent < 0)
                    continue;
                const std::optional<std::size_t> r = units.unitRelations->findRow(dependent);
                if (!r)
                    continue;
                if (unit.isInteriorLocal[local] == 0)
                    needing.push_back(dofMap.pointIdOf(dependent));
                for (const assembly::ConstraintTerm& term : units.unitRelations->row(*r))
                {
                    const std::optional<std::int64_t> termLocal = system.localDofOf(term.dof);
                    if (termLocal && unit.isInteriorLocal[static_cast<std::size_t>(*termLocal)] == 0)
                        needing.push_back(dofMap.pointIdOf(term.dof));
                }
            }
            std::sort(needing.begin(), needing.end());
            needing.erase(std::unique(needing.begin(), needing.end()), needing.end());

            record.unitId = unit.unitId;
            record.boundaryDofCount = system.boundaryDofCount();
            record.interiorDofCount = system.interiorDofCount();
            record.spiderCount = static_cast<std::int64_t>(system.spiderCount());
            record.patternCount = static_cast<std::int64_t>(unit.patternCount);
            record.loadBases = unit.basis.reports();
            record.estimatedKeptBytes = unit.footprint.keptBytes;
            record.estimatedCondensationBytes = unit.footprint.condensationBytes();

            // The interior factor: kept in memory while the budget allows. A Craig-Bampton unit's modal part needs it
            // as a recovery does (reduceModes releases it after). A unit whose equations the memory mode releases
            // releases its factor with them (it was made from them).
            if (isNew && unit.condensation->isCondensedWithoutFactor())
            {
                record.factorBytes = 0;
                record.factorRetention = "not kept (condensed without its factor)";
                record.factorRetentionReason =
                    "the memory mode condenses the units without their interior factor (its condensation holds its "
                    "fronts only): its recovery factorises its interior again, to the same bits";
            }
            else if (isNew && unit.releasesEquations)
            {
                record.factorBytes = unit.condensation->interiorFactorBytes();
                record.factorRetention = "released (equations released)";
                record.factorRetentionReason =
                    "the memory mode releases the unit's equations after its condensation: its recovery builds them "
                    "again and factorises its interior again, to the same bits";
                unit.condensation->releaseInteriorFactor();
            }
            else if (isNew)
            {
                record.factorBytes = unit.condensation->interiorFactorBytes();
                // The load basis handed over: the factors of the units the basis vectors load (their loads are
                // condensed with them), and with every recovery at once those of every unit (its blocks read them).
                const bool isReadEverywhere = inputs.handsOverLoadBasis && m_memory->plan->strategy().recovery ==
                                                                               reduction::RecoveryStrategy::AllAtOnce;
                const reduction::RetentionDecision decision =
                    retention.decide(unit.unitId, record.factorBytes,
                                     unit.isRecovered || inputs.isCraigBampton || isReadEverywhere ||
                                         (inputs.handsOverLoadBasis && unit.loadPatternCount > 0),
                                     budget.available(), scheduler.reserve());
                record.factorRetention = reduction::nameOf(decision);
                record.factorRetentionReason = retention.records().back().reason;
                unit.isFactorKept = decision == reduction::RetentionDecision::Kept;
                if (!unit.isFactorKept)
                    unit.condensation->releaseInteriorFactor();
            }
            // Its equations released, what the analysis reads of its build kept (its numbering, roles, singularities,
            // raw boundary blocks, held DOFs and boundary equations: the system without equations of a cache hit); its
            // recovery builds them again from the snapshot.
            if (unit.releasesEquations && system.hasEquations())
            {
                unit.condensation->releaseInteriorFactor();
                unit.lightSystem = std::make_shared<const reduction::UnitSystem>(units.unitModel, unit.description,
                                                                                 system.boundaryData(budget), budget);
                unit.condensation->rebindSystem(unit.lightSystem, equationNamerOf(unit.lightSystem.get()));
                record.isEquationReleased = true;
                unit.system = unit.lightSystem; // the last holder of the equations: released here
            }
            else if (unit.releasesEquations)
                unit.lightSystem = unit.system; // a cache hit built without them
        }
        ++waveIndex;
    }
    m_record.memory.waveCount = waveIndex;
    m_record.memory.keptFactorBytes = retention.keptBytes();
    m_record.memory.keptFactorCount = retention.keptCount();
    m_record.memory.releasedForBudget = retention.releasedForBudget();
    if (!inputs.defersCacheStores)
        finishCacheStores(sink, inputs.pool);
}

void CondensedStatics::finishCacheStores(base::DiagnosticSink& sink, base::ThreadPool* pool)
{
    if (!m_cacheSession)
        return;
    const std::vector<reduction::CacheStore> stores = m_cacheSession->finishStores(pool);
    if (stores.size() != m_storedUnits.size())
        throw base::InternalError(
            base::concatenate({"CondensedStatics::finishCacheStores: ", std::to_string(stores.size()), " answers for ",
                               std::to_string(m_storedUnits.size()), " stores given (an internal error: report it)"}));
    for (std::size_t k = 0; k < stores.size(); ++k)
    {
        const CondensedUnit& unit = m_units->units[m_storedUnits[k]];
        CondensedUnitRecord& record = m_record.units[m_storedUnits[k]];
        record.isCacheStored = stores[k].isStored;
        record.cacheNote = stores[k].note;
        if (!stores[k].isStored)
            sink.report(problem(
                "cache-not-stored", "UNIT", static_cast<base::EntityId>(unit.unitId),
                base::concatenate({"unit ", std::to_string(unit.unitId), ": its condensation is ", stores[k].note}),
                base::Severity::Information));
    }
    m_storedUnits.clear();
}

void CondensedStatics::decideRecoveredUnits(const Inputs& inputs)
{
    Units& units = *m_units;
    const assembly::DofMap& dofMap = inputs.model->dofMap();
    // The units of the elements of the section and cut loads, whatever recover= and the outputs say.
    const auto recoverSectionUnits = [&]
    {
        if (inputs.elementsToRecover.empty())
            return;
        for (CondensedUnit& unit : units.units)
            for (const base::EntityId elementId : inputs.plan->units()[unit.planIndex].elementIds)
                if (std::binary_search(inputs.elementsToRecover.begin(), inputs.elementsToRecover.end(), elementId))
                {
                    unit.isRecovered = true;
                    break;
                }
    };
    // The units a design point drives, whose pseudo-loads read the displacements of their elements.
    if (inputs.recoversDrivenUnits)
        for (CondensedUnit& unit : units.units)
            if (designStampOf(inputs.plan->units()[unit.planIndex], inputs.designStamps) != 0)
                unit.isRecovered = true;
    for (CondensedUnit& unit : units.units)
        if (std::find(inputs.alsoRecoveredUnits.begin(), inputs.alsoRecoveredUnits.end(), unit.unitId) !=
            inputs.alsoRecoveredUnits.end())
            unit.isRecovered = true;
    if (inputs.recoveredUnits)
    {
        for (const std::int64_t id : *inputs.recoveredUnits)
        {
            const auto found = std::find_if(units.units.begin(), units.units.end(),
                                            [id](const CondensedUnit& unit) { return unit.unitId == id; });
            if (found == units.units.end())
                throw std::invalid_argument(base::concatenate(
                    {"recover: unit ", std::to_string(id),
                     " is not a condensed unit of the plan; name the condensed units to recover (the units kept full "
                     "are always solved)"}));
            found->isRecovered = true;
        }
        recoverSectionUnits();
        return;
    }
    // The units whose interior points or elements an output of a subcase asks for.
    constexpr std::array<output::OutputKind, 5> NODAL_KINDS{
        output::OutputKind::Displacement, output::OutputKind::SpcForces, output::OutputKind::MpcForces,
        output::OutputKind::AppliedLoads, output::OutputKind::GridPointForces};
    constexpr std::array<output::OutputKind, 4> ELEMENT_KINDS{output::OutputKind::Stress, output::OutputKind::Strain,
                                                              output::OutputKind::Force,
                                                              output::OutputKind::StrainEnergy};
    const reduction::PlannedUnits& planned = *units.planned;
    for (CondensedUnit& unit : units.units)
    {
        const reduction::UnitPlan& plan = inputs.plan->units()[unit.planIndex];
        // Its interior points: those of its description that the plan makes interior to it (a generated fastener
        // point is no point of the model, no output takes it).
        std::vector<base::EntityId> interior; // bounded by the unit's points
        for (const base::EntityId nodeId : unit.description.nodeIds)
            if (const std::optional<std::int32_t> node = nodeOfPoint(dofMap, nodeId))
                if (const std::optional<std::size_t> owner = planned.unitOfInteriorNode(*node);
                    owner && *owner == unit.planIndex)
                    interior.push_back(nodeId);
        for (const output::OutputPlan& outputs : inputs.plans)
        {
            for (const output::OutputKind kind : NODAL_KINDS)
            {
                const output::OutputRequestPlan& request = outputs.of(kind);
                if (!request.isRequested)
                    continue;
                for (std::size_t k = 0; k < interior.size() && !unit.isRecovered; ++k)
                    unit.isRecovered = request.takes(interior[k]);
            }
            for (const output::OutputKind kind : ELEMENT_KINDS)
            {
                const output::OutputRequestPlan& request = outputs.of(kind);
                if (!request.isRequested)
                    continue;
                for (const base::EntityId elementId : plan.elementIds)
                    if (request.takes(elementId))
                    {
                        unit.isRecovered = true;
                        break;
                    }
            }
            if (unit.isRecovered)
                break;
        }
    }
    recoverSectionUnits();
}

void CondensedStatics::reportUnrecoveredOutputs(const Inputs& inputs, base::DiagnosticSink& sink)
{
    Units& units = *m_units;
    const assembly::DofMap& dofMap = inputs.model->dofMap();
    // The distinct selections of the subcases' plans: the subcases mostly share theirs (a request of the global
    // section), and a pass over a unit's points per subcase costs the model's points times the subcases.
    std::vector<const output::OutputPlan*> selections; // bounded by the subcases
    for (const output::OutputPlan& outputs : inputs.plans)
        if (std::none_of(selections.begin(), selections.end(),
                         [&](const output::OutputPlan* seen) { return isSameSelection(*seen, outputs); }))
            selections.push_back(&outputs);
    // The outputs of the units not recovered: listed once per unit.
    for (const CondensedUnit& unit : units.units)
    {
        if (unit.isRecovered)
            continue;
        std::set<output::OutputKind> kinds; // in the order of OutputKind
        std::vector<std::int32_t> points;   // bounded by the unit's points
        std::vector<std::int32_t> elementIds;
        const reduction::UnitPlan& plan = inputs.plan->units()[unit.planIndex];
        for (const output::OutputPlan* outputs : selections)
            for (std::size_t k = 0; k < output::OUTPUT_KIND_COUNT; ++k)
            {
                const auto kind = static_cast<output::OutputKind>(k);
                const output::OutputRequestPlan& request = outputs->of(kind);
                if (!request.isRequested)
                    continue;
                const bool isElementKind = kind == output::OutputKind::Stress || kind == output::OutputKind::Strain ||
                                           kind == output::OutputKind::Force ||
                                           kind == output::OutputKind::StrainEnergy;
                bool isDropped = false;
                if (isElementKind)
                {
                    for (const base::EntityId elementId : plan.elementIds)
                        if (request.takes(elementId))
                        {
                            elementIds.push_back(elementId);
                            isDropped = true;
                        }
                }
                else
                {
                    for (std::size_t local = 0; local < unit.isInteriorLocal.size(); ++local)
                    {
                        if (unit.modelDofOfLocal[local] < 0)
                            continue;
                        const base::EntityId nodeId = dofMap.pointIdOf(unit.modelDofOfLocal[local]);
                        const bool isNeeded =
                            unit.isInteriorLocal[local] != 0 ||
                            (kind != output::OutputKind::Displacement && kind != output::OutputKind::AppliedLoads &&
                             std::binary_search(unit.pointsNeedingRecovery.begin(), unit.pointsNeedingRecovery.end(),
                                                nodeId));
                        if (isNeeded && request.takes(nodeId))
                        {
                            points.push_back(nodeId);
                            isDropped = true;
                        }
                    }
                }
                if (isDropped)
                    kinds.insert(kind);
            }
        if (kinds.empty())
            continue;
        for (std::vector<std::int32_t>* list : {&points, &elementIds})
        {
            std::sort(list->begin(), list->end());
            list->erase(std::unique(list->begin(), list->end()), list->end());
        }
        std::string kindText;
        for (const output::OutputKind kind : kinds)
            kindText.append(kindText.empty() ? "" : ", ").append(output::requestNameOf(kind));
        std::string message =
            base::concatenate({"unit ", std::to_string(unit.unitId),
                               " is condensed and not recovered: the requested outputs ", kindText, " of its"});
        if (!points.empty())
            message.append(" ")
                .append(std::to_string(points.size()))
                .append(" points (")
                .append(listOf(points))
                .append(")");
        if (!elementIds.empty())
            message.append(points.empty() ? " " : " and ")
                .append(std::to_string(elementIds.size()))
                .append(" elements (")
                .append(listOf(elementIds))
                .append(")");
        message.append(" are not produced; recover the unit (recover=[")
            .append(std::to_string(unit.unitId))
            .append("]) to have them");
        sink.report(problem("output-not-recovered", "UNIT", static_cast<base::EntityId>(unit.unitId),
                            std::move(message), base::Severity::Warning));
    }
}

void CondensedStatics::rebind(const linalg::DenseKernels& kernels) noexcept
{
    m_inputs.kernels = &kernels;
    m_inputs.pool = nullptr;
    m_inputs.cancellation = nullptr;
    // The units' factor options keep the run's token too (their Craig-Bampton modes, their factorisations).
    if (m_units)
        for (CondensedUnit& unit : m_units->units)
            unit.factorOptions.cancellation = nullptr;
}

bool CondensedStatics::hasUnrecoveredUnits() const noexcept
{
    return !m_units->isEveryUnitRecovered;
}

bool CondensedStatics::isRecovered(std::int64_t unitId) const noexcept
{
    return std::any_of(m_units->units.begin(), m_units->units.end(),
                       [unitId](const CondensedUnit& unit) { return unit.unitId == unitId && unit.isRecovered; });
}

bool CondensedStatics::hasInteriorReactions(std::int64_t unitId) const noexcept
{
    for (const CondensedUnit& unit : m_units->units)
    {
        if (unit.unitId != unitId || !unit.system)
            continue;
        const std::span<const std::int64_t> held = unit.system->heldDofs();
        for (std::size_t k = 0; k < held.size(); ++k)
            if (unit.isInteriorLocal[static_cast<std::size_t>(held[k])] != 0 && unit.system->isHeldDofCoupled(k))
                return true;
    }
    return false;
}

void CondensedStatics::keepSet(const GroupInputs& group, std::unique_ptr<const reduction::InterfaceProblem> interface,
                               std::unique_ptr<const assembly::RigidElimination> gm,
                               base::BudgetedBuffer<std::uint8_t> isConstrained,
                               std::vector<std::shared_ptr<const reduction::ExternalUnitAssembly>> placements,
                               base::MemoryBudget& budget, std::shared_ptr<const CondensedRelief> relief)
{
    const Inputs& inputs = m_inputs;
    const AnalysisModel& model = *inputs.model;
    const linalg::DenseKernels& kernels = *inputs.kernels;
    Units& units = *m_units;
    if (!m_keptUnits)
    {
        auto kept = std::make_shared<KeptCondensedUnits>();
        kept->reserve(units.units.size());
        std::shared_ptr<const KeptUnitModel> rebuild; // made once, when some unit's equations are released
        for (const CondensedUnit& unit : units.units)
        {
            KeptCondensedUnit& entry = kept->emplace_back();
            entry.unitId = unit.unitId;
            entry.system = unit.system;
            entry.condensation = unit.condensation;
            const std::size_t localCount = unit.modelDofOfLocal.size();
            entry.modelDofOfLocal = base::BudgetedBuffer<base::DofIndex>(budget, localCount, "kept unit local DOFs");
            entry.isInteriorLocal = base::BudgetedBuffer<std::uint8_t>(budget, localCount, "kept unit local DOFs");
            for (std::size_t local = 0; local < localCount; ++local)
            {
                entry.modelDofOfLocal[local] = unit.modelDofOfLocal[local];
                entry.isInteriorLocal[local] = unit.isInteriorLocal[local];
            }
            entry.isRecovered = unit.isRecovered;
            entry.isDriven = designStampOf(inputs.plan->units()[unit.planIndex], inputs.designStamps) != 0;
            // The load basis in sequence keeps no factor beyond its use, recovers by regions the units whose factor
            // exceeds the plan's bound, and builds the released equations again for each use.
            const reduction::MemoryStrategy& strategy = m_memory->plan->strategy();
            if (inputs.handsOverLoadBasis && strategy.recovery == reduction::RecoveryStrategy::Sequential)
            {
                entry.releasesFactor = true;
                if (strategy.isWithoutFactor && unit.footprint.isRecoveredByRegions(strategy.regionBytes))
                    entry.regionValues = static_cast<std::int64_t>(strategy.regionBytes / sizeof(double));
            }
            if (unit.releasesEquations)
            {
                if (!rebuild)
                    rebuild = std::make_shared<const KeptUnitModel>(units.unitModel, model.sharedIndex(),
                                                                    model.sharedDofMap(), units.elementIndex,
                                                                    units.unitRelations, units.shares, budget);
                entry.description = unit.description;
                entry.rebuild = rebuild;
            }
        }
        m_keptUnits = std::move(kept);
    }
    CondensedFactorisedSetParts parts;
    parts.index = model.sharedIndex();
    parts.dofMap = model.sharedDofMap();
    parts.units = m_keptUnits;
    parts.interface = std::move(interface);
    parts.gm = std::move(gm);
    parts.isConstrained = std::move(isConstrained);
    parts.externalUnits = std::move(placements);
    parts.kernel = kernels.kind();
    for (const std::size_t s : group.cases)
        parts.subcases.push_back(inputs.subcases[s].label);
    parts.relief = std::move(relief);
    m_keptSets.push_back(std::make_shared<const CondensedFactorisedSet>(std::move(parts)));
}

std::vector<std::shared_ptr<const CondensedFactorisedSet>> CondensedStatics::takeKeptSets()
{
    return std::move(m_keptSets);
}

output::StiffnessRecovery::CondensedUnits CondensedStatics::gridPointForceUnits(
    std::span<const output::StiffnessRecovery::UnitRow> rows) const noexcept
{
    return {m_units->excludedElements.span(), rows};
}

std::vector<ReducedUnitResult> CondensedStatics::reducedUnits(base::MemoryBudget& budget) const
{
    std::vector<ReducedUnitResult> results; // bounded by the units
    for (const CondensedUnit& unit : m_units->units)
    {
        if (!unit.reduction || !unit.system)
            throw std::logic_error(
                "CondensedStatics::reducedUnits: the units are not condensed (the MemoryPlan stage "
                "stops before them); run the Condensation or the Complete stage");
        ReducedUnitResult result;
        result.unitId = unit.unitId;
        // Copies in `budget` (the results' own): the cache's entry is budgeted in the cache, which they may outlive.
        const auto copyOf = [&budget](const linalg::DenseMatrix& source, const char* label)
        {
            linalg::DenseMatrix copy(budget, source.rows(), source.columns(), label);
            for (std::size_t j = 0; j < source.columns(); ++j)
                for (std::size_t i = 0; i < source.rows(); ++i)
                    copy(i, j) = source(i, j);
            return copy;
        };
        const reduction::ReducedUnit& cached = *unit.reduction->reduced;
        base::BudgetedBuffer<reduction::UnitDofLabel> labels(budget, cached.labels().size(), "reduced unit labels");
        std::copy(cached.labels().begin(), cached.labels().end(), labels.data());
        result.reduced = std::make_shared<const reduction::ReducedUnit>(
            cached.unitId(), std::move(labels), copyOf(cached.stiffness(), "reduced unit stiffness"),
            copyOf(cached.patterns(), "reduced unit patterns"), cached.patternStamp(),
            copyOf(cached.inertialPatterns(), "reduced unit inertial patterns"), cached.rigidBodyMass());
        result.loadPatternCount = unit.loadPatternCount;
        for (const reduction::UnitBasisVector& vector : unit.basis.vectors())
            result.families.push_back(vector.family);
        const linalg::DenseMatrix& alpha = unit.basis.coefficients().matrix();
        result.coefficients = linalg::DenseMatrix(budget, alpha.rows(), alpha.columns(), "reduced unit coefficients");
        for (std::size_t column = 0; column < alpha.columns(); ++column)
            for (std::size_t row = 0; row < alpha.rows(); ++row)
                result.coefficients(row, column) = alpha(row, column);
        result.interiorDofCount = unit.system->interiorDofCount();
        result.boundaryMass = copyOf(unit.reduction->boundaryMass, "reduced unit boundary mass");
        if (unit.modal)
            result.modal = std::make_shared<const reduction::ModalReduction>(
                copyOf(unit.modal->eigenvalues(), "reduced unit mode eigenvalues"),
                copyOf(unit.modal->modalStiffness(), "reduced unit modal stiffness"),
                copyOf(unit.modal->couplingMass(), "reduced unit coupling mass"),
                copyOf(unit.modal->modalMass(), "reduced unit modal mass"));
        for (const reduction::UnitDofLabel& label : unit.reduction->reduced->labels())
        {
            if (label.kind != reduction::UnitPointKind::FastenerPoint || label.component != 1)
                continue;
            const auto spider =
                std::find_if(unit.description.spiders.begin(), unit.description.spiders.end(),
                             [&](const reduction::UnitSpider& entry)
                             { return entry.key.elementId == label.id && entry.key.side == label.side; });
            if (spider != unit.description.spiders.end())
                result.generatedPoints.push_back({label.id, label.side, spider->generatedPointPosition});
        }
        results.push_back(std::move(result));
    }
    return results;
}

std::vector<base::EntityId> CondensedStatics::caseLabels() const
{
    std::vector<base::EntityId> labels; // bounded by the subcases
    for (const StaticSubcase* subcase : m_loads->cases)
        labels.push_back(subcase->label);
    return labels;
}

output::OutputPlan CondensedStatics::outputPlanOf(const output::OutputPlan& plan) const
{
    const Units& units = *m_units;
    if (units.isEveryUnitRecovered)
        return plan;
    output::OutputPlan filtered = plan;
    // The points every output may take, those whose forces need no unit recovered, and the elements of no unit left
    // unrecovered (listKnownOutputs: made once, not per subcase).
    const auto restrict = [&](output::OutputKind kind, std::span<const base::EntityId> allowed)
    {
        output::OutputRequestPlan request = plan.of(kind);
        if (!request.isRequested)
            return;
        std::vector<std::int32_t> members; // bounded by the model's points or elements (a request's members are)
        if (request.isAll)
            members.assign(allowed.begin(), allowed.end());
        else
            std::set_intersection(request.members.begin(), request.members.end(), allowed.begin(), allowed.end(),
                                  std::back_inserter(members));
        request.isAll = false;
        request.members = std::move(members);
        filtered.setRequest(kind, std::move(request));
    };
    restrict(output::OutputKind::Displacement, units.knownPoints.span());
    restrict(output::OutputKind::AppliedLoads, units.knownPoints.span());
    restrict(output::OutputKind::SpcForces, units.forcePoints.span());
    restrict(output::OutputKind::MpcForces, units.forcePoints.span());
    restrict(output::OutputKind::GridPointForces, units.forcePoints.span());
    for (const output::OutputKind kind :
         {output::OutputKind::Stress, output::OutputKind::Strain, output::OutputKind::Force})
        restrict(kind, units.knownElements.span());
    // The strain energy percentages need every element's energy.
    if (plan.of(output::OutputKind::StrainEnergy).isRequested)
        filtered.setRequest(output::OutputKind::StrainEnergy, output::OutputRequestPlan{});
    return filtered;
}

bool CondensedStatics::recoversInSequence() const noexcept
{
    if (!m_memory->plan || m_memory->plan->strategy().recovery != reduction::RecoveryStrategy::Sequential)
        return false;
    return std::any_of(m_units->units.begin(), m_units->units.end(),
                       [](const CondensedUnit& unit) { return unit.isRecovered; });
}

std::uint64_t CondensedStatics::unitByUnitBytes(std::size_t caseCount) const noexcept
{
    std::uint64_t values = 0;
    for (const CondensedUnit& unit : m_units->units)
        if (unit.isRecovered)
            values += (static_cast<std::uint64_t>(unit.system->heldDofs().size()) + 1) * caseCount;
    return values * sizeof(double);
}

void CondensedStatics::listUnitOutputs() const
{
    if (m_isUnitOutputsListed)
        return;
    const Units& units = *m_units;
    const assembly::DofMap& dofMap = m_inputs.model->dofMap();
    const reduction::PlannedUnits& planned = *units.planned;
    std::vector<std::size_t> positionOfPlanUnit(m_inputs.plan->units().size(), SIZE_MAX); // bounded by the units
    for (std::size_t u = 0; u < units.units.size(); ++u)
        positionOfPlanUnit[units.units[u].planIndex] = u;
    m_unitOutputs.assign(units.units.size(), UnitOutputs{});
    for (std::int32_t node = 0; node < dofMap.nodeCount(); ++node)
    {
        const base::EntityId nodeId =
            node < dofMap.gridCount() ? dofMap.gridId(node) : dofMap.scalarId(node - dofMap.gridCount());
        const std::optional<std::size_t> owner = planned.unitOfInteriorNode(node);
        if (owner && units.isRecoveredPlanUnit[*owner] != 0 && positionOfPlanUnit[*owner] != SIZE_MAX)
            m_unitOutputs[positionOfPlanUnit[*owner]].interiorPoints.push_back(nodeId);
        else
            m_interfacePoints.push_back(nodeId);
    }
    std::vector<base::EntityId> recoveredElements; // bounded by the model's elements
    for (std::size_t u = 0; u < units.units.size(); ++u)
    {
        UnitOutputs& outputs = m_unitOutputs[u];
        std::sort(outputs.interiorPoints.begin(), outputs.interiorPoints.end());
        if (!units.units[u].isRecovered)
        {
            outputs.interiorPoints.clear();
            continue;
        }
        outputs.elements = units.units[u].description.elementIds;
        std::sort(outputs.elements.begin(), outputs.elements.end());
        recoveredElements.insert(recoveredElements.end(), outputs.elements.begin(), outputs.elements.end());
    }
    std::sort(m_interfacePoints.begin(), m_interfacePoints.end());
    std::sort(recoveredElements.begin(), recoveredElements.end());
    for (const assembly::AssemblyBatch& batch : m_inputs.model->elementSet().batches())
        for (std::int64_t row = 0; row < batch.view.rowCount(); ++row)
            if (const base::EntityId elementId = batch.view.elementId(row);
                !std::binary_search(recoveredElements.begin(), recoveredElements.end(), elementId))
                m_interfaceElements.push_back(elementId);
    std::sort(m_interfaceElements.begin(), m_interfaceElements.end());
    m_isUnitOutputsListed = true;
}

namespace
{

// `plan`'s request of `kind` restricted to `allowed` (sorted IDs): its members among them, or them all for a request of
// every point or element.
void restrictRequest(output::OutputPlan& plan, output::OutputKind kind, std::span<const base::EntityId> allowed)
{
    output::OutputRequestPlan request = plan.of(kind);
    if (!request.isRequested)
        return;
    std::vector<std::int32_t> members; // bounded by the model's points or elements (a request's members are)
    if (request.isAll)
        members.assign(allowed.begin(), allowed.end());
    else
        std::set_intersection(request.members.begin(), request.members.end(), allowed.begin(), allowed.end(),
                              std::back_inserter(members));
    request.isAll = false;
    request.members = std::move(members);
    plan.setRequest(kind, std::move(request));
}

constexpr std::array<output::OutputKind, 3> ELEMENT_TABLE_KINDS{output::OutputKind::Stress, output::OutputKind::Strain,
                                                                output::OutputKind::Force};

} // namespace

output::OutputPlan CondensedStatics::interfaceOutputPlanOf(const output::OutputPlan& plan) const
{
    listUnitOutputs();
    output::OutputPlan filtered = plan;
    restrictRequest(filtered, output::OutputKind::Displacement, m_interfacePoints);
    for (const output::OutputKind kind : ELEMENT_TABLE_KINDS)
        restrictRequest(filtered, kind, m_interfaceElements);
    return filtered;
}

output::OutputPlan CondensedStatics::unitOutputPlanOf(const output::OutputPlan& plan, std::size_t unit) const
{
    listUnitOutputs();
    if (unit >= m_unitOutputs.size())
        throw std::invalid_argument(
            base::concatenate({"CondensedStatics::unitOutputPlanOf: expected a unit below ",
                               std::to_string(m_unitOutputs.size()), ", found ", std::to_string(unit)}));
    output::OutputPlan filtered = plan;
    for (std::size_t k = 0; k < output::OUTPUT_KIND_COUNT; ++k)
    {
        const auto kind = static_cast<output::OutputKind>(k);
        if (kind != output::OutputKind::Displacement &&
            std::find(ELEMENT_TABLE_KINDS.begin(), ELEMENT_TABLE_KINDS.end(), kind) == ELEMENT_TABLE_KINDS.end())
            filtered.setRequest(kind, output::OutputRequestPlan{});
    }
    restrictRequest(filtered, output::OutputKind::Displacement, m_unitOutputs[unit].interiorPoints);
    for (const output::OutputKind kind : ELEMENT_TABLE_KINDS)
        restrictRequest(filtered, kind, m_unitOutputs[unit].elements);
    return filtered;
}

output::OutputPlan CondensedStatics::unitOnlyOutputPlanOf(const output::OutputPlan& plan, std::int64_t unitId) const
{
    listUnitOutputs();
    const Units& units = *m_units;
    const auto found = std::find_if(units.units.begin(), units.units.end(),
                                    [&](const CondensedUnit& unit) { return unit.unitId == unitId; });
    if (found == units.units.end() || !found->isRecovered)
        throw std::invalid_argument(base::concatenate(
            {"CondensedStatics::unitOnlyOutputPlanOf: unit ", std::to_string(unitId),
             " is not a recovered condensed unit of the plan; recover it (a bug of O_o_Solver when the recovery of "
             "one unit named it: report it)"}));
    const auto u = static_cast<std::size_t>(found - units.units.begin());
    // Its points: the interior ones and the grid and scalar points of its boundary (a generated fastener point has no
    // row of the tables).
    std::vector<base::EntityId> points = m_unitOutputs[u].interiorPoints; // bounded by the unit's points
    for (const reduction::UnitDofLabel& label : found->system->boundaryLabels())
        if (label.kind == reduction::UnitPointKind::Grid || label.kind == reduction::UnitPointKind::Scalar)
            points.push_back(label.id);
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    output::OutputPlan filtered = plan;
    for (std::size_t k = 0; k < output::OUTPUT_KIND_COUNT; ++k)
    {
        const auto kind = static_cast<output::OutputKind>(k);
        if (kind != output::OutputKind::Displacement &&
            std::find(ELEMENT_TABLE_KINDS.begin(), ELEMENT_TABLE_KINDS.end(), kind) == ELEMENT_TABLE_KINDS.end())
            filtered.setRequest(kind, output::OutputRequestPlan{});
    }
    restrictRequest(filtered, output::OutputKind::Displacement, points);
    for (const output::OutputKind kind : ELEMENT_TABLE_KINDS)
        restrictRequest(filtered, kind, m_unitOutputs[u].elements);
    return filtered;
}

// ---------------------------------------------------------------------------------------------------------------------
// One constraint set

namespace
{

// The g-set element forces Σ_e K_e·u_e of the elements not excluded (sorted IDs), in batch and row order.
void addElementForces(const assembly::ElementSet& set, const assembly::DofMap& dofMap,
                      std::span<const double> displacements, std::span<const base::EntityId> excluded,
                      std::span<double> forces, const base::CancellationToken* cancellation)
{
    constexpr std::int64_t RANGE_ELEMENTS = 2048;
    elements::ElementMatrixBlock block;
    for (const assembly::AssemblyBatch& batch : set.batches())
    {
        const elements::ElementFamily& family = *batch.family;
        if (!family.facets().has(elements::Facet::Stiffness))
            continue;
        for (std::int64_t first = 0; first < batch.view.rowCount(); first += RANGE_ELEMENTS)
        {
            if (cancellation != nullptr)
                cancellation->throwIfCancelled();
            const elements::ElementRange range{first, std::min(batch.view.rowCount(), first + RANGE_ELEMENTS)};
            block.clear();
            family.computeStiffness(batch.view, range, block);
            for (std::size_t i = 0; i < block.size(); ++i)
            {
                if (std::binary_search(excluded.begin(), excluded.end(), batch.view.elementId(block.elementRow(i))))
                    continue;
                const std::span<const elements::ElementDof> dofs = block.dofs(i);
                const std::span<const double> k = block.matrix(i);
                const std::size_t n = dofs.size();
                for (std::size_t r = 0; r < n; ++r)
                {
                    double sum = 0.0;
                    for (std::size_t c = 0; c < n; ++c)
                        sum += k[r * n + c] * displacements[static_cast<std::size_t>(dofMap.dofOf(dofs[c]))];
                    forces[static_cast<std::size_t>(dofMap.dofOf(dofs[r]))] += sum;
                }
            }
        }
    }
}

} // namespace

std::string CondensedStatics::solveGroup(const GroupInputs& group, const CaseVisitor& visit, base::MemoryBudget& budget,
                                         base::DiagnosticSink& sink)
{
    const Inputs& inputs = m_inputs;
    const input::ModelSnapshot& snapshot = *inputs.snapshot;
    const AnalysisModel& model = *inputs.model;
    const assembly::DofMap& dofMap = model.dofMap();
    const linalg::DenseKernels& kernels = *inputs.kernels;
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    Units& units = *m_units;
    Loads& loads = *m_loads;
    PhaseClock clock;
    const assembly::SpcSet& spc = *group.spc;
    const std::size_t caseCount = group.cases.size();
    std::vector<std::size_t> caseIndices; // the load cases of the group (bounded by the subcases)
    for (const std::size_t s : group.cases)
        caseIndices.push_back(loads.caseOfSubcase[s]);

    // The full path's relations and sets of the subcases: their problems are reported as the full path reports them.
    const input::ConstraintTable* constraints =
        snapshot.hasTable(input::TableRole::Spc) ? &snapshot.constraints() : nullptr;
    const input::RigidTable* rigidElements =
        snapshot.hasTable(input::TableRole::RigidElements) ? &snapshot.rigidElements() : nullptr;
    const assembly::MpcEquations mpc(constraints, dofMap, group.selection.mpc, budget, sink);
    const assembly::GlueContext glueContext{rigidElements, &mpc, units.fasteners.get(), spc.sbDofs(), spc.sgDofs()};
    const assembly::GlueConstraints glue(snapshot, group.selection.contact, group.selection.subcase, dofMap,
                                         glueContext, budget, sink);
    // On the heap: a kept set (Inputs::keepsSystems) takes it.
    auto fullGmOwner = std::make_unique<const assembly::RigidElimination>(rigidElements, &mpc, dofMap, budget, sink,
                                                                          units.fasteners.get(), &glue);
    const assembly::RigidElimination& fullGm = *fullGmOwner;
    // The interior DOFs of the external units join the permanent constraints (as on the full path: ConstrainedSystem).
    std::vector<base::DofIndex> permanentDofs; // bounded by the constrained DOFs and the zones
    if (inputs.externalUnits != nullptr)
    {
        permanentDofs.assign(spc.sgDofs().begin(), spc.sgDofs().end());
        const std::span<const base::DofIndex> held = inputs.externalUnits->heldDofs();
        permanentDofs.insert(permanentDofs.end(), held.begin(), held.end());
    }
    const std::span<const base::DofIndex> permanent =
        inputs.externalUnits != nullptr ? std::span<const base::DofIndex>(permanentDofs) : spc.sgDofs();
    const assembly::DofSets sets(
        dofMap, assembly::DofSetInputs{fullGm.mDofs(), spc.sbDofs(), permanent, {}, group.supports}, budget, sink);
    if (inputs.externalUnits != nullptr && group.relief == assembly::InertiaReliefMode::OnSupports)
        sink.report(problem("inertia-relief-external-unit", "PARAM InertiaRelief", group.selection.subcase,
                            "inertia relief with external units is not supported on the condensed path of this "
                            "version (the interface takes the rigid-body mass of the condensed units and of its own "
                            "elements, not the plugins' M*): run the full path"));
    if (group.relief == assembly::InertiaReliefMode::Automatic)
        sink.report(
            problem("inertia-relief-condensed", "PARAM InertiaRelief", group.selection.subcase,
                    "PARAM InertiaRelief = -2 is not supported on the condensed path of this version of O_o_Solver: "
                    "give the supports with FreeBodySupport and PARAM InertiaRelief = -1, or run the full path"));
    stopOnErrors(sink, "the constraints");
    base::DiagnosticSink interfaceSink; // the same rows as fullGm's: its diagnostics are reported once
    const assembly::RigidElimination interfaceGm(rigidElements, &mpc, dofMap, budget, interfaceSink,
                                                 units.fasteners.get(), &glue, units.planned->interfaceRowFilter());
    const bool hasRelief = group.relief == assembly::InertiaReliefMode::OnSupports && !group.supports.empty();
    // The interface handed over, or its solve replaced by the caller's values.
    const bool isGivenSolve = group.boundaryMode != reduction::BoundarySolveMode::Solve;
    if ((group.exportsInterface || isGivenSolve) && hasRelief)
        throw base::SolverError(base::concatenate(
            {"inertia relief (PARAM InertiaRelief -1 with FreeBodySupport, subcase ",
             std::to_string(group.selection.subcase),
             ") on the interface problem handed over or solved by the caller: its relieved loads come from the solve "
             "of its support motions inside O_o_Solver; run a complete static analysis of this subcase, or hold "
             "the structure with SinglePointConstrs"}));
    if (group.exportsInterface && isGivenSolve)
        throw std::logic_error(
            "CondensedStatics::solveGroup: an interface handed over is not also solved by given "
            "values (a bug of O_o_Solver: report it)");
    // Inertia relief on six FreeBodySupport DOFs, when Inputs::conditionsReliefSupports: the interface held by six
    // translations of the farthest boundary points of the condensed units (assembly::chooseConditionedSupports, among
    // the translations a unit's own stiffness holds, neither constrained nor dependent) rather than by the
    // FreeBodySupport DOFs, the cases' displacements then brought back to the gauge of the FreeBodySupport DOFs (u ← u
    // − D·u_r): the relieved loads are balanced, so that any statically determinate support gives the same deformation,
    // and a thin body held at one point is ill-conditioned (the full path does the same: StaticSolution).
    std::vector<base::DofIndex> heldSupports = group.supports;
    bool isConditioned = false;
    if (hasRelief && inputs.conditionsReliefSupports && group.supports.size() == assembly::MAX_SUPPORT_DOFS)
    {
        // The candidates: the translations of the condensed units' boundary grid points (rows of their K*), neither
        // constrained nor dependent at the interface (a spider's reference, dependent on the full path, is an
        // independent DOF of the interface).
        base::BudgetedBuffer<std::uint8_t> isCandidate(budget, gSize, "relief support candidates");
        constexpr std::uint8_t TRANSLATION_COMPONENTS = 3;
        for (const CondensedUnit& unit : units.units)
            for (const reduction::UnitDofLabel& label : unit.system->boundaryLabels())
                if (label.kind == reduction::UnitPointKind::Grid && label.component <= TRANSLATION_COMPONENTS)
                    if (const std::optional<base::DofIndex> dof = dofMap.findDof(label.id, label.component))
                        isCandidate[static_cast<std::size_t>(*dof)] = 1;
        for (std::size_t dof = 0; dof < gSize; ++dof)
        {
            const assembly::ExclusiveSet set = sets.exclusiveSetOf(static_cast<base::DofIndex>(dof));
            if (set == assembly::ExclusiveSet::Sb || set == assembly::ExclusiveSet::Sg ||
                set == assembly::ExclusiveSet::AutoSpc || set == assembly::ExclusiveSet::O)
                isCandidate[dof] = 0;
        }
        for (const base::DofIndex dof : interfaceGm.mDofs())
            isCandidate[static_cast<std::size_t>(dof)] = 0;
        std::vector<base::DofIndex> chosen = assembly::chooseConditionedSupports(dofMap, isCandidate.span());
        if (chosen.size() == assembly::MAX_SUPPORT_DOFS)
        {
            heldSupports = std::move(chosen);
            isConditioned = true;
        }
    }

    // The interface problem.
    std::vector<reduction::InterfaceUnit> interfaceUnits; // bounded by the units
    for (const CondensedUnit& unit : units.units)
        interfaceUnits.push_back({&unit.condensation->reducedUnit(), unit.system.get()});
    reduction::InterfaceInputs interfaceInputs;
    interfaceInputs.elements = &model.elementSet();
    interfaceInputs.dofMap = &dofMap;
    interfaceInputs.elementIndex = units.elementIndex.get();
    interfaceInputs.isInteriorNode = units.planned->interiorNodes();
    interfaceInputs.elementIds = units.interfaceElements.span();
    interfaceInputs.units = interfaceUnits;
    interfaceInputs.spiders = units.spiders;
    interfaceInputs.relations = &interfaceGm;
    interfaceInputs.modelSets = &sets;
    if (isConditioned)
        interfaceInputs.conditionedSupports = heldSupports;
    interfaceInputs.singularity = group.singularity;
    interfaceInputs.pivotPolicy = group.pivotPolicy;
    interfaceInputs.computesMass = hasRelief;
    interfaceInputs.massOptions = units.massOptions;
    interfaceInputs.nonStructuralMass = units.shares.get();
    std::vector<reduction::InterfaceExternalBlock> externalBlocks; // bounded by the external units
    std::vector<std::shared_ptr<const reduction::ExternalUnitAssembly>> placements;
    if (inputs.externalUnits != nullptr)
        for (std::size_t e = 0; e < inputs.externalUnits->unitCount(); ++e)
            if (std::shared_ptr<const reduction::ExternalUnitAssembly> placement = inputs.externalUnits->placement(e))
            {
                externalBlocks.push_back(
                    {placement->zone().id, placement->zone().contourDofs, &placement->stiffness()});
                placements.push_back(std::move(placement));
            }
    interfaceInputs.externalBlocks = externalBlocks;
    // The interface problem without its factor when the memory plan's strategy says so, each solve a factorisation
    // streamed, its backward sweep by regions of the strategy's bound (every panel kept when the bound holds the plan's
    // factor).
    const reduction::MemoryStrategy& memoryStrategy = m_memory->plan->strategy();
    if (memoryStrategy.isInterfaceWithoutFactor)
    {
        interfaceInputs.keepsFactor = false;
        interfaceInputs.regionValues =
            memoryStrategy.interfaceRegionBytes >= m_memory->plan->interface().factorBytes
                ? std::numeric_limits<std::int64_t>::max() / 2
                : static_cast<std::int64_t>(memoryStrategy.interfaceRegionBytes / sizeof(double));
    }
    // An interface handed over unsolved, or whose solve the caller's values replace, is not factorised.
    const bool isInterfaceFactorised = !isGivenSolve && (!group.exportsInterface || group.solvesExport);
    interfaceInputs.factorises = isInterfaceFactorised;
    auto interfaceOwner = std::make_unique<const reduction::InterfaceProblem>(interfaceInputs, kernels, budget,
                                                                              inputs.pool, inputs.cancellation);
    const reduction::InterfaceProblem& interface = *interfaceOwner;
    // Its pivots: with its factor, of the factorisation made; without it, of the first solve's (after the basis).
    const auto reportInterfacePivots = [&]()
    {
        m_largestPivotRatio = std::max(m_largestPivotRatio, interface.pivotDiagnostics().largestRatio());
        reportKeptPivots(
            interface.pivotDiagnostics(), [&interface](std::int32_t equation)
            { return interface.nameOf(interface.localDofOfEquation(equation)); }, group.selection.subcase, sink);
    };
    if (isInterfaceFactorised && interface.isFactorKept())
        reportInterfacePivots();
    const auto localDofs = static_cast<std::size_t>(interface.localDofCount());
    const auto equations = static_cast<std::size_t>(interface.equationCount());

    // The table of the singular components: the units' interior points and the interface's.
    std::string singularityTable;
    {
        std::vector<assembly::SingularityRow> rows(interface.singularities().begin(), interface.singularities().end());
        for (const CondensedUnit& unit : units.units)
        {
            const std::span<const assembly::SingularityRow> unitRows = unit.system->interiorSingularities();
            rows.insert(rows.end(), unitRows.begin(), unitRows.end());
        }
        std::sort(rows.begin(), rows.end(), [](const assembly::SingularityRow& a, const assembly::SingularityRow& b)
                  { return a.pointId != b.pointId ? a.pointId < b.pointId : a.component < b.component; });
        std::ostringstream table;
        assembly::writeSingularityTable(table, rows, group.singularity);
        singularityTable = table.str();
    }
    const double interfaceSeconds = clock.lap();

    // Inertia relief on the condensed path: the support motions D = Φ_r·C, C fitted on the interface's D_l =
    // −K_ll⁻¹·K_lr; DᵀMD = Cᵀ·(Σ units' Φ_rᵀMΦ_r + the interface's)·C; ü_r = (DᵀMD)⁻¹·Cᵀ·Φ_rᵀ·P of each case; the
    // relief loads −M·Φ_r·(C·ü_r) are the units' inertial patterns and the interface's inertial loads, with the
    // coefficients a = C·ü_r.
    const std::size_t supportCount = group.supports.size();
    // The supports the interface holds (the FreeBodySupport DOFs, or the conditioned ones) and their local DOFs; and
    // the local DOFs and equations of the FreeBodySupport DOFs when the interface does not hold them (their SPC forces
    // are the residuals of their equations, and the gauge of the displacements).
    std::vector<std::size_t> supportLocals(hasRelief ? supportCount : 0);     // the FreeBodySupport DOFs, at most six
    std::vector<std::int32_t> supportEquations(hasRelief ? supportCount : 0); // their equations when conditioned
    for (std::size_t j = 0; j < supportLocals.size(); ++j)
    {
        const std::optional<std::int64_t> local = interface.localDofOf(group.supports[j]);
        if (!local)
            throw std::invalid_argument(base::concatenate({"inertia relief: the FreeBodySupport DOF of point ",
                                                           std::to_string(dofMap.pointIdOf(group.supports[j])),
                                                           " is an interior point of a condensed unit; the plan puts "
                                                           "the FreeBodySupport points on the boundaries (a "
                                                           "bug of O_o_Solver: report it)"}));
        supportLocals[j] = static_cast<std::size_t>(*local);
        supportEquations[j] = -1;
    }
    if (isConditioned)
        for (std::size_t e = 0; e < equations; ++e)
            for (std::size_t j = 0; j < supportLocals.size(); ++j)
                if (static_cast<std::size_t>(interface.localDofOfEquation(static_cast<std::int32_t>(e))) ==
                    supportLocals[j])
                    supportEquations[j] = static_cast<std::int32_t>(e);
    double reliefMisfit = -1.0;    // the record's
    double reliefReaction = -1.0;  // the record's
    std::vector<double> reliefFit; // C, kept with the set
    linalg::DenseMatrix reliefLoads(budget, localDofs, hasRelief ? RIGID_BODY_MODE_COUNT : 0, "interface relief loads");
    std::vector<double> reliefMass; // DᵀMD = Cᵀ·Mr·C (r × r)
    // Six supports, C = Φ_S⁻¹ exact: the relief made on Φ, a = Mr⁻¹·Φᵀ·P (Mr 6 × 6 row-major) and the FreeBodySupport
    // DOFs' accelerations ü_r = C⁻¹·a = Φ_S·a (Φ_S row-major, row j the FreeBodySupport DOF j's), so that the relieved
    // loads do not depend on the FreeBodySupport DOFs in floating point; empty otherwise.
    std::vector<double> reliefRigidMass;
    std::vector<double> reliefSupportRows;
    if (hasRelief)
    {
        if (supportCount > assembly::MAX_SUPPORT_DOFS)
            throw base::SolverError(base::concatenate({"inertia relief takes at most six FreeBodySupport DOFs, found ",
                                                       std::to_string(supportCount),
                                                       "; keep six at most, a statically determinate support"}));
        // D_l: a unit motion of each held support with the others held.
        std::vector<std::size_t> heldLocals(supportCount); // bounded by the six supports
        linalg::DenseMatrix supportEnforced(budget, localDofs, supportCount, "relief support motions");
        for (std::size_t j = 0; j < supportCount; ++j)
        {
            heldLocals[j] = static_cast<std::size_t>(*interface.localDofOf(heldSupports[j]));
            supportEnforced(heldLocals[j], j) = 1.0;
        }
        const linalg::DenseMatrix zero(budget, localDofs, supportCount, "relief zero loads");
        linalg::DenseMatrix supportRhs(budget, equations, supportCount, "relief support right-hand sides");
        interface.equationLoads(zero.view(), supportEnforced.view(), supportRhs.view());
        linalg::DenseMatrix supportLoads(budget, equations, supportCount, "relief support loads"); // −K_ls·e
        for (std::size_t j = 0; j < supportCount; ++j)
            for (std::size_t e = 0; e < equations; ++e)
                supportLoads(e, j) = supportRhs(e, j);
        // C (6 × r). Six supports (a statically determinate support): C = Φ_S⁻¹, the rigid motions whose value at the
        // supports is the identity, exact for a free body without the round-off of a solve. Fewer: by least squares,
        // (ΦᵀΦ)·C = Φᵀ·D on the solved support modes over the interface's independent free DOFs (the full path's l- and
        // r-sets: a held DOF is zero in D whatever the rigid motion, a dependent DOF follows its relation, rigid or
        // not).
        const linalg::DenseMatrix& phi = interface.rigidBodyModes();
        const auto isMoving = [&](std::size_t local)
        {
            const auto dof = static_cast<std::int64_t>(local);
            return !interface.isHeld(dof) && !interface.isDependent(dof);
        };
        std::array<double, assembly::MAX_SUPPORT_DOFS * RIGID_BODY_MODE_COUNT> atSupports{};
        for (std::size_t j = 0; j < supportCount; ++j)
            for (std::size_t a = 0; a < RIGID_BODY_MODE_COUNT; ++a)
                atSupports[j * RIGID_BODY_MODE_COUNT + a] = phi(heldLocals[j], a);
        const auto exact = assembly::rigidSupportCoefficients(atSupports, supportCount);
        // The support modes D_l are solved for the least-squares fit, and for the misfit report when the
        // interface keeps its factor; without it a solve is a factorisation, so an exact fit leaves the misfit
        // unmeasured (the test of a free body below needs no solve).
        const bool solvesSupportModes = interface.isFactorKept() || !exact;
        linalg::DenseMatrix supportModes(budget, localDofs, solvesSupportModes ? supportCount : 0,
                                         "relief support modes");
        if (solvesSupportModes)
        {
            interface.solve(supportRhs.view(), inputs.cancellation);
            interface.expand(supportRhs.view(), supportEnforced.view(), supportModes.view());
        }
        std::vector<double> fit(RIGID_BODY_MODE_COUNT * supportCount, 0.0);
        if (exact)
            fit.assign(exact->begin(), exact->end());
        else
        {
            std::vector<double> normal(RIGID_BODY_MODE_COUNT * RIGID_BODY_MODE_COUNT, 0.0);
            for (std::size_t local = 0; local < localDofs; ++local)
            {
                if (!isMoving(local))
                    continue;
                for (std::size_t a = 0; a < RIGID_BODY_MODE_COUNT; ++a)
                {
                    for (std::size_t b = 0; b < RIGID_BODY_MODE_COUNT; ++b)
                        normal[a * RIGID_BODY_MODE_COUNT + b] += phi(local, a) * phi(local, b);
                    for (std::size_t j = 0; j < supportCount; ++j)
                        fit[j * RIGID_BODY_MODE_COUNT + a] += phi(local, a) * supportModes(local, j);
                }
            }
            if (!solveSmallSpd(normal, RIGID_BODY_MODE_COUNT, fit))
                throw base::SolverError(
                    "inertia relief on the condensed path: the interface points do not fix a rigid-body motion (fewer "
                    "than three points not in line); run the full path");
        }
        // D = Φ·C on the local DOFs (of the held supports: the misfit and the test below).
        linalg::DenseMatrix rigidLocal(budget, localDofs, supportCount, "relief rigid motions");
        for (std::size_t j = 0; j < supportCount; ++j)
            for (std::size_t local = 0; local < localDofs; ++local)
            {
                double rigid = 0.0;
                for (std::size_t a = 0; a < RIGID_BODY_MODE_COUNT; ++a)
                    rigid += phi(local, a) * fit[j * RIGID_BODY_MODE_COUNT + a];
                rigidLocal(local, j) = rigid;
            }
        // The misfit of the solved support modes to D, reported: the round-off of the interface's stiffness on
        // rigid motions amplified by the condition number of the structure held at its supports (the full path's own
        // support modes miss by 1.6e-4 on a free barrel of four frames at 1/32, 1.3 M DOFs), not a measure of what
        // resists a rigid motion (the reactions below are).
        if (solvesSupportModes)
        {
            double largest = 0.0;
            double misfit = 0.0;
            for (std::size_t local = 0; local < localDofs; ++local)
                for (std::size_t j = 0; j < supportCount && isMoving(local); ++j)
                {
                    largest = std::max(largest, std::abs(supportModes(local, j)));
                    misfit = std::max(misfit, std::abs(rigidLocal(local, j) - supportModes(local, j)));
                }
            reliefMisfit = misfit / std::max(largest, 1.0e-300);
        }
        // The reactions of D (the test of a free body: assembly::RIGID_SUPPORT_TOLERANCE): K·D on the equations and at
        // the held DOFs, the supports at their values (Φ·C)_S and the other held DOFs at rest, over the largest force
        // of a unit support motion. For a free body the round-off of the interface elements' sums and of the units' K*
        // on rigid motions, not amplified by a solve; the stiffness of a constraint or of a grounded element otherwise.
        linalg::DenseMatrix rigidEnforced(budget, localDofs, supportCount, "relief rigid support values");
        for (std::size_t j = 0; j < supportCount; ++j)
            for (const std::size_t local : heldLocals)
                rigidEnforced(local, j) = rigidLocal(local, j);
        linalg::DenseMatrix rigidEquations(budget, equations, supportCount, "relief rigid motions on the equations");
        for (std::size_t e = 0; e < equations; ++e)
        {
            const auto local = static_cast<std::size_t>(interface.localDofOfEquation(static_cast<std::int32_t>(e)));
            for (std::size_t j = 0; j < supportCount; ++j)
                rigidEquations(e, j) = rigidLocal(local, j);
        }
        linalg::DenseMatrix rigidRhs(budget, equations, supportCount, "relief rigid right-hand sides");
        interface.equationLoads(zero.view(), rigidEnforced.view(), rigidRhs.view());
        linalg::DenseMatrix rigidReactions(budget, equations, supportCount, "relief rigid reactions");
        interface.residuals(rigidEquations.view(), rigidRhs.view(), rigidReactions.view());
        linalg::DenseMatrix rigidExpanded(budget, localDofs, supportCount, "relief rigid motions expanded");
        interface.expand(rigidEquations.view(), rigidEnforced.view(), rigidExpanded.view());
        linalg::DenseMatrix heldReactions(budget, interface.heldDofs().size(), supportCount,
                                          "relief rigid held reactions");
        interface.heldReactions(rigidExpanded.view(), zero.view(), heldReactions.view());
        double stiffness = 0.0;
        double reaction = 0.0;
        for (std::size_t j = 0; j < supportCount; ++j)
        {
            for (std::size_t e = 0; e < equations; ++e)
            {
                stiffness = std::max(stiffness, std::abs(supportLoads(e, j)));
                reaction = std::max(reaction, std::abs(rigidReactions(e, j)));
            }
            for (std::size_t k = 0; k < heldReactions.rows(); ++k)
                reaction = std::max(reaction, std::abs(heldReactions(k, j)));
        }
        reliefReaction = reaction / std::max(stiffness, 1.0e-300);
        if (reaction > assembly::RIGID_SUPPORT_TOLERANCE * stiffness)
        {
            std::array<char, 32> share{};
            std::snprintf(share.data(), share.size(), "%.3g", reliefReaction);
            throw base::SolverError(base::concatenate(
                {"inertia relief on the condensed path: the structure is not free apart from its FreeBodySupport DOFs "
                 "(a rigid "
                 "motion of the supports makes reactions of ",
                 share.data(),
                 " of a support's stiffness: a constraint or an element resists a rigid motion): the units' rigid-body "
                 "mass properties do not give its inertia; remove the constraint, keep the units full, or run the full "
                 "path"}));
        }
        // The relief's support motions D = Φ·C of the FreeBodySupport DOFs (the held ones' when not conditioned).
        if (isConditioned)
        {
            std::array<double, assembly::MAX_SUPPORT_DOFS * RIGID_BODY_MODE_COUNT> atUserSupports{};
            for (std::size_t j = 0; j < supportCount; ++j)
                for (std::size_t a = 0; a < RIGID_BODY_MODE_COUNT; ++a)
                    atUserSupports[j * RIGID_BODY_MODE_COUNT + a] = phi(supportLocals[j], a);
            const auto exact = assembly::rigidSupportCoefficients(atUserSupports, supportCount);
            if (!exact)
                throw base::SolverError(
                    "inertia relief on the condensed path: the FreeBodySupport DOFs do not fix every rigid-body "
                    "motion; give "
                    "six FreeBodySupport DOFs that do (a statically determinate support), or run the full path");
            fit.assign(exact->begin(), exact->end());
        }
        reliefFit = fit;
        // DᵀMD = Cᵀ·Mr·C.
        base::MatN<6> rigidMass = interface.rigidBodyMass();
        for (const CondensedUnit& unit : units.units)
            rigidMass += unit.system->rigidBodyMass();
        if (exact || isConditioned)
        {
            reliefRigidMass.resize(RIGID_BODY_MODE_COUNT * RIGID_BODY_MODE_COUNT);
            reliefSupportRows.resize(supportCount * RIGID_BODY_MODE_COUNT);
            for (std::size_t a = 0; a < RIGID_BODY_MODE_COUNT; ++a)
                for (std::size_t b = 0; b < RIGID_BODY_MODE_COUNT; ++b)
                    reliefRigidMass[a * RIGID_BODY_MODE_COUNT + b] = rigidMass(a, b);
            for (std::size_t j = 0; j < supportCount; ++j)
                for (std::size_t a = 0; a < RIGID_BODY_MODE_COUNT; ++a)
                    reliefSupportRows[j * RIGID_BODY_MODE_COUNT + a] = phi(supportLocals[j], a);
        }
        reliefMass.assign(supportCount * supportCount, 0.0);
        for (std::size_t i = 0; i < supportCount; ++i)
            for (std::size_t j = 0; j < supportCount; ++j)
            {
                double sum = 0.0;
                for (std::size_t a = 0; a < RIGID_BODY_MODE_COUNT; ++a)
                    for (std::size_t b = 0; b < RIGID_BODY_MODE_COUNT; ++b)
                        sum +=
                            fit[i * RIGID_BODY_MODE_COUNT + a] * rigidMass(a, b) * fit[j * RIGID_BODY_MODE_COUNT + b];
                reliefMass[i * supportCount + j] = sum;
            }
        // The relief loads of the interface problem: the units' condensed inertial patterns and the interface's own.
        for (std::size_t u = 0; u < units.units.size(); ++u)
        {
            const CondensedUnit& unit = units.units[u];
            const linalg::DenseMatrix& condensed = unit.condensation->reducedUnit().patterns();
            const std::span<const std::int64_t> locals = interface.unitLocalDofs(u);
            for (std::size_t mode = 0; mode < RIGID_BODY_MODE_COUNT; ++mode)
                for (std::size_t i = 0; i < locals.size(); ++i)
                    reliefLoads(static_cast<std::size_t>(locals[i]), mode) +=
                        condensed(i, unit.loadPatternCount + mode);
        }
        if (interface.inertialLoads().columns() == RIGID_BODY_MODE_COUNT)
            for (std::size_t mode = 0; mode < RIGID_BODY_MODE_COUNT; ++mode)
                for (std::size_t local = 0; local < localDofs; ++local)
                    reliefLoads(local, mode) += interface.inertialLoads()(local, mode);
    }

    // The load basis handed over (GroupInputs::solvesCases false): the set kept with its interface factor, no case.
    if (!group.solvesCases)
    {
        if (!inputs.keepsSystems)
            throw std::logic_error(
                "CondensedStatics::solveGroup: a constraint set solved without its cases must be kept "
                "(Inputs::keepsSystems; a bug of O_o_Solver: report it)");
        CondensedSetRecord setRecord;
        setRecord.constraintSet = group.constraintSet;
        setRecord.interfaceDofCount = interface.localDofCount();
        setRecord.equationCount = interface.equationCount();
        setRecord.interfaceThreadCount = static_cast<std::int32_t>(interface.factorisationThreadCount());
        setRecord.interfaceSeconds = interfaceSeconds;
        setRecord.interfaceFactorValues = interface.factorValueCount();
        setRecord.isInterfaceFactorKept = interface.isFactorKept();
        setRecord.reliefMisfit = reliefMisfit;
        setRecord.reliefReaction = reliefReaction;
        for (std::size_t j = 0; isConditioned && j < heldSupports.size(); ++j)
            setRecord.reliefSolveSupports.push_back(
                {dofMap.pointIdOf(heldSupports[j]), dofMap.labelOf(heldSupports[j]).component});
        m_record.constraintSets.push_back(setRecord);
        // Inertia relief of the load basis: what its basis vectors are relieved and recovered with.
        std::shared_ptr<CondensedRelief> relief;
        if (hasRelief)
        {
            relief = std::make_shared<CondensedRelief>();
            relief->supportCount = supportCount;
            relief->fit = reliefFit;
            relief->reducedMass = reliefMass;
            relief->rigidMass = reliefRigidMass;
            relief->reliefLoads = std::move(reliefLoads);
            relief->supportLocals = supportLocals;
            relief->isConditioned = isConditioned;
        }
        // The held DOFs: with conditioned supports the FreeBodySupport DOFs, not the DOFs the interface holds in their
        // place.
        base::BudgetedBuffer<std::uint8_t> heldDofs(budget, gSize, "condensed constrained DOFs");
        for (const std::int64_t local : interface.heldDofs())
            if (const std::optional<base::DofIndex> dof = interface.modelDofOf(local))
                heldDofs[static_cast<std::size_t>(*dof)] =
                    isConditioned && std::find(heldSupports.begin(), heldSupports.end(), *dof) != heldSupports.end()
                        ? 0
                        : 1;
        if (isConditioned)
            for (const base::DofIndex dof : group.supports)
                heldDofs[static_cast<std::size_t>(dof)] = 1;
        for (const CondensedUnit& unit : units.units)
            for (const std::int64_t local : unit.system->heldDofs())
                if (unit.modelDofOfLocal[static_cast<std::size_t>(local)] >= 0)
                    heldDofs[static_cast<std::size_t>(unit.modelDofOfLocal[static_cast<std::size_t>(local)])] = 1;
        keepSet(group, std::move(interfaceOwner), std::move(fullGmOwner), std::move(heldDofs), std::move(placements),
                budget, std::move(relief));
        m_solvedCaseCount += caseCount;
        m_record.memory.measuredPeakBytes = budget.peak();
        m_record.memory.keptFactorBytes = m_memory->retention.keptBytes();
        m_record.memory.keptFactorCount = m_memory->retention.keptCount();
        m_record.memory.releasedForBudget = m_memory->retention.releasedForBudget();
        return singularityTable;
    }

    // The coefficients of the group's cases: each unit's load patterns, the interface's vectors.
    std::vector<reduction::BasisCoefficients> unitAlphas; // bounded by the units
    for (const CondensedUnit& unit : units.units)
        unitAlphas.push_back(columnsOf(unit.basis.coefficients(), caseIndices, budget));
    const reduction::BasisCoefficients interfaceAlpha =
        columnsOf(loads.interfaceBasis.coefficients(), caseIndices, budget);
    // The interface's vectors on its local DOFs.
    const std::size_t vectorCount = loads.interfaceBasis.size();
    linalg::DenseMatrix interfaceLoads(budget, localDofs, vectorCount, "interface vector loads");
    linalg::DenseMatrix interfaceEnforced(budget, localDofs, loads.hasEnforced ? vectorCount : 0,
                                          "interface vector enforced values");
    for (std::size_t k = 0; k < vectorCount; ++k)
        for (auto t = loads.offsets[k]; t < loads.offsets[k + 1]; ++t)
        {
            const auto entry = static_cast<std::size_t>(t);
            const std::optional<std::int64_t> local = interface.localDofOf(loads.dofs[entry]);
            if (!local)
                continue;
            interfaceLoads(static_cast<std::size_t>(*local), k) = loads.loadValues[entry];
            if (loads.hasEnforced)
                interfaceEnforced(static_cast<std::size_t>(*local), k) = loads.enforcedValues[entry];
        }

    // The external units: the loads of their zones' interior DOFs, per vector of the interface's basis, go to their
    // plugins (ExternalUnitPath::condenseLoads), whose condensed loads f* act on their contours (as the full path's
    // applyToBasisVector).
    if (inputs.externalUnits != nullptr)
    {
        std::vector<linalg::DenseMatrix> interior; // per external unit, interior × vectors
        for (std::size_t e = 0; e < inputs.externalUnits->unitCount(); ++e)
        {
            const std::shared_ptr<const reduction::ExternalUnitAssembly> placement = inputs.externalUnits->placement(e);
            interior.emplace_back(budget, placement ? placement->interiorSize() : 0, vectorCount,
                                  "external unit interior loads");
            if (!placement)
                continue;
            const std::vector<base::DofIndex>& dofs = placement->zone().interiorDofs;
            for (std::size_t i = 0; i < dofs.size(); ++i)
                if (const std::optional<std::int64_t> local = interface.localDofOf(dofs[i]))
                    for (std::size_t k = 0; k < vectorCount; ++k)
                        interior.back()(i, k) = interfaceLoads(static_cast<std::size_t>(*local), k);
        }
        inputs.externalUnits->condenseLoads(group.constraintSet, group.cases, std::move(interior),
                                            interfaceAlpha.matrix(), budget, sink);
        stopOnErrors(sink, "the external units");
        for (std::size_t e = 0; e < inputs.externalUnits->unitCount(); ++e)
        {
            const std::shared_ptr<const reduction::ExternalUnitAssembly> placement = inputs.externalUnits->placement(e);
            if (!placement)
                continue;
            for (const base::DofIndex dof : placement->zone().interiorDofs)
                if (const std::optional<std::int64_t> local = interface.localDofOf(dof))
                    for (std::size_t k = 0; k < vectorCount; ++k)
                        interfaceLoads(static_cast<std::size_t>(*local), k) = 0.0;
            const linalg::DenseMatrix& condensed = inputs.externalUnits->currentLoads(e);
            const std::vector<base::DofIndex>& contour = placement->zone().contourDofs;
            for (std::size_t a = 0; a < contour.size(); ++a)
                if (const std::optional<std::int64_t> local = interface.localDofOf(contour[a]))
                    for (std::size_t k = 0; k < vectorCount; ++k)
                        interfaceLoads(static_cast<std::size_t>(*local), k) += condensed(a, k);
        }
    }

    // The accelerations of the cases: ü_r = (DᵀMD)⁻¹·Cᵀ·Φ_rᵀ·P of each case, a = C·ü_r the coefficients of the relief
    // loads.
    linalg::DenseMatrix reliefCoefficients(budget, RIGID_BODY_MODE_COUNT, hasRelief ? caseCount : 0,
                                           "relief coefficients");
    std::vector<std::vector<double>> accelerations(caseCount); // bounded by the cases × the supports
    if (hasRelief)
    {
        const std::vector<double>& fit = reliefFit;
        // The resultant Φ_rᵀ·P of each case (about the basic origin): from each vector's own resultant.
        std::vector<double> resultants(RIGID_BODY_MODE_COUNT * caseCount, 0.0);
        const auto addResultants = [&](const linalg::DenseMatrix& modes, linalg::ConstDenseMatrixView vectors,
                                       const reduction::BasisCoefficients& alpha, std::size_t vectorTotal)
        {
            for (std::size_t v = 0; v < vectorTotal; ++v)
            {
                std::array<double, RIGID_BODY_MODE_COUNT> own{};
                for (std::size_t local = 0; local < vectors.rows; ++local)
                {
                    const double value = vectors(local, v);
                    if (value == 0.0)
                        continue;
                    for (std::size_t a = 0; a < RIGID_BODY_MODE_COUNT; ++a)
                        own[a] += modes(local, a) * value;
                }
                for (std::size_t c = 0; c < caseCount; ++c)
                    for (std::size_t a = 0; a < RIGID_BODY_MODE_COUNT; ++a)
                        resultants[c * RIGID_BODY_MODE_COUNT + a] += alpha(v, c) * own[a];
            }
        };
        for (std::size_t u = 0; u < units.units.size(); ++u)
            addResultants(units.units[u].system->makeRigidBodyModes(budget), units.units[u].patternLoads.view(),
                          unitAlphas[u], units.units[u].loadPatternCount);
        addResultants(interface.rigidBodyModes(), interfaceLoads.view(), interfaceAlpha, vectorCount);
        // C exact (six supports): a = Mr⁻¹·R, ü_r = Φ_S·a.
        if (!reliefRigidMass.empty())
        {
            if (!solveSmallSpd(reliefRigidMass, RIGID_BODY_MODE_COUNT, resultants))
                throw base::SolverError(
                    "inertia relief: the rigid-body mass of the free body is singular (a rigid-body motion that "
                    "moves no mass); check the masses of the model");
            for (std::size_t c = 0; c < caseCount; ++c)
            {
                accelerations[c].assign(supportCount, 0.0);
                for (std::size_t a = 0; a < RIGID_BODY_MODE_COUNT; ++a)
                {
                    const double amplitude = resultants[c * RIGID_BODY_MODE_COUNT + a];
                    reliefCoefficients(a, c) = amplitude;
                    for (std::size_t j = 0; j < supportCount; ++j)
                        accelerations[c][j] += reliefSupportRows[j * RIGID_BODY_MODE_COUNT + a] * amplitude;
                }
            }
        }
        // Otherwise (fewer supports, C fitted): ü_r = (DᵀMD)⁻¹·Cᵀ·R, a = C·ü_r.
        std::vector<double> rhs(reliefRigidMass.empty() ? supportCount * caseCount : 0, 0.0);
        for (std::size_t c = 0; c < caseCount && !rhs.empty(); ++c)
            for (std::size_t j = 0; j < supportCount; ++j)
            {
                double sum = 0.0;
                for (std::size_t a = 0; a < RIGID_BODY_MODE_COUNT; ++a)
                    sum += fit[j * RIGID_BODY_MODE_COUNT + a] * resultants[c * RIGID_BODY_MODE_COUNT + a];
                rhs[c * supportCount + j] = sum;
            }
        if (!rhs.empty() && !solveSmallSpd(reliefMass, supportCount, rhs))
            throw base::SolverError(
                "inertia relief: DᵀMD is singular (a FreeBodySupport DOF whose motion moves no mass); "
                "place the FreeBodySupport DOFs where their rigid-body motion moves mass, or check the "
                "masses of the model");
        for (std::size_t c = 0; c < caseCount && !rhs.empty(); ++c)
        {
            accelerations[c].assign(rhs.begin() + static_cast<std::ptrdiff_t>(c * supportCount),
                                    rhs.begin() + static_cast<std::ptrdiff_t>((c + 1) * supportCount));
            for (std::size_t a = 0; a < RIGID_BODY_MODE_COUNT; ++a)
            {
                double sum = 0.0;
                for (std::size_t j = 0; j < supportCount; ++j)
                    sum += fit[j * RIGID_BODY_MODE_COUNT + a] * rhs[c * supportCount + j];
                reliefCoefficients(a, c) = sum;
            }
        }
    }
    const reduction::BasisCoefficients reliefAlpha(std::move(reliefCoefficients));

    // The boundary basis.
    std::vector<reduction::BoundaryUnitPatterns> unitPatterns; // bounded by the units
    for (std::size_t u = 0; u < units.units.size(); ++u)
    {
        const CondensedUnit& unit = units.units[u];
        const linalg::DenseMatrix& condensed = unit.condensation->reducedUnit().patterns();
        unitPatterns.push_back({condensed.view().block(0, 0, condensed.rows(), unit.loadPatternCount), &unitAlphas[u]});
    }
    std::vector<reduction::BoundaryLocalLoads> blocks{
        {interfaceLoads.view(), interfaceEnforced.view(), &interfaceAlpha}};
    if (hasRelief)
        blocks.push_back({reliefLoads.view(), {}, &reliefAlpha});
    // The basis's solve skipped (the interface handed over) or replaced by the caller's values.
    reduction::BoundarySolveOptions solveOptions;
    linalg::DenseMatrix givenLocal; // GivenLocal: the unit's boundary rows put on the local DOFs, the others zero
    if (group.exportsInterface)
    {
        solveOptions.mode =
            group.solvesExport ? reduction::BoundarySolveMode::Solve : reduction::BoundarySolveMode::RightHandSidesOnly;
        solveOptions.keepsEquationSolutions = group.solvesExport;
    }
    else if (group.boundaryMode == reduction::BoundarySolveMode::GivenEquations)
    {
        solveOptions.mode = group.boundaryMode;
        solveOptions.solvesCases = group.isGivenPerCase;
        solveOptions.given = group.given;
    }
    else if (group.boundaryMode == reduction::BoundarySolveMode::GivenLocal)
    {
        const auto found = std::find_if(units.units.begin(), units.units.end(),
                                        [&](const CondensedUnit& unit) { return unit.unitId == group.givenUnitId; });
        if (found == units.units.end())
            throw std::invalid_argument(base::concatenate(
                {"recover: unit ", std::to_string(group.givenUnitId),
                 " is not condensed by the plan; name a unit the plan condenses, or recover from the interface's "
                 "solution"}));
        const auto u = static_cast<std::size_t>(found - units.units.begin());
        const std::span<const std::int64_t> locals = interface.unitLocalDofs(u);
        if (group.given.rows != locals.size() || group.given.columns != caseCount)
            throw std::invalid_argument(base::concatenate(
                {"recover: the boundary displacements of unit ", std::to_string(group.givenUnitId), " are ",
                 std::to_string(group.given.rows), " x ", std::to_string(group.given.columns), " for constraint set ",
                 std::to_string(group.constraintSet), "; expected ", std::to_string(locals.size()),
                 " rows (the boundary DOFs of its reduced unit) by ", std::to_string(caseCount), " cases"}));
        givenLocal = linalg::DenseMatrix(budget, localDofs, caseCount, "given boundary displacements");
        for (std::size_t c = 0; c < caseCount; ++c)
            for (std::size_t i = 0; i < locals.size(); ++i)
                givenLocal(static_cast<std::size_t>(locals[i]), c) = group.given(i, c);
        solveOptions.mode = group.boundaryMode;
        solveOptions.solvesCases = true;
        solveOptions.given = givenLocal.view();
    }
    const reduction::BoundaryBasis basis(interface, unitPatterns, blocks, caseCount, kernels, budget,
                                         inputs.cancellation, solveOptions);
    if (isInterfaceFactorised && !interface.isFactorKept())
        reportInterfacePivots();
    CondensedSetRecord setRecord;
    setRecord.constraintSet = group.constraintSet;
    setRecord.reliefMisfit = reliefMisfit;
    setRecord.reliefReaction = reliefReaction;
    for (std::size_t j = 0; isConditioned && j < heldSupports.size(); ++j)
        setRecord.reliefSolveSupports.push_back(
            {dofMap.pointIdOf(heldSupports[j]), dofMap.labelOf(heldSupports[j]).component});
    setRecord.interfaceDofCount = interface.localDofCount();
    setRecord.equationCount = interface.equationCount();
    setRecord.distinctRightHandSides = static_cast<std::int64_t>(basis.distinctRightHandSides());
    setRecord.solvedColumnCount = static_cast<std::int64_t>(basis.columnCount());
    setRecord.isCaseBasis = basis.isCaseBasis();
    setRecord.interfaceThreadCount = static_cast<std::int32_t>(interface.factorisationThreadCount());
    setRecord.interfaceSeconds = interfaceSeconds;
    setRecord.basisSeconds = clock.lap();
    setRecord.interfaceFactorValues = interface.factorValueCount();
    setRecord.isInterfaceFactorKept = interface.isFactorKept();
    {
        const reduction::InterfaceSolveFigures figures = interface.solveFigures();
        setRecord.interfaceSolveCount = figures.solveCount;
        setRecord.interfaceSolvedColumns = figures.columnCount;
        setRecord.interfaceRegionRuns = figures.regionRunCount;
        setRecord.interfaceRecomputedFlops = figures.recomputedFlops;
        setRecord.interfaceKeptPanelValues = figures.keptPanelValues;
    }

    // The interface problem handed over, nothing recovered.
    if (group.exportsInterface)
    {
        InterfaceExport handed;
        handed.constraintSet = group.constraintSet;
        for (const std::size_t s : group.cases)
            handed.subcases.push_back(inputs.subcases[s].label);
        handed.localLabels = base::BudgetedBuffer<reduction::UnitDofLabel>(budget, localDofs, "interface DOF labels");
        for (std::int32_t point = 0; point < static_cast<std::int32_t>(interface.points().size()); ++point)
        {
            const reduction::InterfacePoint& entry = interface.points()[static_cast<std::size_t>(point)];
            for (std::int64_t k = 0; k < static_cast<std::int64_t>(entry.dofCount); ++k)
            {
                reduction::UnitDofLabel label = entry.label;
                label.component = label.kind == reduction::UnitPointKind::Scalar ? 0 : static_cast<std::uint8_t>(k + 1);
                handed.localLabels[static_cast<std::size_t>(interface.firstDof(point) + k)] = label;
            }
        }
        handed.localOfEquation = base::BudgetedBuffer<std::int64_t>(budget, equations, "interface equation DOFs");
        for (std::size_t e = 0; e < equations; ++e)
            handed.localOfEquation[e] = interface.localDofOfEquation(static_cast<std::int32_t>(e));
        handed.heldLocals =
            base::BudgetedBuffer<std::int64_t>(budget, interface.heldDofs().size(), "interface held DOFs");
        std::copy(interface.heldDofs().begin(), interface.heldDofs().end(), handed.heldLocals.data());
        const reduction::LocalRelations& relations = interface.relations();
        handed.relationDependents =
            base::BudgetedBuffer<std::int64_t>(budget, relations.rowCount(), "interface relation rows");
        handed.relationOffsets =
            base::BudgetedBuffer<std::int64_t>(budget, relations.rowCount() + 1, "interface relation rows");
        handed.relationTermDofs =
            base::BudgetedBuffer<std::int64_t>(budget, relations.termCount(), "interface relation terms");
        handed.relationTermCoefficients =
            base::BudgetedBuffer<double>(budget, relations.termCount(), "interface relation terms");
        std::size_t term = 0;
        for (std::size_t r = 0; r < relations.rowCount(); ++r)
        {
            handed.relationDependents[r] = relations.dependent(r);
            handed.relationOffsets[r] = static_cast<std::int64_t>(term);
            for (const reduction::LocalTerm& entry : relations.terms(r))
            {
                handed.relationTermDofs[term] = entry.dof;
                handed.relationTermCoefficients[term] = entry.coefficient;
                ++term;
            }
        }
        handed.relationOffsets[relations.rowCount()] = static_cast<std::int64_t>(term);
        const linalg::SymmetricCsc& matrix = interface.matrix();
        const auto copyOf = [&budget](auto source, auto& target, const char* label)
        {
            using Value = std::remove_const_t<typename decltype(source)::element_type>;
            target = base::BudgetedBuffer<Value>(budget, source.size(), label);
            std::copy(source.begin(), source.end(), target.data());
        };
        copyOf(matrix.columnOffsets(), handed.matrixOffsets, "interface matrix columns");
        copyOf(matrix.rowIndices(), handed.matrixRows, "interface matrix rows");
        copyOf(matrix.values(), handed.matrixValues, "interface matrix values");
        const auto copyMatrix = [&budget](const linalg::DenseMatrix& source, const char* label)
        {
            linalg::DenseMatrix copy(budget, source.rows(), source.columns(), label);
            for (std::size_t j = 0; j < source.columns(); ++j)
                for (std::size_t i = 0; i < source.rows(); ++i)
                    copy(i, j) = source(i, j);
            return copy;
        };
        handed.rightHandSides = copyMatrix(basis.rightHandSides(), "interface right-hand sides");
        handed.enforced = copyMatrix(basis.enforcedValues(), "interface enforced values");
        handed.coefficients = copyMatrix(basis.coefficients().matrix(), "interface case coefficients");
        handed.isCaseBasis = basis.isCaseBasis();
        for (const reduction::BoundaryColumn& column : basis.columns())
        {
            InterfaceExport::Column entry;
            if (column.unit >= 0)
            {
                entry.unitId = units.units[static_cast<std::size_t>(column.unit)].unitId;
                entry.pattern = column.vector;
            }
            else
            {
                entry.block = column.block;
                entry.vector = column.vector;
            }
            handed.columns.push_back(entry);
        }
        if (group.solvesExport)
            handed.solutions = copyMatrix(basis.equationSolutions(), "interface solutions");
        for (std::size_t u = 0; u < units.units.size(); ++u)
        {
            handed.unitIds.push_back(units.units[u].unitId);
            const std::span<const std::int64_t> locals = interface.unitLocalDofs(u);
            handed.unitLocals.emplace_back(locals.begin(), locals.end());
        }
        for (std::size_t b = 0; b < interface.externalBlockCount(); ++b)
        {
            handed.externalZoneNames.push_back(placements[b]->zone().name);
            const std::span<const std::int64_t> locals = interface.externalLocalDofs(b);
            handed.externalLocals.emplace_back(locals.begin(), locals.end());
        }
        m_interfaceExports.push_back(std::move(handed));
        m_record.constraintSets.push_back(setRecord);
        m_solvedCaseCount += caseCount;
        m_record.memory.measuredPeakBytes = budget.peak();
        return singularityTable;
    }

    // Each unit's coefficients over all its patterns (with relief, the six inertial ones: a), and the recovery of
    // the units asked for.
    std::vector<reduction::BasisCoefficients> fullAlphas; // bounded by the units
    for (std::size_t u = 0; u < units.units.size(); ++u)
    {
        const CondensedUnit& unit = units.units[u];
        const std::size_t patternCount = unit.patternLoads.columns();
        linalg::DenseMatrix alpha(budget, patternCount, caseCount, "condensed unit coefficients");
        for (std::size_t c = 0; c < caseCount; ++c)
        {
            for (std::size_t k = 0; k < unit.loadPatternCount; ++k)
                alpha(k, c) = unitAlphas[u](k, c);
            for (std::size_t mode = 0; hasRelief && mode < RIGID_BODY_MODE_COUNT; ++mode)
                alpha(unit.loadPatternCount + mode, c) = reliefAlpha(mode, c);
        }
        fullAlphas.emplace_back(std::move(alpha));
    }
    // The recoveries (the strategy of the memory plan). All at once: every recovered unit's recovery made here, in
    // waves of consecutive units side by side (below), before the cases: a unit whose interior factor is not in memory
    // (released for the budget, or never made: the cache) factorises its interior again, to the bits of its
    // condensation (its K* checked against the reduced unit's), alone on every thread when it is large and that fits
    // the budget left; once its basis solutions are made the factor goes again unless the unit solves its cases one by
    // one (regime "load cases", the factor read through the cases) or FactorRetention keeps it for the next constraint
    // sets. So the factors made for a recovery in memory at a time beside the kept ones are one wave's, within half the
    // bytes free (one factor when no more fit). In sequence: each unit's recovery made in its wave of each batch of
    // cases (below), its equations built again first when the memory mode released them, and released with its factor
    // and those equations once its cases are handed over.
    const bool isSequential = m_memory->plan->strategy().recovery == reduction::RecoveryStrategy::Sequential;
    std::vector<std::unique_ptr<reduction::UnitRecovery>> recoveries(units.units.size());
    // The loads of each recovered unit's patterns on its equations: the right-hand sides of its interior residuals.
    std::vector<linalg::DenseMatrix> recoveredEquationLoads(units.units.size()); // bounded by the units
    std::vector<reduction::UnitBasisRows> unitRows;                              // bounded by the units
    unitRows.reserve(units.units.size());
    for (std::size_t u = 0; u < units.units.size(); ++u)
        unitRows.push_back(basis.unitRows(u, budget));
    std::vector<std::size_t> releasedAfterCases; // bounded by the units
    const std::size_t threadCount = inputs.pool != nullptr ? inputs.pool->threadCount() : 1;
    linalg::DenseKernelOptions serialOptions = kernels.options();
    serialOptions.pool = nullptr;
    const linalg::DenseKernels serialKernels(kernels.kind(), serialOptions);
    // The interior factor an earlier analysis kept for the unit's equations, taken in place of a factorisation again
    // (its K* checked against the condensation's to the bit); false when none is kept or its bytes do not fit the
    // budget (it is then freed).
    const auto adoptKeptFactor =
        [&](CondensedUnit& unit, CondensedUnitRecord& record, const linalg::DenseKernels& factorKernels)
    {
        if (inputs.keptFactors == nullptr || inputs.isCraigBampton)
            return false;
        std::optional<reduction::KeptInteriorFactor> kept =
            inputs.keptFactors->take(unit.identity, *unit.system, factorKernels.kind(), unit.factorOptions.pivotPolicy);
        if (!kept)
            return false;
        try
        {
            unit.condensation->adoptInteriorFactor(std::move(*kept), factorKernels, budget);
        }
        catch (const base::MemoryBudgetError&)
        {
            return false;
        }
        record.isInteriorFactorReused = true;
        return true;
    };
    // A unit's recovery for the set: its equations built again when released (the same bits: its numbering checked
    // here, its K* when its interior is factorised again), its interior factorised again when its factor is not in
    // memory (or the one an earlier analysis kept taken), its UnitRecovery and the loads of its patterns on its
    // equations. Returns whether its factor was made for it. Writes only the unit's own state and record (units of a
    // wave run it side by side).
    const auto prepareRecovery = [&](std::size_t u, const linalg::DenseKernels& factorKernels,
                                     const linalg::DenseKernels& recoveryKernels, base::ThreadPool* buildPool)
    {
        CondensedUnit& unit = units.units[u];
        CondensedUnitRecord& record = m_record.units[u];
        if (!unit.system->hasEquations())
        {
            auto equations = std::make_shared<const reduction::UnitSystem>(
                units.unitModel, unit.description, reduction::UnitSystem::WithoutMass{unit.system->rigidBodyMass()},
                budget, buildPool);
            if (!isSameNumbering(unit.modelDofOfLocal.span(), *equations))
                throw base::InternalError(base::concatenate(
                    {"CondensedStatics: unit ", std::to_string(unit.unitId),
                     ": its equations built again for its recovery number its DOFs otherwise than its condensation's "
                     "(an internal error: report it)"}));
            unit.condensation->rebindSystem(equations, equationNamerOf(equations.get()));
            unit.system = std::move(equations);
            ++record.rebuildCount;
        }
        bool isMade = false;
        const reduction::MemoryStrategy& strategy = m_memory->plan->strategy();
        if (!unit.condensation->isInteriorFactorKept() && strategy.isWithoutFactor &&
            unit.footprint.isRecoveredByRegions(strategy.regionBytes))
        {
            // Its factor does not fit the bound: its interior solved by regions of its elimination tree, factorised
            // again in the recovery's solves, none above the bound (the same bits).
            unit.condensation->enableRecoveryWithoutFactor(
                static_cast<std::int64_t>(strategy.regionBytes / sizeof(double)), budget);
            record.isRecoveredByRegions = true;
        }
        else if (!unit.condensation->isInteriorFactorKept() && adoptKeptFactor(unit, record, factorKernels))
            isMade = true;
        else if (!unit.condensation->isInteriorFactorKept())
        {
            const bool isFirst = unit.condensation->isFromCache() && !record.isRefactorisedForRecovery;
            unit.condensation->recomputeInteriorFactor(factorKernels, budget, inputs.cancellation);
            record.isRefactorisedForRecovery = record.isRefactorisedForRecovery || isFirst;
            ++record.refactorisationCount;
            isMade = true;
        }
        recoveries[u] = std::make_unique<reduction::UnitRecovery>(
            *unit.condensation, unit.patterns(), fullAlphas[u],
            reduction::UnitBoundaryBasis{unitRows[u].values.view(), basis.coefficients().matrix().view(),
                                         unitRows[u].patternOfColumn},
            recoveryKernels, budget, inputs.cancellation);
        recoveredEquationLoads[u] = linalg::DenseMatrix(budget, static_cast<std::size_t>(unit.system->equationCount()),
                                                        unit.patternCount, "unit recovery equation loads");
        if (unit.patternCount > 0)
            unit.system->equationLoads(unit.patternLoads.view(), unit.patternEnforced.view(),
                                       recoveredEquationLoads[u].view(), budget);
        return isMade;
    };
    // In sequence, after a unit's pass: its recovery, the loads of its patterns, its factor (but one FactorRetention
    // kept) and its equations built again are released.
    const auto releaseRecovery = [&](std::size_t u)
    {
        CondensedUnit& unit = units.units[u];
        recoveries[u].reset();
        recoveredEquationLoads[u] = linalg::DenseMatrix();
        if (unit.condensation->isRecoveryWithoutFactor())
        {
            m_record.units[u].regionRecoveryCount = unit.condensation->regionRecoveryCount();
            unit.condensation->disableRecoveryWithoutFactor();
        }
        if (!unit.isFactorKept)
            unit.condensation->releaseInteriorFactor();
        if (unit.lightSystem && unit.system != unit.lightSystem)
        {
            unit.condensation->rebindSystem(unit.lightSystem, equationNamerOf(unit.lightSystem.get()));
            unit.system = unit.lightSystem;
        }
    };
    // A unit whose interior factorisation is large (its work at least Inputs::largeUnitMinimumFlops) is recovered alone
    // with every thread; the others up to WAVE_UNITS_PER_THREAD a thread side by side, one thread each (the same bits:
    // a unit's kernels do not depend on the threads).
    const auto isLargeUnit = [&](std::size_t u)
    { return threadCount > 1 && units.units[u].footprint.flopCount >= inputs.largeUnitMinimumFlops; };
    const std::size_t largestWave = threadCount * WAVE_UNITS_PER_THREAD;
    if (!isSequential)
    {
        // PERFORMANCE: the recoveries made in waves of consecutive recovered units side by side (each unit on one
        // thread; its decisions after the wave, in order), where they were made one after the other with the threaded
        // kernels: the small units' factorisations made again gain nothing from the threads.
        // Measured: 3 000 plates of 10 M DOFs in the mode speed, 2 182 interior factors made again, the recoveries
        // 8.7 s before.
        std::vector<std::size_t> recovered; // the recovered units in order, bounded by the units
        for (std::size_t u = 0; u < units.units.size(); ++u)
            if (units.units[u].isRecovered)
                recovered.push_back(u);
        const auto prepareBytesOf = [&](std::size_t u)
        {
            const reduction::UnitFootprint& footprint = units.units[u].footprint;
            return footprint.factorisationBytes + footprint.recoveryBytes;
        };
        std::size_t begin = 0;
        while (begin < recovered.size())
        {
            std::size_t end = begin + 1;
            if (inputs.pool != nullptr && !isLargeUnit(recovered[begin]))
            {
                const auto room =
                    static_cast<std::uint64_t>(WAVE_BUDGET_FRACTION * static_cast<double>(budget.available()));
                std::uint64_t needed = prepareBytesOf(recovered[begin]);
                while (end < recovered.size() && end - begin < largestWave && !isLargeUnit(recovered[end]) &&
                       needed + prepareBytesOf(recovered[end]) <= room)
                    needed += prepareBytesOf(recovered[end++]);
            }
            std::vector<std::uint8_t> isMade(end - begin, 0); // bounded by a wave
            if (end - begin == 1)
            {
                const CondensedUnit& unit = units.units[recovered[begin]];
                const bool isThreaded = threadCount > 1 && !unit.condensation->isInteriorFactorKept() &&
                                        linalg::SupernodalFactor::estimateMemoryBytes(
                                            unit.system->symbolic(), threadCount,
                                            linalg::PivotingMode::PositiveDefinite) <= budget.available();
                isMade.front() = prepareRecovery(recovered[begin], isThreaded ? kernels : serialKernels, kernels,
                                                 isThreaded ? inputs.pool : nullptr)
                                     ? 1
                                     : 0;
            }
            else
                inputs.pool->runBatch(
                    end - begin,
                    [&](std::size_t k)
                    {
                        isMade[k] =
                            prepareRecovery(recovered[begin + k], serialKernels, serialKernels, nullptr) ? 1 : 0;
                    });
            // The factors made in the wave after a unit, held until their own decision: free for its decision, as when
            // the units were prepared one after the other.
            std::uint64_t laterFactorBytes = 0;
            for (std::size_t k = begin; k < end; ++k)
                if (isMade[k - begin] != 0)
                    laterFactorBytes += units.units[recovered[k]].condensation->interiorFactorBytes();
            for (std::size_t k = begin; k < end; ++k)
            {
                const std::size_t u = recovered[k];
                CondensedUnit& unit = units.units[u];
                CondensedUnitRecord& record = m_record.units[u];
                record.recoveries.push_back(recoveries[u]->statistics());
                if (isMade[k - begin] == 0)
                    continue;
                laterFactorBytes -= unit.condensation->interiorFactorBytes();
                if (unit.condensation->isFromCache() && record.factorRetention.empty())
                {
                    // The first factor of a unit from the cache: kept for the next constraint sets when the budget
                    // holds it beside the later phases of the analysis (their estimate, an upper bound of what
                    // remains).
                    record.factorBytes = unit.condensation->interiorFactorBytes();
                    const reduction::RetentionDecision decision = m_memory->retention.decide(
                        unit.unitId, record.factorBytes, true, budget.available() + laterFactorBytes,
                        m_memory->plan->laterPhaseBytes());
                    record.factorRetention = reduction::nameOf(decision);
                    record.factorRetentionReason = m_memory->retention.records().back().reason;
                    unit.isFactorKept = decision == reduction::RetentionDecision::Kept;
                }
                if (unit.isFactorKept)
                    continue;
                if (recoveries[u]->hasBasis())
                    unit.condensation->releaseInteriorFactor();
                else
                    releasedAfterCases.push_back(u);
            }
            begin = end;
        }
    }
    setRecord.recoverySeconds = clock.lap();

    // The output Gm of the MPC forces: the full path's, without the rows of the units not recovered.
    std::unique_ptr<assembly::RigidElimination> outputGm;
    if (!units.isEveryUnitRecovered)
    {
        base::DiagnosticSink ignored;
        const std::span<const std::uint8_t> interior = units.planned->interiorNodes();
        const Units* unitsView = &units;
        outputGm = std::make_unique<assembly::RigidElimination>(
            rigidElements, &mpc, dofMap, budget, ignored, units.fasteners.get(), &glue,
            [&dofMap, interior, unitsView](const assembly::ConstraintRowOrigin& origin)
            {
                const std::int32_t node = dofMap.nodeDofOf(origin.dependent).node;
                if (interior[static_cast<std::size_t>(node)] == 0)
                    return true;
                const std::optional<std::size_t> owner = unitsView->planned->unitOfInteriorNode(node);
                return owner && unitsView->isRecoveredPlanUnit[*owner] != 0;
            });
    }
    const assembly::RigidElimination& mpcGm = outputGm ? *outputGm : fullGm;
    const bool hasMset = fullGm.rowCount() > 0;

    // The interface's held DOFs and the units' on the g set: with conditioned supports, the FreeBodySupport DOFs
    // rather than the DOFs the interface holds in their place.
    const auto isConditionedSupport = [&](base::DofIndex dof)
    { return isConditioned && std::find(heldSupports.begin(), heldSupports.end(), dof) != heldSupports.end(); };
    base::BudgetedBuffer<std::uint8_t> isConstrained(budget, gSize, "condensed constrained DOFs");
    for (const std::int64_t local : interface.heldDofs())
        if (const std::optional<base::DofIndex> dof = interface.modelDofOf(local))
            isConstrained[static_cast<std::size_t>(*dof)] = isConditionedSupport(*dof) ? 0 : 1;
    if (isConditioned)
        for (const base::DofIndex dof : group.supports)
            isConstrained[static_cast<std::size_t>(dof)] = 1;
    for (const CondensedUnit& unit : units.units)
        for (const std::int64_t local : unit.system->heldDofs())
            if (unit.modelDofOfLocal[static_cast<std::size_t>(local)] >= 0)
                isConstrained[static_cast<std::size_t>(unit.modelDofOfLocal[static_cast<std::size_t>(local)])] = 1;
    // The DOFs whose SPC force a batch adds to (the interface's held DOFs, the recovered units'): set to zero at the
    // start of each batch, the others never written.
    std::vector<base::DofIndex> reactionDofs; // bounded by the held DOFs
    for (const std::int64_t local : interface.heldDofs())
        if (const std::optional<base::DofIndex> dof = interface.modelDofOf(local))
            if (!isConditionedSupport(*dof))
                reactionDofs.push_back(*dof);
    if (isConditioned)
        reactionDofs.insert(reactionDofs.end(), group.supports.begin(), group.supports.end());
    for (const CondensedUnit& unit : units.units)
        if (unit.isRecovered)
            for (const std::int64_t local : unit.system->heldDofs())
                if (unit.modelDofOfLocal[static_cast<std::size_t>(local)] >= 0)
                    reactionDofs.push_back(unit.modelDofOfLocal[static_cast<std::size_t>(local)]);
    // Whether each unit holds a DOF in every subcase with a stiffness inside it (hasInteriorReactions), once per set
    // (a search of the units per unit and per case before: 10 000 units).
    std::vector<std::uint8_t> isInteriorReacting(units.units.size(), 0); // bounded by the units
    for (std::size_t u = 0; u < units.units.size(); ++u)
    {
        const CondensedUnit& unit = units.units[u];
        const std::span<const std::int64_t> held = unit.system->heldDofs();
        for (std::size_t k = 0; k < held.size() && isInteriorReacting[u] == 0; ++k)
            isInteriorReacting[u] =
                unit.isInteriorLocal[static_cast<std::size_t>(held[k])] != 0 && unit.system->isHeldDofCoupled(k) ? 1
                                                                                                                 : 0;
    }
    // Whether some case of the set reads the element forces on the g set (its MPC forces, the grid point force rows of
    // the units not recovered): the vector is made only then.
    bool needsElementForces = false;
    for (const std::size_t s : group.cases)
    {
        const output::OutputPlan& outputs = inputs.plans[s];
        needsElementForces =
            needsElementForces ||
            (hasMset && (outputs.of(output::OutputKind::MpcForces).isRequested ||
                         outputs.of(output::OutputKind::GridPointForces).isRequested)) ||
            (!units.isEveryUnitRecovered && outputs.of(output::OutputKind::GridPointForces).isRequested);
    }

    // The vectors of one case of one unit in its pass (the largest unit counts: a wave holds one unit at least): a
    // unit not recovered its loads and applied loads and its reaction resultants; a recovered unit its solution, loads
    // and K·u on its equations, its enforced values, displacements, loads and applied loads, its reactions.
    const auto unitCaseBytesOf = [&](std::size_t u)
    {
        const CondensedUnit& unit = units.units[u];
        const auto unitLocal = static_cast<std::uint64_t>(unit.modelDofOfLocal.size());
        std::uint64_t vectors = UNIT_BATCH_VECTORS * unitLocal + unit.patternCount + RIGID_BODY_MODE_COUNT;
        if (unit.isRecovered)
            vectors += RECOVERED_EQUATION_VECTORS * static_cast<std::uint64_t>(unit.system->equationCount()) +
                       RECOVERED_LOCAL_VECTORS * unitLocal + unit.system->heldDofs().size();
        return vectors * sizeof(double);
    };
    // The cases, batch by batch (the batch size lowered when needed, and named): the vectors of one case of a batch
    // counted from the sizes of the interface, of the g set and of the largest unit, the budget left beside the result
    // tables of the cases still to come (their share of Inputs::resultBytes), and BATCH_BUDGET_FRACTION of it for the
    // batch.
    std::size_t batch = std::min(LARGEST_CASE_BATCH, std::max<std::size_t>(caseCount, 1));
    {
        std::uint64_t largestUnit = 0;
        for (std::size_t u = 0; u < units.units.size(); ++u)
            largestUnit = std::max(largestUnit, unitCaseBytesOf(u));
        const std::uint64_t perCase = (static_cast<std::uint64_t>(INTERFACE_BATCH_VECTORS) * localDofs +
                                       CASE_BATCH_G_VECTORS * gSize + RIGID_BODY_MODE_COUNT + 2) *
                                          sizeof(double) +
                                      largestUnit;
        const std::size_t totalCases = m_loads->cases.size();
        const std::uint64_t resultsToCome =
            totalCases == 0 ? 0
                            : static_cast<std::uint64_t>(static_cast<double>(inputs.resultBytes) *
                                                         static_cast<double>(totalCases - m_solvedCaseCount) /
                                                         static_cast<double>(totalCases));
        const std::uint64_t available = budget.available();
        const std::uint64_t room = available > resultsToCome ? available - resultsToCome : 0;
        const auto affordable = static_cast<std::size_t>(BATCH_BUDGET_FRACTION * static_cast<double>(room) /
                                                         static_cast<double>(std::max<std::uint64_t>(perCase, 1)));
        const std::size_t largest = batch;
        batch = std::max<std::size_t>(1, std::min(batch, affordable));
        setRecord.caseBatchSize = static_cast<std::int64_t>(batch);
        if (batch == largest)
            setRecord.caseBatchNote =
                base::concatenate({std::to_string(batch), batch == caseCount ? " cases: every case of the set at once"
                                                                             : " cases: the largest batch"});
        else
            setRecord.caseBatchNote = base::concatenate(
                {"lowered from ", std::to_string(largest), " to ", std::to_string(batch),
                 " cases by the memory budget: ", base::formatBytes(room), " left beside the result tables to come (",
                 base::formatBytes(resultsToCome), "), a case of a batch takes ", base::formatBytes(perCase)});
    }
    m_record.constraintSets.push_back(setRecord);
    CondensedSetRecord& setRecordKept = m_record.constraintSets.back();
    linalg::DenseMatrix interfaceDisplacements(budget, localDofs, batch, "condensed batch interface displacements");
    linalg::DenseMatrix interfaceReduced(budget, localDofs, batch, "condensed batch interface loads");
    linalg::DenseMatrix interfaceRhs(budget, equations, batch, "condensed batch right-hand sides");
    linalg::DenseMatrix interfaceResiduals(budget, equations, batch, "condensed batch residuals");
    linalg::DenseMatrix interfaceSolution(budget, equations, batch, "condensed batch solutions");
    linalg::DenseMatrix interfaceReactions(budget, interface.heldDofs().size(), batch, "condensed batch reactions");
    linalg::DenseMatrix interfaceApplied(budget, localDofs, batch, "condensed batch interface applied loads");
    linalg::DenseMatrix interfaceAll(budget, localDofs, batch, "condensed batch interface loads");
    // The case vectors of a batch on the g set, one column per case: filled by the interface and by each unit
    // on its own DOFs in the order of the sums of each case (the interface first, then the units in their order), so
    // that no column is filled whole per case: a DOF of the interface or the interior of a unit is first assigned
    // (0 + its first term, the value a sum from +0 gives), the SPC forces of the held DOFs set to zero per batch, the
    // interiors of the units not recovered never written (zero) but for their loads.
    linalg::DenseMatrix batchDisplacements(budget, gSize, batch, "condensed batch displacements", inputs.pool);
    linalg::DenseMatrix batchApplied(budget, gSize, batch, "condensed batch applied loads", inputs.pool);
    linalg::DenseMatrix batchLoads(budget, gSize, batch, "condensed batch loads", inputs.pool);
    linalg::DenseMatrix batchSpcForces(budget, gSize, batch, "condensed batch SPC forces", inputs.pool);
    base::BudgetedBuffer<double> mpcForces(budget, hasMset ? gSize : 0, "condensed case MPC forces");
    bool isMpcForcesZero = true;
    base::BudgetedBuffer<double> elementForces(budget, needsElementForces ? gSize : 0, "condensed case element forces");
    // Per case of a batch: the external work and the residual work of the interface, then the units' residual work in
    // their order; the resultant of the reactions of the units not recovered.
    std::vector<double> batchWork(batch, 0.0);     // bounded by LARGEST_CASE_BATCH
    std::vector<double> batchResidual(batch, 0.0); // bounded by LARGEST_CASE_BATCH
    std::vector<std::array<double, RIGID_BODY_MODE_COUNT>> batchReactions(batch);
    // The interface's applied loads (the mechanical vectors) and all its loads, as local vectors per case.
    linalg::DenseMatrix interfaceMechanical(budget, localDofs, vectorCount, "interface mechanical vectors");
    for (std::size_t k = 0; k < vectorCount; ++k)
        if (loads.isThermalVector[k] == 0)
            for (std::size_t local = 0; local < localDofs; ++local)
                interfaceMechanical(local, k) = interfaceLoads(local, k);
    std::vector<UnitBoundaryForces> unitBoundaryForces; // of one case: the units not recovered (bounded by the units)
    // The largest boundary and pattern count of a unit: the scratch of one unit's boundary forces in a case.
    std::size_t largestBoundary = 0;
    std::size_t largestPatterns = 0;
    for (std::size_t u = 0; u < units.units.size(); ++u)
    {
        largestBoundary = std::max(largestBoundary, interface.unitLocalDofs(u).size());
        largestPatterns = std::max(largestPatterns, fullAlphas[u].size());
    }
    // The boundary forces K*·u_a − f*·α of a unit not recovered in a case (exact, no recovery needed), into `forces`.
    const auto boundaryForcesOf = [&](std::size_t u, std::size_t c, std::size_t caseInGroup,
                                      linalg::DenseMatrix& boundaryScratch, linalg::DenseMatrix& coefficientScratch,
                                      linalg::DenseMatrixView forces, const linalg::DenseKernels& unitKernels)
    {
        const reduction::ReducedUnit& reduced = units.units[u].condensation->reducedUnit();
        const std::span<const std::int64_t> locals = interface.unitLocalDofs(u);
        const linalg::DenseMatrixView boundaryDisplacements = boundaryScratch.view().block(0, 0, locals.size(), 1);
        for (std::size_t i = 0; i < locals.size(); ++i)
            boundaryDisplacements(i, 0) = interfaceDisplacements(static_cast<std::size_t>(locals[i]), c);
        const linalg::DenseMatrixView coefficients = coefficientScratch.view().block(0, 0, fullAlphas[u].size(), 1);
        for (std::size_t k = 0; k < fullAlphas[u].size(); ++k)
            coefficients(k, 0) = fullAlphas[u](k, caseInGroup);
        reduced.boundaryForces(boundaryDisplacements, coefficients, forces, unitKernels);
    };
    // A unit's part of a batch: computed in its wave (side by side with the others), added to the case vectors after
    // in the order of the units.
    struct UnitBatch
    {
        linalg::DenseMatrix displacements;
        linalg::DenseMatrix allLoads;
        linalg::DenseMatrix applied;
        linalg::DenseMatrix reactions;
        std::vector<double> residualWork; // per case of the batch: u_oᵀ·(f_o − K_o·u) of its interior
        // A unit not recovered: per case of the batch, the six components of Φᵀ·P_unit + Φ_aᵀ·F_a (its reactions'
        // resultant is their opposite).
        std::vector<double> reactionSums;
        // Some case of the batch loads the unit (a coefficient of its patterns is not zero): its loads are made;
        // otherwise they are zero and neither made nor added (the loads of every unit per batch and per case would
        // cost the g set each time, where a case loads a few units).
        bool isLoaded = false;
    };
    std::size_t waveCount = 0;
    std::int64_t rebuiltCount = 0;

    // The recovery unit by unit (GroupInputs::visitUnit). In sequence, the batches below made each recovered unit's
    // recovery again in every batch (its equations built again, its interior factorised again, its basis solutions: 94
    // times for 6 000 cases); here each recovered unit's recovery is made once, its part of every batch computed from
    // it (the same calls as the batches' on the same values) and its displacements handed over, then released. The
    // reactions of the DOFs it holds and its residual work are kept per case, and the batches add them to the cases'
    // SPC forces and EPSILON where they added their own, in the order of the units: the same bits. The units one after
    // the other, each with every thread when its factorisation fits.
    const bool isUnitByUnit =
        isSequential && static_cast<bool>(group.visitUnit) &&
        std::any_of(units.units.begin(), units.units.end(), [](const CondensedUnit& unit) { return unit.isRecovered; });
    setRecordKept.isUnitByUnit = isUnitByUnit;
    std::vector<linalg::DenseMatrix> keptReactions(units.units.size()); // per recovered unit: held DOFs × cases
    std::vector<base::BudgetedBuffer<double>> keptResidualWork(units.units.size()); // per recovered unit: per case
    if (isUnitByUnit)
    {
        // The gauge of the FreeBodySupport DOFs of each case (conditioned supports): −C·u_r from the interface's
        // displacements at the FreeBodySupport DOFs, the batches' values there (BasisCoefficients::combine computes
        // each element alone).
        linalg::DenseMatrix gauge(budget, RIGID_BODY_MODE_COUNT, isConditioned ? caseCount : 0,
                                  "condensed unit by unit gauge");
        if (isConditioned)
        {
            const linalg::DenseMatrix& boundarySolutions = basis.displacements();
            linalg::DenseMatrix supportRows(budget, supportCount, boundarySolutions.columns(),
                                            "condensed unit by unit FreeBodySupport rows");
            for (std::size_t j = 0; j < supportCount; ++j)
            {
                const std::optional<std::int64_t> local = interface.localDofOf(group.supports[j]);
                if (!local)
                    throw base::InternalError(
                        "CondensedStatics: a FreeBodySupport DOF of conditioned supports is not a DOF of the interface "
                        "(an "
                        "internal error: report it)");
                for (std::size_t q = 0; q < boundarySolutions.columns(); ++q)
                    supportRows(j, q) = boundarySolutions(static_cast<std::size_t>(*local), q);
            }
            linalg::DenseMatrix atSupports(budget, supportCount, caseCount, "condensed unit by unit SUPORT values");
            basis.coefficients().combine(supportRows.view(), 0, atSupports.view(), kernels);
            for (std::size_t c = 0; c < caseCount; ++c)
                for (std::size_t j = 0; j < supportCount; ++j)
                    for (std::size_t a = 0; a < RIGID_BODY_MODE_COUNT; ++a)
                        gauge(a, c) -= reliefFit[j * RIGID_BODY_MODE_COUNT + a] * atSupports(j, c);
        }
        linalg::DenseMatrix unitDisplacements(budget, gSize, batch, "condensed unit by unit displacements",
                                              inputs.pool);
        // The recovered units in waves of consecutive units, as the batches' waves (a large unit alone with every
        // thread, the others side by side, one thread each, within WAVE_BUDGET_FRACTION of the bytes free): their
        // recoveries made, then for every batch their parts side by side and their displacements handed over in the
        // order of the units, then their recoveries released. A unit's values do not depend on the threads.
        struct UnitByUnitPart
        {
            std::vector<std::int64_t> boundaryRowOfLocal; // its row of boundaryRows, −1: none
            std::vector<std::int32_t> gridRows;           // the grid points it moves with the gauge
            linalg::DenseMatrix boundaryRows;             // its rows of the boundary basis solutions
            linalg::DenseMatrix boundaryValues;           // their values in a batch's cases
            linalg::DenseMatrix displacements;            // its local DOFs in a batch's cases
        };
        std::vector<std::size_t> recovered; // bounded by the units
        for (std::size_t u = 0; u < units.units.size(); ++u)
            if (units.units[u].isRecovered)
                recovered.push_back(u);
        const auto waveBytesOf = [&](std::size_t u)
        {
            const CondensedUnit& unit = units.units[u];
            return unit.footprint.passPeakBytes(!unit.system->hasEquations()) +
                   unitCaseBytesOf(u) * static_cast<std::uint64_t>(batch);
        };
        const linalg::DenseMatrix& boundarySolutions = basis.displacements();
        std::size_t begin = 0;
        while (begin < recovered.size())
        {
            if (inputs.cancellation != nullptr)
                inputs.cancellation->throwIfCancelled();
            std::size_t end = begin + 1;
            if (inputs.pool != nullptr && !isLargeUnit(recovered[begin]))
            {
                const auto room =
                    static_cast<std::uint64_t>(WAVE_BUDGET_FRACTION * static_cast<double>(budget.available()));
                std::uint64_t needed = waveBytesOf(recovered[begin]);
                while (end < recovered.size() && end - begin < largestWave && !isLargeUnit(recovered[end]) &&
                       needed + waveBytesOf(recovered[end]) <= room)
                    needed += waveBytesOf(recovered[end++]);
            }
            const std::size_t waveSize = end - begin;
            std::vector<UnitByUnitPart> parts(waveSize);
            std::vector<std::uint8_t> wasLight(waveSize, 0);
            // Alone: every thread when its factorisation fits the budget left.
            const bool isThreaded =
                waveSize == 1 && threadCount > 1 &&
                units.units[recovered[begin]].footprint.factorisationBytes * threadCount <= budget.available();
            const linalg::DenseKernels& aloneKernels = isThreaded ? kernels : serialKernels;
            // a. Each unit's recovery, and what it reads of the interface.
            const auto prepare = [&](std::size_t k, const linalg::DenseKernels& unitKernels, base::ThreadPool* pool)
            {
                const std::size_t u = recovered[begin + k];
                CondensedUnit& unit = units.units[u];
                wasLight[k] = unit.system->hasEquations() ? 0 : 1;
                static_cast<void>(prepareRecovery(u, unitKernels, unitKernels, pool));
                const reduction::UnitSystem& system = *unit.system;
                const auto unitLocal = static_cast<std::size_t>(system.localDofCount());
                UnitByUnitPart& part = parts[k];
                part.boundaryRowOfLocal.assign(unitLocal, -1);
                std::vector<std::size_t> interfaceRows; // bounded by its boundary DOFs
                for (std::size_t local = 0; local < unitLocal; ++local)
                {
                    const base::DofIndex dof = unit.modelDofOfLocal[local];
                    if (dof < 0)
                        continue;
                    if (const std::int32_t node = dofMap.nodeDofOf(dof).node; node < dofMap.gridCount())
                        part.gridRows.push_back(node);
                    if (unit.isInteriorLocal[local] != 0)
                        continue;
                    if (const std::optional<std::int64_t> interfaceLocal = interface.localDofOf(dof))
                    {
                        part.boundaryRowOfLocal[local] = static_cast<std::int64_t>(interfaceRows.size());
                        interfaceRows.push_back(static_cast<std::size_t>(*interfaceLocal));
                    }
                }
                std::sort(part.gridRows.begin(), part.gridRows.end());
                part.gridRows.erase(std::unique(part.gridRows.begin(), part.gridRows.end()), part.gridRows.end());
                part.boundaryRows = linalg::DenseMatrix(budget, interfaceRows.size(), boundarySolutions.columns(),
                                                        "condensed unit by unit boundary rows");
                for (std::size_t q = 0; q < boundarySolutions.columns(); ++q)
                    for (std::size_t i = 0; i < interfaceRows.size(); ++i)
                        part.boundaryRows(i, q) = boundarySolutions(interfaceRows[i], q);
                part.boundaryValues =
                    linalg::DenseMatrix(budget, interfaceRows.size(), batch, "condensed unit by unit boundary values");
                keptReactions[u] = linalg::DenseMatrix(budget, system.heldDofs().size(), caseCount,
                                                       "condensed unit by unit reactions");
                keptResidualWork[u] =
                    base::BudgetedBuffer<double>(budget, caseCount, "condensed unit by unit residual");
            };
            if (waveSize == 1)
                prepare(0, aloneKernels, isThreaded ? inputs.pool : nullptr);
            else
                inputs.pool->runBatch(waveSize, [&](std::size_t k) { prepare(k, serialKernels, nullptr); });
            for (std::size_t k = 0; k < waveSize; ++k)
            {
                const std::size_t u = recovered[begin + k];
                m_record.units[u].recoveries.push_back(recoveries[u]->statistics());
                rebuiltCount += wasLight[k];
            }
            // b. Every batch: the units' parts, as their parts of a batch make them, then their displacements on their
            // DOFs (the interior's, the interface's at their boundary points, the gauge of the FreeBodySupport DOFs:
            // the values of the batches' case vectors there) handed over in the order of the units.
            for (std::size_t first = 0; first < caseCount; first += batch)
            {
                if (inputs.cancellation != nullptr)
                    inputs.cancellation->throwIfCancelled();
                const std::size_t count = std::min(batch, caseCount - first);
                const auto makePart = [&](std::size_t k, const linalg::DenseKernels& unitKernels)
                {
                    const std::size_t u = recovered[begin + k];
                    CondensedUnit& unit = units.units[u];
                    const reduction::UnitSystem& system = *unit.system;
                    const auto unitLocal = static_cast<std::size_t>(system.localDofCount());
                    const auto equationCount = static_cast<std::size_t>(system.equationCount());
                    const auto interiorCount = static_cast<std::size_t>(system.interiorDofCount());
                    const std::size_t heldCount = system.heldDofs().size();
                    linalg::DenseMatrix allLoads(budget, unitLocal, count, "unit batch loads");
                    fullAlphas[u].combine(unit.patternLoads.view(), first, allLoads.view(), unitKernels);
                    linalg::DenseMatrix equationValues(budget, equationCount, count, "unit batch");
                    linalg::DenseMatrix enforced(budget, unitLocal, recoveries[u]->hasEnforcedValues() ? count : 0,
                                                 "unit batch");
                    recoveries[u]->recoverEquations(first, equationValues.view(), enforced.view(), unitKernels, budget,
                                                    inputs.cancellation);
                    linalg::DenseMatrix& displacements = parts[k].displacements;
                    displacements = linalg::DenseMatrix(budget, unitLocal, count, "unit batch displacements");
                    system.expandSolution(equationValues.view(), enforced.view(), displacements.view());
                    linalg::DenseMatrix reactions(budget, heldCount, count, "unit batch reactions");
                    system.heldReactions(equationValues.view(), enforced.view(), allLoads.view(), reactions.view(),
                                         budget);
                    for (std::size_t c = 0; c < count; ++c)
                        for (std::size_t h = 0; h < heldCount; ++h)
                            keptReactions[u](h, first + c) = reactions(h, c);
                    linalg::DenseMatrix caseLoads(budget, equationCount, count, "unit batch equation loads");
                    fullAlphas[u].combine(recoveredEquationLoads[u].view(), first, caseLoads.view(), unitKernels);
                    linalg::DenseMatrix product(budget, equationCount, count, "unit batch K u");
                    system.matrix().multiply(equationValues.view(), product.view());
                    for (std::size_t c = 0; c < count; ++c)
                    {
                        double residualWork = 0.0;
                        for (std::size_t i = 0; i < interiorCount; ++i)
                            residualWork += equationValues(i, c) * (caseLoads(i, c) - product(i, c));
                        keptResidualWork[u][first + c] = residualWork;
                    }
                };
                if (waveSize == 1)
                    makePart(0, aloneKernels);
                else
                    inputs.pool->runBatch(waveSize, [&](std::size_t k) { makePart(k, serialKernels); });
                for (std::size_t k = 0; k < waveSize; ++k)
                {
                    const std::size_t u = recovered[begin + k];
                    const CondensedUnit& unit = units.units[u];
                    UnitByUnitPart& part = parts[k];
                    const std::size_t unitLocal = unit.modelDofOfLocal.size();
                    const linalg::DenseMatrixView values =
                        part.boundaryValues.view().block(0, 0, part.boundaryRows.rows(), count);
                    basis.coefficients().combine(part.boundaryRows.view(), first, values, kernels);
                    for (std::size_t c = 0; c < count; ++c)
                    {
                        for (std::size_t local = 0; local < unitLocal; ++local)
                        {
                            const base::DofIndex dof = unit.modelDofOfLocal[local];
                            if (dof < 0)
                                continue;
                            const auto g = static_cast<std::size_t>(dof);
                            if (unit.isInteriorLocal[local] != 0)
                                unitDisplacements(g, c) = part.displacements(local, c);
                            else
                                unitDisplacements(g, c) =
                                    part.boundaryRowOfLocal[local] >= 0
                                        ? values(static_cast<std::size_t>(part.boundaryRowOfLocal[local]), c)
                                        : 0.0;
                        }
                        if (!isConditioned)
                            continue;
                        std::array<double, RIGID_BODY_MODE_COUNT> amplitudes{};
                        for (std::size_t a = 0; a < RIGID_BODY_MODE_COUNT; ++a)
                            amplitudes[a] = gauge(a, first + c);
                        assembly::addRigidMotionAt(dofMap, amplitudes,
                                                   std::span<double>(&unitDisplacements(0, c), gSize), part.gridRows,
                                                   isConstrained.span());
                        for (const base::DofIndex dof : group.supports)
                            unitDisplacements(static_cast<std::size_t>(dof), c) = 0.0;
                    }
                    part.displacements = linalg::DenseMatrix();
                    GroupInputs::UnitCases unitCases;
                    unitCases.unit = u;
                    unitCases.unitId = unit.unitId;
                    unitCases.firstCaseInGroup = first;
                    unitCases.displacements = unitDisplacements.view().block(0, 0, gSize, count);
                    group.visitUnit(unitCases);
                }
            }
            for (std::size_t k = 0; k < waveSize; ++k)
                releaseRecovery(recovered[begin + k]);
            ++waveCount;
            begin = end;
        }
    }

    for (std::size_t first = 0; first < caseCount; first += batch)
    {
        const std::size_t count = std::min(batch, caseCount - first);
        const auto columnsOfBatch = [&](linalg::DenseMatrix& matrix)
        { return matrix.view().block(0, 0, matrix.rows(), count); };
        basis.coefficients().combine(basis.displacements().view(), first, columnsOfBatch(interfaceDisplacements),
                                     kernels);
        basis.coefficients().combine(basis.reducedLoads().view(), first, columnsOfBatch(interfaceReduced), kernels);
        basis.coefficients().combine(basis.rightHandSides().view(), first, columnsOfBatch(interfaceRhs), kernels);
        basis.coefficients().combine(basis.residuals().view(), first, columnsOfBatch(interfaceResiduals), kernels);
        interfaceAlpha.combine(interfaceMechanical.view(), first, columnsOfBatch(interfaceApplied), kernels);
        interfaceAlpha.combine(interfaceLoads.view(), first, columnsOfBatch(interfaceAll), kernels);
        interface.heldReactions(columnsOfBatch(interfaceDisplacements), columnsOfBatch(interfaceReduced),
                                columnsOfBatch(interfaceReactions));
        for (std::size_t c = 0; c < count; ++c)
            for (std::size_t e = 0; e < equations; ++e)
                interfaceSolution(e, c) = interfaceDisplacements(
                    static_cast<std::size_t>(interface.localDofOfEquation(static_cast<std::int32_t>(e))), c);
        // The interface's part of each case: its points' displacements and loads, the relief loads, its reactions,
        // its external and residual work (EPSILON and the external work: uᵀ·f = u_lᵀ·f_l of the interface + Σ αᵀ·G·α of
        // the units' patterns, added per case below; the residual work of the recovered units follows in their order).
        for (const base::DofIndex dof : reactionDofs)
            for (std::size_t c = 0; c < count; ++c)
                batchSpcForces(static_cast<std::size_t>(dof), c) = 0.0;
        for (std::size_t c = 0; c < count; ++c)
        {
            const std::size_t caseInGroup = first + c;
            for (std::size_t local = 0; local < localDofs; ++local)
                if (const std::optional<base::DofIndex> dof = interface.modelDofOf(static_cast<std::int64_t>(local)))
                {
                    const auto g = static_cast<std::size_t>(*dof);
                    batchDisplacements(g, c) = interfaceDisplacements(local, c);
                    batchApplied(g, c) = 0.0 + interfaceApplied(local, c);
                    batchLoads(g, c) = 0.0 + interfaceAll(local, c);
                }
            if (hasRelief)
                for (std::size_t local = 0; local < localDofs; ++local)
                    if (const std::optional<base::DofIndex> dof =
                            interface.modelDofOf(static_cast<std::int64_t>(local)))
                    {
                        double relief = 0.0;
                        for (std::size_t mode = 0; mode < RIGID_BODY_MODE_COUNT; ++mode)
                            relief += interface.inertialLoads()(local, mode) * reliefAlpha(mode, caseInGroup);
                        batchApplied(static_cast<std::size_t>(*dof), c) += relief;
                        batchLoads(static_cast<std::size_t>(*dof), c) += relief;
                    }
            for (std::size_t k = 0; k < interface.heldDofs().size(); ++k)
                if (const std::optional<base::DofIndex> dof = interface.modelDofOf(interface.heldDofs()[k]))
                    if (!isConditionedSupport(*dof))
                        batchSpcForces(static_cast<std::size_t>(*dof), c) += interfaceReactions(k, c);
            // The FreeBodySupport DOFs solved free (conditioned supports): their reaction K·u − P is less the residual
            // of their equation (round-off: the relieved loads are balanced).
            for (std::size_t j = 0; isConditioned && j < supportEquations.size(); ++j)
                batchSpcForces(static_cast<std::size_t>(group.supports[j]), c) -=
                    interfaceResiduals(static_cast<std::size_t>(supportEquations[j]), c);
            double work = 0.0;
            double residualWork = 0.0;
            for (std::size_t e = 0; e < equations; ++e)
            {
                work += interfaceSolution(e, c) * interfaceRhs(e, c);
                residualWork += interfaceSolution(e, c) * interfaceResiduals(e, c);
            }
            batchWork[c] = work;
            batchResidual[c] = residualWork;
            batchReactions[c] = {};
        }

        // The units, in waves of consecutive units bounded by the budget left: a unit's part of the batch
        // computed in its wave (in sequence, its recovery made first and released after), then added to the case
        // vectors in the order of the units. A unit whose interior factorisation is large (its work at least
        // Inputs::largeUnitMinimumFlops) is a wave of its own with every thread; otherwise up to WAVE_UNITS_PER_THREAD
        // units per thread side by side, each on one thread (the same bits: a unit's kernels do not depend on the
        // threads).
        const auto unitPart =
            [&](std::size_t u, const linalg::DenseKernels& unitKernels, base::ThreadPool* unitPool, UnitBatch& entry)
        {
            CondensedUnit& unit = units.units[u];
            const auto unitLocal = static_cast<std::size_t>(unit.system->localDofCount());
            for (std::size_t c = 0; c < count && !entry.isLoaded; ++c)
                for (std::size_t k = 0; k < fullAlphas[u].size() && !entry.isLoaded; ++k)
                    entry.isLoaded = fullAlphas[u](k, first + c) != 0.0;
            // A recovered unit's loads are made whatever its coefficients: its reactions read them.
            if (unit.isRecovered)
            {
                entry.isLoaded = true;
                if (!recoveries[u] && !isUnitByUnit)
                {
                    static_cast<void>(prepareRecovery(u, unitKernels, unitKernels, unitPool));
                    if (first == 0)
                        m_record.units[u].recoveries.push_back(recoveries[u]->statistics());
                }
            }
            if (entry.isLoaded)
            {
                // The unit's loads of the cases (all of them, and the mechanical ones).
                linalg::DenseMatrix mechanicalAlpha(budget, fullAlphas[u].size(), count,
                                                    "unit mechanical coefficients");
                for (std::size_t c = 0; c < count; ++c)
                    for (std::size_t k = 0; k < fullAlphas[u].size(); ++k)
                        mechanicalAlpha(k, c) = unit.isThermalPattern[k] != 0 ? 0.0 : fullAlphas[u](k, first + c);
                entry.allLoads = linalg::DenseMatrix(budget, unitLocal, count, "unit batch loads");
                fullAlphas[u].combine(unit.patternLoads.view(), first, entry.allLoads.view(), unitKernels);
                entry.applied = linalg::DenseMatrix(budget, unitLocal, count, "unit batch applied loads");
                const reduction::BasisCoefficients mechanical(std::move(mechanicalAlpha));
                mechanical.combine(unit.patternLoads.view(), 0, entry.applied.view(), unitKernels);
            }
            if (!unit.isRecovered)
            {
                // Not recovered: the resultant of its reactions by its equilibrium (forces and moments about the basic
                // origin), −(Φᵀ·P_unit + Φ_aᵀ·F_a), F_a = K*·u_a − f*·α the force on its boundary.
                const reduction::UnitSystem& system = *unit.system;
                const linalg::DenseMatrix modes = system.makeRigidBodyModes(budget);
                const std::span<const std::int64_t> locals = interface.unitLocalDofs(u);
                linalg::DenseMatrix boundaryScratch(budget, locals.size(), 1, "unit boundary displacements");
                linalg::DenseMatrix coefficientScratch(budget, fullAlphas[u].size(), 1, "unit case coefficients");
                linalg::DenseMatrix forces(budget, locals.size(), 1, "unit boundary forces");
                entry.reactionSums.assign(count * RIGID_BODY_MODE_COUNT, 0.0);
                for (std::size_t c = 0; c < count; ++c)
                {
                    boundaryForcesOf(u, c, first + c, boundaryScratch, coefficientScratch, forces.view(), unitKernels);
                    for (std::size_t a = 0; a < RIGID_BODY_MODE_COUNT; ++a)
                    {
                        double sum = 0.0;
                        for (std::size_t local = 0; local < unitLocal && entry.isLoaded; ++local)
                            sum += modes(local, a) * entry.allLoads(local, c);
                        for (std::size_t i = 0; i < locals.size(); ++i)
                            sum += modes(static_cast<std::size_t>(system.localDofOfEquation(
                                             system.interiorDofCount() + static_cast<std::int32_t>(i))),
                                         a) *
                                   forces(i, 0);
                        entry.reactionSums[c * RIGID_BODY_MODE_COUNT + a] = sum;
                    }
                }
                return;
            }
            // Recovered unit by unit: its displacements, reactions and residual work were made then.
            if (isUnitByUnit)
                return;
            const reduction::UnitSystem& system = *unit.system;
            linalg::DenseMatrix equationValues(budget, static_cast<std::size_t>(system.equationCount()), count,
                                               "unit batch");
            linalg::DenseMatrix enforced(budget, unitLocal, recoveries[u]->hasEnforcedValues() ? count : 0,
                                         "unit batch");
            recoveries[u]->recoverEquations(first, equationValues.view(), enforced.view(), unitKernels, budget,
                                            inputs.cancellation);
            entry.displacements = linalg::DenseMatrix(budget, unitLocal, count, "unit batch displacements");
            system.expandSolution(equationValues.view(), enforced.view(), entry.displacements.view());
            entry.reactions = linalg::DenseMatrix(budget, system.heldDofs().size(), count, "unit batch reactions");
            system.heldReactions(equationValues.view(), enforced.view(), entry.allLoads.view(), entry.reactions.view(),
                                 budget);
            // The residual of its interior equations (EPSILON from the residual of the interface and of the units
            // recovered), r_o = f_o − K_o·u with f the case's loads on its equations and K its matrix (both triangles),
            // K·u summed in the fixed order of SymmetricCsc::multiply; its work u_oᵀ·r_o per case, added in the order
            // of the units.
            const auto equationCount = static_cast<std::size_t>(system.equationCount());
            const auto interiorCount = static_cast<std::size_t>(system.interiorDofCount());
            linalg::DenseMatrix caseLoads(budget, equationCount, count, "unit batch equation loads");
            fullAlphas[u].combine(recoveredEquationLoads[u].view(), first, caseLoads.view(), unitKernels);
            linalg::DenseMatrix product(budget, equationCount, count, "unit batch K u");
            system.matrix().multiply(equationValues.view(), product.view());
            entry.residualWork.assign(count, 0.0);
            for (std::size_t c = 0; c < count; ++c)
                for (std::size_t i = 0; i < interiorCount; ++i)
                    entry.residualWork[c] += equationValues(i, c) * (caseLoads(i, c) - product(i, c));
        };
        // A unit's part added to the case vectors (in the order of the units: the order of every sum of a case).
        const auto addUnitPart = [&](std::size_t u, const UnitBatch& entry)
        {
            const CondensedUnit& unit = units.units[u];
            const std::size_t unitLocal = unit.modelDofOfLocal.size();
            for (std::size_t c = 0; c < count; ++c)
            {
                for (std::size_t local = 0; local < unitLocal; ++local)
                {
                    const base::DofIndex dof = unit.modelDofOfLocal[local];
                    if (dof < 0)
                        continue;
                    const auto g = static_cast<std::size_t>(dof);
                    const bool isInterior = unit.isInteriorLocal[local] != 0;
                    if (entry.isLoaded && isInterior)
                    {
                        batchApplied(g, c) = 0.0 + entry.applied(local, c);
                        batchLoads(g, c) = 0.0 + entry.allLoads(local, c);
                    }
                    else if (entry.isLoaded)
                    {
                        batchApplied(g, c) += entry.applied(local, c);
                        batchLoads(g, c) += entry.allLoads(local, c);
                    }
                    else if (isInterior)
                    {
                        // A unit no case of the batch loads adds nothing (its loads are zero): its interior is zero.
                        batchApplied(g, c) = 0.0;
                        batchLoads(g, c) = 0.0;
                    }
                    if (unit.isRecovered && isInterior && !isUnitByUnit)
                        batchDisplacements(g, c) = entry.displacements(local, c);
                }
                if (unit.isRecovered)
                {
                    const std::span<const std::int64_t> held = unit.system->heldDofs();
                    for (std::size_t k = 0; k < held.size(); ++k)
                        if (const base::DofIndex dof = unit.modelDofOfLocal[static_cast<std::size_t>(held[k])];
                            dof >= 0)
                            batchSpcForces(static_cast<std::size_t>(dof), c) +=
                                isUnitByUnit ? keptReactions[u](k, first + c) : entry.reactions(k, c);
                    batchResidual[c] += isUnitByUnit ? keptResidualWork[u][first + c] : entry.residualWork[c];
                    continue;
                }
                for (std::size_t a = 0; a < RIGID_BODY_MODE_COUNT; ++a)
                    batchReactions[c][a] -= entry.reactionSums[c * RIGID_BODY_MODE_COUNT + a];
            }
        };
        // What a unit's part takes while it is made: its vectors of the batch's cases and, in sequence, its pass.
        const auto partBytesOf = [&](std::size_t u)
        {
            const CondensedUnit& unit = units.units[u];
            std::uint64_t bytes = unitCaseBytesOf(u) * count + static_cast<std::uint64_t>(unit.modelDofOfLocal.size()) *
                                                                   RIGID_BODY_MODE_COUNT * sizeof(double);
            if (unit.isRecovered && !recoveries[u] && !isUnitByUnit)
                bytes += unit.footprint.passPeakBytes(!unit.system->hasEquations());
            return bytes;
        };
        std::size_t begin = 0;
        while (begin < units.units.size())
        {
            std::size_t end = begin + 1;
            const bool isAlone = isLargeUnit(begin) || inputs.pool == nullptr;
            if (!isAlone)
            {
                const auto room =
                    static_cast<std::uint64_t>(WAVE_BUDGET_FRACTION * static_cast<double>(budget.available()));
                std::uint64_t needed = partBytesOf(begin);
                while (end < units.units.size() && end - begin < largestWave && !isLargeUnit(end))
                {
                    const std::uint64_t bytes = partBytesOf(end);
                    if (needed + bytes > room)
                        break;
                    needed += bytes;
                    ++end;
                }
            }
            std::vector<UnitBatch> parts(end - begin); // bounded by a wave
            std::vector<std::uint8_t> wasLight(end - begin, 0);
            for (std::size_t u = begin; u < end; ++u)
                wasLight[u - begin] = units.units[u].system->hasEquations() ? 0 : 1;
            if (end - begin == 1)
            {
                // Alone: every thread when its factorisation fits the budget left (a released unit's estimate).
                const CondensedUnit& unit = units.units[begin];
                const bool isThreaded =
                    threadCount > 1 && (!unit.isRecovered || recoveries[begin] || isUnitByUnit ||
                                        unit.footprint.factorisationBytes * threadCount <= budget.available());
                unitPart(begin, isThreaded ? kernels : serialKernels, isThreaded ? inputs.pool : nullptr,
                         parts.front());
            }
            else
                inputs.pool->runBatch(end - begin,
                                      [&](std::size_t k) { unitPart(begin + k, serialKernels, nullptr, parts[k]); });
            for (std::size_t u = begin; u < end; ++u)
            {
                addUnitPart(u, parts[u - begin]);
                if (isUnitByUnit)
                    continue;
                rebuiltCount += wasLight[u - begin] != 0 && units.units[u].isRecovered ? 1 : 0;
                if (isSequential && units.units[u].isRecovered)
                    releaseRecovery(u);
            }
            ++waveCount;
            begin = end;
        }

        // Conditioned supports: the displacements brought back to the gauge of the FreeBodySupport DOFs, u ← u −
        // Φ·C·u_r (C = Φ_r⁻¹, D = Φ·C the relief's support motions) on every DOF but the held ones (as the full path's
        // D, zero on the s set); the reactions, residuals and works above do not depend on it (the relieved loads are
        // balanced and K·Φ is the round-off of the sums).
        for (std::size_t c = 0; isConditioned && c < count; ++c)
        {
            std::array<double, RIGID_BODY_MODE_COUNT> amplitudes{};
            for (std::size_t j = 0; j < supportCount; ++j)
            {
                const double atSupport = batchDisplacements(static_cast<std::size_t>(group.supports[j]), c);
                for (std::size_t a = 0; a < RIGID_BODY_MODE_COUNT; ++a)
                    amplitudes[a] -= reliefFit[j * RIGID_BODY_MODE_COUNT + a] * atSupport;
            }
            assembly::addRigidMotion(dofMap, amplitudes, std::span<double>(&batchDisplacements(0, c), gSize),
                                     isConstrained.span());
            // The FreeBodySupport DOFs, flagged held, at rest: D_r = I, so u_r − u_r (the full path's exact zero).
            for (const base::DofIndex dof : group.supports)
                batchDisplacements(static_cast<std::size_t>(dof), c) = 0.0;
        }

        // The displacements of every case of the batch at once, for the element tables recovered by blocks
        // (GroupInputs::visitBatch): the interface's points and the interiors of the recovered units.
        if (group.visitBatch)
            group.visitBatch(first, columnsOfBatch(batchDisplacements));

        // The boundary displacements, coefficients and forces of one unit not recovered for one case, made once for
        // the batch (not three allocations per unit and per case, with 10 000 units).
        linalg::DenseMatrix boundaryScratch(budget, largestBoundary, 1, "unit boundary displacements");
        linalg::DenseMatrix coefficientScratch(budget, largestPatterns, 1, "unit case coefficients");
        linalg::DenseMatrix forceScratch(budget, largestBoundary, 1, "unit boundary forces");
        for (std::size_t c = 0; c < count; ++c)
        {
            if (inputs.cancellation != nullptr)
                inputs.cancellation->throwIfCancelled();
            const std::size_t caseInGroup = first + c;
            const auto columnOf = [&](const linalg::DenseMatrix& matrix)
            { return std::span<const double>(matrix.data() + c * matrix.leadingDimension(), gSize); };
            const std::span<const double> displacements = columnOf(batchDisplacements);
            const std::span<const double> allLoads = columnOf(batchLoads);
            // The boundary forces of the units not recovered (kept for the contour forces of a zone that is the
            // unit).
            CaseResult result;
            unitBoundaryForces.clear();
            for (std::size_t u = 0; u < units.units.size(); ++u)
            {
                const CondensedUnit& unit = units.units[u];
                if (unit.isRecovered)
                    continue;
                const std::span<const std::int64_t> locals = interface.unitLocalDofs(u);
                const linalg::DenseMatrixView forces = forceScratch.view().block(0, 0, locals.size(), 1);
                boundaryForcesOf(u, c, caseInGroup, boundaryScratch, coefficientScratch, forces, kernels);
                UnitBoundaryForces& boundary = unitBoundaryForces.emplace_back();
                boundary.unitId = unit.unitId;
                boundary.hasInteriorReactions = isInteriorReacting[u] != 0;
                for (std::size_t i = 0; i < locals.size(); ++i)
                    if (const std::optional<base::DofIndex> dof = interface.modelDofOf(locals[i]))
                    {
                        boundary.dofs.push_back(*dof);
                        boundary.forces.push_back(forces(i, 0));
                    }
            }
            result.unrecoveredReactions = batchReactions[c];
            // The MPC forces, when an output takes them: q_m = (K·u − P)_m from the element forces, q_n = −Gmᵀ·q_m
            // (assembly::RigidElimination::computeMpcForces), the units not recovered standing by their boundary
            // forces.
            const output::OutputPlan& outputs = inputs.plans[group.cases[caseInGroup]];
            const bool needsMpcForces = hasMset && (outputs.of(output::OutputKind::MpcForces).isRequested ||
                                                    outputs.of(output::OutputKind::GridPointForces).isRequested);
            std::vector<output::StiffnessRecovery::UnitRow> gridPointUnitRows; // bounded by the boundaries
            const bool needsUnitRows =
                !units.isEveryUnitRecovered && outputs.of(output::OutputKind::GridPointForces).isRequested;
            if (needsMpcForces || needsUnitRows)
            {
                std::fill(elementForces.data(), elementForces.data() + gSize, 0.0);
                if (needsMpcForces)
                    addElementForces(model.elementSet(), dofMap, displacements, units.excludedElements.span(),
                                     elementForces.span(), inputs.cancellation);
                std::size_t unrecovered = 0;
                for (std::size_t u = 0; u < units.units.size(); ++u)
                {
                    const CondensedUnit& unit = units.units[u];
                    if (unit.isRecovered)
                        continue;
                    const UnitBoundaryForces& boundary = unitBoundaryForces[unrecovered++];
                    const std::span<const std::int64_t> locals = interface.unitLocalDofs(u);
                    std::map<std::int32_t, std::array<double, 6>> rowsOfNode; // bounded by the unit's boundary points
                    std::size_t next = 0;
                    for (std::size_t i = 0; i < locals.size(); ++i)
                    {
                        if (!interface.modelDofOf(locals[i]))
                            continue; // a generated fastener point
                        const base::DofIndex dof = boundary.dofs[next];
                        const double force = boundary.forces[next++];
                        elementForces[static_cast<std::size_t>(dof)] += force;
                        const linalg::NodeDof nodeDof = dofMap.nodeDofOf(dof);
                        rowsOfNode[nodeDof.node][static_cast<std::size_t>(nodeDof.component)] -= force;
                    }
                    for (const auto& [node, values] : rowsOfNode)
                        gridPointUnitRows.push_back({node, static_cast<std::int32_t>(unit.unitId), values});
                }
                std::stable_sort(gridPointUnitRows.begin(), gridPointUnitRows.end(),
                                 [](const output::StiffnessRecovery::UnitRow& a,
                                    const output::StiffnessRecovery::UnitRow& b) { return a.node < b.node; });
                if (needsMpcForces)
                {
                    mpcGm.computeMpcForces(elementForces.span(), allLoads, mpcForces.span());
                    isMpcForcesZero = false;
                }
            }
            if (hasMset && !needsMpcForces && !isMpcForcesZero)
            {
                std::fill(mpcForces.data(), mpcForces.data() + gSize, 0.0);
                isMpcForcesZero = true;
            }
            output::StiffnessRecovery::CondensedUnits gridPointUnits{units.excludedElements.span(), gridPointUnitRows};

            // EPSILON and the external work on the interface and the units' patterns: uᵀ·f = u_lᵀ·f_l of the interface
            // + Σ αᵀ·G·α (StaticCondensation::patternWork).
            double work = batchWork[c];
            for (std::size_t u = 0; u < units.units.size(); ++u)
            {
                const linalg::DenseMatrix& g = units.units[u].work();
                const reduction::BasisCoefficients& alpha = fullAlphas[u];
                for (std::size_t a = 0; a < alpha.size(); ++a)
                {
                    const double alphaA = alpha(a, caseInGroup);
                    if (alphaA == 0.0)
                        continue;
                    for (std::size_t b = 0; b < alpha.size(); ++b)
                        work += alphaA * g(a, b) * alpha(b, caseInGroup);
                }
            }
            constexpr double EXTERNAL_WORK_FACTOR = 0.5;
            result.displacements = displacements;
            result.applied = columnOf(batchApplied);
            result.spcForces = columnOf(batchSpcForces);
            result.mpcForces = hasMset ? std::span<const double>(mpcForces.span()) : std::span<const double>{};
            result.mpcRelations = hasMset ? &mpcGm : nullptr;
            result.isConstrained = isConstrained.span();
            result.epsilon = work != 0.0 ? batchResidual[c] / work : 0.0;
            result.externalWork = EXTERNAL_WORK_FACTOR * work;
            if (hasRelief)
                result.supportAccelerations = accelerations[caseInGroup];
            result.gridPointForceUnits = units.isEveryUnitRecovered ? nullptr : &gridPointUnits;
            result.unitBoundaryForces = unitBoundaryForces;
            visit(caseInGroup, result);
        }
    }
    // The factors made for a recovery that solved its cases one by one: no longer read.
    for (const std::size_t u : releasedAfterCases)
        units.units[u].condensation->releaseInteriorFactor();
    setRecordKept.recoveryWaveCount = static_cast<std::int64_t>(waveCount);
    setRecordKept.rebuiltUnitCount = rebuiltCount;
    m_record.memory.recoveryWaveCount += static_cast<std::int64_t>(waveCount);
    m_record.memory.rebuiltUnitCount += rebuiltCount;
    const double caseSeconds = clock.lap();
    setRecordKept.caseSeconds = caseSeconds;
    // The constraint set kept with its interface factor and the units, for the sensitivities of a design
    // point (CondensedFactorisedSet); the units are shared by the sets, made once.
    if (inputs.keepsSystems)
    {
        std::shared_ptr<CondensedRelief> relief;
        if (hasRelief)
        {
            relief = std::make_shared<CondensedRelief>();
            relief->supportCount = supportCount;
            relief->fit = reliefFit;
            relief->accelerations = accelerations;
        }
        keepSet(group, std::move(interfaceOwner), std::move(fullGmOwner), std::move(isConstrained),
                std::move(placements), budget, std::move(relief));
    }
    m_solvedCaseCount += caseCount;
    CondensedMemoryRecord& memory = m_record.memory;
    memory.measuredPeakBytes = budget.peak();
    memory.keptFactorBytes = m_memory->retention.keptBytes();
    memory.keptFactorCount = m_memory->retention.keptCount();
    memory.releasedForBudget = m_memory->retention.releasedForBudget();
    return singularityTable;
}

// ---------------------------------------------------------------------------------------------------------------------
// Eigenvalue analyses: Craig-Bampton

namespace
{

// The interior factor of a unit, computed again when it was released (or never made: the cache), to the bits of its
// condensation (StaticCondensation::recomputeInteriorFactor), on every thread when that fits the budget left.
// `cancellation`: the token of the current call (CondensedStatics::m_inputs), never the one of the unit's condensation:
// a kept structure (a trim's linearisation) solves after its run, whose token is gone.
void ensureInteriorFactor(CondensedUnit& unit, CondensedUnitRecord& record, const linalg::DenseKernels& kernels,
                          const base::CancellationToken* cancellation, base::MemoryBudget& budget)
{
    if (unit.condensation->isInteriorFactorKept())
        return;
    const std::size_t threads = kernels.options().pool != nullptr ? kernels.options().pool->threadCount() : 1;
    linalg::DenseKernelOptions serialOptions = kernels.options();
    serialOptions.pool = nullptr;
    const linalg::DenseKernels serialKernels(kernels.kind(), serialOptions);
    const bool isThreaded = threads > 1 && linalg::SupernodalFactor::estimateMemoryBytes(
                                               unit.system->symbolic(), threads,
                                               linalg::PivotingMode::PositiveDefinite) <= budget.available();
    record.isRefactorisedForRecovery = record.isRefactorisedForRecovery || unit.condensation->isFromCache();
    unit.condensation->recomputeInteriorFactor(isThreaded ? kernels : serialKernels, budget, cancellation);
    ++record.refactorisationCount;
}

// Whether two columns of eigenvalues have the same bits.
bool isSameEigenvalues(const linalg::DenseMatrix& a, const linalg::DenseMatrix& b)
{
    if (a.rows() != b.rows())
        return false;
    for (std::size_t k = 0; k < a.rows(); ++k)
        if (std::bit_cast<std::uint64_t>(a(k, 0)) != std::bit_cast<std::uint64_t>(b(k, 0)))
            return false;
    return true;
}

} // namespace

void CondensedStatics::reduceModes(const reduction::ModeRequest& request, const CraigBamptonRecord& record,
                                   bool needsEveryUnitsModes, base::MemoryBudget& budget, base::DiagnosticSink& sink)
{
    if (!m_inputs.isCraigBampton)
        throw std::logic_error(
            "CondensedStatics::reduceModes: the units were condensed without their mass; set Inputs::isCraigBampton");
    const linalg::DenseKernels& kernels = *m_inputs.kernels;
    const base::CancellationToken* cancellation = m_inputs.cancellation;
    m_record.craigBampton = record;
    Units& units = *m_units;
    for (std::size_t u = 0; u < units.units.size(); ++u)
    {
        if (cancellation != nullptr)
            cancellation->throwIfCancelled();
        CondensedUnit& unit = units.units[u];
        CondensedUnitRecord& unitRecord = m_record.units[u];
        unit.modal.reset();
        unit.modes.reset();
        unitRecord.modalOutcome.clear();
        unitRecord.modesOutcome.clear();
        unitRecord.modesNote.clear();
        unitRecord.isModesStored = false;
        const bool needsModes = unit.isRecovered || needsEveryUnitsModes;
        // Its key with the request: the boundary part's, the modes asked for added (the key includes the mode
        // request).
        std::optional<reduction::CacheKey> modalKey;
        if (m_cacheSession && unit.key)
        {
            modalKey = *unit.key;
            modalKey->modeCount = request.isCount() ? request.count : reduction::CacheKey::MODE_CUTOFF_REQUEST;
            modalKey->modeCutoff = request.isCount() ? 0 : std::bit_cast<std::uint64_t>(request.cutoffEigenvalue);
            const reduction::CacheLookup<reduction::CachedReduction> found =
                m_cacheSession->lookupReduction(*modalKey, budget);
            if (found.isHit() && found.entry->modal)
            {
                unit.modal = found.entry->modal;
                unitRecord.modalOutcome = std::string(reduction::cacheOutcomeName(found.outcome));
            }
        }
        const auto storeModes = [&](const reduction::CachedRecoveryData& data)
        {
            if (!modalKey)
                return;
            const reduction::CacheStore stored = m_cacheSession->storeRecoveryData(*modalKey, data);
            unitRecord.isModesStored = stored.isStored;
            if (!stored.isStored)
            {
                unitRecord.modesNote = stored.note;
                sink.report(
                    problem("cache-not-stored", "UNIT", static_cast<base::EntityId>(unit.unitId),
                            base::concatenate({"unit ", std::to_string(unit.unitId),
                                               ": its fixed-interface modes (recovery data) are ", stored.note}),
                            base::Severity::Information));
            }
        };
        const auto computeModes = [&]
        {
            reduction::FixedInterfaceModes computed =
                reduction::CraigBampton::computeModes(*unit.system, request, kernels, unit.factorOptions, budget);
            unitRecord.modeFactorisationCount += static_cast<std::int64_t>(computed.factorisationCount);
            auto data = std::make_shared<reduction::CachedRecoveryData>();
            data->modes = std::move(computed.modes);
            data->eigenvalues = std::move(computed.eigenvalues);
            return data;
        };
        if (!unit.modal)
        {
            // Its modes and the q part of its reduction, from its interior factor (computed again when released).
            const std::shared_ptr<reduction::CachedRecoveryData> data = computeModes();
            ensureInteriorFactor(unit, unitRecord, kernels, m_inputs.cancellation, budget);
            unit.modal = std::make_shared<const reduction::ModalReduction>(reduction::CraigBampton::reduceModes(
                *unit.condensation, data->modes, data->eigenvalues, kernels, budget, cancellation));
            unitRecord.modalOutcome = "computed";
            if (modalKey)
            {
                reduction::CachedReduction entry;
                entry.modal = unit.modal;
                entry.largestPivotRatio = unit.condensation->largestPivotRatio();
                entry.interiorDofCount = unit.system->interiorDofCount();
                const reduction::CacheStore stored = m_cacheSession->storeReduction(*modalKey, entry);
                if (!stored.isStored)
                    sink.report(problem("cache-not-stored", "UNIT", static_cast<base::EntityId>(unit.unitId),
                                        base::concatenate({"unit ", std::to_string(unit.unitId),
                                                           ": its Craig-Bampton modal part is ", stored.note}),
                                        base::Severity::Information));
            }
            storeModes(*data);
            if (needsModes)
            {
                unit.modes = data;
                unitRecord.modesOutcome = "computed";
            }
        }
        else if (needsModes)
        {
            // The modes of a modal part from the cache: its recovery data, or the modes computed again, to the
            // same bits, when they have been evicted.
            std::string cause = "the cache has no recovery data of the unit for this request";
            if (modalKey)
            {
                const reduction::CacheLookup<reduction::CachedRecoveryData> found =
                    m_cacheSession->lookupRecoveryData(*modalKey, budget);
                if (found.isHit())
                {
                    unit.modes = found.entry;
                    unitRecord.modesOutcome = std::string(reduction::cacheOutcomeName(found.outcome));
                }
                else
                    cause = found.cause;
            }
            if (!unit.modes)
            {
                const std::shared_ptr<reduction::CachedRecoveryData> data = computeModes();
                if (!isSameEigenvalues(data->eigenvalues, unit.modal->eigenvalues()))
                    throw base::CacheError(base::concatenate(
                        {"the cached Craig-Bampton modal part of unit ", std::to_string(unit.unitId),
                         " has other fixed-interface eigenvalues than its modes computed again: the cache entry does "
                         "not describe the unit as it is; clear the cache and run again, "
                         "and report the case (a cache key that misses a change)"}));
                unit.modes = data;
                storeModes(*data);
                unitRecord.modesOutcome = "recomputed";
                unitRecord.modesNote = unitRecord.isModesStored
                                           ? cause
                                           : base::concatenate({cause, "; not stored again: ", unitRecord.modesNote});
            }
        }
        unitRecord.modeCount = static_cast<std::int64_t>(unit.modal->modeCount());
        const linalg::DenseMatrix& eigenvalues = unit.modal->eigenvalues();
        unitRecord.highestModeEigenvalue = eigenvalues.rows() > 0 ? eigenvalues(eigenvalues.rows() - 1, 0) : 0.0;
        // The factor stays for what reads it (a recovery, the projections of a buckling analysis), else goes.
        if (!needsModes && unit.condensation->isInteriorFactorKept())
        {
            unit.condensation->releaseInteriorFactor();
            unit.isFactorKept = false;
        }
    }
}

CondensedStatics::EigenGroup CondensedStatics::solveEigenGroup(const EigenGroupInputs& inputs,
                                                               base::MemoryBudget& budget, base::DiagnosticSink& sink)
{
    if (!m_inputs.isCraigBampton)
        throw std::logic_error(
            "CondensedStatics::solveEigenGroup: the units were condensed without their mass; set "
            "Inputs::isCraigBampton");
    const Inputs& in = m_inputs;
    const input::ModelSnapshot& snapshot = *in.snapshot;
    const AnalysisModel& model = *in.model;
    const assembly::DofMap& dofMap = model.dofMap();
    const linalg::DenseKernels& kernels = *in.kernels;
    Units& units = *m_units;
    const GroupInputs& group = inputs.group;
    const assembly::SpcSet& spc = *group.spc;

    // The full path's relations and sets of the subcases (their problems reported as the full path reports them), the
    // relations of the interface.
    const input::ConstraintTable* constraints =
        snapshot.hasTable(input::TableRole::Spc) ? &snapshot.constraints() : nullptr;
    const input::RigidTable* rigidElements =
        snapshot.hasTable(input::TableRole::RigidElements) ? &snapshot.rigidElements() : nullptr;
    const assembly::MpcEquations mpc(constraints, dofMap, group.selection.mpc, budget, sink);
    const assembly::GlueContext glueContext{rigidElements, &mpc, units.fasteners.get(), spc.sbDofs(), spc.sgDofs()};
    const assembly::GlueConstraints glue(snapshot, group.selection.contact, group.selection.subcase, dofMap,
                                         glueContext, budget, sink);
    const assembly::RigidElimination fullGm(rigidElements, &mpc, dofMap, budget, sink, units.fasteners.get(), &glue);
    const assembly::DofSets sets(
        dofMap, assembly::DofSetInputs{fullGm.mDofs(), spc.sbDofs(), spc.sgDofs(), {}, group.supports}, budget, sink);
    stopOnErrors(sink, "the constraints");
    base::DiagnosticSink interfaceSink; // the same rows as fullGm's: its diagnostics are reported once
    const assembly::RigidElimination interfaceGm(rigidElements, &mpc, dofMap, budget, interfaceSink,
                                                 units.fasteners.get(), &glue, units.planned->interfaceRowFilter());

    // The preload of a buckling analysis on the g set (six values per grid point row, then the scalar points).
    const std::size_t gridValues = 6 * static_cast<std::size_t>(dofMap.gridCount());
    const elements::NodalDisplacements preload{
        inputs.preload.empty() ? inputs.preload : inputs.preload.first(gridValues),
        inputs.preload.empty() ? inputs.preload : inputs.preload.subspan(gridValues), inputs.temperatures};
    if (inputs.isBuckling && inputs.preload.size() != static_cast<std::size_t>(dofMap.gSetSize()))
        throw std::invalid_argument(base::concatenate(
            {"CondensedStatics::solveEigenGroup: a buckling solve needs the g-set preload (",
             std::to_string(dofMap.gSetSize()), " values), found ", std::to_string(inputs.preload.size())}));

    // Each unit's part of the second matrix on its boundary DOFs then its modal coordinates.
    std::vector<linalg::DenseMatrix> seconds(units.units.size()); // bounded by the units
    std::vector<reduction::InterfaceUnit> interfaceUnits;         // bounded by the units
    for (std::size_t u = 0; u < units.units.size(); ++u)
    {
        CondensedUnit& unit = units.units[u];
        CondensedUnitRecord& record = m_record.units[u];
        const reduction::ModalReduction* modal = inputs.includesModes ? unit.modal.get() : nullptr;
        if (inputs.includesModes && modal == nullptr)
            throw std::logic_error(
                "CondensedStatics::solveEigenGroup: the modal parts are not made; call reduceModes "
                "first");
        const auto boundary = static_cast<std::size_t>(unit.system->boundaryDofCount());
        const std::size_t modes = modal != nullptr ? modal->modeCount() : 0;
        if (!inputs.isBuckling)
        {
            // M_CB = [M_aa M_aq; M_qa M_qq].
            const linalg::DenseMatrix& boundaryMass = unit.reduction->boundaryMass;
            seconds[u] = linalg::DenseMatrix(budget, boundary + modes, boundary + modes, "unit Craig-Bampton mass");
            linalg::DenseMatrix& mass = seconds[u];
            for (std::size_t j = 0; j < boundary; ++j)
                for (std::size_t i = 0; i < boundary; ++i)
                    mass(i, j) = boundaryMass(i, j);
            for (std::size_t q = 0; q < modes; ++q)
            {
                for (std::size_t i = 0; i < boundary; ++i)
                {
                    mass(i, boundary + q) = modal->couplingMass()(i, q);
                    mass(boundary + q, i) = modal->couplingMass()(i, q);
                }
                for (std::size_t p = 0; p < modes; ++p)
                    mass(boundary + p, boundary + q) = modal->modalMass()(p, q);
            }
        }
        else
        {
            // T_CBᵀ·K_d·T_CB from the unit's elements under the preload (computed at each analysis, never
            // cached).
            const linalg::SymmetricCsc differential =
                unit.system->differentialStiffness(units.unitModel, unit.description, preload, budget, in.pool);
            ensureInteriorFactor(unit, record, kernels, m_inputs.cancellation, budget);
            const linalg::DenseMatrix noModes(budget, static_cast<std::size_t>(unit.system->interiorDofCount()), 0,
                                              "unit without modes");
            const linalg::DenseMatrix& unitModes = modal != nullptr && unit.modes ? unit.modes->modes : noModes;
            if (modal != nullptr && unitModes.columns() != modes)
                throw std::logic_error(
                    "CondensedStatics::solveEigenGroup: a buckling solve needs every unit's modes "
                    "(reduceModes with needsEveryUnitsModes)");
            seconds[u] = reduction::CraigBampton::project(*unit.condensation, differential, unitModes, kernels, budget,
                                                          in.cancellation);
        }
        interfaceUnits.push_back({&unit.condensation->reducedUnit(), unit.system.get(), modal, &seconds[u]});
    }

    reduction::InterfaceInputs interfaceInputs;
    interfaceInputs.elements = &model.elementSet();
    interfaceInputs.dofMap = &dofMap;
    interfaceInputs.elementIndex = units.elementIndex.get();
    interfaceInputs.isInteriorNode = units.planned->interiorNodes();
    interfaceInputs.elementIds = units.interfaceElements.span();
    interfaceInputs.units = interfaceUnits;
    interfaceInputs.spiders = units.spiders;
    interfaceInputs.relations = &interfaceGm;
    interfaceInputs.modelSets = &sets;
    interfaceInputs.singularity = group.singularity;
    interfaceInputs.pivotPolicy = group.pivotPolicy;
    interfaceInputs.massOptions = units.massOptions;
    interfaceInputs.nonStructuralMass = units.shares.get();
    interfaceInputs.second = inputs.isBuckling ? reduction::InterfaceSecondMatrix::DifferentialStiffness
                                               : reduction::InterfaceSecondMatrix::Mass;
    interfaceInputs.preload = inputs.isBuckling ? &preload : nullptr;
    interfaceInputs.factorises = false;
    EigenGroup result;
    result.interface =
        std::make_unique<reduction::InterfaceProblem>(interfaceInputs, kernels, budget, in.pool, in.cancellation);
    seconds.clear(); // copied into the interface's second matrix
    const reduction::InterfaceProblem& interface = *result.interface;
    result.equationCount = interface.equationCount();
    for (std::size_t u = 0; u < units.units.size(); ++u)
        result.modalCoordinateCount += interface.modalDofCount(u);
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    result.isAnalysisSet = base::BudgetedBuffer<std::uint8_t>(budget, gSize, "condensed analysis set");
    for (std::size_t dof = 0; dof < gSize; ++dof)
        result.isAnalysisSet[dof] = sets.contains(assembly::DofSet::F, static_cast<base::DofIndex>(dof)) ? 1 : 0;

    // The table of the singular components: the units' interior points and the interface's.
    {
        std::vector<assembly::SingularityRow> rows(interface.singularities().begin(), interface.singularities().end());
        for (const CondensedUnit& unit : units.units)
        {
            const std::span<const assembly::SingularityRow> unitRows = unit.system->interiorSingularities();
            rows.insert(rows.end(), unitRows.begin(), unitRows.end());
        }
        std::sort(rows.begin(), rows.end(), [](const assembly::SingularityRow& a, const assembly::SingularityRow& b)
                  { return a.pointId != b.pointId ? a.pointId < b.pointId : a.component < b.component; });
        std::ostringstream table;
        assembly::writeSingularityTable(table, rows, group.singularity);
        result.singularityTable = table.str();
    }

    // The pencil of the interface and its roots.
    if (interface.equationCount() == 0)
    {
        result.solution.isComplete = true;
        return result;
    }
    linalg::FactorOptions factorOptions;
    factorOptions.cancellation = in.cancellation;
    if (inputs.isBuckling)
    {
        linalg::BucklingPencil pencil(interface.symbolic(), interface.matrix(), interface.secondMatrix(), kernels,
                                      factorOptions, budget);
        result.solution = linalg::BlockLanczos(inputs.lanczos).solve(pencil, inputs.request, kernels, budget);
    }
    else
    {
        linalg::SparsePencil pencil(interface.symbolic(), interface.matrix(), interface.secondMatrix(), kernels,
                                    factorOptions, budget);
        result.solution = linalg::BlockLanczos(inputs.lanczos).solve(pencil, inputs.request, kernels, budget);
    }
    return result;
}

reduction::ModeRequest CondensedStatics::chooseModeRequest(const GroupInputs& group,
                                                           const linalg::EigenRequest& reference,
                                                           const linalg::LanczosOptions& lanczos,
                                                           CraigBamptonRecord& record, base::MemoryBudget& budget,
                                                           base::DiagnosticSink& sink, double cutoffFactor)
{
    constexpr double TWO_PI = 6.283185307179586;
    const auto frequencyOf = [](double eigenvalue) { return std::sqrt(std::max(eigenvalue, 0.0)) / TWO_PI; };
    if (!(cutoffFactor >= 1.0))
        throw std::invalid_argument(
            base::concatenate({"CondensedStatics::chooseModeRequest: expected a cut-off "
                               "factor of at least 1, found ",
                               std::to_string(cutoffFactor)}));
    record = CraigBamptonRecord{};
    record.cutoffFactor = cutoffFactor;
    reduction::ModeRequest request;
    // The plan's count (the same for every unit of a plan, PlanRequest::modeCount).
    for (const reduction::UnitPlan& unit : m_inputs.plan->units())
        if (unit.decision == reduction::UnitDecision::Condensed && unit.modeCount > 0)
            request.count = unit.modeCount;
    if (request.isCount())
    {
        record.modeCount = request.count;
        record.reference = "count";
        return request;
    }
    double highest = 0.0;
    if (reference.modeCount > 0)
    {
        // The highest root of the request on the statically condensed model (the Craig-Bampton model without
        // modes): the Rayleigh-Ritz roots of a subspace bound the true ones from above.
        EigenGroupInputs guyan;
        guyan.group = group;
        guyan.request = reference;
        guyan.lanczos = lanczos;
        guyan.includesModes = false;
        const EigenGroup solved = solveEigenGroup(guyan, budget, sink);
        const std::vector<double>& roots = solved.solution.eigenvalues;
        highest = roots.empty() ? (reference.hasUpperBound() ? reference.upperBound : 0.0) : roots.back();
        record.reference = "statically condensed";
    }
    else
    {
        highest = reference.hasUpperBound() ? reference.upperBound : 0.0;
        record.reference = "V2";
    }
    record.referenceEigenvalue = highest;
    record.referenceFrequency = frequencyOf(highest);
    // f_c = CB_CUTOFF_FACTOR·f_max, so λ_c = CB_CUTOFF_FACTOR²·λ_max.
    request.cutoffEigenvalue = cutoffFactor * cutoffFactor * std::max(highest, 0.0);
    record.cutoffEigenvalue = request.cutoffEigenvalue;
    record.cutoffFrequency = frequencyOf(request.cutoffEigenvalue);
    return request;
}

CondensedStatics::ModeVectors CondensedStatics::recoverModes(const EigenGroup& group, base::MemoryBudget& budget)
{
    const assembly::DofMap& dofMap = m_inputs.model->dofMap();
    const linalg::DenseKernels& kernels = *m_inputs.kernels;
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    const reduction::InterfaceProblem& interface = *group.interface;
    const std::size_t modes = group.solution.eigenvalues.size();
    ModeVectors result{linalg::DenseMatrix(budget, gSize, modes, "condensed eigenvectors"),
                       base::BudgetedBuffer<std::uint8_t>(budget, gSize, "condensed eigenvector DOFs"),
                       base::BudgetedBuffer<std::uint8_t>(budget, gSize, "condensed eigenvector analysis set")};
    const auto finish = [&]
    {
        for (std::size_t dof = 0; dof < gSize; ++dof)
            result.isAnalysisSet[dof] =
                result.isKnown[dof] != 0 && group.isAnalysisSet.size() == gSize && group.isAnalysisSet[dof] != 0 ? 1
                                                                                                                 : 0;
    };
    if (modes == 0)
    {
        finish();
        return result;
    }
    // The interface's points: its equations expanded (held DOFs zero, the relations of the interface).
    const auto localDofs = static_cast<std::size_t>(interface.localDofCount());
    linalg::DenseMatrix local(budget, localDofs, modes, "condensed interface eigenvectors");
    interface.expand(group.solution.eigenvectors.view(), {}, local.view());
    for (std::size_t l = 0; l < localDofs; ++l)
        if (const std::optional<base::DofIndex> dof = interface.modelDofOf(static_cast<std::int64_t>(l)))
        {
            for (std::size_t m = 0; m < modes; ++m)
                result.vectors(static_cast<std::size_t>(*dof), m) = local(l, m);
            result.isKnown[static_cast<std::size_t>(*dof)] = 1;
        }
    // The interiors of the recovered units: u_o = Ψ·u_a + Φ·q, then their relations and spiders expanded.
    Units& units = *m_units;
    for (std::size_t u = 0; u < units.units.size(); ++u)
    {
        CondensedUnit& unit = units.units[u];
        if (!unit.isRecovered)
            continue;
        if (m_inputs.cancellation != nullptr)
            m_inputs.cancellation->throwIfCancelled();
        const reduction::UnitSystem& system = *unit.system;
        const std::span<const std::int64_t> boundaryLocals = interface.unitLocalDofs(u);
        const auto modalCount = static_cast<std::size_t>(interface.modalDofCount(u));
        linalg::DenseMatrix boundaryValues(budget, boundaryLocals.size(), modes, "unit mode boundary values");
        linalg::DenseMatrix modalValues(budget, modalCount, modes, "unit mode modal values");
        for (std::size_t m = 0; m < modes; ++m)
        {
            for (std::size_t i = 0; i < boundaryLocals.size(); ++i)
                boundaryValues(i, m) = local(static_cast<std::size_t>(boundaryLocals[i]), m);
            for (std::size_t q = 0; q < modalCount; ++q)
                modalValues(q, m) = local(static_cast<std::size_t>(interface.firstModalDof(u)) + q, m);
        }
        const linalg::DenseMatrix noModes(budget, static_cast<std::size_t>(system.interiorDofCount()), 0,
                                          "unit without modes");
        const linalg::DenseMatrix& unitModes = modalCount > 0 && unit.modes ? unit.modes->modes : noModes;
        if (unitModes.columns() != modalCount)
            throw std::logic_error(
                "CondensedStatics::recoverModes: a recovered unit's modes are not in memory "
                "(reduceModes keeps them)");
        ensureInteriorFactor(unit, m_record.units[u], kernels, m_inputs.cancellation, budget);
        linalg::DenseMatrix equationValues(budget, static_cast<std::size_t>(system.equationCount()), modes,
                                           "unit mode equations");
        reduction::CraigBampton::recover(*unit.condensation, unitModes, boundaryValues.view(), modalValues.view(),
                                         equationValues.view(), kernels, m_inputs.cancellation);
        linalg::DenseMatrix displacements(budget, static_cast<std::size_t>(system.localDofCount()), modes,
                                          "unit mode displacements");
        system.expandSolution(equationValues.view(), {}, displacements.view());
        for (std::size_t l = 0; l < unit.modelDofOfLocal.size(); ++l)
        {
            const base::DofIndex dof = unit.modelDofOfLocal[l];
            if (dof < 0 || unit.isInteriorLocal[l] == 0)
                continue;
            for (std::size_t m = 0; m < modes; ++m)
                result.vectors(static_cast<std::size_t>(dof), m) = displacements(l, m);
            result.isKnown[static_cast<std::size_t>(dof)] = 1;
        }
        if (!unit.isFactorKept)
            unit.condensation->releaseInteriorFactor();
    }
    finish();
    return result;
}

// ---------------------------------------------------------------------------------------------------------------------
// Loads a solution makes itself: the static aeroelastic trim on the condensed path

struct CondensedStatics::GroupConstraints
{
    std::vector<base::DofIndex> supports; // the r set (DofSets reads it)
    std::unique_ptr<assembly::MpcEquations> mpc;
    assembly::GlueContext glueContext; // GlueConstraints reads it; its spans view the caller's SpcSet
    std::unique_ptr<assembly::GlueConstraints> glue;
    std::unique_ptr<assembly::RigidElimination> fullGm;
    std::unique_ptr<assembly::DofSets> sets;
    std::unique_ptr<assembly::RigidElimination> interfaceGm;
};

CondensedStatics::LoadGroup CondensedStatics::prepareLoadGroup(const GroupInputs& group, base::MemoryBudget& budget,
                                                               base::DiagnosticSink& sink)
{
    const Inputs& in = m_inputs;
    const input::ModelSnapshot& snapshot = *in.snapshot;
    const AnalysisModel& model = *in.model;
    const assembly::DofMap& dofMap = model.dofMap();
    const linalg::DenseKernels& kernels = *in.kernels;
    Units& units = *m_units;
    const assembly::SpcSet& spc = *group.spc;

    // The full path's relations and sets of the subcases (their problems reported as the full path reports them), the
    // relations of the interface; the supports held (the r set), with no inertia relief: the solution balances the
    // structure itself.
    const input::ConstraintTable* constraints =
        snapshot.hasTable(input::TableRole::Spc) ? &snapshot.constraints() : nullptr;
    const input::RigidTable* rigidElements =
        snapshot.hasTable(input::TableRole::RigidElements) ? &snapshot.rigidElements() : nullptr;
    auto held = std::make_shared<GroupConstraints>();
    held->supports = group.supports;
    held->mpc = std::make_unique<assembly::MpcEquations>(constraints, dofMap, group.selection.mpc, budget, sink);
    held->glueContext =
        assembly::GlueContext{rigidElements, held->mpc.get(), units.fasteners.get(), spc.sbDofs(), spc.sgDofs()};
    held->glue = std::make_unique<assembly::GlueConstraints>(snapshot, group.selection.contact, group.selection.subcase,
                                                             dofMap, held->glueContext, budget, sink);
    held->fullGm = std::make_unique<assembly::RigidElimination>(rigidElements, held->mpc.get(), dofMap, budget, sink,
                                                                units.fasteners.get(), held->glue.get());
    held->sets = std::make_unique<assembly::DofSets>(
        dofMap, assembly::DofSetInputs{held->fullGm->mDofs(), spc.sbDofs(), spc.sgDofs(), {}, held->supports}, budget,
        sink);
    stopOnErrors(sink, "the constraints");
    base::DiagnosticSink interfaceSink; // the same rows as fullGm's: its diagnostics are reported once
    held->interfaceGm = std::make_unique<assembly::RigidElimination>(
        rigidElements, held->mpc.get(), dofMap, budget, interfaceSink, units.fasteners.get(), held->glue.get(),
        units.planned->interfaceRowFilter());

    std::vector<reduction::InterfaceUnit> interfaceUnits; // bounded by the units
    for (const CondensedUnit& unit : units.units)
        interfaceUnits.push_back({&unit.condensation->reducedUnit(), unit.system.get()});
    reduction::InterfaceInputs interfaceInputs;
    interfaceInputs.elements = &model.elementSet();
    interfaceInputs.dofMap = &dofMap;
    interfaceInputs.elementIndex = units.elementIndex.get();
    interfaceInputs.isInteriorNode = units.planned->interiorNodes();
    interfaceInputs.elementIds = units.interfaceElements.span();
    interfaceInputs.units = interfaceUnits;
    interfaceInputs.spiders = units.spiders;
    interfaceInputs.relations = held->interfaceGm.get();
    interfaceInputs.modelSets = held->sets.get();
    interfaceInputs.singularity = group.singularity;
    interfaceInputs.pivotPolicy = group.pivotPolicy;
    // The interface elements' mass: their inertial loads and rigid-body mass beside the units'.
    interfaceInputs.computesMass = true;
    interfaceInputs.massOptions = units.massOptions;
    interfaceInputs.nonStructuralMass = units.shares.get();
    LoadGroup result;
    result.interface =
        std::make_unique<reduction::InterfaceProblem>(interfaceInputs, kernels, budget, in.pool, in.cancellation);
    const reduction::InterfaceProblem& interface = *result.interface;
    m_largestPivotRatio = std::max(m_largestPivotRatio, interface.pivotDiagnostics().largestRatio());
    reportKeptPivots(
        interface.pivotDiagnostics(), [&interface](std::int32_t equation)
        { return interface.nameOf(interface.localDofOfEquation(equation)); }, group.selection.subcase, sink);

    // The table of the singular components: the units' interior points and the interface's.
    {
        std::vector<assembly::SingularityRow> rows(interface.singularities().begin(), interface.singularities().end());
        for (const CondensedUnit& unit : units.units)
        {
            const std::span<const assembly::SingularityRow> unitRows = unit.system->interiorSingularities();
            rows.insert(rows.end(), unitRows.begin(), unitRows.end());
        }
        std::sort(rows.begin(), rows.end(), [](const assembly::SingularityRow& a, const assembly::SingularityRow& b)
                  { return a.pointId != b.pointId ? a.pointId < b.pointId : a.component < b.component; });
        std::ostringstream table;
        assembly::writeSingularityTable(table, rows, group.singularity);
        result.singularityTable = table.str();
    }

    // The held DOFs of the interface and of the units on the g set, and the rigid-body mass of the structure.
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    result.isConstrained = base::BudgetedBuffer<std::uint8_t>(budget, gSize, "condensed constrained DOFs");
    for (const std::int64_t local : interface.heldDofs())
        if (const std::optional<base::DofIndex> dof = interface.modelDofOf(local))
            result.isConstrained[static_cast<std::size_t>(*dof)] = 1;
    result.rigidBodyMass = interface.rigidBodyMass();
    for (const CondensedUnit& unit : units.units)
    {
        for (const std::int64_t local : unit.system->heldDofs())
            if (unit.modelDofOfLocal[static_cast<std::size_t>(local)] >= 0)
                result.isConstrained[static_cast<std::size_t>(unit.modelDofOfLocal[static_cast<std::size_t>(local)])] =
                    1;
        result.rigidBodyMass += unit.system->rigidBodyMass();
    }
    result.gm = held->fullGm.get();
    result.constraints = std::move(held);
    return result;
}

void CondensedStatics::solveLoads(const LoadGroup& group, linalg::ConstDenseMatrixView loadsG,
                                  linalg::ConstDenseMatrixView accelerations, bool recoversUnits,
                                  linalg::DenseMatrixView displacementsG, base::MemoryBudget& budget)
{
    const reduction::InterfaceProblem& interface = *group.interface;
    const assembly::DofMap& dofMap = m_inputs.model->dofMap();
    const linalg::DenseKernels& kernels = *m_inputs.kernels;
    Units& units = *m_units;
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    const std::size_t columns = loadsG.columns;
    const bool hasAccelerations = accelerations.columns > 0;
    if (loadsG.rows != gSize || displacementsG.rows != gSize || displacementsG.columns != columns ||
        (hasAccelerations && (accelerations.rows != RIGID_BODY_MODE_COUNT || accelerations.columns != columns)))
        throw std::invalid_argument(base::concatenate(
            {"CondensedStatics::solveLoads: expected loads and displacements of ", std::to_string(gSize),
             " rows and the same columns, and accelerations of 6 rows (or none), found loads ",
             std::to_string(loadsG.rows), " x ", std::to_string(columns), ", displacements ",
             std::to_string(displacementsG.rows), " x ", std::to_string(displacementsG.columns), ", accelerations ",
             std::to_string(accelerations.rows), " x ", std::to_string(accelerations.columns)}));
    const auto localDofs = static_cast<std::size_t>(interface.localDofCount());
    const auto equations = static_cast<std::size_t>(interface.equationCount());

    // The loads on the interface's local DOFs: those at its points, then the inertial loads of the accelerations
    // (each unit's condensed inertial patterns f*, the interface elements' −M·Φ_r).
    linalg::DenseMatrix local(budget, localDofs, columns, "interface loads of the solution");
    for (std::size_t dof = 0; dof < gSize; ++dof)
    {
        const std::optional<std::int64_t> localDof = interface.localDofOf(static_cast<base::DofIndex>(dof));
        for (std::size_t c = 0; c < columns; ++c)
        {
            const double value = loadsG(dof, c);
            if (value == 0.0)
                continue;
            if (!localDof)
            {
                const linalg::NodeDof nodeDof = dofMap.nodeDofOf(static_cast<base::DofIndex>(dof));
                throw std::invalid_argument(base::concatenate(
                    {"CondensedStatics::solveLoads: a load on component ", std::to_string(nodeDof.component + 1),
                     " of point ", std::to_string(dofMap.pointIdOf(static_cast<base::DofIndex>(dof))),
                     ", an interior point of a condensed unit; the plan must keep the loaded points on the "
                     "boundaries of their units (retain them)"}));
            }
            local(static_cast<std::size_t>(*localDof), c) += value;
        }
    }
    if (hasAccelerations)
    {
        for (std::size_t u = 0; u < units.units.size(); ++u)
        {
            const CondensedUnit& unit = units.units[u];
            if (unit.patternCount < unit.loadPatternCount + RIGID_BODY_MODE_COUNT)
                throw std::logic_error(
                    "CondensedStatics::solveLoads: the units were condensed without their inertial patterns; set "
                    "Inputs::hasInertiaRelief");
            const linalg::DenseMatrix& condensed = unit.condensation->reducedUnit().patterns();
            const std::span<const std::int64_t> locals = interface.unitLocalDofs(u);
            for (std::size_t c = 0; c < columns; ++c)
                for (std::size_t mode = 0; mode < RIGID_BODY_MODE_COUNT; ++mode)
                {
                    const double a = accelerations(mode, c);
                    if (a == 0.0)
                        continue;
                    for (std::size_t i = 0; i < locals.size(); ++i)
                        local(static_cast<std::size_t>(locals[i]), c) += condensed(i, unit.loadPatternCount + mode) * a;
                }
        }
        if (interface.inertialLoads().columns() == RIGID_BODY_MODE_COUNT)
            for (std::size_t c = 0; c < columns; ++c)
                for (std::size_t mode = 0; mode < RIGID_BODY_MODE_COUNT; ++mode)
                {
                    const double a = accelerations(mode, c);
                    if (a == 0.0)
                        continue;
                    for (std::size_t l = 0; l < localDofs; ++l)
                        local(l, c) += interface.inertialLoads()(l, mode) * a;
                }
    }

    // The interface problem: P_n = P̄_n + Gmᵀ·P_m, its equations solved, u_m = Gm·u_n, the held DOFs zero.
    interface.reduceLoads(local.view());
    const linalg::DenseMatrix noEnforced(budget, localDofs, 0, "no enforced values");
    linalg::DenseMatrix rhs(budget, equations, columns, "interface right-hand sides of the solution");
    interface.equationLoads(local.view(), noEnforced.view(), rhs.view());
    if (equations > 0)
        interface.solve(rhs.view(), kernels, m_inputs.cancellation);
    linalg::DenseMatrix displacementsLocal(budget, localDofs, columns, "interface displacements of the solution");
    interface.expand(rhs.view(), noEnforced.view(), displacementsLocal.view());
    for (std::size_t c = 0; c < columns; ++c)
        for (std::size_t dof = 0; dof < gSize; ++dof)
            displacementsG(dof, c) = 0.0;
    for (std::size_t l = 0; l < localDofs; ++l)
        if (const std::optional<base::DofIndex> dof = interface.modelDofOf(static_cast<std::int64_t>(l)))
            for (std::size_t c = 0; c < columns; ++c)
                displacementsG(static_cast<std::size_t>(*dof), c) = displacementsLocal(l, c);
    if (!recoversUnits)
        return;

    // The interiors of the recovered units: u_o = K_oo⁻¹·(f_o − K_oa·u_a), f_o their inertial loads.
    for (std::size_t u = 0; u < units.units.size(); ++u)
    {
        CondensedUnit& unit = units.units[u];
        if (!unit.isRecovered)
            continue;
        if (m_inputs.cancellation != nullptr)
            m_inputs.cancellation->throwIfCancelled();
        const reduction::UnitSystem& system = *unit.system;
        const auto unitLocal = static_cast<std::size_t>(system.localDofCount());
        linalg::DenseMatrix unitLoads(budget, unitLocal, columns, "unit loads of the solution");
        for (std::size_t c = 0; c < columns && hasAccelerations; ++c)
            for (std::size_t mode = 0; mode < RIGID_BODY_MODE_COUNT; ++mode)
            {
                const double a = accelerations(mode, c);
                if (a == 0.0)
                    continue;
                for (std::size_t l = 0; l < unitLocal; ++l)
                    unitLoads(l, c) += unit.patternLoads(l, unit.loadPatternCount + mode) * a;
            }
        const linalg::DenseMatrix noUnitEnforced(budget, unitLocal, 0, "no unit enforced values");
        linalg::DenseMatrix equationValues(budget, static_cast<std::size_t>(system.equationCount()), columns,
                                           "unit equations of the solution");
        system.equationLoads(unitLoads.view(), noUnitEnforced.view(), equationValues.view(), budget);
        const std::span<const std::int64_t> boundaryLocals = interface.unitLocalDofs(u);
        linalg::DenseMatrix boundaryValues(budget, boundaryLocals.size(), columns, "unit boundary displacements");
        for (std::size_t c = 0; c < columns; ++c)
            for (std::size_t i = 0; i < boundaryLocals.size(); ++i)
                boundaryValues(i, c) = displacementsLocal(static_cast<std::size_t>(boundaryLocals[i]), c);
        ensureInteriorFactor(unit, m_record.units[u], kernels, m_inputs.cancellation, budget);
        unit.condensation->recover(equationValues.view(), boundaryValues.view(), kernels, m_inputs.cancellation);
        linalg::DenseMatrix unitDisplacements(budget, unitLocal, columns, "unit displacements of the solution");
        system.expandSolution(equationValues.view(), noUnitEnforced.view(), unitDisplacements.view());
        for (std::size_t l = 0; l < unitLocal; ++l)
        {
            const base::DofIndex dof = unit.modelDofOfLocal[l];
            if (dof < 0 || unit.isInteriorLocal[l] == 0)
                continue;
            for (std::size_t c = 0; c < columns; ++c)
                displacementsG(static_cast<std::size_t>(dof), c) = unitDisplacements(l, c);
        }
        if (!unit.isFactorKept)
            unit.condensation->releaseInteriorFactor();
    }
}

void CondensedStatics::addInertialLoads(const LoadGroup& group, std::span<const double> acceleration,
                                        std::span<double> loadsG) const
{
    const reduction::InterfaceProblem& interface = *group.interface;
    const Units& units = *m_units;
    if (acceleration.size() != RIGID_BODY_MODE_COUNT ||
        loadsG.size() != static_cast<std::size_t>(m_inputs.model->dofMap().gSetSize()))
        throw std::invalid_argument("CondensedStatics::addInertialLoads: expected 6 accelerations and g-set loads");
    for (const CondensedUnit& unit : units.units)
    {
        const linalg::DenseMatrix& inertial = unit.system->inertialLoads();
        if (inertial.columns() != RIGID_BODY_MODE_COUNT)
            throw std::logic_error("CondensedStatics::addInertialLoads: a unit has no inertial loads (no mass given)");
        for (std::size_t l = 0; l < unit.modelDofOfLocal.size(); ++l)
        {
            const base::DofIndex dof = unit.modelDofOfLocal[l];
            if (dof < 0)
                continue;
            double load = 0.0;
            for (std::size_t mode = 0; mode < RIGID_BODY_MODE_COUNT; ++mode)
                load += inertial(l, mode) * acceleration[mode];
            loadsG[static_cast<std::size_t>(dof)] += load;
        }
    }
    if (interface.inertialLoads().columns() != RIGID_BODY_MODE_COUNT)
        return;
    for (std::size_t l = 0; l < static_cast<std::size_t>(interface.localDofCount()); ++l)
        if (const std::optional<base::DofIndex> dof = interface.modelDofOf(static_cast<std::int64_t>(l)))
        {
            double load = 0.0;
            for (std::size_t mode = 0; mode < RIGID_BODY_MODE_COUNT; ++mode)
                load += interface.inertialLoads()(l, mode) * acceleration[mode];
            loadsG[static_cast<std::size_t>(*dof)] += load;
        }
}

void CondensedStatics::addStructureForces(const LoadGroup& group, std::span<const double> displacementsG,
                                          std::span<const double> acceleration, std::span<double> forcesG,
                                          std::vector<output::StiffnessRecovery::UnitRow>& unitRows,
                                          base::MemoryBudget& budget) const
{
    const reduction::InterfaceProblem& interface = *group.interface;
    const AnalysisModel& model = *m_inputs.model;
    const assembly::DofMap& dofMap = model.dofMap();
    const Units& units = *m_units;
    // K_e·u_e of the elements of the interface and of the recovered units.
    addElementForces(model.elementSet(), dofMap, displacementsG, units.excludedElements.span(), forcesG,
                     m_inputs.cancellation);
    // Each unit not recovered by its boundary force K*·u_a − f*·α (α: the accelerations on its inertial patterns).
    const std::size_t firstRow = unitRows.size();
    for (std::size_t u = 0; u < units.units.size(); ++u)
    {
        const CondensedUnit& unit = units.units[u];
        if (unit.isRecovered)
            continue;
        const reduction::ReducedUnit& reduced = unit.condensation->reducedUnit();
        const std::span<const std::int64_t> locals = interface.unitLocalDofs(u);
        linalg::DenseMatrix boundaryDisplacements(budget, locals.size(), 1, "unit boundary displacements");
        for (std::size_t i = 0; i < locals.size(); ++i)
            if (const std::optional<base::DofIndex> dof = interface.modelDofOf(locals[i]))
                boundaryDisplacements(i, 0) = displacementsG[static_cast<std::size_t>(*dof)];
        linalg::DenseMatrix coefficients(budget, reduced.patterns().columns(), 1, "unit pattern coefficients");
        for (std::size_t mode = 0; mode < RIGID_BODY_MODE_COUNT && !acceleration.empty() &&
                                   unit.loadPatternCount + mode < coefficients.rows();
             ++mode)
            coefficients(unit.loadPatternCount + mode, 0) = acceleration[mode];
        linalg::DenseMatrix forces(budget, locals.size(), 1, "unit boundary forces");
        reduced.boundaryForces(boundaryDisplacements.view(), coefficients.view(), forces.view(), *m_inputs.kernels);
        std::map<std::int32_t, std::array<double, 6>> rowsOfNode; // bounded by the unit's boundary points
        for (std::size_t i = 0; i < locals.size(); ++i)
        {
            const std::optional<base::DofIndex> dof = interface.modelDofOf(locals[i]);
            if (!dof)
                continue; // a generated fastener point
            forcesG[static_cast<std::size_t>(*dof)] += forces(i, 0);
            const linalg::NodeDof nodeDof = dofMap.nodeDofOf(*dof);
            rowsOfNode[nodeDof.node][static_cast<std::size_t>(nodeDof.component)] -= forces(i, 0);
        }
        for (const auto& [node, values] : rowsOfNode)
            unitRows.push_back({node, static_cast<std::int32_t>(unit.unitId), values});
    }
    std::stable_sort(unitRows.begin() + static_cast<std::ptrdiff_t>(firstRow), unitRows.end(),
                     [](const output::StiffnessRecovery::UnitRow& a, const output::StiffnessRecovery::UnitRow& b)
                     { return a.node < b.node; });
}

} // namespace oos::solutions
