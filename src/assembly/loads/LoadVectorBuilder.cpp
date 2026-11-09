#include "oos/assembly/loads/LoadVectorBuilder.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/elements/core/ElementVectorBlock.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace oos::assembly
{

namespace
{

constexpr std::array<std::string_view, 6> NODAL_CARDS{"PointForceLoad",  "PointForceLoad1",  "PointForceLoad2",
                                                      "PointMomentLoad", "PointMomentLoad1", "PointMomentLoad2"};
// The pressures of loadsTable are applied from pressureTable, which gives them per element with all corners (PLOAD2,
// FacePressureLoad), or per face of grid points (PLOAD: elementId null, face_nodes G1-G3 or G1-G4).
constexpr std::array<std::string_view, 3> PRESSURE_CARDS{"PointsPressureLoad", "ShellPressureLoad", "FacePressureLoad"};
constexpr std::string_view GRID_PRESSURE_CARD = "PointsPressureLoad";
// PointsPressureLoad: each of the two sets of overlapping triangles of a quadrilateral takes half of the pressure.
constexpr double QUADRILATERAL_TRIANGLE_SHARE = 0.5;
// PointsPressureLoad: the load of a triangle is divided into three equal parts at its grid points.
constexpr double TRIANGLE_POINT_SHARE = 1.0 / 3.0;
// A face whose vector area is below this fraction of the square of its longest side is degenerate (collinear or
// coincident points: no direction for the pressure). Far above round-off (1e-16), far below any real face.
constexpr double DEGENERATE_FACE_RATIO = 1.0e-12;
// The line loads of loadsTable are applied from lineLoadsTable.
constexpr std::array<std::string_view, 1> LINE_LOAD_CARDS{"BeamDistributedLoad"};
// The inertial loads, applied by InertialLoads (GRAV from loadsTable, all four from inertialLoadsTable when the
// snapshot has it).
constexpr std::array<std::string_view, 4> INERTIAL_CARDS{"GravityLoad", "AccelFieldLoad", "AccelNodeLoad",
                                                         "RotationalForceLoad"};

bool isOneOf(std::string_view card, std::span<const std::string_view> cards)
{
    return std::find(cards.begin(), cards.end(), card) != cards.end();
}

base::Diagnostic loadProblem(std::string code, std::string_view card, std::int32_t setId, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.code = std::move(code);
    diagnostic.card = std::string(card);
    diagnostic.id = setId;
    diagnostic.message = std::move(message);
    return diagnostic;
}

// The rotation of a vector given in the basic system into the displacement system of a grid point: v_cd = Rᵀ·v_basic.
std::array<double, 3> intoCd(const base::Mat3& cdToBasic, double x, double y, double z)
{
    const std::array<double, 3> basic{x, y, z};
    std::array<double, 3> cd{};
    for (std::size_t k = 0; k < 3; ++k)
        for (std::size_t i = 0; i < 3; ++i)
            cd[k] += cdToBasic(i, k) * basic[i];
    return cd;
}

// Adds the equivalent nodal loads of a facet's block to the g-set vector.
void scatter(const DofMap& dofMap, const elements::ElementVectorBlock& block, std::span<double> loadsG)
{
    for (std::size_t i = 0; i < block.size(); ++i)
    {
        const std::span<const elements::ElementDof> dofs = block.dofs(i);
        const std::span<const double> values = block.values(i);
        for (std::size_t k = 0; k < dofs.size(); ++k)
            loadsG[static_cast<std::size_t>(dofMap.dofOf(dofs[k]))] += values[k];
    }
}

// Runs a load facet's check into a sink of its own: true when it accepted every load (its diagnostics are passed on).
template <typename Check>
bool acceptsLoads(Check&& check, base::DiagnosticSink& sink)
{
    base::DiagnosticSink local;
    check(local);
    for (base::Diagnostic& diagnostic : local.diagnostics())
        sink.report(std::move(diagnostic));
    return !local.hasErrors();
}

} // namespace

LoadVectorBuilder::LoadVectorBuilder(const ElementSet& set, const DofMap& dofMap, base::MemoryBudget& budget)
    : m_set(&set), m_dofMap(&dofMap)
{
    std::size_t count = 0;
    for (const AssemblyBatch& batch : set.batches())
        count += static_cast<std::size_t>(batch.view.rowCount());
    m_elements = base::BudgetedBuffer<ElementLocation>(budget, count, "element locations of the loads");
    std::size_t k = 0;
    for (std::size_t b = 0; b < set.batches().size(); ++b)
    {
        const elements::ElementBatchView& view = set.batches()[b].view;
        for (std::int64_t row = 0; row < view.rowCount(); ++row)
            m_elements[k++] = {view.elementId(row), static_cast<std::int32_t>(b), row};
    }
    std::sort(m_elements.data(), m_elements.data() + count,
              [](const ElementLocation& a, const ElementLocation& b) { return a.elementId < b.elementId; });
    const input::ModelSnapshot& snapshot = set.index().snapshot();
    if (snapshot.hasTable(input::TableRole::Loads))
    {
        m_loadRows = LoadSetIndex(snapshot.loads().loadSetIds(), budget);
        m_pressures = PressureRows(snapshot.loads(), budget);
        m_lineLoadRows = LoadSetIndex(snapshot.loads().lineLoads().values<std::int32_t>("setId"), budget);
        m_loadCards = snapshot.loads().loadCards();
        m_lineCards = snapshot.loads().lineLoads().dictionaryTexts("card");
        m_lineLoadTypes = snapshot.loads().lineLoads().dictionaryTexts("loadType");
        m_lineScaleTypes = snapshot.loads().lineLoads().dictionaryTexts("scaleType");
        const auto isConcentrated = [&](std::span<const std::int64_t> rows)
        {
            return std::any_of(rows.begin(), rows.end(),
                               [&](std::int64_t row)
                               {
                                   const std::string_view card = m_loadCards[row];
                                   return !isOneOf(card, PRESSURE_CARDS) && !isOneOf(card, LINE_LOAD_CARDS) &&
                                          !isOneOf(card, INERTIAL_CARDS);
                               });
        };
        std::size_t count = 0;
        m_loadRows.forEachSet([&](std::int32_t, std::span<const std::int64_t> rows)
                              { count += isConcentrated(rows) ? 1 : 0; });
        m_concentratedSets = base::BudgetedBuffer<std::int32_t>(budget, count, "load sets with concentrated loads");
        std::size_t next = 0;
        m_loadRows.forEachSet(
            [&](std::int32_t setId, std::span<const std::int64_t> rows)
            {
                if (isConcentrated(rows))
                    m_concentratedSets[next++] = setId;
            });
    }
    m_inertial.emplace(snapshot, dofMap, m_loadRows, m_loadCards, budget);
}

std::optional<LoadVectorBuilder::ElementLocation> LoadVectorBuilder::findElement(std::int32_t elementId) const
{
    const ElementLocation* begin = m_elements.data();
    const ElementLocation* end = begin + m_elements.size();
    const ElementLocation* found = std::lower_bound(
        begin, end, elementId, [](const ElementLocation& a, std::int32_t id) { return a.elementId < id; });
    if (found == end || found->elementId != elementId)
        return std::nullopt;
    return *found;
}

std::vector<input::LoadTerm> LoadVectorBuilder::termsOf(base::EntityId loadSetId) const
{
    return m_set->index().snapshot().loadTermsOf(loadSetId);
}

template <typename Visit>
void LoadVectorBuilder::forEachConcentratedValue(base::EntityId loadSetId, base::DiagnosticSink& sink,
                                                 Visit&& visit) const
{
    const input::ModelSnapshot& snapshot = m_set->index().snapshot();
    const input::LoadTable& loads = snapshot.loads();
    const base::ContractReader& table = loads.loads();
    const base::DictionaryView& cards = m_loadCards;
    const std::span<const std::int32_t> targets = table.values<std::int32_t>("targetId", base::NullPolicy::Allow);
    const base::ValidityView validTargets = table.validity("targetId");
    std::array<std::span<const double>, 6> components;
    const std::array<const char*, 6> names{"fx", "fy", "fz", "mx", "my", "mz"};
    for (std::size_t c = 0; c < 6; ++c)
        components[c] = table.values<double>(names[c], base::NullPolicy::Allow);
    const bool hasInertialTable = snapshot.hasTable(input::TableRole::InertialLoads);

    // The rows of each set the LOAD brings, their values times its factor (LoadTable::termsOf: the set
    // alone with the factor 1 when the tables expand the combinations); a problem is named by the set of its card.
    for (const input::LoadTerm& term : termsOf(loadSetId))
    {
        const base::EntityId setId = term.set;
        for (const std::int64_t row : m_loadRows.rowsOf(setId))
        {
            const auto index = static_cast<std::size_t>(row);
            const std::string_view card = cards[row];
            if (isOneOf(card, PRESSURE_CARDS) || isOneOf(card, LINE_LOAD_CARDS))
                continue;
            if (isOneOf(card, INERTIAL_CARDS))
            {
                // GravityLoad is applied from loadsTable; the others need the full rows of inertialLoadsTable, which
                // also holds the GravityLoad rows when the snapshot has it.
                if (card != "GravityLoad" && !hasInertialTable)
                    sink.report(loadProblem(
                        "load-unsupported", card, setId,
                        base::concatenate({"the load card ", card, " of load set ", std::to_string(setId),
                                           " needs its fields, which loadsTable does not give (they are in "
                                           "inertialLoadsTable); install a "
                                           "O_o_Reader that delivers it, or hand the table over to the snapshot"})));
                continue;
            }
            const bool isNodal = isOneOf(card, NODAL_CARDS);
            if (!isNodal && card != "ScalarPointLoad")
            {
                sink.report(loadProblem(
                    "load-unsupported", card, setId,
                    base::concatenate(
                        {"the load card ", card, " of load set ", std::to_string(setId),
                         " is not supported by this version of O_o_Solver (PointForceLoad, PointForceLoad1, "
                         "PointForceLoad2, "
                         "PointMomentLoad, "
                         "PointMomentLoad1, PointMomentLoad2, ScalarPointLoad, GravityLoad, AccelFieldLoad, "
                         "AccelNodeLoad, RotationalForceLoad, BeamDistributedLoad, ShellPressureLoad, FacePressureLoad "
                         "are); remove it"})));
                continue;
            }
            if (!validTargets.isValid(row))
                continue;
            const std::int32_t pointId = targets[index];
            const std::optional<input::PointRef> point = m_dofMap->points().find(pointId);
            if (!point)
            {
                sink.report(
                    loadProblem("load-unknown-point", card, setId,
                                base::concatenate({"expected a grid or scalar point, found ID ",
                                                   std::to_string(pointId), " which the model does not have"})));
                continue;
            }
            const bool isScalar = point->kind == input::PointKind::Scalar;
            if (isScalar == isNodal)
            {
                sink.report(loadProblem(
                    "load-bad-point", card, setId,
                    base::concatenate({card, " on point ", std::to_string(pointId), ": expected a ",
                                       isNodal ? "grid point" : "scalar point", ", found a ",
                                       isScalar ? "scalar point" : "grid point", "; correct the point ID"})));
                continue;
            }
            if (isScalar)
            {
                if (std::isfinite(components[0][index]))
                    visit(m_dofMap->scalarDof(point->row), components[0][index] * term.factor, row, term);
                continue;
            }
            const base::Mat3 rotation = m_dofMap->cdToBasic(point->row);
            // A null component (the moment of a FORCE row) is no load. The factor multiplies the basic components
            // before their rotation, as the expanded rows hold them scaled (by F·S then N: a rounding apart).
            const auto valueOf = [&](std::size_t c)
            { return std::isfinite(components[c][index]) ? components[c][index] * term.factor : 0.0; };
            const std::array<double, 3> force = intoCd(rotation, valueOf(0), valueOf(1), valueOf(2));
            const std::array<double, 3> moment = intoCd(rotation, valueOf(3), valueOf(4), valueOf(5));
            for (int k = 0; k < 3; ++k)
            {
                visit(m_dofMap->gridDof(point->row, k + 1), force[static_cast<std::size_t>(k)], row, term);
                visit(m_dofMap->gridDof(point->row, k + 4), moment[static_cast<std::size_t>(k)], row, term);
            }
        }
    }
}

void LoadVectorBuilder::addConcentratedLoads(base::EntityId loadSetId, std::span<double> loadsG,
                                             base::DiagnosticSink& sink) const
{
    forEachConcentratedValue(loadSetId, sink,
                             [&](base::DofIndex dof, double value, std::int64_t, const input::LoadTerm&)
                             { loadsG[static_cast<std::size_t>(dof)] += value; });
}

std::optional<LoadFamily> LoadVectorBuilder::familyOfCard(std::string_view card)
{
    if (isOneOf(card, PRESSURE_CARDS) || isOneOf(card, LINE_LOAD_CARDS))
        return LoadFamily::Distributed;
    if (isOneOf(card, INERTIAL_CARDS))
        return LoadFamily::Inertial;
    if (isOneOf(card, NODAL_CARDS) || card == "ScalarPointLoad")
        return LoadFamily::Concentrated;
    return std::nullopt;
}

bool LoadVectorBuilder::needsMass(base::EntityId loadSetId) const
{
    return m_inertial->hasLoads(loadSetId);
}

bool LoadVectorBuilder::hasFamilyLoads(base::EntityId loadSetId, LoadFamily family) const
{
    const input::ModelSnapshot& snapshot = m_set->index().snapshot();
    if (!snapshot.hasTable(input::TableRole::Loads))
        return false;
    if (family == LoadFamily::Inertial)
        return m_inertial->hasLoads(loadSetId);
    const std::vector<input::LoadTerm> terms = termsOf(loadSetId);
    if (family == LoadFamily::Concentrated)
    {
        const std::span<const std::int32_t> sets = m_concentratedSets.span();
        return std::any_of(terms.begin(), terms.end(), [&](const input::LoadTerm& term)
                           { return std::binary_search(sets.begin(), sets.end(), term.set); });
    }
    if (family == LoadFamily::Distributed)
    {
        return std::any_of(terms.begin(), terms.end(), [&](const input::LoadTerm& term)
                           { return m_pressures.hasRows(term.set) || !m_lineLoadRows.rowsOf(term.set).empty(); });
    }
    return false;
}

bool LoadVectorBuilder::holdsOwnCards(base::EntityId setId) const
{
    return !m_loadRows.rowsOf(setId).empty() || m_pressures.hasRows(setId) || !m_lineLoadRows.rowsOf(setId).empty() ||
           m_inertial->holdsOwnRows(setId);
}

bool LoadVectorBuilder::hasLineLoads(base::EntityId loadSetId) const
{
    const std::vector<input::LoadTerm> terms = termsOf(loadSetId);
    return std::any_of(terms.begin(), terms.end(),
                       [&](const input::LoadTerm& term) { return !m_lineLoadRows.rowsOf(term.set).empty(); });
}

void LoadVectorBuilder::addLoads(base::EntityId loadSetId, const linalg::BlockSparseMatrix* mass,
                                 std::span<double> loadsG, base::MemoryBudget& budget, base::DiagnosticSink& sink) const
{
    if (loadsG.size() != static_cast<std::size_t>(m_dofMap->gSetSize()))
        throw std::invalid_argument(
            base::concatenate({"LoadVectorBuilder: expected a vector of ", std::to_string(m_dofMap->gSetSize()),
                               " values (the g set), found ", std::to_string(loadsG.size())}));
    if (!m_set->index().snapshot().hasTable(input::TableRole::Loads))
        return;
    addConcentratedLoads(loadSetId, loadsG, sink);
    addInertialLoads(loadSetId, mass, loadsG, budget, sink);
    addPressures(loadSetId, loadsG, sink);
    addLineLoads(loadSetId, loadsG, sink);
}

void LoadVectorBuilder::addFamilyLoads(base::EntityId loadSetId, LoadFamily family,
                                       const linalg::BlockSparseMatrix* mass, std::span<double> loadsG,
                                       base::MemoryBudget& budget, base::DiagnosticSink& sink) const
{
    if (loadsG.size() != static_cast<std::size_t>(m_dofMap->gSetSize()))
        throw std::invalid_argument(
            base::concatenate({"LoadVectorBuilder: expected a vector of ", std::to_string(m_dofMap->gSetSize()),
                               " values (the g set), found ", std::to_string(loadsG.size())}));
    if (family == LoadFamily::Thermal || family == LoadFamily::Enforced)
        throw std::invalid_argument(base::concatenate(
            {"LoadVectorBuilder::addFamilyLoads: expected the concentrated, distributed or inertial family, found ",
             nameOf(family), " (thermal loads come from addThermalLoads, enforced displacements from SpcSet)"}));
    if (!m_set->index().snapshot().hasTable(input::TableRole::Loads))
        return;
    if (family == LoadFamily::Concentrated)
    {
        addConcentratedLoads(loadSetId, loadsG, sink);
        return;
    }
    if (family == LoadFamily::Distributed)
    {
        addPressures(loadSetId, loadsG, sink);
        addLineLoads(loadSetId, loadsG, sink);
        return;
    }
    addInertialLoads(loadSetId, mass, loadsG, budget, sink);
}

void LoadVectorBuilder::addInertialAccelerations(base::EntityId loadSetId, std::span<double> accelerationsG,
                                                 base::MemoryBudget& budget, base::DiagnosticSink& sink) const
{
    m_inertial->addAccelerations(loadSetId, accelerationsG, budget, sink);
}

std::vector<InertialTerm> LoadVectorBuilder::inertialTermsOf(base::EntityId loadSetId, base::DiagnosticSink& sink) const
{
    return m_inertial->termsOf(loadSetId, sink);
}

void LoadVectorBuilder::addInertialLoads(base::EntityId loadSetId, const linalg::BlockSparseMatrix* mass,
                                         std::span<double> loadsG, base::MemoryBudget& budget,
                                         base::DiagnosticSink& sink) const
{
    if (!m_inertial->hasLoads(loadSetId))
        return;
    if (mass == nullptr)
    {
        sink.report(loadProblem("load-needs-mass", "GravityLoad", loadSetId,
                                "an inertial load (GravityLoad, AccelFieldLoad, AccelNodeLoad, RotationalForceLoad) "
                                "needs the mass matrix, which the "
                                "caller did not assemble; assemble Mgg for the load sets that hold one "
                                "(LoadVectorBuilder::needsMass)"));
        return;
    }
    m_inertial->addLoads(loadSetId, *mass, loadsG, budget, sink);
}

base::BudgetedBuffer<LoadVectorBuilder::DofValue> LoadVectorBuilder::concentratedEntries(
    base::EntityId loadSetId, base::MemoryBudget& budget, const FieldPrecision* precision) const
{
    const input::ModelSnapshot& snapshot = m_set->index().snapshot();
    if (!snapshot.hasTable(input::TableRole::Loads))
        return base::BudgetedBuffer<DofValue>(budget, 0, "concentrated load entries");
    base::DiagnosticSink ignored; // addLoads() reports the problems of the same rows
    std::size_t count = 0;
    forEachConcentratedValue(loadSetId, ignored,
                             [&](base::DofIndex, double, std::int64_t, const input::LoadTerm&) { ++count; });
    base::BudgetedBuffer<DofValue> entries(budget, count, "concentrated load entries");
    const base::ContractReader& table = snapshot.loads().loads();
    const base::DictionaryView& cards = m_loadCards;
    const std::span<const std::int32_t> sources = table.values<std::int32_t>("sourceSetId", base::NullPolicy::Allow);
    const std::span<const double> scales = snapshot.loads().loadScales();
    std::array<std::span<const double>, 6> components;
    const std::array<const char*, 6> names{"fx", "fy", "fz", "mx", "my", "mz"};
    for (std::size_t c = 0; c < 6; ++c)
        components[c] = table.values<double>(names[c], base::NullPolicy::Allow);
    // The precision of a row's values: the magnitude of its force (or moment) as written, rounded in its field, twice
    // (the magnitude F and the direction N of a FORCE are written separately and both rounded), times the factors: the
    // row's scale (the expanded tables) and its term's (the combination's factor of the set).
    const auto precisionOf = [&](std::int64_t row, base::DofIndex dof, const input::LoadTerm& term) -> double
    {
        if (precision == nullptr)
            return 0.0;
        const auto index = static_cast<std::size_t>(row);
        const bool isMoment = m_dofMap->labelOf(dof).component >= 4;
        const std::size_t first = isMoment ? 3 : 0;
        double magnitude = 0.0;
        for (std::size_t c = first; c < first + 3; ++c)
            if (std::isfinite(components[c][index]))
                magnitude += components[c][index] * components[c][index];
        magnitude = std::sqrt(magnitude);
        const double scale = std::isfinite(scales[index]) && scales[index] != 0.0 ? std::abs(scales[index]) : 1.0;
        std::optional<double> half = precision->halfUnitOf(term.set, cards[row], magnitude / scale);
        if (!half)
            half = precision->halfUnitOf(sources[index], cards[row], magnitude / scale);
        if (!half)
            return std::numeric_limits<double>::quiet_NaN();
        return 2.0 * (scale * std::abs(term.factor)) * *half;
    };
    std::size_t next = 0;
    forEachConcentratedValue(loadSetId, ignored,
                             [&](base::DofIndex dof, double value, std::int64_t row, const input::LoadTerm& term)
                             { entries[next++] = {dof, value, precisionOf(row, dof, term)}; });
    return entries;
}

template <typename Visit>
void LoadVectorBuilder::forEachGridPressureValue(base::EntityId loadSetId, base::DiagnosticSink& sink,
                                                 Visit&& visit) const
{
    for (const input::LoadTerm& term : termsOf(loadSetId))
    {
        const base::EntityId setId = term.set;
        m_pressures.forEachRowOf(
            setId,
            [&](std::int64_t, const PressureRow& pressureRow)
            {
                const std::string_view card = pressureRow.card;
                if (pressureRow.hasElement || card != GRID_PRESSURE_CARD)
                    return;
                const std::span<const std::int32_t> ids =
                    pressureRow.hasFace ? pressureRow.faceNodes : std::span<const std::int32_t>{};
                if (ids.size() != 3 && ids.size() != 4)
                {
                    sink.report(loadProblem(
                        "load-bad-face", card, setId,
                        base::concatenate({"expected the 3 or 4 grid points of a triangle or a "
                                           "quadrilateral (PointsPressureLoad.node1-PointsPressureLoad.node4, "
                                           "PointsPressureLoad.node4 blank or 0 for a triangle), found ",
                                           std::to_string(ids.size()), "; correct the PointsPressureLoad card"})));
                    return;
                }
                std::array<input::PointRef, 4> points{};
                std::array<base::Vec3, 4> xyz{};
                bool isKnown = true;
                for (std::size_t k = 0; k < ids.size() && isKnown; ++k)
                {
                    const std::optional<input::PointRef> point = m_dofMap->points().find(ids[k]);
                    if (!point)
                    {
                        sink.report(
                            loadProblem("load-unknown-point", card, setId,
                                        base::concatenate({"expected a grid point, found ID ", std::to_string(ids[k]),
                                                           " which the model does not have"})));
                        isKnown = false;
                    }
                    else if (point->kind == input::PointKind::Scalar)
                    {
                        sink.report(
                            loadProblem("load-bad-point", card, setId,
                                        base::concatenate({"PointsPressureLoad on point ", std::to_string(ids[k]),
                                                           ": expected a grid point, found a scalar point; correct "
                                                           "the point ID"})));
                        isKnown = false;
                    }
                    else
                    {
                        points[k] = *point;
                        xyz[k] = m_dofMap->positionInBasic(point->row);
                    }
                }
                if (!isKnown)
                    return;
                // PointsPressureLoad on a triangle G1 G2 G3: P·A·n, n by the right-hand rule of the sequence, its
                // vector area A·n = ½ (r2 − r1) × (r3 − r1), in three equal parts at its points. On a
                // quadrilateral is divided into two sets of overlapping triangles, each bounded by two sides and one
                // diagonal (G1 G2 G3 and G3 G4 G1; G2 G3 G4 and G4 G1 G2), each triangle taking half of the pressure as
                // above; the total is P times the vector area of the quadrilateral, ½ (r3 − r1) × (r4 − r2), whether it
                // is flat or warped.
                using Triangle = std::array<std::size_t, 3>;
                constexpr std::array<Triangle, 1> TRIANGLE{{{0, 1, 2}}};
                constexpr std::array<Triangle, 4> QUADRILATERAL{{{0, 1, 2}, {2, 3, 0}, {1, 2, 3}, {3, 0, 1}}};
                const std::span<const Triangle> triangles =
                    ids.size() == 3 ? std::span<const Triangle>(TRIANGLE) : std::span<const Triangle>(QUADRILATERAL);
                const double share = ids.size() == 3 ? 1.0 : QUADRILATERAL_TRIANGLE_SHARE;
                double longestSide = 0.0;
                for (std::size_t k = 0; k < ids.size(); ++k)
                    longestSide = std::max(longestSide, base::norm(xyz[(k + 1) % ids.size()] - xyz[k]));
                const base::Vec3 faceArea = ids.size() == 3 ? 0.5 * base::cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
                                                            : 0.5 * base::cross(xyz[2] - xyz[0], xyz[3] - xyz[1]);
                if (!(base::norm(faceArea) > DEGENERATE_FACE_RATIO * longestSide * longestSide))
                {
                    sink.report(loadProblem(
                        "load-bad-face", card, setId,
                        base::concatenate({"the grid points of the PointsPressureLoad (", std::to_string(ids[0]), ", ",
                                           std::to_string(ids[1]), ", ", std::to_string(ids[2]),
                                           ids.size() == 4 ? ", " : "", ids.size() == 4 ? std::to_string(ids[3]) : "",
                                           ") make a face of no area: no direction for the pressure; "
                                           "correct the grid points or their order"})));
                    return;
                }
                // The pressure times the combination's factor of its set (one product: the expanded table's value to
                // the bit).
                const double pressure =
                    (std::isfinite(pressureRow.corners[0]) ? pressureRow.corners[0] : 0.0) * term.factor;
                std::array<base::Vec3, 4> forces{};
                for (const Triangle& triangle : triangles)
                {
                    const base::Vec3 area =
                        0.5 * base::cross(xyz[triangle[1]] - xyz[triangle[0]], xyz[triangle[2]] - xyz[triangle[0]]);
                    const base::Vec3 pointLoad = (share * pressure * TRIANGLE_POINT_SHARE) * area;
                    for (const std::size_t corner : triangle)
                        forces[corner] = forces[corner] + pointLoad;
                }
                for (std::size_t k = 0; k < ids.size(); ++k)
                {
                    const std::array<double, 3> force =
                        intoCd(m_dofMap->cdToBasic(points[k].row), forces[k][0], forces[k][1], forces[k][2]);
                    for (int c = 0; c < 3; ++c)
                        visit(m_dofMap->gridDof(points[k].row, c + 1), force[static_cast<std::size_t>(c)]);
                }
            });
    }
}

void LoadVectorBuilder::addPressures(base::EntityId loadSetId, std::span<double> loadsG,
                                     base::DiagnosticSink& sink) const
{
    forEachGridPressureValue(loadSetId, sink,
                             [&](base::DofIndex dof, double value) { loadsG[static_cast<std::size_t>(dof)] += value; });
    const std::vector<std::vector<elements::ElementPressure>> byBatch = collectPressures(loadSetId, sink);
    elements::ElementVectorBlock block;
    for (std::size_t b = 0; b < byBatch.size(); ++b)
    {
        if (byBatch[b].empty())
            continue;
        const AssemblyBatch& batch = m_set->batches()[b];
        const std::span<const elements::ElementPressure> pressures = byBatch[b];
        if (!acceptsLoads([&](base::DiagnosticSink& local)
                          { batch.family->checkPressureLoads(batch.view, pressures, local); }, sink))
            continue;
        block.clear();
        batch.family->computePressureLoads(batch.view, pressures, block);
        scatter(*m_dofMap, block, loadsG);
    }
}

void LoadVectorBuilder::checkLoads(base::EntityId loadSetId, const linalg::BlockSparseMatrix* mass,
                                   base::MemoryBudget& budget, base::DiagnosticSink& sink, bool checksInertial) const
{
    if (!m_set->index().snapshot().hasTable(input::TableRole::Loads))
        return;
    forEachConcentratedValue(loadSetId, sink, [](base::DofIndex, double, std::int64_t, const input::LoadTerm&) {});
    forEachGridPressureValue(loadSetId, sink, [](base::DofIndex, double) {});
    const std::vector<std::vector<elements::ElementPressure>> pressures = collectPressures(loadSetId, sink);
    const std::vector<std::vector<elements::ElementLineLoad>> lineLoads = collectLineLoads(loadSetId, sink);
    for (std::size_t b = 0; b < m_set->batches().size(); ++b)
    {
        const AssemblyBatch& batch = m_set->batches()[b];
        if (!pressures[b].empty())
            batch.family->checkPressureLoads(batch.view, pressures[b], sink);
        if (!lineLoads[b].empty())
            batch.family->checkLineLoads(batch.view, lineLoads[b], sink);
    }
    if (checksInertial && m_inertial->hasLoads(loadSetId))
    {
        base::BudgetedBuffer<double> scratch(budget, static_cast<std::size_t>(m_dofMap->gSetSize()),
                                             "inertial load check");
        addInertialLoads(loadSetId, mass, scratch.span(), budget, sink);
    }
}

LoadVectorBuilder::ElementDistributedLoads LoadVectorBuilder::elementDistributedLoads(base::EntityId loadSetId,
                                                                                      base::DiagnosticSink& sink) const
{
    if (!m_set->index().snapshot().hasTable(input::TableRole::Loads))
        return {std::vector<std::vector<elements::ElementPressure>>(m_set->batches().size()),
                std::vector<std::vector<elements::ElementLineLoad>>(m_set->batches().size())};
    return {collectPressures(loadSetId, sink), collectLineLoads(loadSetId, sink)};
}

std::vector<std::vector<elements::ElementPressure>> LoadVectorBuilder::collectPressures(
    base::EntityId loadSetId, base::DiagnosticSink& sink) const
{
    // The pressures of each batch, in table order (bounded by the rows of the set).
    std::vector<std::vector<elements::ElementPressure>> byBatch(m_set->batches().size());
    for (const input::LoadTerm& term : termsOf(loadSetId))
    {
        const base::EntityId setId = term.set;
        m_pressures.forEachRowOf(
            setId,
            [&](std::int64_t, const PressureRow& row)
            {
                const std::string_view card = row.card;
                if (!row.hasElement)
                {
                    // PointsPressureLoad: a face of grid points, applied by forEachGridPressureValue.
                    if (card != GRID_PRESSURE_CARD)
                        sink.report(
                            loadProblem("load-unsupported", card, setId,
                                        base::concatenate({card,
                                                           " without an element is not supported by this version of "
                                                           "O_o_Solver; give the element IDs of the pressure"})));
                    return;
                }
                const std::optional<ElementLocation> location = findElement(row.elementId);
                if (!location)
                {
                    sink.report(
                        loadProblem("load-unknown-element", card, setId,
                                    base::concatenate({"expected an element of the model that O_o_Solver supports, "
                                                       "found element ",
                                                       std::to_string(row.elementId), "; correct the element ID"})));
                    return;
                }
                elements::ElementPressure pressure = elementPressureOf(row, *location);
                pressure.setId = setId;
                for (std::size_t c = 0; c < 4; ++c)
                    pressure.cornerPressures[c] *= term.factor;
                byBatch[static_cast<std::size_t>(location->batchIndex)].push_back(pressure);
            });
    }
    return byBatch;
}

elements::ElementPressure LoadVectorBuilder::elementPressureOf(const PressureRow& row,
                                                               const ElementLocation& location) const
{
    elements::ElementPressure pressure;
    pressure.elementRow = location.row;
    pressure.setId = row.setId;
    pressure.card = row.card;
    // FacePressureLoad: N1-N3 (in CID, the basic system when blank) give the direction of the pressure; all blank or
    // zero, the pressure is normal to the face.
    bool hasDirection = false;
    for (std::size_t c = 0; c < 3; ++c)
    {
        const bool isGiven = std::isfinite(row.direction[c]);
        pressure.direction[c] = isGiven ? row.direction[c] : 0.0;
        hasDirection = hasDirection || pressure.direction[c] != 0.0;
    }
    pressure.hasDirection = hasDirection;
    pressure.directionFrameId = row.hasFrameId ? row.frameId : 0;
    if (row.hasFace)
        pressure.faceNodes = row.faceNodes;
    for (std::size_t c = 0; c < 4; ++c)
        pressure.cornerPressures[c] = std::isfinite(row.corners[c]) ? row.corners[c] : 0.0;
    return pressure;
}

base::BudgetedBuffer<std::int64_t> LoadVectorBuilder::pressureRowsOf(base::EntityId setId,
                                                                     base::MemoryBudget& budget) const
{
    base::BudgetedBuffer<std::int64_t> rows(budget, static_cast<std::size_t>(m_pressures.rowCountOf(setId)),
                                            "pressure rows of a set");
    std::size_t next = 0;
    m_pressures.forEachRowOf(setId, [&](std::int64_t row, const PressureRow&) { rows[next++] = row; });
    return rows;
}

void LoadVectorBuilder::pressuresOfRows(std::span<const std::int64_t> rows, std::vector<ElementLocation>& locations,
                                        std::vector<elements::ElementPressure>& pressures) const
{
    locations.clear();
    pressures.clear();
    for (const std::int64_t index : rows)
    {
        const PressureRow row = m_pressures.rowAt(index);
        std::optional<ElementLocation> location;
        if (row.hasElement)
            location = findElement(row.elementId);
        locations.push_back(location.value_or(ElementLocation{}));
        elements::ElementPressure pressure = elementPressureOf(row, location.value_or(ElementLocation{}));
        pressure.elementRow = location ? location->row : 0;
        pressures.push_back(pressure);
    }
}

void LoadVectorBuilder::checkThermalLoads(const elements::ElementTemperatures& temperatures,
                                          base::DiagnosticSink& sink) const
{
    for (const AssemblyBatch& batch : m_set->batches())
        batch.family->checkThermalLoads(batch.view, temperatures, sink);
    // The elements of the DEFORM set: one-dimensional elements of the model (DEFORM applies to those only), never left
    // out without a word.
    for (const elements::ElementDeformation& deformation : temperatures.deformations)
    {
        const std::optional<ElementLocation> location = findElement(deformation.elementId);
        if (!location)
        {
            sink.report(loadProblem(
                "load-unknown-element", "AxialDeformationLoad", temperatures.deformationSet,
                base::concatenate(
                    {"expected a Rod2, Rod2InlineSection, Tube2, Bar2 or Beam2 of the model that O_o_Solver "
                     "supports (AxialDeformationLoad applies to one-dimensional elements), found element ",
                     std::to_string(deformation.elementId), " in AxialDeformationLoad set ",
                     std::to_string(temperatures.deformationSet), "; correct the element ID"})));
            continue;
        }
        const std::string_view card = m_set->batches()[static_cast<std::size_t>(location->batchIndex)].view.card();
        if (!elements::takesElementDeformations(card))
            sink.report(loadProblem(
                "load-unsupported", "AxialDeformationLoad", temperatures.deformationSet,
                base::concatenate(
                    {"expected a Rod2, Rod2InlineSection, Tube2, Bar2 or Beam2 in AxialDeformationLoad set ",
                     std::to_string(temperatures.deformationSet),
                     " (AxialDeformationLoad: an enforced "
                     "axial deformation of a one-dimensional element), found element ",
                     std::to_string(deformation.elementId), ", a ", std::string(card),
                     "; remove it from the AxialDeformationLoad card"})));
    }
}

void LoadVectorBuilder::addThermalLoads(const elements::ElementTemperatures& temperatures,
                                        std::span<double> loadsG) const
{
    if (loadsG.size() != static_cast<std::size_t>(m_dofMap->gSetSize()))
        throw std::invalid_argument(
            base::concatenate({"LoadVectorBuilder: expected a vector of ", std::to_string(m_dofMap->gSetSize()),
                               " values (the g set), found ", std::to_string(loadsG.size())}));
    elements::ElementVectorBlock block;
    for (const AssemblyBatch& batch : m_set->batches())
    {
        if (batch.family->isThermallyInert() || batch.view.rowCount() == 0)
            continue;
        // A DEFORM alone loads the one-dimensional elements only.
        if (!temperatures.hasTemperatureLoad && !elements::takesElementDeformations(batch.view.card()))
            continue;
        block.clear();
        batch.family->computeThermalLoads(batch.view, elements::wholeBatch(batch.view), temperatures, block);
        scatter(*m_dofMap, block, loadsG);
    }
}

std::vector<std::vector<elements::ElementLineLoad>> LoadVectorBuilder::lineLoadsOf(base::EntityId loadSetId) const
{
    base::DiagnosticSink ignored; // addLoads() reports the problems of the same loads
    std::vector<std::vector<elements::ElementLineLoad>> byBatch = collectLineLoads(loadSetId, ignored);
    for (std::vector<elements::ElementLineLoad>& loads : byBatch)
        std::stable_sort(loads.begin(), loads.end(),
                         [](const elements::ElementLineLoad& left, const elements::ElementLineLoad& right)
                         { return left.elementRow < right.elementRow; });
    return byBatch;
}

void LoadVectorBuilder::addLineLoads(base::EntityId loadSetId, std::span<double> loadsG,
                                     base::DiagnosticSink& sink) const
{
    // The loads in the order of the table rows: the order of the sums of Pg.
    const std::vector<std::vector<elements::ElementLineLoad>> byBatch = collectLineLoads(loadSetId, sink);
    elements::ElementVectorBlock block;
    for (std::size_t b = 0; b < byBatch.size(); ++b)
    {
        if (byBatch[b].empty())
            continue;
        const AssemblyBatch& batch = m_set->batches()[b];
        const std::span<const elements::ElementLineLoad> loads = byBatch[b];
        if (!acceptsLoads([&](base::DiagnosticSink& local) { batch.family->checkLineLoads(batch.view, loads, local); },
                          sink))
            continue;
        block.clear();
        batch.family->computeLineLoads(batch.view, loads, block);
        scatter(*m_dofMap, block, loadsG);
    }
}

std::vector<std::vector<elements::ElementLineLoad>> LoadVectorBuilder::collectLineLoads(
    base::EntityId loadSetId, base::DiagnosticSink& sink) const
{
    const base::ContractReader& table = m_set->index().snapshot().loads().lineLoads();
    const base::DictionaryView& cards = m_lineCards;
    const std::span<const std::int32_t> elementIds = table.values<std::int32_t>("elementId");
    const base::DictionaryView& loadTypes = m_lineLoadTypes;
    const base::DictionaryView& scaleTypes = m_lineScaleTypes;
    const std::span<const double> x1 = table.values<double>("x1", base::NullPolicy::Allow);
    const std::span<const double> p1 = table.values<double>("p1", base::NullPolicy::Allow);
    const std::span<const double> x2 = table.values<double>("x2", base::NullPolicy::Allow);
    const std::span<const double> p2 = table.values<double>("p2", base::NullPolicy::Allow);
    const std::span<const double> scales = table.values<double>("scale", base::NullPolicy::Allow);

    std::vector<std::vector<elements::ElementLineLoad>> byBatch(m_set->batches().size());
    for (const input::LoadTerm& term : termsOf(loadSetId))
    {
        const base::EntityId setId = term.set;
        for (const std::int64_t row : m_lineLoadRows.rowsOf(setId))
        {
            const auto index = static_cast<std::size_t>(row);
            const std::optional<ElementLocation> location = findElement(elementIds[index]);
            if (!location)
            {
                sink.report(
                    loadProblem("load-unknown-element", cards[row], setId,
                                base::concatenate({"expected a bar or beam of the model that O_o_Solver supports, "
                                                   "found element ",
                                                   std::to_string(elementIds[index]), "; correct the element ID"})));
                continue;
            }
            // The row's scale (the expanded tables' LOAD factors) times its term's (the combination's factor of its
            // set).
            const double scale = (std::isfinite(scales[index]) ? scales[index] : 1.0) * term.factor;
            elements::ElementLineLoad load;
            load.elementRow = location->row;
            load.setId = setId;
            load.card = cards[row];
            load.loadType = loadTypes[row];
            load.scaleType = scaleTypes[row];
            load.x1 = x1[index];
            load.p1 = p1[index] * scale;
            // BeamDistributedLoad: a blank X2 is a concentrated load at X1 (O_o_Reader gives X2 = X1 then).
            load.x2 = std::isfinite(x2[index]) ? x2[index] : x1[index];
            load.p2 = (std::isfinite(p2[index]) ? p2[index] : p1[index]) * scale;
            byBatch[static_cast<std::size_t>(location->batchIndex)].push_back(load);
        }
    }
    return byBatch;
}

base::BudgetedBuffer<LoadVectorBuilder::CardValue> LoadVectorBuilder::distributedValues(
    base::EntityId loadSetId, base::MemoryBudget& budget, const FieldPrecision* precision) const
{
    const input::ModelSnapshot& snapshot = m_set->index().snapshot();
    if (!snapshot.hasTable(input::TableRole::Loads))
        return base::BudgetedBuffer<CardValue>(budget, 0, "distributed load card values");
    const base::ContractReader& lines = snapshot.loads().lineLoads();
    // The precision of a value as written: half a unit of the last digit of the row's value over the row's scale (the
    // expanded tables' LOAD factors) in the field of its card, looked up under the set of the card, times the scale and
    // the term's factor (the combination's factor of the set).
    const auto precisionOf = [&](const CardValue& value, double rowValue, double rowScale,
                                 const input::LoadTerm& term) -> double
    {
        if (precision == nullptr)
            return 0.0;
        const double scale = rowScale != 0.0 ? std::abs(rowScale) : 1.0;
        const std::optional<double> half = precision->halfUnitOf(term.set, value.card, rowValue / scale);
        if (!half)
            return std::numeric_limits<double>::quiet_NaN();
        return (scale * std::abs(term.factor)) * *half;
    };
    const std::vector<input::LoadTerm> terms = termsOf(loadSetId);
    std::size_t count = 0;
    for (const input::LoadTerm& term : terms)
        count +=
            4 * static_cast<std::size_t>(m_pressures.rowCountOf(term.set)) + 2 * m_lineLoadRows.rowsOf(term.set).size();
    base::BudgetedBuffer<CardValue> values(budget, count, "distributed load card values");
    std::size_t next = 0;

    for (const input::LoadTerm& term : terms)
        m_pressures.forEachRowOf(term.set,
                                 [&](std::int64_t, const PressureRow& row)
                                 {
                                     const double rowScale = std::isfinite(row.scale) ? row.scale : 1.0;
                                     if (!row.hasElement)
                                     {
                                         // A PointsPressureLoad: one value, its face of grid points as the identity
                                         // (forEachGridPressureValue reports a face it cannot apply; the value is kept,
                                         // its equivalent loads are then zero in every case alike).
                                         if (row.card != GRID_PRESSURE_CARD || !row.hasFace || row.faceNodes.size() > 4)
                                             return;
                                         CardValue value;
                                         value.card = row.card;
                                         std::copy(row.faceNodes.begin(), row.faceNodes.end(), value.faceGrids.begin());
                                         value.scale = rowScale * term.factor;
                                         const double rowValue = std::isfinite(row.corners[0]) ? row.corners[0] : 0.0;
                                         value.value = rowValue * term.factor;
                                         value.precision = precisionOf(value, rowValue, rowScale, term);
                                         values[next++] = value;
                                         return;
                                     }
                                     if (!findElement(row.elementId))
                                         return;
                                     CardValue value;
                                     value.elementId = row.elementId;
                                     value.card = row.card;
                                     value.frameId = row.hasFrameId ? row.frameId : 0;
                                     for (std::size_t c = 0; c < 3; ++c)
                                         value.direction[c] = std::isfinite(row.direction[c]) ? row.direction[c] : 0.0;
                                     value.scale = rowScale * term.factor;
                                     for (std::size_t c = 0; c < 4; ++c)
                                     {
                                         if (!std::isfinite(row.corners[c]))
                                             continue;
                                         value.slot = static_cast<std::int32_t>(c);
                                         value.value = row.corners[c] * term.factor;
                                         value.precision = precisionOf(value, row.corners[c], rowScale, term);
                                         values[next++] = value;
                                     }
                                 });

    const base::DictionaryView& lineCards = m_lineCards;
    const std::span<const std::int32_t> lineElementIds = lines.values<std::int32_t>("elementId");
    const base::DictionaryView& loadTypes = m_lineLoadTypes;
    const base::DictionaryView& scaleTypes = m_lineScaleTypes;
    const std::span<const double> x1 = lines.values<double>("x1", base::NullPolicy::Allow);
    const std::span<const double> p1 = lines.values<double>("p1", base::NullPolicy::Allow);
    const std::span<const double> x2 = lines.values<double>("x2", base::NullPolicy::Allow);
    const std::span<const double> p2 = lines.values<double>("p2", base::NullPolicy::Allow);
    const std::span<const double> lineScales = lines.values<double>("scale", base::NullPolicy::Allow);
    for (const input::LoadTerm& term : terms)
        for (const std::int64_t row : m_lineLoadRows.rowsOf(term.set))
        {
            const auto index = static_cast<std::size_t>(row);
            if (!findElement(lineElementIds[index]))
                continue;
            CardValue value;
            value.elementId = lineElementIds[index];
            value.card = lineCards[row];
            value.loadType = loadTypes[row];
            value.scaleType = scaleTypes[row];
            value.x1 = x1[index];
            // As collectLineLoads reads them: a blank X2 is X1, a blank P2 is P1 (the line loads facet sees the same),
            // P1 and P2 times the row's scale and the term's factor.
            value.x2 = std::isfinite(x2[index]) ? x2[index] : x1[index];
            const double rowScale = std::isfinite(lineScales[index]) ? lineScales[index] : 1.0;
            value.scale = rowScale * term.factor;
            const double secondValue = std::isfinite(p2[index]) ? p2[index] : p1[index];
            value.slot = 0;
            value.value = p1[index] * value.scale;
            value.precision = precisionOf(value, p1[index] * rowScale, rowScale, term);
            values[next++] = value;
            value.slot = 1;
            value.value = secondValue * value.scale;
            value.precision = precisionOf(value, secondValue * rowScale, rowScale, term);
            values[next++] = value;
        }
    // The rows on unknown elements and the blank corners were counted as slots: keep the values written.
    base::BudgetedBuffer<CardValue> written(budget, next, "distributed load card values");
    std::copy(values.data(), values.data() + next, written.data());
    return written;
}

} // namespace oos::assembly
