#include "oos/assembly/loads/InertialLoads.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"

#include <array>
#include <cmath>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace oos::assembly
{

namespace
{

// RotationalForceLoad: A and RACC are in revolutions per unit time (squared): ω = 2π·A.
constexpr double RADIANS_PER_REVOLUTION = 2.0 * std::numbers::pi;
// RotationalForceLoad METHOD: 1 (the mass matrix times the acceleration field, computed) or 2 (refused).
constexpr std::int32_t ROTATIONAL_FORCE_MASS_MATRIX_METHOD = 1;
// A zero or null METHOD is the card's blank: METHOD defaults to 1.
constexpr std::int32_t ROTATIONAL_FORCE_DEFAULT_METHOD = 0;

base::Diagnostic inertialProblem(std::string code, std::string_view card, base::EntityId setId, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.code = std::move(code);
    diagnostic.card = std::string(card);
    diagnostic.id = setId;
    diagnostic.message = std::move(message);
    return diagnostic;
}

// The acceleration field of the rows of one card, one vector per grid point row in the basic system (translations)
// and an angular acceleration (rotations), written into the g-set vector `field` in the CD systems.
class FieldWriter
{
public:
    FieldWriter(const DofMap& dofMap, std::span<double> field) : m_dofMap(&dofMap), m_field(field) {}

    void setTranslation(std::int32_t grid, base::Vec3 basic) { write(grid, basic, 1); }
    void setRotation(std::int32_t grid, base::Vec3 basic) { write(grid, basic, 4); }

private:
    void write(std::int32_t grid, base::Vec3 basic, int firstComponent)
    {
        // v_cd = Rᵀ·v_basic, R the CD axes in basic (DofMap::cdToBasic).
        const base::Vec3 cd = m_dofMap->cdToBasic(grid).transposed() * basic;
        for (int k = 0; k < 3; ++k)
            m_field[static_cast<std::size_t>(m_dofMap->gridDof(grid, firstComponent + k))] +=
                cd[static_cast<std::size_t>(k)];
    }

    const DofMap* m_dofMap;
    std::span<double> m_field;
};

// loads += factor·Mgg·field.
void addMassTimes(const linalg::BlockSparseMatrix& mass, std::span<const double> field, double factor,
                  std::span<double> loadsG, base::MemoryBudget& budget)
{
    base::BudgetedBuffer<double> product(budget, loadsG.size(), "inertial load Mgg a");
    mass.multiply(field, product.span());
    for (std::size_t dof = 0; dof < loadsG.size(); ++dof)
        loadsG[dof] += factor * product[dof];
}

// The factor of an ACCEL table at a coordinate: linear between the points, the end values beyond them.
double interpolate(std::span<const double> locations, std::span<const double> factors, double coordinate)
{
    if (coordinate <= locations.front())
        return factors.front();
    for (std::size_t k = 1; k < locations.size(); ++k)
    {
        if (coordinate <= locations[k])
        {
            const double span = locations[k] - locations[k - 1];
            const double t = span > 0.0 ? (coordinate - locations[k - 1]) / span : 1.0;
            return factors[k - 1] + t * (factors[k] - factors[k - 1]);
        }
    }
    return factors.back();
}

} // namespace

InertialLoads::InertialLoads(const input::ModelSnapshot& snapshot, const DofMap& dofMap, const LoadSetIndex& loadRows,
                             base::DictionaryView loadCards, base::MemoryBudget& budget)
    : m_snapshot(&snapshot), m_dofMap(&dofMap), m_loadRows(&loadRows), m_loadCards(loadCards)
{
    if (snapshot.hasTable(input::TableRole::InertialLoads))
    {
        const base::ContractReader& table = snapshot.loads().inertialLoads();
        m_inertialRows = LoadSetIndex(table.values<std::int32_t>("setId"), budget);
        m_inertialCards = table.dictionaryTexts("card");
    }
}

bool InertialLoads::hasLoads(base::EntityId loadSetId) const
{
    const input::ModelSnapshot& snapshot = *m_snapshot;
    if (!snapshot.hasTable(input::TableRole::Loads))
        return false;
    const bool hasInertialTable = snapshot.hasTable(input::TableRole::InertialLoads);
    for (const input::LoadTerm& term : snapshot.loadTermsOf(loadSetId))
    {
        if (hasInertialTable)
        {
            if (!m_inertialRows.rowsOf(term.set).empty())
                return true;
            continue;
        }
        for (const std::int64_t row : m_loadRows->rowsOf(term.set))
            if (m_loadCards[row] == "GravityLoad")
                return true;
    }
    return false;
}

bool InertialLoads::holdsOwnRows(base::EntityId setId) const
{
    if (!m_snapshot->hasTable(input::TableRole::Loads))
        return false;
    if (m_snapshot->hasTable(input::TableRole::InertialLoads))
        return !m_inertialRows.rowsOf(setId).empty();
    for (const std::int64_t row : m_loadRows->rowsOf(setId))
        if (m_loadCards[row] == "GravityLoad")
            return true;
    return false;
}

void InertialLoads::addLoads(base::EntityId loadSetId, const linalg::BlockSparseMatrix& mass, std::span<double> loadsG,
                             base::MemoryBudget& budget, base::DiagnosticSink& sink) const
{
    if (loadsG.size() != static_cast<std::size_t>(m_dofMap->gSetSize()))
        throw std::invalid_argument(
            base::concatenate({"InertialLoads: expected a vector of ", std::to_string(m_dofMap->gSetSize()),
                               " values (the g set), found ", std::to_string(loadsG.size())}));
    if (!m_snapshot->hasTable(input::TableRole::Loads))
        return;
    const FieldApplier apply = [&](std::span<const double> field, double factor)
    { addMassTimes(mass, field, factor, loadsG, budget); };
    if (m_snapshot->hasTable(input::TableRole::InertialLoads))
        addFromInertialTable(loadSetId, apply, loadsG.size(), budget, sink);
    else
        addFromLoadTable(loadSetId, apply, loadsG.size(), budget);
}

void InertialLoads::addAccelerations(base::EntityId loadSetId, std::span<double> accelerationsG,
                                     base::MemoryBudget& budget, base::DiagnosticSink& sink) const
{
    if (accelerationsG.size() != static_cast<std::size_t>(m_dofMap->gSetSize()))
        throw std::invalid_argument(
            base::concatenate({"InertialLoads: expected a vector of ", std::to_string(m_dofMap->gSetSize()),
                               " values (the g set), found ", std::to_string(accelerationsG.size())}));
    if (!m_snapshot->hasTable(input::TableRole::Loads))
        return;
    const FieldApplier apply = [&](std::span<const double> field, double factor)
    {
        for (std::size_t dof = 0; dof < accelerationsG.size(); ++dof)
            accelerationsG[dof] += factor * field[dof];
    };
    if (m_snapshot->hasTable(input::TableRole::InertialLoads))
        addFromInertialTable(loadSetId, apply, accelerationsG.size(), budget, sink);
    else
        addFromLoadTable(loadSetId, apply, accelerationsG.size(), budget);
}

void InertialLoads::addFromLoadTable(base::EntityId loadSetId, const FieldApplier& apply, std::size_t gSize,
                                     base::MemoryBudget& budget) const
{
    for (const RowSpec& spec : gravityRowsOf(loadSetId))
    {
        base::BudgetedBuffer<double> field(budget, gSize, "inertial acceleration field");
        writeField(spec, field.span());
        apply(field.span(), spec.factor);
    }
    // RotationalForceLoad, ACCEL, AccelNodeLoad rows of loadsTable are refused by LoadVectorBuilder (they need
    // inertialLoadsTable).
}

std::vector<InertialLoads::RowSpec> InertialLoads::gravityRowsOf(base::EntityId loadSetId) const
{
    const base::ContractReader& table = m_snapshot->loads().loads();
    const std::span<const double> ax = table.values<double>("ax", base::NullPolicy::Allow);
    const std::span<const double> ay = table.values<double>("ay", base::NullPolicy::Allow);
    const std::span<const double> az = table.values<double>("az", base::NullPolicy::Allow);
    std::vector<RowSpec> specs; // bounded by the set's rows
    // The rows of each term, their accelerations times its factor (a rounding from the expanded rows,
    // which multiply the GravityLoad magnitude by the factor before its direction).
    for (const input::LoadTerm& term : m_snapshot->loadTermsOf(loadSetId))
        for (const std::int64_t row : m_loadRows->rowsOf(term.set))
        {
            const auto index = static_cast<std::size_t>(row);
            if (m_loadCards[row] != "GravityLoad")
                continue;
            // GravityLoad: Mgg·r, r the rigid motion translating every grid point by the acceleration (rotations and
            // scalar points 0).
            RowSpec spec;
            spec.kind = RowSpec::Kind::Gravity;
            spec.direction = {ax[index] * term.factor, ay[index] * term.factor, az[index] * term.factor};
            specs.push_back(std::move(spec));
        }
    return specs;
}

std::vector<InertialLoads::RowSpec> InertialLoads::inertialRowsOf(base::EntityId loadSetId,
                                                                  base::DiagnosticSink& sink) const
{
    // O_o_Reader inertialLoadsTable (contract 31): a column that does not apply to a card is null, so every column
    // is read with its nulls and each card reads only its own.
    constexpr base::NullPolicy ALLOW = base::NullPolicy::Allow;
    const base::ContractReader& table = m_snapshot->loads().inertialLoads();
    const base::DictionaryView& cards = m_inertialCards;
    const std::span<const std::int32_t> frameIds = table.values<std::int32_t>("frameId", ALLOW);
    const base::ValidityView hasFrameId = table.validity("frameId");
    const std::span<const double> scales = table.values<double>("scale", ALLOW);
    const base::ValidityView hasScale = table.validity("scale");
    const std::array<std::span<const double>, 3> directions{table.values<double>("directionX", ALLOW),
                                                            table.values<double>("directionY", ALLOW),
                                                            table.values<double>("directionZ", ALLOW)};
    const std::span<const double> magnitudes = table.values<double>("acceleration", ALLOW);
    const base::ValidityView hasMagnitude = table.validity("acceleration");
    const std::span<const std::int32_t> centreGrids = table.values<std::int32_t>("centerNodeId", ALLOW);
    const base::ValidityView hasCentreGrid = table.validity("centerNodeId");
    const std::array<std::span<const double>, 3> centres{table.values<double>("centerX", ALLOW),
                                                         table.values<double>("centerY", ALLOW),
                                                         table.values<double>("centerZ", ALLOW)};
    const base::ValidityView hasCentre = table.validity("centerX");
    const std::span<const std::int32_t> methods = table.values<std::int32_t>("method", ALLOW);
    const base::ValidityView hasMethod = table.validity("method");
    const base::ListView<std::int32_t> regions = table.lists<std::int32_t>("nodeIds", ALLOW);
    const base::ValidityView hasRegion = table.validity("nodeIds");
    const std::span<const double> velocities = table.values<double>("angularVelocity", ALLOW);
    const std::span<const double> angularAccelerations = table.values<double>("angularAcceleration", ALLOW);
    const base::DictionaryView axes = table.dictionaryTexts("locationDirection", ALLOW);
    const base::ValidityView hasAxis = table.validity("locationDirection");
    const base::ListView<double> locationLists = table.lists<double>("locations", ALLOW);
    const base::ListView<double> valueLists = table.lists<double>("values", ALLOW);
    const base::ValidityView hasLocations = table.validity("locations");
    const base::ValidityView hasValues = table.validity("values");

    std::vector<RowSpec> specs; // bounded by the set's rows
    for (const input::LoadTerm& term : m_snapshot->loadTermsOf(loadSetId))
    {
        const base::EntityId setId = term.set;
        for (const std::int64_t row : m_inertialRows.rowsOf(setId))
        {
            const auto index = static_cast<std::size_t>(row);
            const std::string_view card = cards[row];
            // The unit direction of the table: the acceleration of GravityLoad, ACCEL, AccelNodeLoad, the axis of
            // RotationalForceLoad.
            const base::Vec3 unit{directions[0][index], directions[1][index], directions[2][index]};
            // GravityLoad, ACCEL, AccelNodeLoad: the acceleration vector, the LOAD factors in its magnitude: the
            // expanded tables' (the row's), or the term's factor times the set's (a rounding from the expanded rows,
            // which multiply the card's magnitude by the factor before the direction).
            const double magnitude = (hasMagnitude.isValid(row) ? magnitudes[index] : 0.0) * term.factor;
            RowSpec spec;
            spec.direction = card == "RotationalForceLoad" ? unit : magnitude * unit;
            if (card == "GravityLoad")
                spec.kind = RowSpec::Kind::Gravity;
            else if (card == "AccelNodeLoad")
            {
                spec.kind = RowSpec::Kind::Region;
                bool isValid = true;
                const std::span<const std::int32_t> region =
                    hasRegion.isValid(row) ? regions[row] : std::span<const std::int32_t>();
                for (const std::int32_t id : region)
                {
                    const std::optional<std::int32_t> grid = m_dofMap->points().findGrid(id);
                    if (!grid)
                    {
                        sink.report(inertialProblem(
                            "load-unknown-point", card, setId,
                            base::concatenate({"expected a grid point of the model, found ID ", std::to_string(id),
                                               " in the grid list of AccelNodeLoad; correct the list"})));
                        isValid = false;
                        continue;
                    }
                    spec.grids.push_back(*grid);
                }
                if (!isValid)
                    continue;
            }
            else if (card == "AccelFieldLoad")
            {
                spec.kind = RowSpec::Kind::Varying;
                if (!hasAxis.isValid(row) || !hasLocations.isValid(row) || !hasValues.isValid(row))
                {
                    sink.report(inertialProblem("load-invalid", card, setId,
                                                "expected the AccelFieldLoad.direction, AccelFieldLoad.locationi and "
                                                "AccelFieldLoad.accelerationi of the AccelFieldLoad card in "
                                                "inertialLoadsTable, found none; correct the AccelFieldLoad card"));
                    continue;
                }
                const std::string_view axis = axes[row];
                spec.locations = locationLists[row];
                spec.factors = valueLists[row];
                spec.axisIndex = axis == "XDirection" ? 0 : axis == "YDirection" ? 1 : axis == "ZDirection" ? 2 : 3;
                if (spec.axisIndex > 2 || spec.locations.size() < 2 || spec.locations.size() != spec.factors.size())
                {
                    sink.report(inertialProblem(
                        "load-invalid", card, setId,
                        base::concatenate({"expected AccelFieldLoad.direction XDirection, YDirection or "
                                           "ZDirection and a table of at least two "
                                           "(AccelFieldLoad.location, AccelFieldLoad.acceleration) "
                                           "pairs, found AccelFieldLoad.direction '",
                                           axis, "' and ", std::to_string(spec.locations.size()), " locations for ",
                                           std::to_string(spec.factors.size()),
                                           " values; correct the AccelFieldLoad card"})));
                    continue;
                }
                const std::int32_t frameId = hasFrameId.isValid(row) ? frameIds[index] : 0;
                if (frameId != 0)
                {
                    std::optional<input::CoordinateFrame> found;
                    if (m_snapshot->hasTable(input::TableRole::CoordinateSystems))
                    {
                        const input::CoordinateSystemTable& systems = m_snapshot->coordinateSystems();
                        const std::span<const std::int32_t> ids = systems.ids();
                        for (std::size_t s = 0; s < ids.size(); ++s)
                            if (ids[s] == frameId)
                                found = systems.frameAt(static_cast<std::int64_t>(s));
                    }
                    if (!found || found->kind != input::CoordinateFrame::Kind::Rectangular)
                    {
                        sink.report(inertialProblem(
                            "load-unsupported", card, setId,
                            base::concatenate(
                                {"expected AccelFieldLoad.frameId 0 or a rectangular coordinate system of the "
                                 "model, found AccelFieldLoad.frameId ",
                                 std::to_string(frameId), found ? " (cylindrical or spherical)" : " (unknown)",
                                 "; give the AccelFieldLoad variation in a rectangular system"})));
                        continue;
                    }
                    spec.frame = *found;
                }
            }
            else if (card == "RotationalForceLoad")
            {
                spec.kind = RowSpec::Kind::Rotation;
                const std::int32_t method = hasMethod.isValid(row) ? methods[index] : ROTATIONAL_FORCE_DEFAULT_METHOD;
                if (method != ROTATIONAL_FORCE_MASS_MATRIX_METHOD && method != ROTATIONAL_FORCE_DEFAULT_METHOD)
                {
                    sink.report(inertialProblem(
                        "load-unsupported", card, setId,
                        base::concatenate(
                            {"RotationalForceLoad RotationalForceLoad.methodName = ", std::to_string(method),
                             " is not supported by this version of O_o_Solver "
                             "(RotationalForceLoad.methodName = 1 is: the mass matrix times the "
                             "acceleration field); write RotationalForceLoad.methodName = 1"})));
                    continue;
                }
                // The point of the axis: G when the card gives it (its position in the model), else the point
                // O_o_Reader gives (the origin of the basic system).
                if (hasCentre.isValid(row))
                    spec.centre = {centres[0][index], centres[1][index], centres[2][index]};
                if (hasCentreGrid.isValid(row) && centreGrids[index] != 0)
                {
                    const std::optional<std::int32_t> grid = m_dofMap->points().findGrid(centreGrids[index]);
                    if (!grid)
                    {
                        sink.report(inertialProblem(
                            "load-unknown-point", card, setId,
                            base::concatenate({"expected a grid point of the model as RotationalForceLoad.node, found ",
                                               std::to_string(centreGrids[index]),
                                               "; correct the RotationalForceLoad card"})));
                        continue;
                    }
                    spec.centre = m_dofMap->positionInBasic(*grid);
                }
                if (!(base::norm(spec.direction) > 0.0))
                {
                    sink.report(inertialProblem("load-invalid", card, setId,
                                                "expected a nonzero rotation axis "
                                                "RotationalForceLoad.radius1-RotationalForceLoad."
                                                "radius3, found a zero vector; correct the "
                                                "RotationalForceLoad card"));
                    continue;
                }
                spec.velocity = velocities[index];
                spec.angularAcceleration = angularAccelerations[index];
                // The force of the field on the mass, −M·a, times the LOAD factors (they scale the load, not A and
                // RACC).
                spec.factor = -((hasScale.isValid(row) ? scales[index] : 1.0) * term.factor);
            }
            else
            {
                sink.report(inertialProblem(
                    "load-unsupported", card, setId,
                    base::concatenate(
                        {"the inertial load card ", card,
                         " is not supported (GravityLoad, AccelFieldLoad, AccelNodeLoad, RotationalForceLoad are)"})));
                continue;
            }
            specs.push_back(std::move(spec));
        }
    }
    return specs;
}

void InertialLoads::writeField(const RowSpec& spec, std::span<double> field) const
{
    FieldWriter writer(*m_dofMap, field);
    switch (spec.kind)
    {
    case RowSpec::Kind::Gravity:
        for (std::int32_t grid = 0; grid < m_dofMap->gridCount(); ++grid)
            writer.setTranslation(grid, spec.direction);
        return;
    case RowSpec::Kind::Region:
        for (const std::int32_t grid : spec.grids)
            writer.setTranslation(grid, spec.direction);
        return;
    case RowSpec::Kind::Varying:
        for (std::int32_t grid = 0; grid < m_dofMap->gridCount(); ++grid)
        {
            // The coordinate of the grid point along DIR in CID: x_local = axesᵀ·(x − origin).
            const base::Vec3 local =
                spec.frame.axes.transposed() * (m_dofMap->positionInBasic(grid) - spec.frame.origin);
            writer.setTranslation(grid,
                                  interpolate(spec.locations, spec.factors, local[spec.axisIndex]) * spec.direction);
        }
        return;
    case RowSpec::Kind::Rotation:
    {
        const double length = base::norm(spec.direction);
        const base::Vec3 axis = spec.direction / length;
        const base::Vec3 omega = RADIANS_PER_REVOLUTION * spec.velocity * axis;
        const base::Vec3 alpha = RADIANS_PER_REVOLUTION * spec.angularAcceleration * axis;
        for (std::int32_t grid = 0; grid < m_dofMap->gridCount(); ++grid)
        {
            const base::Vec3 rho = m_dofMap->positionInBasic(grid) - spec.centre;
            writer.setTranslation(grid, base::cross(omega, base::cross(omega, rho)) + base::cross(alpha, rho));
            writer.setRotation(grid, alpha);
        }
        return;
    }
    }
}

void InertialLoads::addFromInertialTable(base::EntityId loadSetId, const FieldApplier& apply, std::size_t gSize,
                                         base::MemoryBudget& budget, base::DiagnosticSink& sink) const
{
    for (const RowSpec& spec : inertialRowsOf(loadSetId, sink))
    {
        base::BudgetedBuffer<double> field(budget, gSize, "inertial acceleration field");
        writeField(spec, field.span());
        apply(field.span(), spec.factor);
    }
}

std::vector<InertialTerm> InertialLoads::termsOf(base::EntityId loadSetId, base::DiagnosticSink& sink) const
{
    std::vector<InertialTerm> terms; // bounded by the set's rows
    if (!m_snapshot->hasTable(input::TableRole::Loads))
        return terms;
    const std::vector<RowSpec> specs = m_snapshot->hasTable(input::TableRole::InertialLoads)
                                           ? inertialRowsOf(loadSetId, sink)
                                           : gravityRowsOf(loadSetId);
    for (const RowSpec& spec : specs)
    {
        // The generator of the row's field and its coefficients (InertialTerm): the field is linear in the acceleration
        // vector (GRAV, AccelNodeLoad, ACCEL: three components) or in the row's factor (RFORCE: its own field).
        InertialTerm term;
        term.generator.push_back(static_cast<double>(spec.kind));
        switch (spec.kind)
        {
        case RowSpec::Kind::Gravity:
            break;
        case RowSpec::Kind::Region:
            for (const std::int32_t grid : spec.grids)
                term.generator.push_back(static_cast<double>(grid));
            break;
        case RowSpec::Kind::Varying:
            for (std::size_t k = 0; k < 3; ++k)
                term.generator.push_back(spec.frame.origin[k]);
            for (const double value : spec.frame.axes.values)
                term.generator.push_back(value);
            term.generator.push_back(static_cast<double>(spec.axisIndex));
            term.generator.insert(term.generator.end(), spec.locations.begin(), spec.locations.end());
            term.generator.insert(term.generator.end(), spec.factors.begin(), spec.factors.end());
            break;
        case RowSpec::Kind::Rotation:
            for (std::size_t k = 0; k < 3; ++k)
                term.generator.push_back(spec.centre[k]);
            for (std::size_t k = 0; k < 3; ++k)
                term.generator.push_back(spec.direction[k]);
            term.generator.push_back(spec.velocity);
            term.generator.push_back(spec.angularAcceleration);
            break;
        }
        if (spec.kind == RowSpec::Kind::Rotation)
            term.coefficients = {spec.factor};
        else
            term.coefficients = {spec.direction[0], spec.direction[1], spec.direction[2]};
        terms.push_back(std::move(term));
    }
    return terms;
}

} // namespace oos::assembly
