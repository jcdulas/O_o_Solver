#include "oos/aero/geometry/AeroReference.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/base/errors/UnsupportedCardError.hpp"
#include "oos/input/contract/TableRole.hpp"

#include <cmath>
#include <optional>
#include <string>

namespace oos::aero
{

namespace
{

// The value of a real column, or 0 when it is null (an AeroStaticParams gives REFC, REFB and REFS; the format requires
// them).
double realOrZero(const base::ContractReader& table, const char* column, std::int64_t row)
{
    if (!table.validity(column).isValid(row))
        return 0.0;
    return table.values<double>(column, base::NullPolicy::Allow)[static_cast<std::size_t>(row)];
}

std::int32_t integerOrZero(const base::ContractReader& table, const char* column, std::int64_t row)
{
    if (!table.validity(column).isValid(row))
        return 0;
    return table.values<std::int32_t>(column, base::NullPolicy::Allow)[static_cast<std::size_t>(row)];
}

} // namespace

input::CoordinateFrame frameOf(const input::ModelSnapshot& snapshot, std::int32_t frameId, const char* card)
{
    if (frameId == 0)
        return input::CoordinateFrame{};
    if (snapshot.hasTable(input::TableRole::CoordinateSystems))
    {
        const input::CoordinateSystemTable& systems = snapshot.coordinateSystems();
        const std::span<const std::int32_t> ids = systems.ids();
        for (std::size_t row = 0; row < ids.size(); ++row)
            if (ids[row] == frameId)
                return systems.frameAt(static_cast<std::int64_t>(row));
    }
    throw base::SolverError(
        base::concatenate({card, ": expected a coordinate system ", std::to_string(frameId),
                           ", found none in the model; add its coordinate system card or correct the ID"}));
}

AeroReference AeroReference::read(const input::ModelSnapshot& snapshot)
{
    if (!snapshot.hasTable(input::TableRole::AeroReference))
        throw base::SolverError(
            "a static aeroelastic analysis needs an AeroStaticParams card, and the model has none "
            "(or the snapshot was taken without aeroReferenceTable); add the AeroStaticParams card");
    const base::ContractReader& table = snapshot.aeroTable(input::TableRole::AeroReference);
    const base::DictionaryView cards = table.dictionaryTexts("card");
    std::optional<std::int64_t> aeros;
    for (std::int64_t row = 0; row < table.rowCount(); ++row)
    {
        if (cards[row] != "AeroStaticParams")
            continue;
        if (aeros)
            throw base::SolverError("expected one AeroStaticParams card, found several; keep one");
        aeros = row;
    }
    if (!aeros)
        throw base::SolverError(
            "a static aeroelastic analysis needs an AeroStaticParams card, and the model has only "
            "AeroUnsteadyParams; add the AeroStaticParams card with its reference chord, span and area");
    AeroReference reference;
    reference.aeroFrameId = integerOrZero(table, "aeroFrameId", *aeros);
    reference.rigidBodyFrameId = integerOrZero(table, "rigidBodyFrameId", *aeros);
    reference.chord = realOrZero(table, "refChord", *aeros);
    reference.span = realOrZero(table, "refSpan", *aeros);
    reference.area = realOrZero(table, "refArea", *aeros);
    reference.symmetryXz = integerOrZero(table, "symXz", *aeros);
    reference.symmetryXy = integerOrZero(table, "symXy", *aeros);
    if (!(reference.chord > 0.0) || !(reference.span > 0.0) || !(reference.area > 0.0))
        throw base::SolverError(base::concatenate(
            {"AeroStaticParams: expected a positive AeroStaticParams.referenceChord, "
             "AeroStaticParams.referenceSpan and AeroStaticParams.referenceArea, found ",
             std::to_string(reference.chord), ", ", std::to_string(reference.span), ", ",
             std::to_string(reference.area), "; give the reference chord, span and area of the configuration"}));
    reference.aeroFrame = frameOf(snapshot, reference.aeroFrameId, "AeroStaticParams AeroStaticParams.aeroFrameId");
    reference.referenceFrame =
        frameOf(snapshot, reference.rigidBodyFrameId, "AeroStaticParams AeroStaticParams.referenceFrameId");
    for (const input::CoordinateFrame* frame : {&reference.aeroFrame, &reference.referenceFrame})
        if (frame->kind != input::CoordinateFrame::Kind::Rectangular)
            throw base::UnsupportedCardError(base::concatenate(
                {"AeroStaticParams: expected rectangular systems for AeroStaticParams.aeroFrameId and "
                 "AeroStaticParams.referenceFrameId, found a cylindrical or spherical one "
                 "(AeroStaticParams.aeroFrameId ",
                 std::to_string(reference.aeroFrameId), ", AeroStaticParams.referenceFrameId ",
                 std::to_string(reference.rigidBodyFrameId), "); use a RectFrame2"}));
    return reference;
}

base::Vec3 AeroReference::pointInAeroFrame(base::Vec3 basic) const
{
    return aeroFrame.axes.transposed() * (basic - aeroFrame.origin);
}

base::Vec3 AeroReference::directionInAeroFrame(base::Vec3 basic) const
{
    return aeroFrame.axes.transposed() * basic;
}

base::Vec3 AeroReference::directionInReferenceFrame(base::Vec3 aero) const
{
    return referenceFrame.axes.transposed() * (aeroFrame.axes * aero);
}

base::Vec3 AeroReference::referenceOriginInAeroFrame() const
{
    return pointInAeroFrame(referenceFrame.origin);
}

} // namespace oos::aero
