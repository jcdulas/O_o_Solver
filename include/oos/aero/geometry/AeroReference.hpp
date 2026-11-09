#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/input/tables/CoordinateSystemTable.hpp"

#include <cstdint>

namespace oos::aero
{

// The reference values and frames of a static aeroelastic model (the AeroStaticParams card, read from
// aeroReferenceTable):
//
//   | AeroStaticParams | ACSID | RCSID | REFC | REFB | REFS | SYMXZ | SYMXY |
//
// ACSID is the aerodynamic frame, whose x axis is the flow and z axis up: the boxes, the doublet lattice and the
// downwash are computed in it. RCSID is the reference frame of the rigid-body motions (RollRate, PitchRate, YawRate
// about its axes at its origin) and of the stability derivatives (forces along its axes, moments about its origin), the
// reference coordinates. REFC, REFB, REFS divide the forces and moments of the derivatives (CX = Fx / (q REFS), CMX =
// Mx / (q REFS REFB), CMY = My / (q REFS REFC), CMZ = Mz / (q REFS REFB)). SYMXZ and SYMXY (+1 symmetric, −1
// antisymmetric, 0 none; the Case Control AESYMXZ and AESYMXY applied by O_o_Reader) say that only half of the
// configuration is modelled, the other half being its image in the plane (DoubletLatticeSteady).
//
// Thread-safety: immutable. Ownership: plain values.
struct AeroReference
{
    std::int32_t aeroFrameId = 0;
    std::int32_t rigidBodyFrameId = 0;
    double chord = 0.0; // REFC
    double span = 0.0;  // REFB
    double area = 0.0;  // REFS
    std::int32_t symmetryXz = 0;
    std::int32_t symmetryXy = 0;
    input::CoordinateFrame aeroFrame;      // ACSID in the basic system (rectangular)
    input::CoordinateFrame referenceFrame; // RCSID in the basic system (rectangular)

    // The AeroStaticParams of the snapshot. Throws SolverError when the model has no AeroStaticParams (SOL 144 needs
    // one) or more than one, or names a coordinate system it does not have, and UnsupportedCardError for a cylindrical
    // or spherical ACSID or RCSID (the format requires rectangular systems).
    [[nodiscard]] static AeroReference read(const input::ModelSnapshot& snapshot);

    // A point or a direction of the basic system in the aerodynamic frame.
    [[nodiscard]] base::Vec3 pointInAeroFrame(base::Vec3 basic) const;
    [[nodiscard]] base::Vec3 directionInAeroFrame(base::Vec3 basic) const;
    // A direction of the aerodynamic frame in the reference frame (RCSID), and the reference origin in the aerodynamic
    // frame.
    [[nodiscard]] base::Vec3 directionInReferenceFrame(base::Vec3 aero) const;
    [[nodiscard]] base::Vec3 referenceOriginInAeroFrame() const;
};

// The frame of coordinate system `frameId` of the snapshot (the basic system for 0). Throws SolverError when the
// snapshot has no such system, naming `card`.
[[nodiscard]] input::CoordinateFrame frameOf(const input::ModelSnapshot& snapshot, std::int32_t frameId,
                                             const char* card);

} // namespace oos::aero
