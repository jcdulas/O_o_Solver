#pragma once

// The materials of the shells (MAT1, AnisoMat2D, OrthoMat2D), each reduced to what a shell section reads: the
// plane-stress matrix in the material frame, the transverse shear moduli, the in-plane shear modulus (the rotation
// penalty of the drilling membranes), the density, the thermal expansion and its reference temperature, and the
// allowables of the failure indices of a ply.
//
//   | IsoMat | MID | E   | G   | NU  | RHO | A   | TREF | GE  |
//   | AnisoMat2D | MID | G11 | G12 | G13 | G22 | G23 | G33  | RHO | A1 A2 A3 TREF GE ST SC SS MCSID |
//   | OrthoMat2D | MID | E1  | E2  | NU12| G12 | G1Z | G2Z  | RHO | A1 A2 TREF Xt Xc Yt Yc S GE F12 STRN |
//
// IsoMat comes from materialsTable (E, G, NU completed by O_o_Reader); AnisoMat2D and OrthoMat2D from their own card
// tables (model.anisoMat2D.table(), model.orthoMat2D.table(): the Gij, G1Z, G2Z and Ai that materialsTable does not
// give). O_o_Reader gives a blank G1Z or G2Z as 1.0E8, its stored form of a blank field: read as a blank, an infinite
// transverse shear modulus (a rigid direction).

#include "oos/base/math/Mat3.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/core/IsotropicMaterial.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace oos::elements
{

enum class ShellMaterialType : std::uint8_t
{
    IsoMat,
    AnisoMat2D,
    OrthoMat2D
};

// The allowables of the failure indices of a ply (MAT8 Xt, Xc, Yt, Yc, S, F12, STRN; CompProp FT). Xc defaults to
// Xt and Yc to Yt (MAT8); a blank Xt, Yt or S leaves the allowables incomplete (hasStrengths false).
struct PlyAllowables
{
    bool hasStrengths = false;
    double xt = 0.0;
    double xc = 0.0;
    double yt = 0.0;
    double yc = 0.0;
    double s = 0.0;
    double f12 = 0.0;                 // the Tsai-Wu interaction term
    bool areStrainAllowables = false; // STRN = 1.0: Xt ... S are strains (the maximum strain theory)
};

// One shell material in its material frame. The strains are (ε1, ε2, γ12) with the engineering shear.
struct ShellMaterial
{
    std::int32_t matId = 0;
    ShellMaterialType type = ShellMaterialType::IsoMat;
    base::Mat3 planeStress; // IsoMat: E/(1 − ν²)[[1, ν, 0], ...]; AnisoMat2D: Gij; OrthoMat2D: Q of E1, E2...
    std::array<double, 2> transverseShear{}; // the moduli of the x and y transverse shears; 0: rigid (see .cpp)
    double inPlaneShearModulus = 0.0; // IsoMat G, AnisoMat2D G33, OrthoMat2D G12: the drilling membranes' penalty
    double density = 0.0;             // RHO
    std::array<double, 3>
        expansion{}; // (α1, α2, α12): IsoMat (A, A, 0), AnisoMat2D (A1, A2, A3), OrthoMat2D (A1, A2, 0)
    double referenceTemperature = 0.0; // TREF (0 when blank, the default)
    PlyAllowables allowables;          // OrthoMat2D only
    double e1 = 0.0;                   // OrthoMat2D E1, E2 and G12 (MAT1: E, E, G), for the strain allowables
    double e2 = 0.0;
    double g12 = 0.0;
};

// The columns of the material tables a shell reads, once per facet call: materialsTable for IsoMat and the type of
// every material, the AnisoMat2D and OrthoMat2D card tables (null when the snapshot has none).
struct ShellMaterialColumns
{
    IsotropicMaterialColumns isotropic;
    std::span<const double> expansion, referenceTemperature; // a, tref of materialsTable
    base::ValidityView isExpansionGiven, isReferenceTemperatureGiven;
    const input::CardTable* anisoMat2D = nullptr;
    const input::CardTable* orthoMat2D = nullptr;

    explicit ShellMaterialColumns(const ElementBatchView& view);
};

// The material of ID `matId` for the shell of batch row `row`, read as MID1, MID2, MID3 or MID4 (`matIdField`) or as
// the material of a ply ("MID3" of CompProp ply 3...). A missing material, a type other than IsoMat, AnisoMat2D or
// OrthoMat2D, a AnisoMat2D or OrthoMat2D the snapshot has no card table for, or a OrthoMat2D whose E1, E2 are not
// positive are problems.
[[nodiscard]] Resolution<ShellMaterial> resolveShellMaterial(const ElementBatchView& view,
                                                             const ShellMaterialColumns& columns, std::int64_t row,
                                                             std::int32_t matId, const std::string& matIdField);

} // namespace oos::elements
