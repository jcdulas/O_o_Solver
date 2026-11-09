#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/core/IsotropicMaterial.hpp"
#include "oos/elements/core/MaterialMatrices.hpp"
#include "oos/elements/core/PropertyLookup.hpp"
#include "oos/elements/core/SolidShapeFunctions.hpp"

#include <cstdint>
#include <span>
#include <string_view>

namespace oos::elements
{

// The integration of a solid as SolidProp IN and ISOP choose it (resolveSolidSection says which fields give which):
// - Standard, the default of each shape (IN and ISOP blank), a textbook element:
//   - the 8-node SolidHexa: the incompatible-modes hexahedron of Wilson et al. with Taylor's correction, 2 × 2 × 2
//   points
//     (solidHexa/SolidHexaIncompatibleModes.hpp; CalculiX C3D8I);
//   - the 20-node SolidHexa: the serendipity hexahedron with the full 3 × 3 × 3 rule (Zienkiewicz & Taylor vol. 1, the
//     rule exact for the stiffness of a parallelepiped; CalculiX C3D20, Code_Aster's HEXA20);
//   - the SolidPenta: the isoparametric wedge with the rule exact on a straight prism (6 nodes: 3 triangle points × 2;
//     15 nodes: a degree-4 triangle rule × 3; CalculiX C3D6 and C3D15, Code_Aster's PENTA6 and PENTA15 are the same
//     isoparametric elements);
//   - the SolidTetra: one point for 4 nodes (the constant strain tetrahedron), five points of degree 3 for 10 nodes;
//   - the SolidPyramid: the collapsed hexahedron, 2 × 2 × 2 (5 nodes) or 3 × 3 × 3 (13 nodes) points;
// - FullGauss: the plain isoparametric element with its full Gauss rule (ISOP = FULL): the trilinear 8-node SolidHexa
//   with 2 × 2 × 2 points (CalculiX C3D8); for the other shapes the same element as Standard;
// - ReducedGauss: the 20-node SolidHexa with the reduced 2 × 2 × 2 rule (ISOP = REDUCED or IN = TWO; Zienkiewicz &
//   Taylor vol. 1, reduced integration; CalculiX C3D20R), which has zero-energy modes that a mesh of several elements
//   does not let through but a single element can.
enum class SolidIntegration : std::uint8_t
{
    Standard,
    FullGauss,
    ReducedGauss
};

// What a solid element reads from its SolidProp and its material:
// - the constitutive matrix in the material system (MAT1: isotropicSolidMatrix from E and NU; AnisoMat3D: G11-G66 as
//   given), Voigt order of MaterialMatrices.hpp;
// - the density RHO (mass per unit volume);
// - the material system CORDM (PSOLID: 0 the basic system, or a rectangular system), in which the material is
//   given and the stresses and strains are output: `axes` has the material axes as columns, in the basic system;
// - the integration (SolidIntegration);
// - the thermal expansion (MAT1 A, TREF; AnisoMat3D A1-A6, TREF): the thermal strain of a unit temperature change in
// the
//   material system, in the engineering strains of the Voigt order (εx, εy, εz, γxy, γyz, γzx): (A, A, A, 0, 0, 0)
//   for a IsoMat, (A1, ..., A6) for a AnisoMat3D (MAT9 gives the coefficients of the strains in that order), and the
//   reference temperature TREF of the material (the initial temperature of a subcase without TEMPERATURE(INITIAL)).
//
// IsoMat for the solids: the matrix is built from E and NU only, G being ignored. With E = 2.9e7, G = 1.1e7, NU = 0.32,
// the shear strains are the shear stresses divided by E/2(1 + NU) = 1.0985e7, not by G.
struct SolidSection
{
    Mat6 elasticity{};
    double density = 0.0;
    base::Mat3 axes = base::Mat3::identity();
    base::Vec3 origin{};
    SolidIntegration integration = SolidIntegration::Standard;
    Voigt6 expansion{};
    double referenceTemperature = 0.0;
};

// The columns a solid batch reads to resolve its sections, read once per facet call: the SolidProp table and its
// fields (propId, matId, matFrameId, in_, stress, isop, fctn of model.solidProp.table()), the materials table (type, E,
// NU, RHO) and, for AnisoMat3D, the material card table "materials/AnisoMat3D" of the snapshot (id; stiffness: the 21
// terms G11, G12, ..., G66 of the upper triangle by rows, as model.anisoMat3D.table() gives them; thermalExpansion and
// referenceTemperature: A1-A6 and TREF, 0 when the table lacks them).
struct SolidSectionColumns
{
    PropertyRows properties;
    std::span<const std::int32_t> matId;
    std::span<const std::int32_t> matFrameId;
    base::Utf8View integrationNetwork; // IN
    base::Utf8View stressPoints;       // STRESS
    base::Utf8View integrationScheme;  // ISOP
    base::Utf8View function;           // FCTN
    IsotropicMaterialColumns isotropic;
    // The AnisoMat3D card table, or nullptr when the snapshot has none (a snapshot taken without material card tables).
    const base::ContractReader* anisotropic = nullptr;
    std::span<const std::int32_t> anisotropicIds;
    base::FixedListView<double> anisotropicTerms;
    base::FixedListView<double> anisotropicExpansions; // ai, when the table has it
    std::span<const double> anisotropicReferences;     // tref, when the table has it

    explicit SolidSectionColumns(const ElementBatchView& view);
};

// The section of the element of batch row `row`, whose shape is `shape`. The fields IN and ISOP:
// - 8-node SolidHexa: IN blank or BUBBLE with ISOP blank or REDUCED, the incompatible modes (Standard: the modes are the
//   bubble functions 1 − ξ², and they relieve the shear locking a reduced rule would; a one-point hexahedron, which
//   needs hourglass control, is not offered); ISOP = FULL with IN blank, GAUSS or TWO, FullGauss;
// - 20-node SolidHexa: IN blank or THREE with ISOP blank or FULL, 3 × 3 × 3 points; ISOP = REDUCED with IN blank or TWO,
// or
//   IN = TWO with ISOP blank, ReducedGauss;
// - 6-node SolidPenta: IN blank or BUBBLE with ISOP blank or REDUCED, or ISOP = FULL with IN blank, GAUSS or TWO: the
// one
//   isoparametric wedge;
// - 15-node SolidPenta: IN blank or THREE with ISOP blank or FULL;
// - 4-node SolidTetra: IN blank or TWO with ISOP blank, FULL or REDUCED (its strain is constant: every rule gives the
// same
//   matrices);
// - 10-node SolidTetra, SolidPyramid: IN blank with ISOP blank or FULL.
// Problems: the SolidProp or its material missing or of another type; CORDM −1 or −2 (the element systems of the
// solids, not supported yet), a missing or curvilinear CORDM; IN and ISOP other than above; STRESS = GAUSS but on a
// 4-node SolidTetra (whose one Gauss point is its centroid); FCTN other than SMECH; a IsoMat with E ≤ 0 or NU outside
// (−1, 0.5); a AnisoMat3D whose matrix is not positive definite; a material other than IsoMat and AnisoMat3D.
[[nodiscard]] Resolution<SolidSection> resolveSolidSection(const ElementBatchView& view,
                                                           const SolidSectionColumns& columns, std::int64_t row,
                                                           SolidShape shape);

// The constitutive matrix of a AnisoMat3D from the 21 terms of its card (G11, G12, ..., G16, G22, ..., G66: the upper
// triangle by rows), symmetric. Precondition: terms.size() is 21.
[[nodiscard]] Mat6 anisotropicSolidMatrix(std::span<const double> terms);

// Whether a symmetric 6 × 6 matrix is positive definite (Cholesky factorisation with every pivot above a relative
// tolerance).
[[nodiscard]] bool isPositiveDefinite(const Mat6& matrix);

} // namespace oos::elements
