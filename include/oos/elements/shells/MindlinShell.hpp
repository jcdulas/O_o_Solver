#pragma once

#include "oos/base/math/MatN.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/shells/HigherOrderShapes.hpp"
#include "oos/elements/shells/ShellRecovery.hpp"
#include "oos/elements/shells/ShellSection.hpp"
#include "oos/elements/shells/ShellSectionRate.hpp"

#include <array>
#include <cstdint>
#include <optional>

namespace oos::elements
{

// The strain-displacement rows over the 6N local DOFs of a shell of N nodes at a point: the membrane strains (εx, εy,
// γxy), the shell curvatures (κx, κy, 2κxy) and the transverse shear strains (γx, γy), in the tangent frame of the
// point.
template <std::size_t N>
struct ShellStrainRows
{
    base::MatN<3, 6 * N> membrane;
    base::MatN<3, 6 * N> curvature;
    base::MatN<2, 6 * N> shear;
};

// One higher-order shell (ShellQuad8 with SerendipityQuadShape, ShellTria6 with QuadraticTriangleShape), flat or
// curved, formulated in its element frame (x, y, n: HigherOrderShell) on the six local DOFs (u, v, w, θx, θy, θn) of
// each of its N nodes. It is the degenerated (continuum-based) shell of Ahmad, Irons & Zienkiewicz, "Analysis of thick
// and thin shell structures by curved finite elements", IJNME 2 (1970) 419-451, with the conventions of the ShellQuad4
// and of the ShellTria3 (docs/shell-quad8-tria6-formulation.md): the directors and the drilling factor of
// ShellDirectors, the free thermal dilatation. Against
// locking, the ShellQuad8 has the uniform reduced integration of Zienkiewicz, Taylor & Too, "Reduced integration
// technique in general analysis of plates and shells", IJNME 3 (1971) 275-290 (the S8R shell of the open-source
// CalculiX, G. Dhondt, is the same choice), and the ShellTria6 tied covariant strains (below). On a flat element the
// kinematics are the isoparametric membrane plus the Reissner-Mindlin plate of the shape (Mindlin, J. Appl. Mech. 18
// (1951) 31-38; Hughes, The Finite Element Method (2000), chapter 5; Bathe, Finite Element Procedures (1996), §5.4).
//
// Kinematics (Ahmad et al. 1970; Hughes 2000, chapter 6; Bathe 1996, §5.4): the mid-surface and the fibres are
// interpolated from the nodes,
//
//   X(ξ, η, z) = X0 + z d,   X0 = Σ N_a X_a,   d = Σ N_a ν_a,
//   U(ξ, η, z) = u0 + z b,   u0 = Σ N_a u_a,   b = Σ N_a (θ_a × ν_a),
//
// ν_a the unit director of node a (given: the normal its grid point shares with the other elements of its smooth
// group or is given by a ShellNormal card, ShellDirectors, or the element's own normal there, nodalDirectors), z the
// distance along the fibre. A rotation θ_a moves the fibre by θ_a × ν_a, so its component along ν_a moves nothing:
// each node has five DOFs in the shell sense (three translations, two rotations normal to its director) carried by
// the six of the grid point, and the rotation about the director is held by the drilling term alone (below).
//
// Strains (the linear shell strains of these kinematics, written with the strains linear through the thickness and the
// shell shifter taken as one: the thin-shell assumption under which the degenerated solid and the shell theory give
// the same element, Büchter & Ramm, "Shell theory versus degeneration — a comparison in large rotation finite element
// analysis", IJNME 34 (1992) 39-59; Simo & Fox, "On a stress resultant geometrically exact shell model, part I",
// CMAME 72 (1989) 267-304, its strain measures linearised; Hughes 2000, chapter 6): at each point a local orthonormal
// tangent frame (e1, e2, e3), e3 the unit normal of the mid-surface, e1 the element x axis projected on the tangent
// plane (the x axis of the element frame where the element is flat), e2 = e3 × e1, and the derivatives ∂/∂s_i along
// e_i,
//
//   membrane       ε_ij = ½ (e_i·∂u0/∂s_j + e_j·∂u0/∂s_i),
//   bending        k_ij = ½ (e_i·∂b/∂s_j + e_j·∂b/∂s_i + ∂d/∂s_i·∂u0/∂s_j + ∂d/∂s_j·∂u0/∂s_i) − ½ (Bᵀε + εB)_ij,
//   transverse     γ_i  = e_i·b + d·∂u0/∂s_i,                                        ε(z) = ε + z k,
//
// B_ij = e_i·∂d/∂s_j the curvature of the director field. The shell curvature of ShellRecovery is κ = −k (κx =
// −∂θy/∂x, κy = ∂θx/∂y, 2κxy = ∂θx/∂x − ∂θy/∂y on a flat element, whose shear strains are then γx = ∂w/∂x + θy,
// γy = ∂w/∂y − θx). A rigid motion (u = ω × X0, θ = ω) gives zero strains for any director field. The terms
// ∂d/∂s·∂u0/∂s couple the membrane displacements to the bending of a curved shell; the last term, the symmetric
// correction of the first-approximation theory of Sanders ("An improved first-approximation theory for thin shells",
// NASA TR R-24, 1959; Budiansky & Sanders, "On the 'best' first-order linear shell theory", Progress in Applied
// Mechanics, Prager anniversary volume, 1963, 129-140), makes a uniform dilatation u0 = e X0 bend nothing, so that a
// free uniform heating of a curved element is the homothety it is in three dimensions (without it the free shape of a
// heated curved panel is off by a few percent, whatever its thickness). These terms vanish on a flat element (d = n,
// ∂d/∂s = 0), where the element is the Mindlin plate of the shape.
//
// Through the thickness: analytically, by the resultants of the section (N = t·G1·ε, M from D = 12I/T³·t³/12·G2 on
// κ, the coupling [B], Q = S·γ), which is exact for the linear strain of these kinematics and takes every section of
// ShellSection (MID1-MID4, CompProp, CompPropGlobal) as it is given, as resultants; a Gauss rule through z of the 3-D
// degenerated solid would need a 3-D material at each height. The dropped shifter terms are of order t/R (R the radius
// of curvature), small for the thin shells these elements serve.
//
// - membrane (MID1): K_m = Σ_g w dA t B_mᵀ G1 B_m;
// - bending and transverse shear (MID2, MID3): K_b = Σ_g w dA (B_bᵀ D B_b + B_sᵀ S B_s), S = diag(t κG_x, t κG_y),
//   the shear correction κ = TS/T (5/6 by default, the factor of Reissner, J. Appl. Mech. 12 (1945) A69-A77);
// - coupling (MID4, on these fields): K_mb = s Σ_g w dA B_mᵀ [B] B_b, s = SHELL_COUPLING_SIGN;
// - drilling (a section with MID1 and MID2, c_d > 0): K_d = c_d Σ_g w dA t G1_33 rᵀ r, the term of the ShellQuad4
//   (docs/shell-quad4-formulation.md, section 8; Hughes & Brezzi, "On drilling degrees of freedom", CMAME 72 (1989)
//   105-121) with r = ω − θ_n, ω = ½ (e2·∂u0/∂s1 − e1·∂u0/∂s2) the rotation of the surface about its normal and
//   θ_n = Σ N_a (θ_a − θ̄)·ν_a + θ̄·e3 that of the nodes about their directors (θ̄ their mean rotation: r is zero for a
//   rigid motion), c_d = 1e-6 × PARAM ShellDrillingStiffness (ShellDirectors). With the full rule of the shape: the
//   2 × 2 rule would leave the rotations of the edge nodes in ξ² − η², zero at its four points, without stiffness
//   across a whole mesh. It is a stabilisation: it holds the rotation of each node about its director, which nothing
//   else holds where the elements of a grid point share its normal. Without it (c_d = 0, or a section without MID1
//   or without MID2) that rotation is singular there and the automatic constraint of singular DOFs
//   (AutoSingularityConstr, assembly::GridPointSingularities) removes it, along a component axis or along the skewed
//   normal.
// Stiffness integrals with the rule of the shape (Shape::rule()), dA = |∂X0/∂ξ × ∂X0/∂η| dξ dη, t(ξ, η) interpolated
// from the corner thicknesses (the edge nodes take the mean of their corners, so the quadratic interpolation
// reproduces the bilinear or linear field of the corners exactly).
//
// The ShellQuad8: the strains of the displacements with the uniform 2 × 2 rule (Zienkiewicz, Taylor & Too 1971). It
// keeps the spurious modes of that rule an isolated element shows (one of the membrane, one of the bending, flat or
// curved: 8 zero eigenvalues with the 6 rigid motions); they are not communicable, so an assembled mesh does not
// have them (Zienkiewicz, Taylor & Too 1971; Hughes & Cohen, "The 'heterosis' finite element for plate bending",
// Computers & Structures 9 (1978) 445-450).
//
// The ShellTria6: the quadratic triangle has no reduced rule of full rank, and with the strains of its displacements
// it locks in the bending of thin plates and of curved shells (a clamped plate of 8 × 8 cells gave 0.85 of its
// deflection, the pinched hemisphere 0.38). Its covariant transverse shears and membrane strains are tied (assumed
// from their values at tying points: Lee & Bathe, "Development of MITC isotropic triangular shell finite elements",
// Computers & Structures 82 (2004) 945-962; Bathe & Dvorkin, IJNME 22 (1986) 697-722 for the principle), in the
// natural coordinates (ξ, η) of the element, sides η = 0, ξ = 0 and ξ + η = 1:
// - the transverse shears γ̃_α = ∂X0/∂ξ_α·b + d·∂u0/∂ξ_α take the field of eight terms
//   a1 + b1 ξ + c1 η + η (d ξ + e η), a2 + b2 ξ + c2 η − ξ (d ξ + e η), whose tangential component along each side
//   is linear, from the tangential shear at the two Gauss points of each side and the two shears at the centroid;
// - the membrane strains ẽ_ξξ, ẽ_ηη and ẽ_qq = ẽ_ξξ + ẽ_ηη − 2 ẽ_ξη (along the third side) are each linear, from
//   their values at the two Gauss points of their side and at one inner point, the middle of the segment the element
//   cuts on the parallel to that side at 1/√3 of the way to the opposite corner (the position that gives the best
//   pinched hemisphere and barrel vault among those measured from 1/3 to 0.8: the document gives the table);
// - the local strains follow by ε_ij = Σ (g^α·e_i)(g^β·e_j) ẽ_αβ and γ_i = Σ (g^α·e_i) γ̃_α, g^α the contravariant
//   vectors of the surface at the point; the curvatures are those of the displacements;
// - with the seven-point rule (the tied fields are integrated in full).
// On a triangle of straight sides the membrane strains are linear and the tied ones are those of the displacements;
// a state of uniform strains and curvatures is exact, in the element and at every place of its recovery. Its six
// rigid motions are its only zero-energy modes.
//
// Transverse shear with MID3 blank (a design choice of this project): the rigid shear is approximated by the shear of
// the bending material, S = TS/T · G2_33 · t (G2_33 the in-plane shear modulus of MID2, G of an IsoMat: the transverse
// shear of a homogeneous isotropic plate), which is negligible for a thin plate. Not the stiffness c_K · 12 max(D11,
// D22) / A the ShellQuad4 and the ShellTria3 give such a section (EdgeShearShellSection.hpp): that stiffness makes a
// plate of any thickness a thin one for its elements, and the eight nodes of the ShellQuad8 do not hold the uniform
// curvature of a thin plate on a quadrilateral that is not a parallelogram (measured, 3 × 3 cells with their inner
// grids moved by 0.3 of a cell, a plate of thickness a tenth of its side: moments at the places off by 2 % with the
// shear of the material, by 50 % with c_K); the ShellTria6 keeps the same rule.
//
// Mass and pressure loads with the full rule of the shape (Shape::massRule(): 3 × 3 Gauss, the seven-point triangle
// rule), which integrates N_i N_j and N_i p exactly on a flat element of straight edges: M_ij = ∫ (ρ t + NSM) N_i N_j
// dA on the translations, plus the rotary inertia of the fibres ∫ ρ t³/12 b·b dA in the coupled mass (Mindlin 1951;
// on θ_i, θ_j the block ∫ ρ t³/12 N_i N_j dA ((ν_i·ν_j) I − ν_j ν_iᵀ), diag(1, 1, 0) on a flat element); the lumped
// mass is the HRZ diagonal of the translational mass (Hinton, Rock & Zienkiewicz, "A note on mass lumping and related
// processes in the finite element method", Earthquake Eng. Struct. Dyn. 4 (1976) 245-249: the diagonal of the
// consistent mass, scaled to the element's mass; corner 3/76 and edge node 16/76 of a rectangular ShellQuad8, corner
// 3/57 and edge node 16/57 of a straight-sided ShellTria6), with no rotary inertia. The pressure loads F_i = ∫ N_i p e3
// dA of the corner pressures interpolated by the corner (bilinear or linear) shape functions, normal to the surface at
// each point (e3), or along a given direction. Thermal loads with the stiffness rule, the temperature change
// interpolated from the grid point values of every node; the dilatation ε of the free thermal strain (the mean over
// the element of half the trace of its free membrane strain) is taken apart, its loads being K q for the homothety
// q = ε x of the element: a free uniform heating is then that homothety without stress, exactly, on a curved element
// and with directors shared with the neighbours (whose transverse shear the correction of Sanders does not see), as
// for the ShellQuad4 and the ShellTria3. Recovery: the state at the centre and at the corner nodes, in the tangent
// frame of that place; the ShellTria6 from its tied strains there, the ShellQuad8 from the strains at the four
// points of its reduced rule (the points where the strains of a reduced integration are the most accurate: Barlow,
// "Optimal stress locations in finite element models", IJNME 10 (1976) 243-251), carried to the place by the
// bilinear functions through these points. The strains of a place are those of the displacements less what the
// strains of the free dilatation of the element are off those of a flat element.
//
// Thread-safety: immutable once built. A plain value, built per element.
template <typename Shape>
class MindlinShell
{
public:
    static constexpr std::size_t N = Shape::NODES;
    static constexpr std::size_t C = Shape::CORNERS;
    static constexpr std::size_t DOFS = 6 * N;

    // `position` the nodes in the element frame (relative to its origin; all z = 0 on a flat element), `director` the
    // unit director of each node in the element frame (nodalDirectors), `section` in the element frame, `thickness`
    // the thickness at each corner. Precondition: the surface is not singular (the family checks the shape:
    // isValidShape on the plane coordinates, and smallestNormalCosine).
    // `drillingFactor`: c_d of the model (ShellDirectors), zero for no drilling term.
    MindlinShell(const std::array<base::Vec3, N>& position, const std::array<base::Vec3, N>& director,
                 const ShellSection& section, const std::array<double, C>& thickness, double drillingFactor = 0.0);

    // Whether the isoparametric map of the plane coordinates (x, y: the nodes projected on the element plane) has a
    // positive Jacobian at every point of both rules and at every node: the surface does not fold over that plane.
    [[nodiscard]] static bool isValidShape(const std::array<double, N>& x, const std::array<double, N>& y);
    // The director of each node: the unit normal of the isoparametric surface X0 = Σ N_a X_a at the node,
    // ν_a = (∂X0/∂ξ × ∂X0/∂η)/|∂X0/∂ξ × ∂X0/∂η| at (ξ_a, η_a). Precondition: isValidShape.
    [[nodiscard]] static std::array<base::Vec3, N> nodalDirectors(const std::array<base::Vec3, N>& position);
    // The smallest cosine between the normal of the surface and the element normal n, over the nodes and the points of
    // both rules: how far the curved element turns from the plane of its corners.
    [[nodiscard]] static double smallestNormalCosine(const std::array<base::Vec3, N>& position);

    [[nodiscard]] base::MatN<DOFS> stiffness() const;
    // The differential stiffness of the membrane resultants (Nx, Ny, Nxy) of the preload, constant over the element
    // and taken in the tangent frame of each point (ShellDifferential), with the rule of the shape.
    [[nodiscard]] base::MatN<DOFS> differentialStiffness(const std::array<double, 3>& membraneForce) const;
    [[nodiscard]] base::MatN<DOFS> mass(MassFormulation formulation, double areaMass) const;
    // The loads of the corner pressures along the normal of the surface at each point, or along `direction` (element
    // frame, unit) when given.
    [[nodiscard]] std::array<double, DOFS> pressureLoads(const std::array<double, C>& cornerPressures,
                                                         const std::optional<base::Vec3>& direction = {}) const;
    [[nodiscard]] std::array<double, DOFS> thermalLoads(const ShellTemperatureChange& temperature) const;
    // The state at the centre then at the C corner nodes, named by `gridIds` (their corners first), each in the
    // tangent frame of its place.
    [[nodiscard]] std::array<ShellPlaceState, C + 1> placeStates(const std::array<double, DOFS>& q,
                                                                 const std::array<std::int32_t, N>& gridIds,
                                                                 const ShellTemperatureChange& temperature = {}) const;
    // placeStates in parts: what does not depend on the displacements nor on the
    // temperatures (PlaceOperators: the shape functions and strain rows of each place, its thickness, its bending,
    // coupling and transverse shear stiffnesses), what depends on the temperatures only (PlaceThermal), formed once
    // each for many displacements (the unit displacements of the recovery slopes); placeStates(operators, thermal, q,
    // ...) gives the bits of placeStates(q, ...) (the same operations in the same order).
    struct PlaceOperators
    {
        std::array<ShapeValues<N>, C + 1> shape;
        std::array<base::MatN<3, DOFS>, C + 1> membrane;
        std::array<base::MatN<3, DOFS>, C + 1> curvature;
        std::array<base::MatN<2, DOFS>, C + 1> shear;
        std::array<double, C + 1> thickness{};
        std::array<base::Mat3, C + 1> coupling; // [B], zero without coupling
        std::array<base::Mat3, C + 1> bending;  // D (with bending)
        std::array<base::MatN<2>, C + 1> shearStiffness;
    };
    struct PlaceThermal
    {
        std::array<ShellTemperatureChange, C + 1> temperature; // the element's, ΔT at the place
        std::array<ThermalResultants, C + 1> resultants;
        // The strains at the place of the homothety of the dilatation ε of the element (thermalLoads), less (ε, ε, 0)
        // for the membrane: what the strains of a free dilatation are off those of a flat element. Zero without
        // temperature change, and on a flat element whose directors are its normal (to round-off).
        std::array<std::array<double, 3>, C + 1> membraneOfDilatation{};
        std::array<std::array<double, 3>, C + 1> curvatureOfDilatation{};
        std::array<std::array<double, 2>, C + 1> shearOfDilatation{};
    };
    [[nodiscard]] PlaceOperators placeOperators() const;
    [[nodiscard]] PlaceThermal placeThermal(const PlaceOperators& operators,
                                            const ShellTemperatureChange& temperature) const;
    [[nodiscard]] std::array<ShellPlaceState, C + 1> placeStates(const PlaceOperators& operators,
                                                                 const PlaceThermal& thermal,
                                                                 const std::array<double, DOFS>& q,
                                                                 const std::array<std::int32_t, N>& gridIds) const;
    [[nodiscard]] double area() const noexcept { return m_area; }
    // The energy of the drilling term of K under the local displacements `q`: ½ c_d ∫ t G1_33 (r q)² dA, zero for a
    // shell without that term.
    [[nodiscard]] double drillingEnergy(const std::array<double, DOFS>& q) const;

    // The rates along a rate of the section at fixed geometry and thickness (the AnisoMat2D terms of
    // ShellSectionRate): every part of K is linear in G1, D, [B] and the shear S, so dK is K's sum at the rates, the
    // shear's S = t·κG of a flexible direction or t·TS/T·G2_33 of one without MID3 (the class comment)
    // differentiated as such.
    [[nodiscard]] base::MatN<DOFS> stiffnessSectionRate(const ShellSectionRate& rate) const;
    // The rates of the place states of q (placeStates without a temperature change: `states`) at fixed q:
    // dN = t dG1 ε + s d[B] κ, dM = −(dD κ + s d[B] ε), dQ = dS γ; the strains and the curvatures do not move.
    [[nodiscard]] std::array<ShellPlaceState, C + 1> placeStateSectionRates(
        const std::array<ShellPlaceState, C + 1>& states, const std::array<double, DOFS>& q,
        const ShellSectionRate& rate) const;

    // The geometry of the mid-surface at a point (ξ, η): the shape functions and their derivatives along the tangent
    // axes, the area factor, the tangent frame and the director field.
    struct SurfacePoint
    {
        double xi = 0.0;
        double eta = 0.0;
        ShapeValues<N> shape;
        std::array<double, N> dX{}; // ∂N_a/∂s1, along e1
        std::array<double, N> dY{}; // ∂N_a/∂s2, along e2
        double determinant = 0.0;   // dA / (dξ dη) = |∂X0/∂ξ × ∂X0/∂η|
        // The tangents along the tangent axes: [∂X0/∂ξ·e1, ∂X0/∂ξ·e2, ∂X0/∂η·e1, ∂X0/∂η·e2].
        std::array<double, 4> jacobian{};
        std::array<base::Vec3, 3> frame{}; // e1, e2, e3 in the element frame
        base::Vec3 director;               // d = Σ N_a ν_a
        base::Vec3 directorDX;             // ∂d/∂s1
        base::Vec3 directorDY;             // ∂d/∂s2
    };

    using StrainRows = ShellStrainRows<N>;
    // The values the tied strains of the ShellTria6 are built from (the class comment), as rows over the local DOFs:
    // the covariant transverse shears at their eight tying points, the covariant membrane strains at their nine.
    static constexpr std::size_t SHEAR_TIES = 8;
    static constexpr std::size_t MEMBRANE_TIES = 9;
    struct TiedStrains
    {
        base::MatN<SHEAR_TIES, DOFS> shear;
        base::MatN<MEMBRANE_TIES, DOFS> membrane;
    };

private:
    [[nodiscard]] SurfacePoint pointAt(double xi, double eta) const;
    // The tied values of the element (nothing for a shape without tied strains).
    [[nodiscard]] TiedStrains tiedStrains() const;
    // The strain rows of the point: those of the displacements, the membrane strains and the transverse shears
    // replaced by the tied ones for a shape with tied strains.
    [[nodiscard]] StrainRows rowsAt(const SurfacePoint& p, const TiedStrains& tied) const;
    // The drilling row of the point: r = ω − θ_n (the class comment).
    [[nodiscard]] std::array<double, DOFS> drillingRowAt(const SurfacePoint& p) const;
    [[nodiscard]] bool hasDrilling() const noexcept
    {
        return m_section.hasMembrane && m_section.hasBending && m_drillingFactor > 0.0;
    }
    // The dilatation of the free thermal strain of the element: the mean over the element of half the trace of its
    // free membrane strain. Zero for a section without membrane.
    [[nodiscard]] double dilatationOf(const ShellTemperatureChange& temperature) const;
    [[nodiscard]] double thicknessAt(const ShapeValues<N>& shape) const;
    [[nodiscard]] double deltaTAt(const ShellTemperatureChange& temperature, const ShapeValues<N>& shape) const;
    [[nodiscard]] base::MatN<2> shearStiffness(double t) const;
    [[nodiscard]] base::MatN<2> shearStiffnessRate(double t, const ShellSectionRate& rate) const;
    // K += c_d ∫ t · `shearModulus` · rᵀ r dA, r the drilling row, with the full rule of the shape.
    void addDrilling(base::MatN<DOFS>& k, double shearModulus) const;

    std::array<base::Vec3, N> m_position{};
    std::array<base::Vec3, N> m_director{};
    ShellSection m_section;
    std::array<double, N> m_nodeThickness{}; // corners as given, edge nodes the mean of their corners
    double m_area = 0.0;
    double m_drillingFactor = 0.0;
};

using ShellQuad8Shell = MindlinShell<SerendipityQuadShape>;
using ShellTria6Shell = MindlinShell<QuadraticTriangleShape>;

extern template class MindlinShell<SerendipityQuadShape>;
extern template class MindlinShell<QuadraticTriangleShape>;

} // namespace oos::elements
