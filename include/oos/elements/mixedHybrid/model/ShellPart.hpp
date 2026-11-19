#pragma once

#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/geometry/ElementSurface.hpp"
#include "oos/elements/mixedHybrid/geometry/MeshTopology.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"
#include "oos/elements/mixedHybrid/line/LineAnalysis.hpp"
#include "oos/elements/mixedHybrid/line/LineCore.hpp"
#include "oos/elements/mixedHybrid/line/LineKinematics.hpp"
#include "oos/elements/mixedHybrid/line/LineSection.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElement.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElementData.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElementLoads.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElementResults.hpp"
#include "oos/elements/mixedHybrid/shell/ShellReferenceElement.hpp"
#include "oos/elements/mixedHybrid/shell/ShellSection.hpp"
#include "oos/elements/mixedHybrid/shell/ShellTypes.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace oos::elements::mixedHybrid
{

// The shells of a model of the mixed hybrid formulation, and the line elements that lie on their edges (document 34
// of the reference, sections 2.4 to 2.6, 4, 5, 6 and 8.1; docs/mixed-hybrid-formulation-spec.md sections 3.1, 3.2,
// 4.1, 5.4 to 5.7): the reading of their surface from the mesh, their unknowns, the supports read along their edges,
// the mean rotation of the shells at a grid point, their loads, the recovery of their results and the values of an
// affine field on their unknowns. The model of the formulation holds one ShellPart; the line elements alone and the
// solids are kept elsewhere.
//
// The part numbering of the unknowns (the reference's numbering of its shells, so that a port of the rest of the
// reference's analysis follows it line by line):
//   6 g + c                 component c of grid g in the basic system: c = 0, 1, 2 the translations, 3, 4, 5 the
//                           rotations. The shells use the translations only; the lines on shell edges use the rotations
//                           (their rotation Λ at their grids). The other rotations are the integrator's (spec 5.4).
//   then the entity DOFs, entity after entity in the order of entities():
//   - one entity per edge of the shells, in the order of the topology (the shells in their order, edge k of a shell
//     from its corner k to its corner k + 1), each with 3 (k − 1) + 2 k DOFs:
//       3 l + c                 the moment l (l < k − 1) of the displacement trace along the basic axis c, against
//                               P_l(2τ − 1), τ from the lower grid of the edge (its canonical direction);
//       3 (k − 1) + l           the shear γ · t of the edge, Legendre degree l ≤ p = k − 1;
//       3 (k − 1) + k + l       the rotation about the edge, Legendre degree l ≤ p (ShellElement.hpp: â);
//   - one entity per shell, in the order of the shells: its interior displacement functions (three components each,
//     3 m + c) then its interior shear functions (3 n1i + m).
// The entities are the same in the static and in the modal analyses (the reference keeps the shell interiors in
// both). Local unknown i of a shell (ShellElement.hpp ordering) is part unknown dofs[i] of its ShellElementData, with
// the sign signs[i] (local = sign × part); those of a line on a shell edge likewise (LineData::dofs, signs).
//
// The grids are given by a dense index in increasing ID order and the shells by increasing element ID, so that the
// canonical direction of an edge (from its lower grid) and the first shell of an edge (the one whose surface gives
// the curve of the edge and the kinematics of its lines) do not depend on the order of the rows of the model.
//
// Mid-side grid points (ShellQuad8, ShellTria6, spec 4.1; not in the reference): each mid-side grid X_m given fixes
// its edge as the parabola through the two corners and it, the conic of weight 1 whose control point is
// 2 X_m − (X_a + X_b) / 2. It replaces the curve the reading of the mesh gives that edge (Topology::bulges and
// Topology::weights: bulge X_m − (X_a + X_b) / 2, weight 1) before the surfaces of the elements are built, so that
// both elements of the edge and every later step (the fit of the inside of the patches, the edge moments of the map)
// see it. The first shell that gives one wins; another that disagrees is reported. The mid-side grid carries no
// unknown: its position is a point of the edge.
//
// What is built when: the constructor reads the mesh, builds the geometry of every shell (map, section, axes, edges
// and signs: ShellElementData without its tables) and numbers the unknowns. The tables of an element (its jets) and
// its matrices are built on demand (element(), elementWithMatrices(), lineData()): a shell of degree 4 holds about
// 0.2 MB of jets and 0.3 MB of matrices, too much to keep for every element of a large model. With
// ShellPartInput::checkElements the constructor checks every shell and line once to report the elements the
// formulation refuses (a degenerate element, an element that turns too much, a singular section), building the
// matrices of those it has room to keep (elementRefusal); elementRefusal() does the same for one element, from any
// thread.
//
// Reference: src/tdnns/analysis.rs (the shell part of Analysis::new: numbering, geometry, elements, lines on shell
// edges, supports, the loads of the shells; Analysis::fill_rotations, rotation_rows, edge_traction, resultants, state,
// corner_state, stress, transverse_shear, geometric_stiffness, line_kin, line_state, rigid_modes, geometry_report),
// src/tdnns/line.rs::shell_kin, src/tdnns/solid.rs::Context::shell_rows.
//
// Thread-safety: immutable once built; every const member function may be called from several threads at once (the
// reference elements are shared, immutable, built once). Ownership: a ShellPart owns a copy of its input.

// A grid point of the shells: its position in the basic system and its ID (for the messages).
struct ShellPartGrid
{
    SpaceVector position{};
    std::int32_t id = 0;
};

// A shell element.
struct ShellPartShell
{
    std::int32_t elementId = 0;
    // The corner grids (indices of ShellPartInput::grids) in the card's order: 3 or 4.
    ShellCorners corners;
    // The mid-side grid of edge k (from corner k to corner k + 1) of a 6- or 8-node shell, −1 where there is none.
    std::array<std::int64_t, 4> midside{-1, -1, -1, -1};
    // The section in its material axes at its reference thickness.
    Section section;
    // The thicknesses at the corner grids (three for a triangle); empty: the reference thickness of the section.
    std::optional<std::array<double, 4>> gridThicknesses;
    // The material x axis as a vector of the space (projected on the tangent plane point by point); empty: the angle
    // `theta` from the side G1-G2 of the element, about its normal.
    std::optional<SpaceVector> materialDirection;
    double theta = 0.0;
    // The offset of the section along the normal; empty: Section::offset.
    std::optional<double> offset;
    // No transverse shear flexibility (normally: the section has no H): every shear unknown of the element is held at
    // zero (supportsOf), γ = 0 exactly.
    bool kirchhoff = false;
};

// A line element on an edge of the shells (its grids, LineElement::grids, are indices of ShellPartInput::grids).
struct ShellPartLine
{
    std::int32_t elementId = 0;
    LineElement element;
    LineSection section;
};

// What the shells of a model are made of.
struct ShellPartInput
{
    std::vector<ShellPartGrid> grids;
    // By increasing element ID.
    std::vector<ShellPartShell> shells;
    // The line elements that lie on an edge of the shells (a line whose edge has no shell is reported).
    std::vector<ShellPartLine> lines;
    // The declarations of the geometry (MeshReadingInput): raw edges (grid indices), raw elements (shell indices).
    std::set<EdgeKey> rawEdges;
    std::vector<std::size_t> rawElements;
    // The translations (grid index, basic component 0 to 2) held at grids: they orient the common frame of an external
    // edge in the reading of the mesh (MeshReadingInput::heldTranslations). The integrator passes those held in every
    // subcase.
    std::set<std::pair<std::size_t, int>> heldTranslations;
    // The grids of every line element of the formulation (on the shell edges or alone): their rotations are the Λ of
    // the lines, held as the supports state them (only the note on the rotations held elsewhere reads it).
    std::set<std::size_t> lineGrids;
    // The degree k of the displacement (MIXED_HYBRID_DEGREE in the formulation) and the points per direction of the
    // rules (0: 2k + 4).
    std::size_t degree = 4;
    std::size_t pointsPerDirection = 0;
    // The mass matrices and the loads of the inertia are built (elementWithMatrices, lineData).
    bool withMass = false;
    // The spaces of the publications on elements that are not affine (a check against them); false in the formulation.
    bool mapReferenceConstants = false;
    // Build the matrices of every shell and line once in the constructor, to report the elements refused.
    bool checkElements = true;
    // The bytes of shell elements with their matrices that the part keeps once built (elementWithMatrices then copies
    // them instead of building them again: a shell of degree 4 takes a few milliseconds to build and about 0.5 MB to
    // keep), 0 for none. The elements built once the budget is spent are built again at each call.
    std::size_t matrixCacheBytes = 0;
};

// What is attached to a problem: a shell, a line (their index in the input), or the shells as a whole.
struct ShellPartProblem
{
    enum class Subject
    {
        Shell,
        Line,
        Model
    };
    Subject subject = Subject::Model;
    std::size_t index = 0;
    std::string message;
    // The geometry of the element is what the formulation refuses (a degenerate element, an element that turns too
    // much), not its property or its grids.
    bool isGeometry = false;
};

// An entity of the shells: a set of unknowns that belongs to no grid point.
struct ShellPartEntity
{
    enum class Kind
    {
        // The unknowns of an edge (3 (k − 1) + 2k).
        Edge,
        // The interior unknowns of a shell (3 n1i + nci).
        Interior
    };
    Kind kind = Kind::Edge;
    // An edge: its lower and higher grid. An interior: unused.
    std::array<std::size_t, 2> grids{};
    // An edge: its number in the topology. An interior: unused.
    std::size_t edge = 0;
    // An interior: its shell. An edge: its first shell.
    std::size_t shell = 0;
    std::size_t dofCount = 0;
    // Its first unknown in the part numbering.
    std::size_t firstUnknown = 0;
    // A point of it (the middle of the edge curve, the centre of the shell), for the geometric ordering.
    SpaceVector position{};
};

// What an unknown of the part numbering is.
struct ShellPartUnknown
{
    enum class Kind
    {
        // Component `component` (0 to 5, basic) of grid `index`.
        GridComponent,
        // DOF `component` of entity `index`.
        EntityDof
    };
    Kind kind = Kind::GridComponent;
    std::size_t index = 0;
    std::size_t component = 0;
};

// One term of a linear combination of unknowns of the part numbering.
struct ShellPartTerm
{
    std::size_t unknown = 0;
    double coefficient = 0.0;
};

// A direction held at a grid point that is no axis of the basic system (a component of a displacement system turned
// from it): its unit vector in the basic system and the component of the grid point that holds it (0 to 2 a
// translation, 3 to 5 a rotation, in the displacement system of the grid point).
struct ShellSkewAxis
{
    SpaceVector direction{};
    std::size_t component = 0;
};

// The components held at one grid point, in the basic system (0 to 2 the translations, 3 to 5 the rotations), with
// their enforced values (0 where none), and the directions it holds that are no basic axis (`skewAxes`; `values` is
// then the held translation and the held rotation as basic vectors, the parts along the directions not held zero).
struct ShellGridSupport
{
    std::size_t grid = 0;
    std::array<bool, 6> held{};
    std::array<double, 6> values{};
    std::vector<ShellSkewAxis> skewAxes;
    // The grid is not held by a support: a relation of the model moves it (a rigid element). The edges between two
    // such grids follow them wherever they are.
    bool followsRelation = false;
};

// A term of the map that carries the reaction of a derived support to a grid component (0 to 5, basic).
struct ShellGridTerm
{
    std::size_t grid = 0;
    std::size_t component = 0;
    double coefficient = 0.0;
};

// An entity DOF held by the formulation, its value, and the transpose of the map from the held grid values to it:
// the reaction r of the DOF adds r × coefficient to each grid component of `carriedTo` (empty for the DOFs held at
// zero by a rule: the shear of the junctions and of the Kirchhoff elements, the shear of a supported edge).
struct ShellDerivedSupport
{
    std::size_t unknown = 0;
    double value = 0.0;
    std::vector<ShellGridTerm> carriedTo;
};

// A term of a support row on a component of a grid point in its displacement system (0 to 5).
struct ShellAxisTerm
{
    std::size_t grid = 0;
    std::size_t component = 0;
    double coefficient = 0.0;
};

// A constraint that the supports put on the unknowns of an edge along a direction that is no basic axis (a
// translation held in a displacement system turned from the basic one, at both grids of the edge):
// Σ unknowns + Σ grids = 0. It is no support of one unknown: the solver eliminates it as a relation.
struct ShellSupportRow
{
    std::vector<ShellPartTerm> unknowns;
    std::vector<ShellAxisTerm> grids;
};

// The supports derived from those of the grid points, by increasing unknown; the rows of the directions that are no
// basic axis; the notes on what the shells do not represent (a rotation across an edge held with the normal
// displacement free); and the count of edges whose two grids hold directions that differ (a cylindrical displacement
// system along a circle: the grids hold them, the edge between them does not).
struct ShellSupports
{
    std::vector<ShellDerivedSupport> held;
    std::vector<ShellSupportRow> rows;
    std::vector<std::string> notes;
    std::size_t uncarriedEdges = 0;
};

// A point of a shell where a load per unit area is read: its reference coordinates, its position on the reference
// surface of the section (the surface of the grid points moved by the offset along the normal) and the unit normal.
struct ShellLoadPoint
{
    double xi = 0.0;
    double eta = 0.0;
    SpaceVector position{};
    SpaceVector normal{};
};

// A force per unit area of the reference surface of the section (basic components).
using ShellAreaForce = std::function<SpaceVector(const ShellLoadPoint&)>;

// The load of an edge load on the local unknowns of the shell that carries it (the first shell of the edge).
struct ShellEdgeLoad
{
    std::size_t shell = 0;
    Vector local;
};

// The results of a shell at one point: its state (the resultants about the middle of an offset section), the
// stresses at each fibre distance asked (in the material axes of the section, and in the axes of the ply there) and
// the transverse shear stresses there (same axes).
struct ShellRecoveryPoint
{
    ShellPointState state;
    std::vector<std::pair<Voigt3, Voigt3>> stresses;
    std::vector<std::pair<std::array<double, 2>, std::array<double, 2>>> transverseShear;
};

// The results of a shell at its centre and at its corners (in the corner order of the shell).
struct ShellRecovery
{
    ShellRecoveryPoint centre;
    std::vector<ShellRecoveryPoint> corners;
};

// The displacement of a shell at a point as rows on the unknowns of the part numbering (three rows: the basic
// components), and the point of its surface.
struct ShellDisplacementRows
{
    std::array<std::vector<ShellPartTerm>, 3> rows;
    SpaceVector point{};
};

// The entity unknowns of the part that no shell or line of it stiffens (ShellPart::unstiffenedUnknowns), and what
// the displacement directions among them are.
struct ShellUnstiffened
{
    // By increasing unknown.
    std::vector<std::size_t> unknowns;
    // The singular directions of displacement functions held, those of them that do not lie along the basic axes (held
    // by the components closest to them), and the entity of the first of those.
    std::size_t displacementDirections = 0;
    std::size_t skewedDirections = 0;
    std::size_t firstSkewedEntity = 0;
};

// The direction of an edge of a shell and the measure of its rotation unknowns at a point of the edge, s the parameter
// of the edge from its lower grid: the unit tangent t = X_s / |X_s| (X_s = dX/ds) and W² |X_s|, W the denominator of
// the map. The rotation unknown l of the edge for a rotation θ is (2l + 1) ∫ W² |X_s| θ · t P_l(2s − 1) ds (document
// 34 of the reference, section 4.2: â, the rotation a_T = s_k â / (|X_τ| W²) about the edge).
struct ShellEdgeRotationFrame
{
    SpaceVector tangent{};
    double measure = 0.0;
};

// The kinematics of a line on edge `localEdge` of a shell element at the element's own parameter `tau` of the edge
// (ShellElement.hpp: its columns are the local unknowns of the shell): u the trace of the shell, du/ds its derivative
// along the line, and θ = a_T t + t × du/ds, that is θ · μ = ν · du/ds and θ · ν = −μ · du/ds (t, μ = t × ν, ν the
// tangent, conormal and normal of the shell), a_T = s_k â(τ_e) / (|X_τ| W²) the rotation about the edge; the
// derivatives of the tangent and of the normal along the line (the metric of an offset axis):
//   d t/ds = (X_ττ − t (t · X_ττ)) / |X_τ|²,  X_ττ = Σ_αβ (Γ^λ_αβ a_λ + b_αβ ν) t̂^α t̂^β,  dν/ds = ∂_α ν t̂^α / |X_τ|.
// `forward`: the line runs along the element's direction of the edge (its tangent, fraction and the derivatives
// along it change sign otherwise).
// Reference: src/tdnns/line.rs::shell_kin
[[nodiscard]] LineKinematics shellLineKinematics(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                                 std::size_t localEdge, double tau, bool forward);

class ShellPart
{
public:
    // Reads the mesh and builds the geometry and the numbering. The problems of the input (a grid index out of range,
    // a shell of coincident corners, a line not on an edge of the shells, an element refused) are reported by
    // problems(), never thrown. When a corner or index problem prevents the reading of the mesh, the part is not
    // usable() and holds nothing else.
    explicit ShellPart(ShellPartInput input);

    [[nodiscard]] bool usable() const { return m_usable; }
    [[nodiscard]] const std::vector<ShellPartProblem>& problems() const { return m_problems; }
    [[nodiscard]] const ShellPartInput& input() const { return m_input; }
    [[nodiscard]] std::size_t degree() const { return m_input.degree; }

    // ---- the reading of the mesh and the elements

    [[nodiscard]] const MeshReadingInput& meshReading() const { return m_reading; }
    [[nodiscard]] const Topology& topology() const { return m_topology; }
    // The reference element of the shells of `cornerCount` corners.
    [[nodiscard]] const ShellReferenceElement& referenceElement(std::size_t cornerCount) const;
    // Shell `shell` with its tables (jets), its matrices empty.
    [[nodiscard]] ShellElementData element(std::size_t shell) const;
    // Shell `shell` with its tables and its matrices (the mass with ShellPartInput::withMass), or the reason the
    // formulation refuses it.
    [[nodiscard]] std::variant<ShellElementData, std::string> elementWithMatrices(std::size_t shell) const;
    // The same with the mass matrix only when `withMass` (and ShellPartInput::withMass): the mass costs about half the
    // stiffness again. Both come from the cache of the part when it holds the element
    // (ShellPartInput::matrixCacheBytes).
    [[nodiscard]] std::variant<ShellElementData, std::string> elementWithMatrices(std::size_t shell,
                                                                                  bool withMass) const;
    // Why the formulation refuses shell `shell`: a section without bending stiffness (without membrane stiffness for a
    // membrane-only section) or a thickness that is not positive (a singular section), a normal that turns by 90° or
    // more from that of its corners at a point of the element (an element that turns too much), a map that is not
    // invertible, membrane functionals that are not unisolvent or a moment block that is not positive definite (a
    // degenerate element). Empty when it is accepted. An element whose matrices the part has no room to keep
    // (ShellPartInput::matrixCacheBytes) is checked without building them: its section and its geometry (the map
    // included); the two refusals that only its matrices show are then those of elementWithMatrices.
    [[nodiscard]] std::optional<std::string> elementRefusal(std::size_t shell) const;
    // Whether shell `shell` carries moments: every shell but a membrane-only one without offset
    // (sectionCarriesMoments). The others have no moment, no transverse shear and no rotation about their edges.
    [[nodiscard]] bool carriesMoments(std::size_t shell) const;
    // The part unknowns of the local unknowns of a shell and their signs (local = sign × part).
    [[nodiscard]] std::span<const std::size_t> localUnknowns(std::size_t shell) const;
    [[nodiscard]] std::span<const double> localSigns(std::size_t shell) const;
    // The shells that have grid `grid` as a corner, in their order, with the corner.
    [[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> shellsAt(std::size_t grid) const;

    // ---- the unknowns

    [[nodiscard]] std::size_t gridCount() const { return m_input.grids.size(); }
    // 6 per grid, then the entity DOFs.
    [[nodiscard]] std::size_t unknownCount() const { return m_unknownCount; }
    // The first unknown of the entities: 6 × gridCount().
    [[nodiscard]] std::size_t firstEntityUnknown() const { return 6 * gridCount(); }
    [[nodiscard]] const std::vector<ShellPartEntity>& entities() const { return m_entities; }
    // The entity of an edge of the topology, and that of the interior of a shell.
    [[nodiscard]] std::size_t edgeEntity(std::size_t edge) const { return edge; }
    [[nodiscard]] std::size_t interiorEntity(std::size_t shell) const { return m_topology.edges.size() + shell; }
    [[nodiscard]] ShellPartUnknown describe(std::size_t unknown) const;
    // The DOFs of an edge entity: per edge 3 (k − 1) displacement moments, then k shear, then k rotation.
    [[nodiscard]] std::size_t edgeDofCount() const { return 3 * (degree() - 1) + 2 * degree(); }
    [[nodiscard]] std::size_t edgeShearDof(std::size_t l) const { return 3 * (degree() - 1) + l; }
    [[nodiscard]] std::size_t edgeRotationDof(std::size_t l) const { return 3 * (degree() - 1) + degree() + l; }

    // ---- the lines on the edges of the shells

    [[nodiscard]] std::size_t lineCount() const { return m_input.lines.size(); }
    // The line's source: edge `localEdge` of shell `shell` (the first shell of its edge), `forward` along the shell's
    // direction of that edge. Empty for a line the part refused (problems()).
    [[nodiscard]] std::optional<LineSource> lineSource(std::size_t line) const;
    // The line built from the kinematics of the first shell of its edge (moments of degree k, shear flexibility, the
    // rule of the shell's edges), on the part unknowns (the shell's, then the rotations 6 a + 3 and 6 b + 3 of its
    // grids A and B), or the message of a problem.
    // Reference: the lines on shell edges of tdnns/analysis.rs::Analysis::new
    [[nodiscard]] std::variant<LineData, std::string> lineData(std::size_t line) const;
    // The same with the section `section` in place of the input's (a non-structural mass added by the model) and the
    // mass matrix only when `withMass` (and ShellPartInput::withMass).
    [[nodiscard]] std::variant<LineData, std::string> lineData(std::size_t line, const LineSection& section,
                                                               bool withMass) const;
    // The kinematics of a line at the fraction `fraction` of its chord from its grid A: the parameter of the edge where
    // the chord coordinate of the edge point is the fraction (bisection, 60 halvings).
    // Reference: tdnns/analysis.rs::Analysis::line_kin (LineSource::Shell)
    [[nodiscard]] LineKinematics lineKinematics(std::size_t line, double fraction) const;
    // The state of a line at the fraction `fraction` of its chord for its local unknowns x and the temperature field
    // of the case (possibly empty).
    // Reference: tdnns/analysis.rs::Analysis::line_state
    [[nodiscard]] std::variant<LineState, std::string> lineState(std::size_t line, const LineData& data,
                                                                 double fraction, std::span<const double> x,
                                                                 const LineTemperatureField& temperature) const;

    // ---- supports

    // The entity DOFs held for the grid supports `grids` (basic components; grids not listed hold nothing): the shear
    // of the junction edges (three shells or more) and of the Kirchhoff shells, at zero; then, along each edge
    // whose two grids hold components (edgeHeld): an external edge, an edge inside the shells one of whose grids is
    // inside them too (a line of supports across a plate), and any edge between two grids that follow a relation; not
    // an inner edge between two grids of the outline (the diagonal of a triangle at a corner, the edge across a strip
    // one element wide), which would hold what lies between two supported sides. On it: the displacement moments of
    // each component held, with the
    // grid values carried along the edge as by a rod (linear along the chord, cubic Hermite across it with the
    // rotations held at both grids, moments on the isoparametric edge); where the tangent is in the span of the held
    // rotation axes, the edge rotation (2l + 1) ∫ W² θ · X_τ P_l with θ linear along the chord; where the conormal is
    // in that span and the normal displacement is held, the shear at zero. A DOF given by two rules takes the last.
    // The notes: a rotation across an edge held with the normal displacement free (not represented), and the rotations
    // held with a value at corners of shells that are not grids of lines (ShellPartInput::lineGrids): they act only
    // through the edges that hold them.
    // Reference: the supports of tdnns/analysis.rs::Analysis::new
    [[nodiscard]] ShellSupports supportsOf(std::span<const ShellGridSupport> grids) const;

    // The entity unknowns that no shell or line of the part stiffens, to be held at zero in every constraint set (spec
    // sections 4.1 and 5.8; not in the reference). Only the entities whose shells are all membrane-only or bending-only
    // are looked at:
    // - the shear of an edge none of whose shells has a transverse shear (a shell without moments has none), and the
    //   interior shear of a shell without moments;
    // - the rotation about an edge none of whose shells carries moments, unless a line on it turns it (its diagonal
    //   terms there above 1e-14 of its largest: not a rod without torsion constant);
    // - the displacement functions of an entity (an edge function l, an interior function) that `followed` does not
    //   name (the entities whose displacement a solid follows, which the solids stiffen): the 3 × 3 block of the
    //   stiffness of the shells of the entity and of the lines on the edge on the three components of the function,
    //   whose eigenvalues below 1e-8 of the largest are singular directions (the normal of a flat membrane, the plane
    //   of a flat bending-only plate). s singular directions V (orthonormal columns) are held by the s components C
    //   that maximise |det V_C|, the lowest on a tie, as the singularity processor of the grid points does; they lie
    //   along the components held when |det V_C| reaches cos 1°.
    // The elements are built (through the cache of the part).
    [[nodiscard]] ShellUnstiffened unstiffenedUnknowns(const std::function<bool(std::size_t entity)>& followed) const;

    // ---- rotations at grid points

    // The mean rotation of the shells at grid `grid`, basic component `component` (0 to 2): the rotation of each shell
    // at its corner there (rotationRowsOf), averaged with equal weights over the shells that carry moments there (over
    // all of them where none does: a membrane alone turns only with its surface; rotationShellsAt), as a row on the
    // part unknowns (by increasing unknown; the terms whose magnitude is at most `relativeThreshold` times the largest
    // are left out: 0 keeps every term that is not zero, the reference's tie of a line alone at a grid of shells uses
    // 1e-14). Empty at a grid of no shell. Reference: tdnns/analysis.rs::Analysis::fill_rotations and the rows of the
    // lines alone at grids of shells
    [[nodiscard]] std::vector<ShellPartTerm> meanRotationRow(std::size_t grid, std::size_t component,
                                                             double relativeThreshold = 0.0) const;
    // The shells whose rotation the mean rotation at grid `grid` takes, with their corner there (shellsAt, without the
    // shells that carry no moments when one there does).
    [[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> rotationShellsAt(std::size_t grid) const;
    // The three rows at once (the tables of the shells there are built once).
    [[nodiscard]] std::array<std::vector<ShellPartTerm>, 3> meanRotationRows(std::size_t grid,
                                                                             double relativeThreshold = 0.0) const;
    // The unit normal the shells of rotationShellsAt(grid) share at the grid (up to its sign, within a degree): the
    // rotation about it is the rotation of their membrane field in its plane (the drilling rotation of a shell with
    // rotations at its grid points). Empty at a grid of no shell, and where the shells meet at an angle (a fold: the
    // rotation about the normal of one bends the other).
    [[nodiscard]] std::optional<SpaceVector> commonNormalAt(std::size_t grid) const;

    // ---- loads (on the local unknowns of one shell or line)
    //
    // The loads of a line on a shell edge, its geometric stiffness and its free strains are those of LineAnalysis.hpp
    // applied to its lineData(): lineInertiaLoad, lineThermalLoad (the thermal field of the line), lineFreeStrains,
    // lineGeometricStiffness.

    // A force per unit area of the reference surface of the section: f_3i+c = ∫ N_i f_c μ0 dA, μ0 the area of that
    // surface over that of the grid points (offsetArea).
    // Reference: the traction of tdnns/analysis.rs::Analysis::new
    [[nodiscard]] Vector areaForceLoad(const ShellElementData& element, const ShellAreaForce& force) const;
    // A pressure along the normal of the element, given at its corners (ShellCorners order) and interpolated in the
    // reference coordinates (bilinear on a quadrilateral, linear on a triangle).
    // Reference: the pressure of tdnns/analysis.rs::Analysis::new
    [[nodiscard]] Vector pressureLoad(const ShellElementData& element, const std::array<double, 4>& corners) const;
    // The inertia of a uniform acceleration (inertiaLoad).
    [[nodiscard]] Vector inertiaLoad(const ShellElementData& element, const SpaceVector& acceleration) const;
    // The free strains of the temperatures (T0, T1) at the points of the surface of the grid points: the forces, and
    // the terms the recovery (momentTerm) and the geometric stiffness (momentTerm, membraneTerm) need. The element
    // must have its matrices.
    [[nodiscard]] ShellThermalLoad thermalLoad(const ShellElementData& element,
                                               const TemperatureAtPosition& temperatures) const;
    // The load of forces and couples per length (f, m, basic) on the edge between grids a and b, given as a function
    // of the chord fraction from grid a, on the first shell of the edge: the force on the displacement, the couple
    // about the edge on the edge rotation (∫ m · t P_l / W² dτ, W² from the measure of â), the couple about the
    // conormal μ on −β · t. A couple about the normal is refused, as an edge that is not one of the shells.
    // Reference: tdnns/analysis.rs::Analysis::edge_traction (an edge of shells)
    [[nodiscard]] std::variant<ShellEdgeLoad, std::string> edgeLoad(std::size_t gridA, std::size_t gridB,
                                                                    const EdgeTractionField& load) const;

    // ---- results

    // The results at the centre and the corners from the local unknowns x (ShellElementData::local), the thermal
    // moment term of the case (null without) and its temperatures (empty without), at the fibre distances `fibres`
    // from the middle of the section. The element must have its matrices.
    // Reference: tdnns/analysis.rs::Analysis::state, corner_state, stress, transverse_shear
    [[nodiscard]] ShellRecovery recover(const ShellElementData& element, std::span<const double> x,
                                        const Vector* momentTerm, const TemperatureAtPosition& temperatures,
                                        std::span<const double> fibres) const;
    // The geometric stiffness of a shell for its local preload unknowns x (with the thermal terms of the preload
    // case, null without, and the pressure of the case when it follows the surface, null otherwise).
    // Reference: the shells of tdnns/analysis.rs::Analysis::geometric_stiffness
    [[nodiscard]] Matrix geometricStiffness(const ShellElementData& element, std::span<const double> x,
                                            const Vector* momentTerm, const Vector* membraneTerm,
                                            const PressureAt* followerPressure) const;
    // The gradient of φᵀ·K_g(x)·φ with respect to the local preload unknowns x, φ local unknowns too (the geometric
    // stiffness without a following pressure; geometricStiffnessGradient of ShellNonlinear.hpp).
    [[nodiscard]] Vector geometricStiffnessGradient(const ShellElementData& element,
                                                    std::span<const double> phi) const;

    // ---- fields

    // The values on the entity DOFs (unknown − firstEntityUnknown(), unknownCount() − firstEntityUnknown() values) of
    // the affine displacement u(x) = b + A x (A[i][j] = ∂u_i/∂x_j, basic): the displacement coefficients
    // b c_i + A x_i (c_i those of the constant 1, x_i those of the map: P = (b + A X) W exactly), the edge rotations of
    // the rotation ω of the skew part of A, the shear zero. Every entity DOF is written (the shared ones by the last
    // shell that has them, as the reference does).
    // Reference: tdnns/analysis.rs::Analysis::rigid_modes (the shells)
    void affineField(const std::array<SpaceVector, 3>& gradient, const SpaceVector& translation,
                     std::span<double> entityValues) const;
    // The displacement of shell `shell` at (ξ, η) as rows on the part unknowns of the displacement functions whose
    // place `keep` accepts (all with an empty function), and the point of the surface there.
    // Reference: tdnns/solid.rs::Context::shell_rows
    [[nodiscard]] ShellDisplacementRows displacementRows(
        std::size_t shell, double xi, double eta, const std::function<bool(const FunctionKind&)>& keep = {}) const;

    // ---- points of the shells (the ties of the solids, SolidPart::ties)

    // The same rows at several reference points of one shell, one per point in their order, without building the
    // tables of the element (the displacement functions are evaluated at the points themselves): the ties of the
    // solids ask for a hundred points of a shell at a time.
    // Reference: tdnns/solid.rs::Context::shell_rows
    [[nodiscard]] std::vector<ShellDisplacementRows> displacementRowsAt(
        std::size_t shell, std::span<const ReferencePoint> points,
        const std::function<bool(const FunctionKind&)>& keep = {}) const;
    // The motion of shell `shell` at (ξ, η) as rows on the part unknowns, without building the tables of the element:
    // the displacement (rows 0 to 2, basic) and the rotation θ = ν × β + ω ν (rows 3 to 5, rotationRowsOf), each by
    // increasing unknown, and the point of its surface there (a fastener or a weld that pierces the shell,
    // FormulationModel::shellPointMotion; not in the reference).
    struct PointRows
    {
        std::array<std::vector<ShellPartTerm>, 6> rows;
        SpaceVector point{};
    };
    [[nodiscard]] PointRows pointRows(std::size_t shell, double xi, double eta) const;
    // pointRows for a point that may lie on an edge of the shell (the point a glued contact ties to a face,
    // FormulationModel::facePointMotion). On an edge the rotation about the edge is that of the edge, its rotation
    // unknowns a(s) = Σ_l a_l P_l(2s − 1) over the measure W² |X_s| (ShellEdgeRotationFrame), in place of that of
    // the displacement of the element: the unknowns of the edge are the rotation about it in this formulation, one
    // value for the shells of the edge, and the one a support of the edge holds (the slope of the displacement of an
    // element across a clamped edge is not zero at a point of it). At a corner the rotation about the two edges of
    // the shell there. The other components stay those of the element.
    [[nodiscard]] PointRows pointRowsOnEdges(std::size_t shell, double xi, double eta) const;
    // The unknowns of edge `edge` of the topology as moments of a motion along the edge, at the points of the edge
    // rule (a glued contact ties them to the motion of the face under the edge, FormulationModel::edgeTraces; the
    // moments are those of the supports along an edge, supportsOf):
    //   displacement unknown (l, c) = Σ_g displacementWeights[l][g]·u_c(points[g]),
    //                                 the moment ∫ W u_c P_l(2s − 1) ds, l = 0 … k − 2;
    //   rotation unknown l = Σ_g rotationWeights[l][g]·θ(points[g]),
    //                        the coefficient (2l + 1) ∫ W² (θ · X_s) P_l(2s − 1) ds, l = 0 … k − 1,
    // s the parameter of the edge from its lower grid, u and θ in the basic system. Empty for an edge no shell has.
    struct EdgeMoments
    {
        std::vector<SpaceVector> points;
        std::vector<std::size_t> displacementUnknowns; // (l, c) at 3 l + c
        std::vector<std::vector<double>> displacementWeights;
        std::vector<std::size_t> rotationUnknowns;
        std::vector<std::vector<SpaceVector>> rotationWeights;
    };
    [[nodiscard]] EdgeMoments edgeMoments(std::size_t edge) const;
    // The reference point of shell `shell` at the parameter s ∈ [0, 1] of its edge `localEdge` (from its corner
    // localEdge to its corner localEdge + 1), s from the lower grid of the edge: the parameter of the shell is τ = s
    // where the shell runs the edge from its lower grid, 1 − s otherwise.
    // Reference: tdnns/solid.rs::Context::edge_point_of
    [[nodiscard]] ReferencePoint edgeReferencePoint(std::size_t shell, std::size_t localEdge, double s) const;
    // The tangent of edge `localEdge` of shell `shell` and the measure of its rotation unknowns at the parameter s of
    // the edge from its lower grid (ShellEdgeRotationFrame): X_s = s_k (d_1 a_1 + d_2 a_2), d the direction of the
    // edge in the reference domain, s_k its edge sign.
    // Reference: the geometry of the edges of shells in tdnns/solid.rs::Solids::ties
    [[nodiscard]] ShellEdgeRotationFrame edgeRotationFrame(std::size_t shell, std::size_t localEdge, double s) const;

    // ---- the geometry report

    // Per edge of the topology: its grids, its number of shells, whether it is declared raw, and the largest angle
    // between the normals of two shells along it (the isoparametric surfaces of the elements at the points of the
    // edge rule).
    // Reference: tdnns/analysis.rs::Analysis::geometry_report
    [[nodiscard]] std::vector<EdgeGeometry> geometryReport() const;

private:
    // Where a line takes its kinematics.
    struct LineEdge
    {
        std::size_t edge = 0;
        std::size_t shell = 0;
        std::size_t localEdge = 0;
        bool forward = true;
    };

    // The structural checks of the input (indices, corners); false when the mesh cannot be read.
    bool checkInput();
    // The mid-side parabolas in place of the curves read from the mesh.
    void applyMidsideGrids();
    // The geometry of every shell (surfaces, edge moments, map, section, axes), its numbering and the entities.
    void buildShells();
    // The entities, their unknowns and their positions on the surfaces of the shells.
    void buildEntities(std::span<const ElementSurface> surfaces);
    // The shell `shell` with its tables, and its matrices when `withMatrices`, or the reason it is refused.
    [[nodiscard]] std::variant<ShellElementData, std::string> build(std::size_t shell, bool withMatrices,
                                                                    bool withMass) const;
    // The edges of the lines.
    void placeLines();
    // The refusals of the shells and lines (ShellPartInput::checkElements).
    void checkElements();
    // The local edge of shell `shell` that is edge `edge` of the topology.
    [[nodiscard]] std::size_t localEdgeOf(std::size_t shell, std::size_t edge) const;
    [[nodiscard]] const InteriorProjection& projectionOf(std::size_t cornerCount) const;
    void addProblem(ShellPartProblem::Subject subject, std::size_t index, std::string message,
                    bool isGeometry = false);
    [[nodiscard]] std::string gridName(std::size_t grid) const;

    // The shells with their matrices kept once built (ShellPartInput::matrixCacheBytes), by shell, and the bytes they
    // take; guarded by its mutex (the part is shared by the threads of the assembly).
    struct MatrixCache
    {
        std::mutex mutex;
        std::vector<std::shared_ptr<const ShellElementData>> elements;
        std::vector<std::size_t> entryBytes;
        std::size_t bytes = 0;
    };

    ShellPartInput m_input;
    std::unique_ptr<MatrixCache> m_cache = std::make_unique<MatrixCache>();
    std::size_t m_pointsPerDirection = 0;
    bool m_usable = false;
    std::vector<ShellPartProblem> m_problems;
    MeshReadingInput m_reading;
    Topology m_topology;
    // The shells without their tables and matrices.
    std::vector<ShellElementData> m_shells;
    // The interior projections of the triangle and of the quadrilateral (built when a shell needs them).
    std::optional<InteriorProjection> m_triangleProjection;
    std::optional<InteriorProjection> m_quadrilateralProjection;
    std::vector<ShellPartEntity> m_entities;
    std::size_t m_unknownCount = 0;
    std::vector<std::optional<LineEdge>> m_lineEdges;
};

} // namespace oos::elements::mixedHybrid
