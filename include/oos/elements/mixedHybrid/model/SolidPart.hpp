#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/elements/core/EntityLayout.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/geometry/MeshTopology.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"
#include "oos/elements/mixedHybrid/line/LineKinematics.hpp"
#include "oos/elements/mixedHybrid/solid/SolidData.hpp"
#include "oos/elements/mixedHybrid/solid/VolumeGeometry.hpp"
#include "oos/elements/mixedHybrid/solid/VolumeKernel.hpp"
#include "oos/elements/mixedHybrid/solid/VolumeShape.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
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

struct EdgeProjection;
// The faces of the skin of the solids, read at points (SolidPartTies.cpp).
class SkinFaces;

// The solids of a model of the mixed hybrid formulation (document 35 of the reference, sections 3.4 and 3.5; document
// 34, section 8.2; docs/mixed-hybrid-formulation-spec.md sections 3.3, 5.6, 5.7 and 10.3): their edges and faces,
// the geometry of the edges and faces (mid-side parabolas, the skin read from the mesh), the entities that carry the
// unknowns shared by the solids, the map of the local unknowns of each solid onto the grid translations and the
// entities, the kernel of each solid, its loads and stresses, the supports derived from the grid supports, and the
// values of an affine field on the entities. The integrator holds one SolidPart in the model of the formulation and
// maps its grid indices and entities onto its own numbering.
//
// Grids are given by a dense index in increasing ID order: every orientation (an edge from its lower grid, the
// functions of a face from the numbers of its grids) follows the grid IDs, so that the results do not depend on the
// order of the elements nor on the corner an element starts from.
//
// Geometry, in the priority of document 35, section 3.4:
//   0. an edge or a face that is tied to the shells takes, with traces and faces of degree 4, the same projection of
//      the surface of the shell it follows (a rigid motion of the shell is then a rigid motion of the solid); an edge
//      of lines alone is straight;
//   1. an edge with a mid-side grid X_m is the parabola through it: the coefficient 4 (X_m − (X_a + X_b)/2) on its
//      first function (a partial set of mid-side grids is accepted; the first element by ID gives the grid of an edge);
//      a face all of whose edges have one keeps no curve of its own;
//   2. with traces and faces of degree 4, the rest of the skin (the faces of one solid) is read from the mesh across
//      the declared sharp edges (readSolidSkin, skinFacePatch): each edge curve projected on the functions of its edge,
//      each patch on those of its face (SolidPartProjections.hpp);
//   3. the interior edges and faces are straight.
//
// Ties with the shells and the lines (spec section 3.4, ties 1 to 4; SolidPartTies.cpp), when the input gives the
// shells (SolidPartInput::shells) and the edges of lines alone (SolidPartInput::aloneEdges):
//   1. an edge of the solids that is an edge of shells (or the diagonal of a quadrilateral shell that covers two
//      triangles of the solids) follows the shell: its functions are the L2 projection of the trace of the shell less
//      the linear part of its two grids; where none of the shells of the edge lies on a face of the solids and the
//      skin holds their fibre, the trace of the shell is that of the mean of the skin over its thickness instead
//      (the force of the shell spread over its thickness; not in the reference);
//   2. a face of the solids that a shell covers (a triangle on a triangle, a quadrilateral on a quadrilateral, half of
//      a quadrilateral on a triangle) follows the shell: its functions are the same projection of the displacement of
//      the shell less what its grids and edges carry;
//   3. the displacement functions of the lines alone on an edge of the solids follow those of the solid edge;
//   4. the rotation unknowns about an edge of shells or of lines alone that the solids have, where a shell of the
//      edge does not lie on a face of the solids (a fin standing on a block, a beam along its edge), follow the
//      rotation of the solids about the edge: the slope of the displacement of their skin across the fibre of that
//      shell, over its thickness (not in the reference), or, for the lines alone and where the skin does not hold the
//      fibre, half the curl of their displacement, the mean over the solids of the edge.
// An edge or a face tied to a shell or a line has `follows` set (SolidPartEdge::tie, SolidPartFace::tie): its
// geometry is that of the shell it follows (straight under lines alone), and the supports do not hold it (it is held
// through what it is tied to). ties() gives the rows; the integrator maps them onto its unknowns.
//
// Unknowns: three translations per grid (basic axes); per edge its 3 (p − 1) functions; per face its 3 fc functions
// (fc = faceFunctionCount); per solid of positive density, in a modal analysis only, its interior fields
// (SolidOptions::interiorFieldCount). Entity DOF 3 m + c is component c (basic) of function m of the edge or face,
// the functions written from the grid IDs (VolumeShape.hpp): unknown m of an edge or face is the same unknown for
// every element that has it, and no sign is needed. Every entity DOF has stiffness from the solids that have it (the
// reference gives a solid without interior fields a dummy diagonal for them; here such a solid has no interior
// entity). A mid-side grid carries no unknown: it follows its edge (midSideTerms).
//
// Thread-safety: immutable once built; every const function may be called concurrently. Ownership: a value that
// holds its own copy of the input, but for the shells (SolidPartInput::shells), which it only points to.

class ShellPart;

// No mid-side grid on that edge.
inline constexpr std::size_t NO_MID_SIDE_GRID = static_cast<std::size_t>(-1);

// A solid element as the integrator gives it.
struct SolidPartElement
{
    std::int32_t elementId = 0;
    // The corner grids (indices) in the order of the card: tetrahedron 4, pyramid 5 (apex last), pentahedron 6,
    // hexahedron 8.
    std::vector<std::size_t> corners;
    // One per edge of the shape in the order of shapeEdges (the order of the mid-side nodes of the card),
    // NO_MID_SIDE_GRID where an edge has none; empty when the element has none at all.
    std::vector<std::size_t> midSideGrids;
    // The law in the material axes (SolidMaterial::stiffness), its density and expansion.
    SolidMaterial material;
    // The material axes: column i is axis i written in the basic system (identity: the basic axes).
    base::Mat3 materialAxes = base::Mat3::identity();
};

// An edge of line elements that no shell has (an edge alone, LineKinematics.hpp), as the integrator builds it: its
// grids, its number in the integrator's list of edges (TieUnknown::index of its unknowns), and the edge itself
// (straight, the frame of its displacement functions, its degree k). Its entity DOFs: 3 l + c the displacement
// function l (l < k − 1) along axis c of AloneEdge::frame, then 3 (k − 1) + l its twist coefficient l (l < k).
struct SolidPartAloneEdge
{
    EdgeKey grids;
    std::size_t edge = 0;
    AloneEdge alone;
};

// Everything SolidPart reads.
struct SolidPartInput
{
    // The grids by dense index, increasing ID: positions in the basic system, and IDs (for the messages).
    std::vector<SolidPoint> gridPositions;
    std::vector<std::int32_t> gridIds;
    std::vector<SolidPartElement> solids;
    // The declarations of the skin (SkinDeclarations), by grid indices: the edges declared sharp and those declared
    // smooth, the fold angle (radians) that decides the others (empty: they are smooth), and the mesh lines declared.
    std::set<EdgeKey> sharpEdges;
    std::set<EdgeKey> smoothEdges;
    std::optional<double> foldAngle;
    std::map<std::pair<std::size_t, std::size_t>, std::set<std::size_t>> declaredLines;
    // The degrees and the interior fields (defaults: trace 4, face 4, stress 4, three interior fields).
    SolidOptions options;
    // The bytes of kernels the part keeps once built (kernelOf), 0 for none: a hexahedron of degree 4 takes about
    // 40 ms to build and 3 MB to keep. The kernels built once the budget is spent are built again at each call.
    std::size_t kernelCacheBytes = 0;
    // The shells of the model (ties 1, 2 and 4), built on the same grids (the same dense indices), or null when the
    // model has none. Not copied: the ShellPart must outlive the SolidPart and its copies.
    const ShellPart* shells = nullptr;
    // The edges of lines alone (ties 3 and 4); an edge that a shell has is not one.
    std::vector<SolidPartAloneEdge> aloneEdges;
};

// What an edge of the solids follows (src/tdnns/solid.rs::EdgeKind).
struct SolidEdgeTie
{
    enum class Kind
    {
        // Nothing: it carries its own unknowns.
        Own,
        // Edge `localEdge` of shell `shell`, the first shell of the edge (tie 1).
        Shell,
        // An edge of lines alone, SolidPartInput::aloneEdges[aloneEdge]: its lines follow the edge (tie 3).
        Alone,
        // The diagonal of the quadrilateral shell `shell` that covers two faces of the solids, from its corner
        // `lowerCorner` (at the lower grid of the edge) to `higherCorner` (tie 1).
        Diagonal
    };

    Kind kind = Kind::Own;
    std::size_t shell = 0;
    std::size_t localEdge = 0;
    std::size_t lowerCorner = 0;
    std::size_t higherCorner = 0;
    std::size_t aloneEdge = 0;
};

// The shell a face of the solids follows (src/tdnns/solid.rs::FaceKind): `corners[i]` is the corner of the shell at
// grid i of the face (SolidPartFace::grids order) (tie 2).
struct SolidFaceTie
{
    bool coveredByShell = false;
    std::size_t shell = 0;
    std::array<std::size_t, 4> corners{};
};

// An edge of the solids: its grids (lower first), the solids that have it (increasing element ID), whether it bounds
// a face of the skin, its mid-side grid, and the coefficients of its geometry on its edgeFunctionCount(p) functions.
struct SolidPartEdge
{
    EdgeKey grids;
    std::vector<std::size_t> solids;
    bool onSkin = false;
    std::size_t midSideGrid = NO_MID_SIDE_GRID;
    std::vector<SolidPoint> geometry;
    // What the edge is tied to, and whether it is tied (tie.kind is not Own, the reference's EdgeKind other than Own):
    // its geometry is then that of the shell it follows (straight under lines alone) and the supports do not hold it.
    SolidEdgeTie tie;
    bool follows = false;
};

// A face of the solids: its grids in a canonical cyclic order (from its lowest grid towards the lower of its two
// neighbours), the solids that have it (one on the skin, two inside), and the coefficients of its geometry on its
// faceFunctionCount(pf, count) interior functions.
struct SolidPartFace
{
    std::array<std::size_t, 4> grids{};
    std::size_t count = 3;
    std::vector<std::size_t> solids;
    std::vector<SolidPoint> geometry;
    // The shell the face follows, and whether it follows one (tie.coveredByShell): its geometry is then that of the
    // shell and the supports do not hold it.
    SolidFaceTie tie;
    bool follows = false;

    [[nodiscard]] std::span<const std::size_t> view() const { return {grids.data(), count}; }
    [[nodiscard]] bool onSkin() const { return solids.size() == 1; }
};

// An unknown that a tie writes or reads, in the numbering of the part that holds it:
//   Grid           translation `dof` (basic, 0 … 2) of grid `index` (a grid translation of the shells is one too);
//   SolidEntity    DOF `dof` of entity `index` of entities(EntityLayout::Kind::Static) (dofOffset numbering);
//   ShellUnknown   unknown `index` of the ShellPart numbering (an entity DOF of the shells: never a grid component);
//                  `dof` is 0;
//   AloneEdge      DOF `dof` of the edge of lines alone whose SolidPartAloneEdge::edge is `index` (its entity DOFs:
//                  3 l + c the displacement function l along axis c of its frame, 3 (k − 1) + l its twist l).
struct TieUnknown
{
    enum class Kind
    {
        Grid,
        SolidEntity,
        ShellUnknown,
        AloneEdge
    };

    Kind kind = Kind::Grid;
    std::size_t index = 0;
    std::size_t dof = 0;

    friend constexpr auto operator<=>(const TieUnknown&, const TieUnknown&) = default;
};

// An unknown and its coefficient in a tie.
struct TieTerm
{
    TieUnknown unknown;
    double coefficient = 0.0;
};

// A tie: the dependent unknown is Σ coefficient × unknown over the terms (by increasing unknown, no zero coefficient;
// a term may be the dependent of another tie, the chains are the integrator's to resolve). The dependent is a
// SolidEntity (ties 1 and 2), an AloneEdge (ties 3 and 4) or a ShellUnknown (tie 4: a rotation of a shell edge), never
// a Grid.
struct SolidTie
{
    TieUnknown dependent;
    std::vector<TieTerm> terms;
};

// An edge or a face of the skin of the solids with what gives its functions from a displacement field known at
// points (SolidPart::skinTraces): the points of the projection on the edge or the face (basic), and, for the
// function m along the basic axis c, the DOF 3 m + c of `entity`,
//     q(m, c) = Σ_p weights(m, p) u_c(points[p]) + Σ_t terms[m][t].coefficient × (the component c of its unknown),
// the terms taking out what the grids and, on a face, the functions of its edges carry: a Grid unknown (its `dof`
// 0: the translation c of the grid) or a SolidEntity unknown (its `dof` 3 n, the function n of an edge: its DOF
// 3 n + c). The projections are those of the ties with the shells (EdgeProjection, FaceProjection).
struct SolidSkinTrace
{
    std::vector<std::size_t> grids; // the two grids of an edge, or the grids of a face in its cyclic order
    std::size_t entity = 0;
    std::vector<SolidPoint> points;
    Matrix weights; // functions × points
    std::vector<std::vector<TieTerm>> terms;
};

// An entity of the solids: an edge, a face or the interior of one solid, with its DOF count and a position (for the
// geometric ordering of the factorisation).
struct SolidPartEntity
{
    enum class Kind
    {
        Edge,
        Face,
        Interior
    };

    Kind kind = Kind::Edge;
    std::size_t index = 0; // the edge, the face or the solid
    std::size_t dofCount = 0;
    SolidPoint position{};
};

// A local unknown of a solid: the translation of a grid along a basic axis, or a DOF of an entity.
struct SolidLocalUnknown
{
    bool onGrid = true;
    std::size_t index = 0;     // the grid index, or the entity index (entities(kind))
    std::size_t component = 0; // the basic component 0 … 2, or the DOF within the entity
};

// A local unknown and its coefficient in a linear combination.
struct SolidTerm
{
    SolidLocalUnknown unknown;
    double coefficient = 0.0;
};

// A translation component held at a grid (basic component 0 … 2) and its value.
struct SolidHeldComponent
{
    std::size_t grid = 0;
    std::size_t component = 0;
    double value = 0.0;
};

// One term of the map from the held grid values to a derived support: grid, basic component, coefficient.
struct SolidSupportTerm
{
    std::size_t grid = 0;
    std::size_t component = 0;
    double coefficient = 0.0;
};

// A support of an entity DOF derived from the grid supports: the entity (entities(Static)), the DOF within it, its
// value Σ coefficient × held value, and the terms of that map: the reaction r of the DOF adds r × coefficient to each
// grid component of `carriedTo` (the transpose of the map). Empty terms: the map is zero (a straight edge or a flat
// face, whose functions are zero for every affine field).
struct SolidDerivedSupport
{
    std::size_t entity = 0;
    std::size_t dof = 0;
    double value = 0.0;
    std::vector<SolidSupportTerm> carriedTo;
};

// A translation direction held at a grid that is no axis of the basic system (unit vector, basic), and the component
// of the displacement system of the grid that holds it (0 to 2).
struct SolidSkewAxis
{
    std::size_t grid = 0;
    SolidPoint direction{};
    std::size_t component = 0;
};

// A constraint the supports put on the functions of an edge or a face along a direction d that is no basic axis, the
// same at every grid of the entity: Σ_c d_c q_(m, c) = 0 for each function m (`dofs`: the DOFs of the entity and
// their coefficients). The value of a function in a motion of its grids vanishes on a straight edge and a flat face.
struct SolidSupportRow
{
    std::size_t entity = 0;
    std::vector<std::pair<std::size_t, double>> dofs;
};

// The rows of the directions off the basic axes, and the count of edges and faces whose grids hold directions that
// differ (their grids hold them, the entity does not).
struct SolidSupportRows
{
    std::vector<SolidSupportRow> rows;
    std::size_t uncarried = 0;
};

// A problem of the model attached to a solid (its index and element ID).
struct SolidPartProblem
{
    enum class Severity
    {
        Error,  // the element cannot be built as given
        Warning // the element is built, something was chosen for it
    };

    Severity severity = Severity::Error;
    std::size_t solid = 0;
    std::int32_t elementId = 0;
    std::string message;
};

// The stresses of a solid at its centre and at its corners (in the order of the corners), basic axes.
struct SolidPointStresses
{
    VoigtVector centre{};
    std::vector<VoigtVector> corners;
};

// Reference: src/tdnns/solid.rs::Solids (with Topo3, TdSolid, Context, build)
class SolidPart
{
public:
    using TemperatureField = std::function<double(const SolidPoint&)>;
    using ForceField = std::function<SolidPoint(const SolidPoint&)>;

    // The points per direction of the rule of the loads on a face: p + 4 (the reference's self.p + 4).
    static constexpr std::size_t FACE_LOAD_EXTRA_POINTS = 4;

    // Builds the topology, what the edges and faces follow (the shells and the edges of lines alone of the input:
    // build them first), the geometry and the entities. Never throws for a problem of the model: a solid that
    // cannot be read (a corner count that is no shape, a grid out of range or repeated) is left out of everything
    // (usable() false) and its problem recorded.
    // Reference: src/tdnns/solid.rs::build (its topology, edge and face kinds and geometry), solid_defs
    explicit SolidPart(SolidPartInput input);

    // ---- the solids, their edges and faces

    [[nodiscard]] std::size_t solidCount() const { return m_input.solids.size(); }
    [[nodiscard]] const SolidPartElement& solid(std::size_t solid) const { return m_input.solids[solid]; }
    [[nodiscard]] bool usable(std::size_t solid) const { return m_usable[solid]; }
    // Precondition: usable(solid).
    [[nodiscard]] VolumeShape shapeOf(std::size_t solid) const { return m_shapes[solid]; }
    [[nodiscard]] std::size_t traceDegree() const { return m_traceDegree; }
    [[nodiscard]] std::size_t faceDegree() const { return m_faceDegree; }
    [[nodiscard]] const SolidOptions& options() const { return m_input.options; }
    [[nodiscard]] std::span<const SolidPoint> gridPositions() const { return m_input.gridPositions; }

    // The edges by key (sorted pairs of grids), the faces by key (sorted grids).
    [[nodiscard]] std::span<const SolidPartEdge> edges() const { return m_edges; }
    [[nodiscard]] std::span<const SolidPartFace> faces() const { return m_faces; }
    [[nodiscard]] std::optional<std::size_t> edgeIndex(const EdgeKey& key) const;
    // The state the reading of the skin gave each edge of two faces of the skin (SkinReading::states).
    [[nodiscard]] const std::map<EdgeKey, EdgeReading>& skinEdgeStates() const { return m_skinStates; }
    // The face that has the grids `grids` (any three of its grids, or all of them).
    [[nodiscard]] std::optional<std::size_t> faceIndex(std::span<const std::size_t> grids) const;
    // The edges and faces of a usable solid, in the order of shapeEdges and shapeFaces.
    [[nodiscard]] std::span<const std::size_t> solidEdges(std::size_t solid) const { return m_solidEdges[solid]; }
    [[nodiscard]] std::span<const std::size_t> solidFaces(std::size_t solid) const { return m_solidFaces[solid]; }

    // The mid-side grids that fix the curve of an edge (one per such edge, increasing grid index), each with its
    // edge. A mid-side grid carries no unknown: its translation is that of its edge at s = 1/2 (midSideTerms).
    [[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> midSideGrids() const;
    // The translation component c (basic) of the mid-side grid of edge `edge` as a combination of the unknowns of the
    // edge: u(1/2) = (u_a + u_b)/2 + Σ_m f_m(1/2) q_mc, f_m the functions of the edge (edgeFunctions).
    [[nodiscard]] std::vector<SolidTerm> midSideTerms(std::size_t edge, std::size_t component) const;

    // The problems found while building (mid-side grids too far from the middle of their edge, different between
    // elements, or corners of solids, solids that cannot be read).
    [[nodiscard]] std::span<const SolidPartProblem> problems() const { return m_problems; }

    // ---- entities and local unknowns

    // The entities of an analysis of the given kind, in a deterministic order: the edges by key, the faces by key,
    // then (modal only) the interior fields by element ID. The static list is a prefix of the modal one.
    [[nodiscard]] std::span<const SolidPartEntity> entities(EntityLayout::Kind kind) const;
    // The DOFs of the entities before `entity` (the offset of its values in affineField).
    [[nodiscard]] std::size_t dofOffset(std::size_t entity) const { return m_dofOffsets[entity]; }
    // The entity of an edge, a face or the interior fields of a solid; empty when it carries no DOF.
    [[nodiscard]] std::optional<std::size_t> edgeEntity(std::size_t edge) const;
    [[nodiscard]] std::optional<std::size_t> faceEntity(std::size_t face) const;
    [[nodiscard]] std::optional<std::size_t> interiorEntity(std::size_t solid) const;

    // The local unknowns of a usable solid in the order of its kernel (VolumeKernel.hpp: 3 j + c for function j and
    // basic component c: the corners, the edges, the faces), followed in a modal analysis by its interior fields.
    [[nodiscard]] std::vector<SolidLocalUnknown> localUnknowns(std::size_t solid, EntityLayout::Kind kind) const;

    // ---- kernels

    // The geometry of a usable solid: its corners, the coefficients of its edges and faces, its functions. Refused
    // when its functions do not reproduce their traces (residual 1e-8 or more).
    [[nodiscard]] SolidResult<VolumeGeometry> geometryOf(std::size_t solid) const;

    // The kernel of a usable solid for an analysis of the given kind, with its mass when `withMass` (and always for a
    // solid with interior fields in a modal analysis: they come from the mass), with the loads `loads` (their results
    // in kernel.loads(), in the same order). Refused for a folded element, a trace residual of 1e-8 or more, matrices
    // that are not positive definite. Nothing is cached: each call builds the functions of the element (a minimum-norm
    // inverse of its trace matrix, kept per orientation: elementFunctions) and its matrices: at degree 4, native and on
    // one thread, about 0.04 s for a hexahedron, 0.015 s for a pentahedron, a few milliseconds for a tetrahedron or a
    // pyramid (measured November 2026), the mass and the interior fields adding a fifth. kernelOf keeps them.
    [[nodiscard]] SolidResult<VolumeKernel> buildKernel(std::size_t solid, EntityLayout::Kind kind, bool withMass,
                                                        std::span<const SolidLoad> loads = {}) const;

    // The kernel of a usable solid without loads, as buildKernel builds it, kept by the part once built when its
    // budget allows (SolidPartInput::kernelCacheBytes): a kernel with the mass and the interior fields serves the
    // calls that ask for less; a kept kernel without the mass gets it from VolumeKernel::withMass when a call asks for
    // the mass without the interior fields (about a third of a build). Or the reason the solid is refused.
    using SharedKernel = std::shared_ptr<const VolumeKernel>;
    [[nodiscard]] std::variant<SharedKernel, std::string> kernelOf(std::size_t solid, EntityLayout::Kind kind,
                                                                   bool withMass) const;

    // The problem of a usable solid that its kernel finds (a folded element, a trace residual...), as buildKernel
    // finds it for a static analysis without mass: costs one kernel (then kept: kernelOf). Empty when the solid is
    // accepted.
    [[nodiscard]] std::optional<SolidPartProblem> validate(std::size_t solid) const;

    // The stiffness and the mass of a solid on its local unknowns (localUnknowns(solid, kind)): the kernel's on its
    // parameters, and in a modal analysis its interior fields (stiffness λ_i on their diagonal; mass 1 on their
    // diagonal and the coupling with the parameters).
    // Reference: src/tdnns/solid.rs::Solids::assemble (one solid)
    [[nodiscard]] Matrix localStiffness(std::size_t solid, const VolumeKernel& kernel, EntityLayout::Kind kind) const;
    [[nodiscard]] Matrix localMass(std::size_t solid, const VolumeKernel& kernel, EntityLayout::Kind kind) const;

    // ---- loads (the volume loads go to buildKernel; their nodal forces are kernel.loads()[i].force, on the
    // parameters, the first local unknowns)

    // The body force ρ a of a uniform acceleration a (inertia), ρ the density of the solid.
    [[nodiscard]] SolidLoad accelerationLoad(std::size_t solid, const SolidPoint& acceleration) const;
    // A body force per unit volume.
    [[nodiscard]] static SolidLoad bodyForceLoad(ForceField force) { return SolidLoad::body(std::move(force)); }
    // The free expansion α ΔT(x) of a temperature rise field.
    [[nodiscard]] SolidLoad thermalLoad(std::size_t solid, TemperatureField temperatureRise) const;
    // The unit inertia load `mode` of a rigid acceleration (SolidLoad::rigidInertia), for the stresses of an inertia
    // relief.
    [[nodiscard]] SolidLoad rigidInertiaLoad(std::size_t solid, std::size_t mode) const;

    // The local face of a usable solid that has the grids `faceGrids` (three of them, or all); empty if none.
    [[nodiscard]] std::optional<std::size_t> localFaceOf(std::size_t solid,
                                                         std::span<const std::size_t> faceGrids) const;
    // The first solid (by element ID) that has the face of the grids `grids` (any three of its grids), and its local
    // face; empty if none.
    // Reference: src/tdnns/solid.rs::Solids::face_by_grids
    [[nodiscard]] std::optional<std::pair<std::size_t, std::size_t>> solidFaceOf(
        std::span<const std::size_t> grids) const;

    // The loads on the parameters of a traction (force per area), a stress acting on the face, a pressure along the
    // inward normal (position fields), on local face `localFace`, rule of p + 4 points per direction.
    // Reference: src/tdnns/solid.rs::Solids::loads (the faces)
    [[nodiscard]] Vector faceTractionLoad(const VolumeKernel& kernel, std::size_t localFace,
                                          const VolumeKernel::TractionField& traction) const;
    [[nodiscard]] Vector faceStressLoad(const VolumeKernel& kernel, std::size_t localFace,
                                        const VolumeKernel::StressField& stress) const;
    [[nodiscard]] Vector facePressureLoad(const VolumeKernel& kernel, std::size_t localFace,
                                          const VolumeKernel::PressureField& pressure) const;
    // The same for a pressure given at the corners of the face (`cornerGrids`, `cornerPressures`: the face pressure
    // of the input format) and interpolated over the face coordinates (faceWeights), along the inward normal.
    [[nodiscard]] Vector faceCornerPressureLoad(const VolumeKernel& kernel, std::size_t solid, std::size_t localFace,
                                                std::span<const std::size_t> cornerGrids,
                                                std::span<const double> cornerPressures) const;
    // The load stiffness of a pressure that follows the face (symmetric part; added to the geometric stiffness).
    // Reference: src/tdnns/solid.rs::Solids::follower (one face)
    [[nodiscard]] Matrix followerPressureStiffness(const VolumeKernel& kernel, std::size_t localFace,
                                                   const VolumeKernel::PressureField& pressure) const;

    // ---- stresses

    // The stress coefficients for the local unknown values `local` (localUnknowns order; the interior fields of a
    // modal analysis do not enter) and the particular stresses of the kernel's loads times `loadFactors` (one per
    // kernel load, or empty): β = stressOperator q + Σ_l factor_l β_l. A load case passes 1 for its loads; an inertia
    // relief passes −a_i for the six rigidInertiaLoad loads.
    // Reference: src/tdnns/solid.rs::Solids::beta, src/tdnns/analysis.rs::solid_stress_relief_at
    [[nodiscard]] static Vector stressCoefficients(const VolumeKernel& kernel, std::span<const double> local,
                                                   std::span<const double> loadFactors);
    // The stresses at the centre and the corners of the solid (basic axes).
    // Reference: src/tdnns/analysis.rs::solid_stress_at (at Shape::centre and Shape::vertices)
    [[nodiscard]] static SolidPointStresses pointStresses(const VolumeKernel& kernel, std::span<const double> beta);
    // A stress of a solid in its material axes: σ' = Aᵀ σ A, A the material axes (columns).
    [[nodiscard]] VoigtVector toMaterialAxes(std::size_t solid, const VoigtVector& basic) const;
    // The stresses at the grids: at each grid, the mean of the corner stresses of the solids that have it
    // (`cornerStresses[solid]`, in the order of its corners; an empty list leaves the solid out). Empty at a grid
    // that none has.
    // Reference: src/tdnns/analysis.rs::solid_grid_stresses
    [[nodiscard]] std::vector<std::optional<VoigtVector>> gridStresses(
        std::span<const std::vector<VoigtVector>> cornerStresses) const;
    // The geometric stiffness on the parameters for the stresses of a preload (stressCoefficients).
    // Reference: src/tdnns/solid.rs::Solids::geometric (one solid)
    [[nodiscard]] static SolidResult<Matrix> geometricStiffness(const VolumeKernel& kernel,
                                                                std::span<const double> beta);

    // ---- supports and affine fields

    // The supports of the entity DOFs derived from the held grid translations `held` (section 5.6 of the
    // specification): an edge of the skin whose two grids hold component c holds its functions along c, a face of the
    // skin whose grids all hold c holds its functions along c. The value of each is the image of the held grid values
    // by a linear map exact for the rigid motions (and the uniform strains along the face or edge) that the held
    // values see; see SolidPartSupports.cpp.
    // Reference: src/tdnns/solid.rs::Solids::supports (edge_bc), with values and reaction maps
    // `skew`: the translation directions held off the basic axes; an edge or a face whose grids all hold the same one
    // gives the rows of `rows` (when not null).
    [[nodiscard]] std::vector<SolidDerivedSupport> deriveSupports(std::span<const SolidHeldComponent> held,
                                                                  std::span<const SolidSkewAxis> skew = {},
                                                                  SolidSupportRows* rows = nullptr) const;

    // The values on the entity DOFs (entities(kind) order, dofOffset) of the affine field u(x) = b + A x, A =
    // `gradient` (∂u_i/∂x_j in row i): A times the geometry coefficients of each edge and face function (the corner
    // functions are a partition of unity, so b goes to the grids alone); zero on the interior fields.
    // Reference: src/tdnns/solid.rs::Solids::rigid (generalised to any affine field)
    [[nodiscard]] std::vector<double> affineField(const base::Mat3& gradient, EntityLayout::Kind kind) const;

    // ---- ties with the shells and the lines alone (see the head of this file)

    // The ties of the solids with the shells and the lines alone, as rows: per tied unknown, the combination of the
    // others it equals. In this order: the edges by key (tie 1, then tie 3 for an edge of lines alone), the faces by
    // key (tie 2), the rotations about the edges by key (tie 4). A dependent that `isHeld` says is held (a support) is
    // left out, and the first row of a dependent wins. Nothing without shells and edges of lines alone. Tie 4 builds
    // the kernels of the solids around its edges (kernelOf); a solid whose kernel is refused is left out of the mean.
    // The ties of a shell that stands on the solids read their skin over its thickness: they build the kernels of the
    // solids whose faces hold its fibre.
    // Reference: src/tdnns/solid.rs::Solids::ties
    [[nodiscard]] std::vector<SolidTie> ties(const std::function<bool(const TieUnknown&)>& isHeld = {}) const;

    // The edges and the faces of the skin whose grids are all among `grids` (increasing), with what gives their
    // functions from a displacement field known at their points (SolidSkinTrace): a glued contact makes them follow
    // the face they lie on. The edges first, then the faces, by key; an edge or a face that follows a shell or a line
    // is left to its tie, and a face one of whose edges carries no functions is left out.
    [[nodiscard]] std::vector<SolidSkinTrace> skinTraces(std::span<const std::size_t> grids) const;

private:
    // A row of a tie being written: coefficient by unknown.
    using TieRow = std::map<TieUnknown, double>;
    struct PendingTie
    {
        TieUnknown dependent;
        TieRow row;
    };

    void readSolids();
    void buildTopology();
    void findTies();
    void findEdgeTies(const ShellPart* shells);
    void coverFaces(std::size_t shell);
    void followDiagonal(std::size_t face);
    void buildGeometry();
    void readSkin(const EdgeProjection& projection);
    void followTies(const EdgeProjection& projection);
    void buildEntities();
    // The rows of the ties of one edge (ties 1 and 3), one face (tie 2) and the rotation about one edge (tie 4).
    void addEdgeTies(std::size_t edge, const EdgeProjection& projection, SkinFaces& skin,
                     std::vector<PendingTie>& out) const;
    // Tie 1 of an edge whose shells stand on the solids, over their thickness; false where it does not apply.
    [[nodiscard]] bool addFibreEdgeTies(std::size_t edge, SkinFaces& skin, std::vector<PendingTie>& out) const;
    // Whether a shell of edge `edge` lies on a face of the solids.
    [[nodiscard]] bool liesOnFace(std::size_t shell, const SolidPartEdge& edge) const;
    void addAloneEdgeTies(std::size_t edge, std::vector<PendingTie>& out) const;
    void addFaceTies(std::size_t face, std::vector<PendingTie>& out) const;
    void addRotationTies(std::size_t edge, SkinFaces& skin, std::vector<PendingTie>& out) const;
    // The rows of the shell displacement at points of shell `shell` (all its functions, or those of its corners
    // `lowerCorner`, `higherCorner` and of its edge `localEdge` only when `onEdge`), and the points of its surface.
    [[nodiscard]] std::vector<std::array<TieRow, 3>> shellRows(std::size_t shell,
                                                               std::span<const ReferencePoint> points,
                                                               const std::optional<std::size_t>& onEdge) const;
    [[nodiscard]] std::vector<SolidPoint> shellPoints(std::size_t shell, std::span<const ReferencePoint> points) const;
    // The reference points of the shell an edge or a face follows, at the parameters s of the edge (from its lower
    // grid) or at the coordinates c of the face (faceRule, in the order of SolidPartFace::grids).
    [[nodiscard]] std::vector<ReferencePoint> followedEdgePoints(std::size_t edge,
                                                                 std::span<const double> parameters) const;
    [[nodiscard]] std::vector<ReferencePoint> followedFacePoints(std::size_t face,
                                                                 std::span<const std::array<double, 2>> points) const;
    void addProblem(std::size_t solid, SolidPartProblem::Severity severity, std::string message);
    [[nodiscard]] std::vector<base::EntityId> orderOf(std::size_t solid) const;
    // The curves of the skin and of the shells are taken with traces and faces of degree 4 only, as the reference
    // does (`quartic` in solid.rs::build); below, the edges and faces are straight but for the mid-side grids.
    [[nodiscard]] bool takesCurves() const;

    // The kernels kept once built (kernelOf), by solid, with what each holds and the bytes they take; guarded by its
    // mutex (the part is shared by the threads of the assembly). Shared by the copies of the part.
    struct KernelCache
    {
        std::mutex mutex;
        std::vector<SharedKernel> kernels;
        std::vector<bool> hasMass;
        std::vector<bool> hasInterior;
        std::vector<std::size_t> entryBytes;
        std::size_t bytes = 0;
    };

    SolidPartInput m_input;
    std::shared_ptr<KernelCache> m_kernelCache = std::make_shared<KernelCache>();
    std::size_t m_traceDegree = 0;
    std::size_t m_faceDegree = 0;
    std::vector<bool> m_usable;
    std::vector<VolumeShape> m_shapes;
    std::vector<SolidMaterial> m_basicMaterials; // the law in the basic axes
    std::vector<std::size_t> m_byElementId;      // usable solids by increasing element ID
    std::vector<SolidPartEdge> m_edges;
    std::vector<SolidPartFace> m_faces;
    std::map<EdgeKey, std::size_t> m_edgeIndex;
    std::map<EdgeKey, EdgeReading> m_skinStates; // readSkin
    std::map<std::vector<std::size_t>, std::size_t> m_faceIndex; // by sorted grids
    std::map<std::array<std::size_t, 3>, std::size_t> m_faceByThree;
    std::vector<std::vector<std::size_t>> m_solidEdges;
    std::vector<std::vector<std::size_t>> m_solidFaces;
    std::vector<SolidPartEntity> m_entities;
    std::size_t m_staticEntityCount = 0;
    std::vector<std::size_t> m_dofOffsets;
    std::vector<std::optional<std::size_t>> m_edgeEntities;
    std::vector<std::optional<std::size_t>> m_faceEntities;
    std::vector<std::optional<std::size_t>> m_interiorEntities;
    std::vector<SolidPartProblem> m_problems;
};

} // namespace oos::elements::mixedHybrid
