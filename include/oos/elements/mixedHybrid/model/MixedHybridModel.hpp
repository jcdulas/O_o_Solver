#pragma once

#include "oos/base/diagnostics/Diagnostic.hpp"
#include "oos/elements/bars_beams/BeamFormulation.hpp"
#include "oos/elements/core/ElementTemperatures.hpp"
#include "oos/elements/core/FormulationModel.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"
#include "oos/elements/mixedHybrid/line/LineAnalysis.hpp"
#include "oos/elements/mixedHybrid/line/LineKinematics.hpp"
#include "oos/elements/mixedHybrid/line/LineSection.hpp"
#include "oos/elements/mixedHybrid/model/ShellPart.hpp"
#include "oos/elements/mixedHybrid/model/SolidPart.hpp"
#include "oos/elements/rods/RodFormulation.hpp"
#include "oos/elements/shells/ShellSection.hpp"
#include "oos/elements/solids/SolidSection.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace oos::elements::mixedHybrid
{

// The degree k of the displacement of the formulation (docs/mixed-hybrid-formulation-spec.md section 15, decision
// 7): 4 unless the model parameter MixedHybridDegree gives 2 or 3. Degree 4 is that of the reference and of its
// validation. A lower degree costs less on the same mesh (an element of degree 3 about half, one of degree 2 about a
// fifth) and is less accurate on a coarse one; the shells, the lines and the solids of a model take it together, so
// that the functions of the edges they share are the same. Below 4 the skin of the solids is read with straight
// edges between its grid points (SolidPart::takesCurves).
inline constexpr std::size_t MIXED_HYBRID_DEFAULT_DEGREE = 4;
inline constexpr std::size_t MIXED_HYBRID_LOWEST_DEGREE = 2;
inline constexpr std::size_t MIXED_HYBRID_HIGHEST_DEGREE = 4;
inline constexpr std::string_view MIXED_HYBRID_DEGREE_PARAMETER = "MixedHybridDegree";
// The value of that parameter that gives each shell the degree 3 or 4 its shape asks (ShellDegreeChoice.hpp): the
// lowest on the elements of a regular mesh, higher ones on those a taper, a warp or a curvature takes away from an
// affine map. The lines alone and the solids then keep the default degree, and so do the shells that share an edge with
// a solid.
inline constexpr std::string_view MIXED_HYBRID_AUTOMATIC_DEGREE = "Auto";

// A component released at an end of a line (the pin flags PA, PB of a bar or beam, section 4.2): the line's own
// unknown there instead of the grid point's, DOF `index` of the entity `entity` (the end of the line), along the unit
// `direction` of the element system (in basic).
struct LineRelease
{
    std::size_t end = 0; // 0 grid A, 1 grid B
    bool isRotation = false;
    std::array<double, 3> direction{};
    std::int32_t entity = -1;
    int index = 0;
};

// A line element of the formulation (Rod2, Rod2InlineSection, Tube2, Bar2, Beam2): its card, batch row and ID, its
// grid points (node table rows, A then B), what the line kernel reads of it (the grids as indices of the model's
// grids), the data its card's other family reads (for the rows of its output tables), and its edge.
struct MixedLine
{
    std::string_view card; // one of MIXED_HYBRID_CARDS (static storage)
    std::int64_t batchRow = 0;
    std::int32_t elementId = 0;
    std::array<std::int32_t, 2> gridRows{};
    LineElement element;
    LineSection section;
    std::variant<RodData, BeamData> classic;
    std::int32_t edge = -1;
    bool forward = true; // grid A is the lower grid of the edge
    std::vector<LineRelease> releases;
    // On an edge of shells: its index among the lines of the ShellPart (−1 for a line alone).
    std::int32_t shellLine = -1;
};

// A shell element of the formulation (ShellQuad4, ShellTria3, ShellQuad4Drilling, ShellTria3Drilling, ShellQuad8,
// ShellTria6): its card, batch row and ID, its corner and mid-side grid points (node table rows, −1 where absent), the
// section its card's other families read (for the fibres and plies of the output rows), and its index in the
// ShellPart (the shells by increasing element ID).
struct MixedShell
{
    std::string_view card; // one of MIXED_HYBRID_CARDS (static storage)
    std::int64_t batchRow = 0;
    std::int32_t elementId = 0;
    std::size_t cornerCount = 4;
    std::array<std::int32_t, 4> gridRows{};
    std::array<std::int32_t, 4> midRows{-1, -1, -1, -1};
    elements::ShellSection classicSection;
    ShellPartShell part; // what the ShellPart reads (its grids filled with the model's grid indices)
    std::size_t partIndex = 0;
};

// A solid element of the formulation (SolidTetra, SolidPyramid, SolidPenta, SolidHexa): its card, batch row and ID, its
// corner and mid-side grid points (node table rows in the order of the card, −1 for an absent mid-side node), the
// section its card's other families read (the material system CORDM of the output rows, the expansion), whether its
// expansion is the same along every direction (the kernel takes one coefficient), and what the SolidPart reads (its
// grids filled with the model's grid indices) with its index there (the solids by increasing element ID).
struct MixedSolid
{
    std::string_view card; // one of MIXED_HYBRID_CARDS (static storage)
    std::int64_t batchRow = 0;
    std::int32_t elementId = 0;
    std::vector<std::int32_t> cornerRows;
    std::vector<std::int32_t> midRows; // one per edge of the shape (empty: none at all)
    elements::SolidSection classicSection;
    bool isotropicExpansion = true;
    SolidPartElement part;
    std::size_t partIndex = 0;
};

// An edge of the formulation: its two grid points (indices of the model's grids, the lower grid point ID first, so
// that the orientation of the edge does not depend on the element order or on the grid order of an element), the
// lines on it (by increasing element ID: the first gives the frame of an edge alone), the kernel's edge of lines
// alone, and its entity.
struct MixedEdge
{
    std::array<std::int32_t, 2> grids{};
    std::vector<std::int32_t> lines;
    std::optional<AloneEdge> alone;
    std::int32_t entity = -1; // its row in the layouts (the same in both)
};

// The model of the mixed hybrid formulation of one snapshot (docs/mixed-hybrid-formulation-spec.md section 5.2): what
// its element families share, built once per snapshot index. It holds the elements of every card in the formulation,
// the grid points they use (numbered by increasing ID), their edges with their owners, the entities of each kind of
// analysis, the ties, and the problems found while building it (each attached to the element it concerns, which its
// family reports in check(), or to the model as a whole, reported by reportProblems()).
//
// Thread-safety: immutable once built; concurrent reads are safe. Ownership: built by of() into the snapshot index,
// whose lifetime it shares; it holds the node table rows of the snapshot, not its tables.
class MixedHybridModel final : public FormulationModel
{
public:
    // The model of the snapshot of `index`, built on the first call (with the threads of `pool`, when given: the
    // same model). Must not be called from inside another SnapshotIndex::computedOnce build of the same index.
    [[nodiscard]] static const MixedHybridModel& of(const SnapshotIndex& index, base::ThreadPool* pool = nullptr);

    // An analysis may initialise its own fresh index with a cubic pilot or a frozen enrichment.
    // Solid interfaces retain their full trace degree in both passes. `pool`: the threads that validate the solids.
    explicit MixedHybridModel(const SnapshotIndex& index, const std::set<std::int32_t>* quarticShells = nullptr,
                              base::ThreadPool* pool = nullptr);

    // FormulationModel.
    [[nodiscard]] const EntityLayout& layout(EntityLayout::Kind kind) const override;
    void reportProblems(base::DiagnosticSink& sink) const override;
    void ties(EntityLayout::Kind kind, std::span<const HeldGridComponent> held, std::vector<ElementDof>& dependents,
              std::vector<std::size_t>& offsets, std::vector<FormulationRelationTerm>& terms,
              base::DiagnosticSink& sink) const override;
    void determinedGridComponents(EntityLayout::Kind kind, std::vector<HeldGridComponent>& components) const override;
    void determinedRow(EntityLayout::Kind kind, std::int32_t gridRow, int component,
                       std::vector<FormulationRelationTerm>& terms) const override;
    void deriveSupports(EntityLayout::Kind kind, std::span<const HeldGridComponent> held,
                        std::vector<DerivedSupport>& supports, base::DiagnosticSink& sink) const override;
    void supportRows(EntityLayout::Kind kind, std::span<const HeldGridComponent> held,
                     std::vector<std::vector<FormulationRelationTerm>>& rows) const override;
    [[nodiscard]] std::optional<base::Vec3> commonShellNormal(std::int32_t gridRow) const override;
    void shellEdges(std::vector<std::array<std::int32_t, 2>>& edges) const override;
    void affineField(EntityLayout::Kind kind, const base::Mat3& gradient, const base::Vec3& translation,
                     std::span<double> entityValues) const override;

    void fillDeterminedComponents(EntityLayout::Kind kind, const NodalDisplacements& displacements,
                                  const std::function<void(std::int32_t, int, double)>& write) const override;

    // The line element of a batch row of a line card, or null when it could not be read (its problem is in
    // elementProblems()).
    [[nodiscard]] const MixedLine* lineOf(std::string_view card, std::int64_t batchRow) const;
    // The shell element of a batch row of a shell card, or null when it could not be read.
    [[nodiscard]] const MixedShell* shellOf(std::string_view card, std::int64_t batchRow) const;
    // The shells of the model (null without shells, or when their reading failed: the problems say why).
    [[nodiscard]] const ShellPart* shellPart() const noexcept { return m_shellPart.get(); }
    // The solid element of a batch row of a solid card, or null when it could not be read.
    [[nodiscard]] const MixedSolid* solidOf(std::string_view card, std::int64_t batchRow) const;
    // The solids of the model (null without solids).
    [[nodiscard]] const SolidPart* solidPart() const noexcept { return m_solidPart.get(); }
    [[nodiscard]] const MixedEdge& edge(std::int32_t index) const { return m_edges[static_cast<std::size_t>(index)]; }
    // The node table row of a grid of the model.
    [[nodiscard]] std::int32_t gridRow(std::size_t gridIndex) const { return m_gridRows[gridIndex]; }
    // The problems of the elements of a card, which its family reports in check().
    [[nodiscard]] std::vector<base::Diagnostic> elementProblems(std::string_view card) const;

    // The DOFs of the local unknowns of a line alone (the order of the line kernel: aloneLineUnknowns), and for each
    // the grid point whose CD system its component is in the basic system of (−1 for an entity DOF): the family
    // turns the grid blocks into the CD systems.
    struct LocalDofs
    {
        std::vector<ElementDof> dofs;
        std::vector<std::int32_t> basicGridRows;
        // The sign of each local unknown (−1 where an element sees an unknown of its edge in the other direction).
        std::vector<double> signs;
    };
    [[nodiscard]] LocalDofs aloneLineDofs(const MixedLine& line) const;
    // The DOFs of unknowns of the ShellPart's numbering (a shell's localUnknowns, a line's LineData::dofs) with their
    // signs.
    [[nodiscard]] LocalDofs partDofs(std::span<const std::size_t> unknowns, std::span<const double> signs) const;
    // The DOFs of the local unknowns of a solid in an analysis of the given kind (SolidPart::localUnknowns: the grid
    // translations in the basic system, the entity DOFs, the interior fields of a modal analysis).
    [[nodiscard]] LocalDofs solidDofs(const MixedSolid& solid, EntityLayout::Kind kind) const;
    // Corner temperature changes, followed by edge values when any mid-side grid has thermal data.
    // Missing edge data then inherit the corresponding corner field.
    [[nodiscard]] std::vector<double> solidNodalChanges(const MixedSolid& solid,
                                                         const ElementTemperatures* temperatures) const;
    // Whether a solid takes a thermal strain from its nodal changes (an expansion and a change that is not zero).
    [[nodiscard]] static bool takesThermalStrain(const MixedSolid& solid, std::span<const double> changes);
    // The pressure over a face of a solid given at the grids of the face (`faceGrids`, grid indices, with
    // `cornerPressures`): at a position, the pressures of the corners interpolated by the functions of the corners of
    // the straight element at the point nearest to it, so that a pressure linear in the
    // position is read as such whatever the curves the skin reads between the grids.
    [[nodiscard]] VolumeKernel::PressureField solidFacePressure(const MixedSolid& solid,
                                                                std::span<const std::size_t> faceGrids,
                                                                std::span<const double> cornerPressures) const;
    // The kernel with the thermal expansion of the case as its only load.
    [[nodiscard]] VolumeKernel solidThermalKernel(const MixedSolid& solid, EntityLayout::Kind kind,
                                                  std::span<const double> changes) const;
    // The same nodal temperature field drives the thermal load and strain recovery.
    [[nodiscard]] SolidPart::TemperatureField solidTemperatureField(const MixedSolid& solid,
                                                                  std::span<const double> changes) const;
    [[nodiscard]] SolidLoad solidThermalLoad(const MixedSolid& solid, std::span<const double> changes) const;
    // Adds to the stress coefficients `beta` of a solid the particular stresses of the body force ρ a(x) of the
    // acceleration field of a case (NodalDisplacements::bodyAcceleration; nothing without a field or without
    // density), from its kernel.

    // Adds the particular stresses of an affine body force density times acceleration.
    static void addBodyForceStresses(double density, const BodyAcceleration* acceleration, const VolumeKernel& kernel,
                                     Vector& beta);
    // The stress coefficients of a solid for the displacements of a case, with the kernel they are of (the kept one,
    // or one with the free expansion of the case's temperatures as its load) and the nodal changes (empty without a
    // thermal strain).
    struct SolidCaseStresses
    {
        std::shared_ptr<const VolumeKernel> kernel;
        Vector beta;
        std::vector<double> nodalChanges;
    };
    [[nodiscard]] SolidCaseStresses solidStresses(const MixedSolid& solid, EntityLayout::Kind kind,
                                                  const NodalDisplacements& displacements) const;
    void gridPointStresses(EntityLayout::Kind kind, const NodalDisplacements& displacements,
                           const SolidSelection& takes, std::vector<GridPointStress>& stresses) const override;
    [[nodiscard]] std::optional<FormulationFace> faceOf(std::span<const std::int32_t> gridRows) const override;
    // The forces at the grid points of the outline of the shells read as tractions (ShellPart::forcesAlongEdges).
    [[nodiscard]] EdgeForceReading readForcesAlongEdges(std::span<const GridForce> forces) const override;
    // The grids of the elements of two rings around the element's grids (describeSurroundings).
    [[nodiscard]] std::vector<std::int32_t> readGridRows(std::string_view card, std::int64_t batchRow) const override;
    // From the name of the entity: the element of an interior or a line end, the elements at its first grid point that
    // have its other grid points.
    [[nodiscard]] std::vector<std::int32_t> entityElements(EntityLayout::Kind kind, std::int32_t row) const override;
    [[nodiscard]] std::vector<std::int32_t> entityGridRows(EntityLayout::Kind kind, std::int32_t row) const override;
    // The elements of two rings around the grid points (those that have one of them, then those that have a grid
    // point of the first ring: the normals at a grid point are read from the elements there, the curve of an edge from
    // the normals at its ends), each with its kind, ID, card and the IDs and positions of its grid points, then the
    // geometry declarations of the model.
    void describeSurroundings(std::span<const std::int32_t> gridRows,
                              const std::function<void(double)>& mix) const override;
    // The motion of a point of a shell pierced by a fastener or a weld (section 5.10): the mean, over the samples of
    // the footprint of the fastener, of the displacement and rotation of the shell (ShellPart::pointRows) carried
    // rigidly to the piercing point; each sample is placed on its shell by the shape functions of its card (corner
    // grids at the reference corners, mid-side grids at the middles of their edges). A point load on a shell is
    // singular (its displacement in the plane and its rotation under a moment grow without bound as the element
    // resolves them), so the motion at the point alone depends on the mesh and on where the point lies in it; over
    // the footprint of the fastener it converges.
    [[nodiscard]] std::optional<PointMotion> shellPointMotion(std::int32_t elementId, const base::Vec3& point,
                                                              std::span<const ShellSample> samples) const override;
    // The motion of a point of a face of a shell or of a solid of the formulation (faceOf: the face of its
    // corners), placed by the shape functions of the face: of the shell at its reference point (its translations,
    // its rotation and the point of its surface, ShellPart::pointRowsOnEdges), of the solid at the point of its
    // reference shape (its translations: the functions of its kernel there; its interior fields vanish on a face).
    [[nodiscard]] std::optional<PointMotion> facePointMotion(std::span<const std::int32_t> faceGridRows,
                                                             std::span<const double> shapeValues) const override;
    // The edges of the shells between two of the grid points `gridRows`, each with the points of its rule and its
    // displacement and rotation unknowns as moments of the motion of those points (ShellPart::edgeMoments); then
    // the edges and faces of the skin of the solids whose grid points are all among them, each with the points of
    // its projection and its functions as projections of the motion of those points (SolidPart::skinTraces).
    void boundaryTraces(std::span<const std::int32_t> gridRows, std::vector<BoundaryTrace>& traces) const override;
    // The CD-to-basic rotation of a grid of the model.
    [[nodiscard]] const base::Mat3& gridFrame(std::size_t gridIndex) const { return m_gridFrames[gridIndex]; }
    // The index of a grid of the model from its node table row (−1 when no element of the formulation uses it).
    [[nodiscard]] std::int32_t gridIndexOfRow(std::int32_t row) const;
    [[nodiscard]] std::vector<SurfaceEdgeReading> surfaceEdges() const override { return m_surfaceEdges; }

    // The degree k of the displacement of this model (MIXED_HYBRID_DEGREE_PARAMETER).
    [[nodiscard]] std::size_t degree() const noexcept { return m_degree; }
    // Whether each shell takes the degree its shape asks, up to degree() (MIXED_HYBRID_AUTOMATIC_DEGREE).
    [[nodiscard]] bool choosesShellDegrees() const noexcept { return m_choosesShellDegrees; }

private:
    // The degree of the model, from its parameter; a value that is neither a whole number of
    // [MIXED_HYBRID_LOWEST_DEGREE, MIXED_HYBRID_HIGHEST_DEGREE] nor MIXED_HYBRID_AUTOMATIC_DEGREE is a problem of the
    // model, which then keeps the default degree.
    void readDegree(const SnapshotIndex& index);
    // The geometry declarations of the model (section 4.4): those of the program, with those of the deck (the
    // SurfaceEdge, SurfaceFacet and SurfaceLine cards) added where the program does not name the same edge or the
    // same mesh line; an empty declaration when the deck says only that it was prepared (the GeometryDeclarations
    // parameter); none when neither gives any.
    void readDeclarations(const SnapshotIndex& index);
    // The problems of the model as a whole (section 4.4): no geometry declaration, or declarations that give one
    // edge side two states.
    void checkDeclarations(const SnapshotIndex& index);
    // The index of the grid of the model of a grid point ID (−1 when no element of the formulation uses it).
    [[nodiscard]] std::int32_t gridIndexOfId(const SnapshotIndex& index, std::int32_t id) const;
    // The fold angle of the model and the declarations of the smooth and sharp edges and of the mesh lines, given to
    // the reading of the shells where they name edges of shells.
    void declareShellGeometry(const SnapshotIndex& index, ShellPartInput& input);
    // The same given to the reading of the skin of the solids.
    void declareSkinGeometry(const SnapshotIndex& index, SolidPartInput& input) const;
    // Marks the declarations that name edges of the solids (once the SolidPart is built).
    void markSkinDeclarationsUsed(const SnapshotIndex& index);
    // The declarations that name no edge of the shells nor of the solids, as problems of the model.
    void reportUnknownDeclarations();
    // The line elements of every line card of the snapshot.
    void readLines(const SnapshotIndex& index);
    // The model's grids (every grid point an element uses, by increasing ID) and the edges of the lines.
    void buildGridsAndEdges(const SnapshotIndex& index);
    // The kernel's edges of lines alone (before the solids, which tie to them).
    void buildAloneEdges(const SnapshotIndex& index);
    // The entities of the edges of lines alone and of the ends of lines with pin flags.
    void buildEntities(const SnapshotIndex& index);
    // The ties of the solids with the shells and the lines alone (SolidPart::ties) as relations on ElementDofs.
    void solidTies(std::span<const HeldGridComponent> held, std::vector<ElementDof>& dependents,
                   std::vector<std::size_t>& offsets, std::vector<FormulationRelationTerm>& terms,
                   base::DiagnosticSink& sink) const;
    // The ElementDofs of an unknown of a tie, with their coefficients (a grid translation in the basic system is a
    // combination of the CD components).
    void appendTieTerms(const TieUnknown& unknown, double factor, std::vector<FormulationRelationTerm>& terms) const;
    // The edges and faces of the skin of the solids whose grid points are all among `gridRows` (boundaryTraces).
    void solidBoundaryTraces(std::span<const std::int32_t> gridRows, std::vector<BoundaryTrace>& traces) const;
    // The shell elements of every shell card of the snapshot.
    void readShells(const SnapshotIndex& index);
    // The ShellPart of the shells (and of the lines on their edges), and its entities.
    void buildShellPart(const SnapshotIndex& index, const std::set<std::int32_t>* quarticShells);
    // The edges of the shells between two grid points of the solids (a superset of those the solids share).
    [[nodiscard]] std::set<EdgeKey> shellEdgesOnSolids() const;
    // The geometry report of the shells (spec section 10.5), as diagnostics of the model: its summary, and the edges
    // named, warnings where the reading takes as smooth an edge that kinks or an outline that turns.
    void reportGeometry();
    // The largest angle along each edge of the shells between the normal of the surface read and the normal of the
    // plane of the corners of a shell of the edge (the rounding of the reading).
    [[nodiscard]] std::map<EdgeKey, double> edgeRoundings() const;
    // The report of the skin of the solids: its edges in the table of the edges read, a summary, and the edges that
    // the fold angle alone made sharp with an angle near it (a coarse curved skin, or a shallow fold).
    void reportSkinGeometry();
    // The solid elements of every solid card of the snapshot.
    void readSolids(const SnapshotIndex& index);
    // The SolidPart of the solids, its entities in both layouts (the interior fields of a modal analysis apart:
    // addModalInteriors), the grid points that only solids have, and their mid-side grid points.
    void buildSolidPart(const SnapshotIndex& index, base::ThreadPool* pool);
    // The interior fields of the solids, in the modal layout only, after every other entity (so that the rows of the
    // other entities are the same in both layouts).
    void addModalInteriors(const SnapshotIndex& index);
    // m_elementsAtGrid, for describeSurroundings.
    void indexElementsByGrid();
    // The elements of `rings` rings around the grids `grids` (indices), and the grids they have, `grids` included.
    void collectRings(std::set<std::int32_t>& grids, std::set<std::pair<std::uint8_t, std::size_t>>& elements,
                      int rings) const;
    // The grid indices of element `element` of kind `kind` (ElementKind), its mid-side grids included.
    [[nodiscard]] std::vector<std::int32_t> gridsOfElement(std::uint8_t kind, std::size_t element) const;
    // The entity DOFs of the membrane-only and bending-only shells that no element stiffens
    // (ShellPart::unstiffenedUnknowns; the solids stiffen the entities they follow), added to m_unstiffenedEntityDofs,
    // and an information on the singular displacement directions held.
    void holdUnstiffenedShellDofs();
    // The directions of the rotation of the grid points of shells whose line elements all lie on shell edges that
    // those lines do not stiffen (m_unstiffenedLineRotations): a rod has no bending stiffness, a bar may bend in one
    // plane only. The rotation of the grid point follows the shells in those directions (ties()).
    void findUnstiffenedLineRotations();
    // The terms of a row of ShellPart unknowns carried to ElementDofs, the grid components to the CD systems.
    void appendPartTerms(std::span<const ShellPartTerm> partTerms, double factor,
                         std::vector<FormulationRelationTerm>& terms) const;
    // The same for a row of SolidPart local unknowns (grid translations in the basic system, entity DOFs).
    void appendSolidTerms(std::span<const SolidTerm> solidTerms, double factor,
                          std::vector<FormulationRelationTerm>& terms) const;

    // The parts of the FormulationModel functions that the shells and the solids answer.
    void shellDeterminedComponents(std::vector<HeldGridComponent>& components) const;
    void shellDeterminedRow(std::int32_t grid, int component, std::vector<FormulationRelationTerm>& terms) const;
    void fillShellComponents(const NodalDisplacements& displacements,
                             const std::function<void(std::int32_t, int, double)>& write) const;
    void solidDeterminedComponents(std::vector<HeldGridComponent>& components) const;
    void solidDeterminedRow(std::int32_t grid, int component, std::vector<FormulationRelationTerm>& terms) const;
    void fillSolidComponents(const NodalDisplacements& displacements,
                             const std::function<void(std::int32_t, int, double)>& write) const;
    // The components held at each grid of the model in the basic system (section 5.6): a basic axis is held when it
    // lies in the span of the held CD axes; `values` the held translation and rotation as basic vectors; `isHeldInCd`
    // the components held in the displacement system; `skew` the held CD axes that are no held basic axis (their
    // direction in the basic system, their component 0 to 5).
    struct BasicSupport
    {
        std::array<bool, 6> isHeld{};
        std::array<double, 6> values{};
        std::array<bool, 6> isHeldInCd{};
        std::vector<ShellSkewAxis> skew;
        // A component of the grid point follows a relation of the model (HeldGridComponent::followsRelation).
        bool followsRelation = false;
    };
    [[nodiscard]] std::map<std::int32_t, BasicSupport> basicSupportsOf(std::span<const HeldGridComponent> held) const;
    // The supports of the entity DOFs of the shells (`supports`) and the rows of the directions off the basic axes
    // (`rows`), either null when not asked; a warning counts the edges whose grids hold different directions.
    void shellSupports(const std::map<std::int32_t, BasicSupport>& basic, std::vector<DerivedSupport>* supports,
                       std::vector<std::vector<FormulationRelationTerm>>* rows, base::DiagnosticSink& sink) const;
    void solidSupports(const std::map<std::int32_t, BasicSupport>& basic, std::vector<DerivedSupport>* supports,
                       std::vector<std::vector<FormulationRelationTerm>>* rows, base::DiagnosticSink& sink) const;
    void solidAffineField(EntityLayout::Kind kind, const base::Mat3& gradient, std::span<double> entityValues) const;
    void addElementProblem(std::string_view card, std::int64_t batchRow, base::Diagnostic problem);

    std::size_t m_degree = MIXED_HYBRID_DEFAULT_DEGREE;
    bool m_choosesShellDegrees = false;
    EntityLayout m_staticLayout{EntityLayout::Kind::Static};
    EntityLayout m_modalLayout{EntityLayout::Kind::Modal};
    std::vector<base::Diagnostic> m_problems;
    // The problems of elements, by card and batch row.
    std::map<std::pair<std::string, std::int64_t>, base::Diagnostic> m_elementProblems;

    std::vector<std::int32_t> m_gridRows;                  // node table row of each grid of the model
    std::vector<std::int32_t> m_gridIds;                   // and its ID
    std::vector<base::Vec3> m_gridPositions;               // and its position in the basic system
    std::map<std::int32_t, std::int32_t> m_gridIndexOfRow; // and back
    std::vector<base::Mat3> m_gridFrames;                  // the CD-to-basic rotation of each grid of the model
    std::set<std::int32_t> m_lineGrids;                    // the grid indices of the line elements
    std::set<std::int32_t> m_shellGrids;                   // those of the shells, corners and mid-side grid points
    std::vector<MixedLine> m_lines;
    std::map<std::string, std::vector<std::int32_t>, std::less<>> m_lineOfRow; // per line card: per batch row, or −1
    std::vector<MixedEdge> m_edges;
    // The entity DOFs no element stiffens (the functions across a rod alone, the shear and the rotations of the edges
    // of membrane-only shells, the plane of a flat bending-only plate...), held at 0 in every constraint set where no
    // support holds them with a value.
    std::vector<ElementDof> m_unstiffenedEntityDofs;
    // By grid index: the unit directions (basic system) of the grid point's rotation that its lines do not stiffen.
    std::map<std::int32_t, std::vector<base::Vec3>> m_unstiffenedLineRotations;

    std::vector<MixedShell> m_shells; // by increasing element ID (their index in the ShellPart)
    std::map<std::string, std::vector<std::int32_t>, std::less<>> m_shellOfRow; // per shell card: per batch row, or −1
    std::map<std::int32_t, std::size_t> m_shellOfId; // the shell of each element ID (the first of an ID given twice)
    std::unique_ptr<ShellPart> m_shellPart;
    std::vector<std::int32_t> m_shellEntityRow; // the layout row of each entity of the ShellPart
    // The mid-side grids of the shells that only their edge carries: grid index → (shell, reference point).
    struct MidSideGrid
    {
        std::size_t shell = 0;
        double xi = 0.0;
        double eta = 0.0;
    };
    std::map<std::int32_t, MidSideGrid> m_midSideGrids;

    std::vector<MixedSolid> m_solids; // by increasing element ID (their index in the SolidPart)
    std::map<std::string, std::vector<std::int32_t>, std::less<>> m_solidOfRow; // per solid card: per batch row, or −1
    std::unique_ptr<SolidPart> m_solidPart;
    std::vector<std::int32_t> m_solidEntityRow; // the layout row of each entity of the SolidPart (modal list)
    std::set<std::int32_t> m_solidOnlyGrids;    // the grids of solids that no line or shell has: no rotation
    std::map<std::int32_t, std::size_t> m_solidMidSideGrids; // the mid-side grids of the solids: grid → its edge

    // The geometry declarations of the model (readDeclarations), and the elements at each grid (index): (kind, index
    // in m_lines, m_shells or m_solids), for describeSurroundings.
    std::optional<input::FormulationGeometry> m_declarations;
    // The edges of the surfaces read (surfaceEdges), filled by the two geometry reports.
    std::vector<SurfaceEdgeReading> m_surfaceEdges;
    // The fold angle of the model (radians): the model's ShellNormalAngle, which decides the state of the edges that
    // nothing declares (zero or negative: they are sharp).
    double m_foldAngle = 0.0;
    // For each smooth edge, sharp edge and mesh line of the declarations (in their order): a surface of the model has
    // the edges it names, or it was already reported.
    struct DeclarationUse
    {
        std::vector<char> smoothEdges;
        std::vector<char> sharpEdges;
        std::vector<char> lineContinuations;
        std::vector<char> lineEnds;
    };
    DeclarationUse m_declarationUse;
    std::vector<std::vector<std::pair<std::uint8_t, std::size_t>>> m_elementsAtGrid;
};

} // namespace oos::elements::mixedHybrid
