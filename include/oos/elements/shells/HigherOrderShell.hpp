#pragma once

#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/shells/HigherOrderGeometry.hpp"
#include "oos/elements/shells/MindlinShell.hpp"
#include "oos/elements/shells/ShellConnectivity.hpp"
#include "oos/elements/shells/ShellNodeFrames.hpp"
#include "oos/elements/shells/ShellRecovery.hpp"
#include "oos/elements/shells/ShellSection.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace oos::elements
{

// The facets shared by the higher-order shells ShellQuad8 and ShellTria6, on MindlinShell:
//
//   | ShellQuad8 | EID | PID | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 | T1 | T2 | T3 | T4 | THETA or MCID | ZOFFS | TFLAG
//   | | ShellTria6 | EID | PID | G1 | G2 | G3 | G4 | G5 | G6 | THETA or MCID | ZOFFS | T1 | T2 | T3 | TFLAG |
//
// The element frame is that of the corners (the ShellQuad4 frame of the mean plane for the ShellQuad8, the ShellTria3
// frame for the ShellTria6), the origin at the mean of the corners; the material axis, ZOFFS, the corner thicknesses and
// the sections as for the ShellQuad4 and the ShellTria3 (the material angle measured from G1→G2 in the element plane, or
// the MCID x axis projected on it, not along the lines of constant ξ of the element: a design choice of this project).
//
// Flat and curved elements: an element whose nodes are all within HIGHER_ORDER_SHELL_FLATNESS of its size from the
// mean plane of its corners is flat, its nodes projected on that plane (HigherOrderGeometry); any other is curved,
// MindlinShell then the degenerated shell of its nodes as given. The director of each node is the one its grid point
// shares with the other ShellQuad8 and ShellTria6 of its smooth group there (HigherOrderShellDirectors: the mean of their
// own normals, separate across a fold, a T-junction or a stiffener), or the element's own normal (that of its
// isoparametric surface, n on a flat element) where it is alone in its group or where the group's normals agree to
// round-off: a flat mesh is then the Mindlin plate of the shape, bit for bit. The stresses, strains and forces of a
// place are in its tangent frame: e3 the normal of the surface there, e1 the element x axis projected on the tangent
// plane, e2 = e3 × e1 (the element frame on a flat element); the material axis is at the material angle from e1. ZOFFS
// moves each node along its director (u_ref = u + θ × ZOFFS·ν_i: ShellNodeFrames::buildWithNodeOffsets), and the
// reference surface with it (X_i + ZOFFS·ν_i), unless every director is n (ShellNodeFrames::build).
//
// The rotation about a node's director takes no stiffness (MindlinShell). Where the elements of a grid point share
// one director (a flat mesh, a smooth curved mesh), that rotation is singular, and the automatic constraint of
// singular DOFs (AUTOSPC) removes it, along a component axis or, when the director is skewed in the displacement
// system, by constraining the component best aligned with it (assembly::GridPointSingularities; exact, the direction
// having no stiffness). Where elements of different directions meet (a fold, a stiffener, a ShellQuad4 at a ShellQuad8),
// each holds the others' rotation by its bending, and nothing is constrained. K6ROT plays no part.
//
// Not supported (refused with a diagnostic): an element without all its edge grid points (a ShellQuad4 or ShellTria3
// serves there), a map whose Jacobian on the element plane is not positive (an edge node far from the middle third of
// its edge, or a surface folding over the plane of its corners), a curved element whose surface normal turns by more
// than 45° from the element normal (HIGHER_ORDER_SHELL_MIN_NORMAL_COSINE: one quadratic element cannot follow so curved
// a surface), and everything the ShellQuad4 refuses.

// The columns of a ShellQuad8 or ShellTria6 batch, read once per facet call.
struct HigherOrderShellColumns
{
    ShellConnectivity connectivity;
    ShellSectionColumns sections;
    std::span<const double> theta;
    std::span<const std::int32_t> matFrameId;
    std::span<const double> offset;
    std::span<const std::int32_t> thicknessFlag;
    base::FixedListView<double> cornerThickness;
    const HigherOrderShellDirectors* directors; // the directors shared at the grid points of the snapshot

    explicit HigherOrderShellColumns(const ElementBatchView& view);
};

// Everything the facets need about one element.
template <typename Shape>
struct HigherOrderShellData
{
    static constexpr std::size_t N = Shape::NODES;
    static constexpr std::size_t C = Shape::CORNERS;

    base::Mat3 axes = base::Mat3::identity(); // columns x, y, n of the element frame in basic
    bool isCurved = false;
    // The nodes on the reference surface (moved by ZOFFS along their directors on a curved element), in the element
    // frame from its origin; z = 0 on a flat element.
    std::array<base::Vec3, N> position{};
    std::array<base::Vec3, N> director{}; // unit, in the element frame; n on a flat element
    ShellSection section;                 // in the element frame
    std::array<double, C> thickness{};
    std::array<std::int32_t, N> gridIds{};
    ShellNodeFrames<N> frames;
    double materialAngle = 0.0;

    [[nodiscard]] MindlinShell<Shape> shell() const
    {
        return MindlinShell<Shape>(position, director, section, thickness);
    }
};

template <typename Shape>
[[nodiscard]] Resolution<HigherOrderShellData<Shape>> resolveHigherOrderShell(const ElementBatchView& view,
                                                                              const HigherOrderShellColumns& columns,
                                                                              std::int64_t row);

// The facet loops (each as its ElementFamily facet).
template <typename Shape>
void checkHigherOrderShells(const ElementBatchView& view, base::DiagnosticSink& sink);
template <typename Shape>
void appendHigherOrderShellStiffness(const ElementBatchView& view, ElementRange range, ElementMatrixBlock& block);
template <typename Shape>
void appendHigherOrderShellMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                                ElementMatrixBlock& block);
template <typename Shape>
void appendHigherOrderShellPressures(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                                     ElementVectorBlock& block);
template <typename Shape>
void checkHigherOrderShellThermalLoads(const ElementBatchView& view, const ElementTemperatures& temperatures,
                                       base::DiagnosticSink& sink);
template <typename Shape>
void appendHigherOrderShellThermalLoads(const ElementBatchView& view, ElementRange range,
                                        const ElementTemperatures& temperatures, ElementVectorBlock& block);
template <typename Shape>
void appendHigherOrderShellDifferential(const ElementBatchView& view, ElementRange range,
                                        const NodalDisplacements& preload, ElementMatrixBlock& block);
template <typename Shape>
void recoverHigherOrderShells(const ElementBatchView& view, ElementRange range, ElementOutput output,
                              const NodalDisplacements& displacements, ElementResultBlock& block);
// The rows of several element and layer outputs from one resolution and one shell per element (recoverOutputs).
template <typename Shape>
void recoverHigherOrderShellOutputs(const ElementBatchView& view, ElementRange range,
                                    std::span<const RecoveryOutput> outputs, const NodalDisplacements& displacements,
                                    std::span<ElementResultBlock> blocks);
template <typename Shape>
void recoverHigherOrderShellLayers(const ElementBatchView& view, ElementRange range, LayerOutput output,
                                   const NodalDisplacements& displacements, ElementResultBlock& block);
// The recovery slopes of several element and layer outputs of the element of batch row `row`
// (ElementFamily::recoverySlopesOfOutputs): the shell built once, its place states at the unit local
// displacements (shellRecoverySlopesOfOutputs), as the ShellQuad4's.
template <typename Shape>
void higherOrderShellSlopes(const ElementBatchView& view, std::int64_t row, std::span<const RecoveryOutput> outputs,
                            std::span<RecoverySlopes> slopes);
// The closed forms in a AnisoMat2D stiffness term of the ShellProp (ShellSectionRate): dK
// (MindlinShell::stiffnessSectionRate), dK_d at a fixed preload (the differential stiffness of the centre's membrane
// resultants' rate) and the rows' rates at fixed displacements (placeStateSectionRates, appendShellRowRates); false
// for another field, a section whose rate shellSectionRate does not give, or a temperature change of the preload or
// state (its thermal resultants move with G1).
template <typename Shape>
[[nodiscard]] bool higherOrderShellStiffnessRate(const ElementBatchView& view, std::int64_t row,
                                                 const DesignedField& field, ElementMatrixBlock& block);
template <typename Shape>
[[nodiscard]] bool higherOrderShellDifferentialRate(const ElementBatchView& view, std::int64_t row,
                                                    const DesignedField& field, const NodalDisplacements& preload,
                                                    ElementMatrixBlock& block);
template <typename Shape>
[[nodiscard]] bool higherOrderShellRecoveryRates(const ElementBatchView& view, std::int64_t row,
                                                 const DesignedField& field, ElementOutput output,
                                                 const NodalDisplacements& displacements, ElementResultBlock& rates);
// And in a ply thickness of a CompProp or CompPropGlobal: the same, from the rates of the section's products
// (LaminateSectionRate::ofProducts; refused with a direction rigid in transverse shear), the forces only, and the ply
// rows' rates (appendLaminateRowRates).
template <typename Shape>
[[nodiscard]] bool higherOrderShellLayerRates(const ElementBatchView& view, std::int64_t row,
                                              const DesignedField& field, LayerOutput output,
                                              const NodalDisplacements& displacements, ElementResultBlock& rates);
// The failure index rows of the blocks from the ply states (completeLaminateFailureRows), each element resolved once.
template <typename Shape>
void completeHigherOrderShellFailureRows(const ElementBatchView& view, std::span<const ElementResultBlock> sources,
                                         std::span<ElementResultBlock> blocks);

} // namespace oos::elements
