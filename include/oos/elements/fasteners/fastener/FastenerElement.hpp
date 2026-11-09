#pragma once

// Fastener — fastener between two shells (CFAST, FastenerProp):
//
//   | Fastener | EID | PID  | TYPE  | IDA | IDB | GS   | GA    | GB    |
//   |       | XS  | YS   | ZS    |     |     |      |       |       |
//   | FastenerProp | PID | D    | MCID  | MFLAG | KT1 | KT2 | KT3  | KR1   |
//   |       | KR2 | KR3  | MASS  | GE  |     |      |       |       |
//
// Connectivity (fasteners/FastenerConnection): the piercing points A and B on the shells of sides A and B (TYPE PROP:
// the shells of properties IDA, IDB; ELEM: the shell elements IDA, IDB; located by GS, by XS-ZS or given as GA, GB),
// each moving with the reference point of an RigidInterpolation interpolation element whose independent points are the
// grid points of the shell element it pierces, weighted by the shape functions of that element at the point
// (fasteners/PiercingInterpolation): the piercing points, their elements and shape functions come from the fastener
// piercing table the reader computes. The auxiliary points are eliminated at element level: the element matrix couples
// the grid points of the two pierced elements (FastenerConnection::condense). FastenerProp D plays no part in the
// stiffness.
//
// Stiffness (PFAST: KT1-KT3, KR1-KR3 the translational and rotational stiffnesses in the fastener system): six
// uncoupled springs on the relative motion of B and A at the middle point p of A-B, in the fastener system E (x from
// A to B; MCID = −1: y the basic axis least aligned with x, fastenerAxes; MCID ≥ 0 with MFLAG = 1: the axes of MCID),
//   Δu = Eᵀ·((u_B + θ_B × (p − x_B)) − (u_A + θ_A × (p − x_A))),   Δθ = Eᵀ·(θ_B − θ_A),   f = diag(KT, KR)·(Δu, Δθ),
// as a Bushing whose spring point is the middle of the fastener (so that a rigid motion gives no force; the spring
// point is a choice of this implementation).
//
// Facets: stiffness; mass (PFAST MASS lumped on the translations of A and B, half each, carried to the grid points of
// the pierced elements by the same map; not validated); recovery: forces fx, fy, fz, mx, my, mz in the fastener system
// (f above), and the "stresses" of the data contract (stress.fastener), filled with the same forces (the stress and
// force tables of a Fastener are identical). GE plays no part in SOL 101, 103, 105.
//
// Not supported, refused by check(): MCID ≥ 0 with MFLAG = 0 (a relative fastener system, whose construction is not
// implemented); a curvilinear MCID. Problems: a PID that is not a FastenerProp, the problems of FastenerConnection (no
// piercing table), an unknown MCID.

#include "oos/base/math/MatN.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/core/PropertyLookup.hpp"
#include "oos/elements/fasteners/FastenerConnection.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace oos::elements
{

// The element family of Fastener. Thread-safety: immutable; facets may run concurrently on distinct ranges.
class FastenerElement final : public ElementFamily
{
public:
    [[nodiscard]] std::string_view card() const noexcept override { return "Fastener"; }
    [[nodiscard]] FacetSet facets() const noexcept override
    {
        return {Facet::Stiffness, Facet::Mass, Facet::Recovery, Facet::Derivatives};
    }
    [[nodiscard]] FieldUse designFieldUse(const DesignedField& field) const override;
    [[nodiscard]] std::vector<std::string> propertyCards() const override { return {"FastenerProp"}; }
    // FastenerProp ALPHA (the thermal expansion of the fastener) and COINL (the length below which GA and GB are taken
    // as coincident) are refused unless blank: neither is implemented.
    [[nodiscard]] std::vector<input::FieldRule> fieldRules() const override;

    void check(const ElementBatchView& view, base::DiagnosticSink& sink) const override;
    void computeStiffness(const ElementBatchView& view, ElementRange range, ElementMatrixBlock& block) const override;
    void computeMass(const ElementBatchView& view, ElementRange range, const MassOptions& options,
                     ElementMatrixBlock& block) const override;
    [[nodiscard]] std::vector<std::string> outputColumns(ElementOutput output) const override;
    void recover(const ElementBatchView& view, ElementRange range, ElementOutput output,
                 const NodalDisplacements& displacements, ElementResultBlock& block) const override;
    // The vectorised recovery (LinearRowRecovery.hpp): every column Linear, the slopes in closed form.
    [[nodiscard]] std::vector<ColumnForm> outputColumnForms(ElementOutput output) const override;
    bool recoverySlopes(const ElementBatchView& view, std::int64_t row, ElementOutput output,
                        RecoverySlopes& slopes) const override;
    bool recoverySlopesOfOutputs(const ElementBatchView& view, std::int64_t row,
                                 std::span<const RecoveryOutput> outputs,
                                 std::span<RecoverySlopes> slopes) const override;
    [[nodiscard]] std::size_t vectorisedRecoveryMinCases(ElementOutput output) const override;
    void completeRecoveredRows(const ElementBatchView& view, ElementOutput output,
                               std::span<ElementResultBlock> blocks) const override;

    // One resolved Fastener: its connection, H (the relative motion in the fastener system over w = (u_A, θ_A, u_B,
    // θ_B) in basic), its stiffnesses and mass.
    struct Fastener
    {
        FastenerConnection connection;
        base::MatN<6, 12> relativeMotion; // H
        std::array<double, 6> stiffness{};
        double mass = 0.0;
    };

private:
    struct Columns
    {
        PropertyRows properties;
        std::span<const std::int32_t> matFrameId;
        std::span<const std::uint8_t> mflag;
        base::FixedListView<double> kt;
        base::FixedListView<double> kr;
        std::span<const double> mass;

        explicit Columns(const ElementBatchView& view);
    };

    [[nodiscard]] static Resolution<Fastener> resolve(const ElementBatchView& view, const Columns& columns,
                                                      std::int64_t row);
};

} // namespace oos::elements
