#pragma once

#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/output/arrays/ElementResultBuilder.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace oos::output
{

// The output kinds of a static subcase (the Case Control requests DISPLACEMENT, SPCFORCES, MPCFORCES,
// OLOAD, STRESS, STRAIN, FORCE, GPFORCE, ESE).
enum class OutputKind : std::uint8_t
{
    Displacement,
    SpcForces,
    MpcForces,
    AppliedLoads,
    Stress,
    Strain,
    Force,
    GridPointForces,
    StrainEnergy
};

inline constexpr std::size_t OUTPUT_KIND_COUNT = 9;

// The neutral output request name ("DisplacementOutput") and the table kind it gives ("displacements", "appliedLoads",
// "stress"...).
[[nodiscard]] std::string_view requestNameOf(OutputKind kind) noexcept;
[[nodiscard]] std::string_view tableKindOf(OutputKind kind) noexcept;

// What one request asks for in one subcase: whether it is on, the points or elements of its SET (all of them for
// "= ALL"), and the describers that change the tables (of STRESS, STRAIN, FORCE): CORNER (or BILIN, CUBIC, SGAGE) for
// the corner rows of the shells, CENTER for the centre rows only (the default); FIBER for the strains at the fibres,
// STRCUR (the default) for the membrane strains and curvatures; VONMISES for the von Mises stress column, MAXS (the
// default) for the maximum shear.
struct OutputRequestPlan
{
    bool isRequested = false;
    bool isAll = true;
    std::vector<std::int32_t> members; // sorted, when !isAll; bounded by the SET the deck writes
    bool isCorner = false;
    bool isFiber = false;
    bool isVonMises = false;

    // Whether the request takes a point or element of this ID.
    [[nodiscard]] bool takes(std::int32_t id) const;
};

// The modal effective mass request of a subcase (Case Control MEFFMASS, read by SOL 103): the reference point of
// its describer GRID = gid (0, the origin of the basic system, by default). Its other describers (PRINT, PUNCH,
// SUMMARY, PARTFAC, MEFFM, MEFFW, FRACSUM) choose parts of a printed report: the solution gives every part.
struct ModalEffectiveMassRequest
{
    bool isRequested = false;
    std::int32_t referencePoint = 0;
};

// The outputs of one subcase: the requests of output_requests_table and the members of their SETs, SORT1 (a
// SORT2 request is given in SORT1 with a warning: O_o_Solver writes no SORT2 table).
//
// Without the requests table (when O_o_Reader does not give it), the tables of every point and element are given:
// displacements, SPC forces, MPC forces when there is an m set, and the stresses, strains and forces
// of every element at the centre and corners with the von Mises column; GPFORCE, ESE and OLOAD are then not produced
// (a warning says the requests are not known).
//
// A request the solver does not produce (GPSTRESS, ACCELERATION, ...) is a warning, never a refusal; a SET that the
// subcase cannot see is an error Diagnostic.
//
// Thread-safety: an immutable value.
class OutputPlan
{
public:
    // The plan of the subcase `subcase` (its ID in the Case Control tables; 0 the global section of a deck without
    // SUBCASE).
    [[nodiscard]] static OutputPlan forSubcase(const input::ModelSnapshot& snapshot, std::int32_t subcase,
                                               base::DiagnosticSink& sink);
    // The plan used without a requests table.
    [[nodiscard]] static OutputPlan everything();

    [[nodiscard]] const OutputRequestPlan& of(OutputKind kind) const noexcept
    {
        return m_requests[static_cast<std::size_t>(kind)];
    }
    [[nodiscard]] bool isFromCaseControl() const noexcept { return m_isFromCaseControl; }
    [[nodiscard]] const ModalEffectiveMassRequest& modalEffectiveMass() const noexcept { return m_modalEffectiveMass; }
    // The aerodynamic outputs of a trim (SOL 144: APRESSURE, AEROF), which are given when asked for.
    [[nodiscard]] bool isAeroPressureRequested() const noexcept { return m_isAeroPressureRequested; }
    [[nodiscard]] bool isAeroForceRequested() const noexcept { return m_isAeroForceRequested; }
    void setRequest(OutputKind kind, OutputRequestPlan request);
    // Adds the outputs `required` asks for (the outputs the design responses read, whatever the Case Control asks:
    // StaticOptions::requiredOutputs): a kind requested by both takes the union of their points or elements with this
    // plan's describers, a kind this plan does not request takes the describers of `required`.
    void addRequired(const std::array<OutputRequestPlan, OUTPUT_KIND_COUNT>& required);
    // Keeps only the outputs `required` asks for (StaticOptions::usesCaseControlOutputs false): a kind both ask for
    // takes the points or elements of `required` with this plan's describers, a kind only `required` asks for takes its
    // describers, a kind only this plan asks for is dropped.
    void restrictTo(const std::array<OutputRequestPlan, OUTPUT_KIND_COUNT>& required);

private:
    std::array<OutputRequestPlan, OUTPUT_KIND_COUNT> m_requests;
    bool m_isFromCaseControl = false;
    ModalEffectiveMassRequest m_modalEffectiveMass;
    bool m_isAeroPressureRequested = false;
    bool m_isAeroForceRequested = false;
};

// The element tables a plan asks for (STRESS, STRAIN, FORCE with their SETs, CENTER or CORNER, FIBER or STRCUR). The
// maximum shear describer (MAXS) is not applied: O_o_Reader's result tables hold the stress tables of the shells and
// plies with their von Mises column only, so the von Mises stress is given whatever
// the describer, the maximum shear (σ1 − σ2)/2 following from the principal stresses of the table. The selection
// borrows the members of the plan, which must outlive it.
[[nodiscard]] ElementOutputSelection elementSelectionOf(const OutputPlan& plan);

} // namespace oos::output
