#pragma once

#include "oos/design/external/ExternalEvaluator.hpp"
#include "oos/design/external/ExternalResponse.hpp"
#include "oos/design/model/DesignModel.hpp"
#include "oos/design/responses/DesignResp1Response.hpp"
#include "oos/design/responses/ElementCatalog.hpp"
#include "oos/design/responses/ResponseContext.hpp"
#include "oos/design/responses/ResponseDefinitions.hpp"
#include "oos/design/responses/ResponseValues.hpp"
#include "oos/design/responses/equation/EquationResponse.hpp"
#include "oos/output/arrays/OutputPlan.hpp"
#include "oos/output/sections/SectionLoadRun.hpp"
#include "oos/output/sections/ZoneRun.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <vector>

namespace oos::design
{

// The subcases each response is evaluated in, by response ID (subcase 0 for a global response).
using ResponseSubcases = std::map<std::int32_t, std::vector<std::int32_t>>;

// The design responses of a problem: the DesignResp1 resolved against the snapshot, the equation responses
// against the design model, in an order where every equation follows the responses it reads.
//
// A response is global (evaluated once, subcase 0) when its type is (WEIGHT, VOLUME, FRMASS) or, for an equation by
// the card rules, when every response it reads is global; the others are evaluated in subcases. Which subcases is
// decided by DesignConstraints (the constraints and the objective that use a response, its own subcases=, else every
// analysed subcase); closeSubcases() then adds, to each response an equation reads, the subcases of that equation.
//
// The response IDs are unique across DesignResp1, DesignResp2 and DesignResp3; a DesignResp2 that reads itself,
// directly or through others, is refused.
//
// The external responses (ExternalResponse) are evaluated after the DesignResp1 and the equations that do not
// read one (a select= callable sees them), and before the equations that read one; an equation names one as a
// DesignResp2 argument.
//
// Thread-safety: immutable once built; evaluate() runs on the calling thread (it may call the caller's callables).
// Ownership: owns its resolution; borrows nothing past the constructor.
class DesignResponses
{
public:
    DesignResponses() = default;
    // Throws OptimisationError or UnsupportedCardError (throwDesignDiagnostics) listing every problem of the parts.
    DesignResponses(ResponseParts parts, const DesignModel& model, const input::ModelSnapshot& snapshot);

    [[nodiscard]] bool isEmpty() const noexcept
    {
        return m_designResp1.empty() && m_equations.empty() && m_externals.empty();
    }
    [[nodiscard]] const ResponseParts& parts() const noexcept { return m_parts; }
    // The IDs of every response, DesignResp1 first then the equations, in their order of definition.
    [[nodiscard]] std::vector<std::int32_t> ids() const;
    [[nodiscard]] bool hasResponse(std::int32_t id) const noexcept { return m_kinds.count(id) != 0; }
    [[nodiscard]] bool isGlobal(std::int32_t id) const;
    // Whether a response is evaluated in the subcases of an analysis: a static DesignResp1 and an external
    // response in STATICS subcases, EIGN and FREQ in MODES subcases, LAMA in BUCK subcases, a global response in any
    // (once, at subcase 0), an equation where every response it reads is.
    [[nodiscard]] bool isEvaluatedIn(std::int32_t id, SubcaseAnalysis analysis) const;
    // The label of a response ("" when unknown) and its own subcases= (empty when none).
    [[nodiscard]] std::string label(std::int32_t id) const;
    [[nodiscard]] std::vector<std::int32_t> declaredSubcases(std::int32_t id) const;
    // The DesignResp1 type name ("Stress") of a DesignResp1, "DesignResp2" or "DesignResp3" for an equation (the
    // screening types).
    [[nodiscard]] std::string screeningType(std::int32_t id) const;

    // The DesignResp1, equation or external response of an ID, or nullptr (the sensitivities read their resolution).
    [[nodiscard]] const DesignResp1Response* findDesignResp1(std::int32_t id) const;
    [[nodiscard]] const EquationResponse* findEquation(std::int32_t id) const;
    [[nodiscard]] const ExternalResponse* findExternal(std::int32_t id) const;
    [[nodiscard]] const std::vector<ExternalResponse>& externals() const noexcept { return m_externals; }
    // The equation responses, each after the equations it reads.
    [[nodiscard]] const std::vector<EquationResponse>& equations() const noexcept { return m_equations; }

    // The outputs the analysis must give for the DesignResp1 (StaticOptions::requiredOutputs).
    [[nodiscard]] std::array<output::OutputRequestPlan, output::OUTPUT_KIND_COUNT> requiredOutputs() const;
    // The section and cut loads the analysis must compute for the SECTLOAD and CUTLOAD responses
    // (StaticOptions::sectionLoads): their definitions tagged with the response ID (group), the values kept.
    [[nodiscard]] output::SectionLoadRequest sectionLoadRequest() const;
    // The zones the analysis must give the contour data of (StaticOptions::zones): those of the CONTDISP,
    // CONTFORC and external responses, with the subcases that decided their contours; their values kept for the
    // contour responses and the external responses that are not linear, their data on the basis for the linear ones.
    [[nodiscard]] output::ZoneRequest zoneRequest() const;

    // `subcases` with, for every response an equation reads, the subcases of the equation added (a global response:
    // subcase 0), and every global response at subcase 0.
    [[nodiscard]] ResponseSubcases closeSubcases(ResponseSubcases subcases) const;

    // The values of the responses of `subcases` (closed), in the order of ids(); a response absent from `subcases`
    // is skipped. `selections`, when given, receives the subcases each external response evaluated per zone. Throws as
    // DesignResp1Response::evaluate, EquationResponse::evaluate and ExternalResponse::evaluate.
    [[nodiscard]] std::vector<ResponseValues> evaluate(
        const ResponseSubcases& subcases, ResponseContext& context, const CallableRegistry& callables,
        std::size_t blockItems = EQUATION_BATCH_ITEMS, const ExternalRegistry& externals = {},
        std::map<std::int32_t, ItemSelection>* selections = nullptr) const;

private:
    enum class Kind : std::uint8_t
    {
        DesignResp1,
        Equation,
        External
    };

    ResponseParts m_parts;
    std::vector<DesignResp1Response> m_designResp1;
    std::vector<EquationResponse> m_equations; // in an order where each follows the equations it reads
    std::vector<ExternalResponse> m_externals;
    std::vector<std::uint8_t> m_readsExternal; // per equation (m_equations order): reads an external, at any depth
    std::map<std::int32_t, std::pair<Kind, std::size_t>> m_kinds; // response ID → kind and index
    std::map<std::int32_t, bool> m_isGlobal;
    // The analyses each response is evaluated in, one bit per SubcaseAnalysis (analysisBit).
    std::map<std::int32_t, std::uint8_t> m_analyses;

    static constexpr std::uint8_t EVERY_ANALYSIS = 0b1111;
    [[nodiscard]] static constexpr std::uint8_t analysisBit(SubcaseAnalysis analysis) noexcept
    {
        return static_cast<std::uint8_t>(1U << static_cast<unsigned>(analysis));
    }
};

} // namespace oos::design
