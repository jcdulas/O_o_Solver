#include "oos/assembly/constraints/SpcSet.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/ids/ComponentCode.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace oos::assembly
{

namespace
{

constexpr std::string_view PERMANENT_CARD = input::PERMANENT_CONSTRAINT_CARD; // the card of the PS rows
constexpr std::string_view ENFORCED_MOTION_CARD = "EnforcedMotionConstr";

constexpr std::uint8_t IN_SB = 1;
constexpr std::uint8_t IN_SG = 2;
constexpr std::uint8_t VALUE_FROM_CONSTRAINT = 4;
constexpr std::uint8_t VALUE_FROM_ENFORCED_MOTION = 8;

base::Diagnostic spcDiagnostic(std::string code, std::string_view card, base::EntityId setId, std::string field,
                               std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.code = std::move(code);
    diagnostic.card = std::string(card);
    diagnostic.id = setId;
    diagnostic.field = std::move(field);
    diagnostic.message = std::move(message);
    return diagnostic;
}

// The g-set work arrays of the construction (model-sized, hence budgeted).
struct SpcWork
{
    base::BudgetedBuffer<std::uint8_t> flags;
    base::BudgetedBuffer<double> values;

    SpcWork(base::MemoryBudget& budget, std::size_t size)
        : flags(budget, size, "SPC flags"), values(budget, size, "SPC enforced values")
    {
    }
};

// Records the value of one constrained DOF from a row of `card`; reports a conflict with an earlier value of the same
// kind.
void recordValue(SpcWork& work, base::DofIndex dof, double value, bool isEnforcedMotion, std::string_view card,
                 base::EntityId setId, base::EntityId nodeId, base::DiagnosticSink& sink)
{
    const auto index = static_cast<std::size_t>(dof);
    std::uint8_t& flags = work.flags[index];
    const std::uint8_t source = isEnforcedMotion ? VALUE_FROM_ENFORCED_MOTION : VALUE_FROM_CONSTRAINT;
    if ((flags & source) != 0)
    {
        if (work.values[index] != value)
            sink.report(
                spcDiagnostic("spc-conflicting-values", card, setId, "D",
                              base::concatenate({"grid point ", std::to_string(nodeId),
                                                 " is given two enforced values in set ", std::to_string(setId), " (",
                                                 std::to_string(work.values[index]), " and ", std::to_string(value),
                                                 "); the first is kept: give each constrained component one value"})));
        return;
    }
    if (!isEnforcedMotion && (flags & VALUE_FROM_ENFORCED_MOTION) != 0)
    {
        flags |= source;
        return; // EnforcedMotionConstr: the EnforcedMotionConstr value overrides the SPC value of the same DOF
    }
    flags |= source;
    work.values[index] = value;
}

} // namespace

SpcSet::SpcSet(const input::ConstraintTable* constraints, const input::LoadTable* loads, const DofMap& dofMap,
               std::optional<base::EntityId> spcSetId, std::optional<base::EntityId> loadSetId,
               base::MemoryBudget& budget, base::DiagnosticSink& sink)
    : m_spcSetId(spcSetId), m_loadSetId(loadSetId)
{
    // The sets whose EnforcedMotionConstr rows the load set brings, with their factors: the set alone with the factor 1
    // when the tables expand the LOAD combinations (or without load tables). Bounded by the sets of one combination.
    std::vector<input::LoadTerm> loadTerms;
    if (loadSetId)
        loadTerms = loads != nullptr ? loads->termsOf(*loadSetId) : std::vector<input::LoadTerm>{{*loadSetId, 1.0}};
    SpcWork work(budget, static_cast<std::size_t>(dofMap.gSetSize()));
    // sg: the PS field of the grid points (a component code as an integer of digits, 0 for none).
    const std::span<const std::int32_t> permanent = dofMap.nodes().permanentConstraints();
    for (std::int32_t row = 0; row < dofMap.gridCount(); ++row)
    {
        for (std::int32_t digits = permanent[static_cast<std::size_t>(row)]; digits > 0; digits /= 10)
        {
            const int component = digits % 10;
            if (component >= 1 && component <= 6)
                work.flags[static_cast<std::size_t>(dofMap.gridDof(row, component))] |= IN_SG;
        }
    }
    if (constraints != nullptr && (spcSetId || loadSetId))
    {
        const std::span<const std::int32_t> setIds = constraints->spcSetIds();
        const base::ValidityView validSetIds = constraints->spcReader().validity("setId");
        const std::span<const std::int32_t> nodeIds = constraints->spcNodeIds();
        const base::Utf8View components = constraints->spcComponents();
        const base::DictionaryView cards = constraints->spcCards();
        const std::span<const double> enforced = constraints->enforcedValues();
        const base::ValidityView validValues = constraints->spcReader().validity("enforced");
        for (std::int64_t row = 0; row < constraints->spcRowCount(); ++row)
        {
            const auto index = static_cast<std::size_t>(row);
            const std::string_view card = cards[row];
            if (card == PERMANENT_CARD || !validSetIds.isValid(row))
                continue; // the PS rows: read from the node table above
            const bool isEnforcedMotion = card == ENFORCED_MOTION_CARD;
            double factor = 1.0; // an EnforcedMotionConstr's term's factor
            if (isEnforcedMotion)
            {
                const auto term = std::find_if(loadTerms.begin(), loadTerms.end(),
                                               [&](const input::LoadTerm& t) { return t.set == setIds[index]; });
                if (term == loadTerms.end())
                    continue;
                factor = term->factor;
            }
            else if (!spcSetId || setIds[index] != *spcSetId)
                continue;
            const base::EntityId setId = setIds[index];
            const base::EntityId nodeId = nodeIds[index];
            const std::optional<input::PointRef> point = dofMap.points().find(nodeId);
            if (!point)
            {
                sink.report(spcDiagnostic(
                    "spc-unknown-point", card, setId, "nodeId",
                    base::concatenate({"set ", std::to_string(setId), " constrains point ", std::to_string(nodeId),
                                       ", which the model does not define; add the Node or "
                                       "ScalarPoint, or remove it from the card"})));
                continue;
            }
            base::ComponentCode code;
            bool isValidCode = true;
            try
            {
                code = base::ComponentCode::parse(components[row]);
            }
            catch (const std::invalid_argument&)
            {
                isValidCode = false;
            }
            const bool isScalar = point->kind == input::PointKind::Scalar;
            // SPC: a scalar point takes component 0 or blank; a grid point components 1 to 6.
            if (!isValidCode || (isScalar && !code.isEmpty()) || (!isScalar && (code.isScalar() || code.isEmpty())))
            {
                sink.report(spcDiagnostic(
                    "spc-bad-component", card, setId, "components",
                    base::concatenate({"set ", std::to_string(setId), " gives components '", components[row], "' to ",
                                       isScalar ? "scalar point " : "grid point ", std::to_string(nodeId),
                                       isScalar ? "; expected 0 or blank for a scalar point"
                                                : "; expected digits 1 to 6 for a grid point"})));
                continue;
            }
            // The EnforcedMotionConstr value times its set's factor in the combination (one product: the expanded
            // table's to the bit).
            const double value = (validValues.isValid(row) ? enforced[index] : 0.0) * factor;
            const auto constrain = [&](base::DofIndex dof)
            {
                work.flags[static_cast<std::size_t>(dof)] |= IN_SB;
                recordValue(work, dof, value, isEnforcedMotion, card, setId, nodeId, sink);
            };
            if (isScalar)
                constrain(dofMap.scalarDof(point->row));
            else
                for (int component = 1; component <= 6; ++component)
                    if (code.contains(component))
                        constrain(dofMap.gridDof(point->row, component));
        }
    }
    std::size_t sbCount = 0;
    std::size_t sgCount = 0;
    std::size_t valueCount = 0;
    for (std::size_t dof = 0; dof < work.flags.size(); ++dof)
    {
        sbCount += (work.flags[dof] & IN_SB) != 0 ? 1 : 0;
        sgCount += (work.flags[dof] & IN_SG) != 0 ? 1 : 0;
        valueCount += (work.flags[dof] & IN_SB) != 0 && work.values[dof] != 0.0 ? 1 : 0;
    }
    auto arrays = std::make_shared<Arrays>();
    arrays->sb = base::BudgetedBuffer<base::DofIndex>(budget, sbCount, "sb DOFs");
    arrays->sg = base::BudgetedBuffer<base::DofIndex>(budget, sgCount, "sg DOFs");
    arrays->enforced = base::BudgetedBuffer<EnforcedDisplacement>(budget, valueCount, "enforced displacements");
    sbCount = 0;
    sgCount = 0;
    valueCount = 0;
    for (std::size_t dof = 0; dof < work.flags.size(); ++dof)
    {
        const auto index = static_cast<base::DofIndex>(dof);
        if ((work.flags[dof] & IN_SB) != 0)
        {
            arrays->sb[sbCount++] = index;
            if (work.values[dof] != 0.0)
                arrays->enforced[valueCount++] = {index, work.values[dof]};
        }
        if ((work.flags[dof] & IN_SG) != 0)
            arrays->sg[sgCount++] = index;
    }
    m_arrays = std::move(arrays);
}

SpcSet SpcSet::forSubcase(const input::ConstraintTable* constraints, const input::LoadTable* loads,
                          const input::SubcaseTable& subcases, const DofMap& dofMap, base::EntityId subcase,
                          base::MemoryBudget& budget, base::DiagnosticSink& sink)
{
    return SpcSet(constraints, loads, dofMap, subcases.selectedSet(subcase, "spc"),
                  subcases.selectedSet(subcase, "load"), budget, sink);
}

std::vector<base::EntityId> SpcSet::enforcedSetIds(const input::ConstraintTable* constraints)
{
    std::vector<base::EntityId> ids; // bounded by the EnforcedMotionConstr rows
    if (constraints == nullptr)
        return ids;
    const std::span<const std::int32_t> setIds = constraints->spcSetIds();
    const base::ValidityView validSetIds = constraints->spcReader().validity("setId");
    const base::DictionaryView cards = constraints->spcCards();
    for (std::int64_t row = 0; row < constraints->spcRowCount(); ++row)
        if (validSetIds.isValid(row) && cards[row] == ENFORCED_MOTION_CARD)
            ids.push_back(setIds[static_cast<std::size_t>(row)]);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

std::vector<input::LoadTerm> SpcSet::enforcedTermsOf(const input::LoadTable* loads,
                                                     std::span<const base::EntityId> enforcedSets,
                                                     std::optional<base::EntityId> loadSetId)
{
    std::vector<input::LoadTerm> terms; // bounded by the terms of the load set
    if (!loadSetId || enforcedSets.empty())
        return terms;
    for (const input::LoadTerm& term :
         loads != nullptr ? loads->termsOf(*loadSetId) : std::vector<input::LoadTerm>{{*loadSetId, 1.0}})
        if (std::binary_search(enforcedSets.begin(), enforcedSets.end(), term.set))
            terms.push_back(term);
    return terms;
}

SpcSet SpcSet::withLoadSet(std::optional<base::EntityId> loadSetId) const
{
    SpcSet result;
    result.m_spcSetId = m_spcSetId;
    result.m_loadSetId = loadSetId;
    result.m_arrays = m_arrays;
    return result;
}

bool SpcSet::hasSameConstrainedDofs(const SpcSet& other) const noexcept
{
    const auto equal = [](std::span<const base::DofIndex> a, std::span<const base::DofIndex> b)
    {
        if (a.size() != b.size())
            return false;
        for (std::size_t k = 0; k < a.size(); ++k)
            if (a[k] != b[k])
                return false;
        return true;
    };
    return equal(sbDofs(), other.sbDofs()) && equal(sgDofs(), other.sgDofs());
}

} // namespace oos::assembly
