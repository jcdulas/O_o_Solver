#pragma once

#include <cstdint>
#include <initializer_list>
#include <string_view>

namespace oos::elements
{

// A real field the design model may drive (DVPREL, DVMREL, DVCREL): the card that holds it — the
// element's property ("ShellProp"), one of its materials ("IsoMat") or the element's own card ("ShellQuad4") — and its
// column in the data contract ("t", "e", "offset"); `id` the ID of that card (the PID, MID or EID; 0 when the caller
// does not name it) and `item` the item of a fixed-list column ("stiffness" of AnisoMat2D: G11 is item 0), -1 for a
// scalar column. A family whose closed form depends on which of the element's cards holds the field (a AnisoMat2D read
// as MID1 or MID4) needs `id`; a field whose `id` is 0 gets no such closed form (the caller differences it).
struct DesignedField
{
    std::string_view card;
    std::string_view column;
    std::int32_t id = 0;
    std::int32_t item = -1;
};

// What a designed field changes in the computations of an element (the Derivatives facet): the
// semi-analytic derivatives (design/sensitivity/ElementDerivatives: central differences of the element's facets on the
// snapshots with the field moved by ±h) compute only these. Leaving a computation out is an optimisation, never a
// change of result: a computation the field does not enter gives the same bits at +h and −h, a zero difference.
// `changesLoads` covers the thermal loads an element computes itself, `changesDistributedLoads` its pressure and line
// loads (a section or material field does not move them, their derivative is zero); the inertial loads
// follow the mass.
struct FieldUse
{
    bool changesStiffness = false;
    bool changesMass = false;
    bool changesLoads = false;
    bool changesRecovery = false;
    bool changesDistributedLoads = false;

    // Every computation: what a field nothing is known about gets.
    [[nodiscard]] static constexpr FieldUse all() noexcept { return {true, true, true, true, true}; }
    [[nodiscard]] static constexpr FieldUse none() noexcept { return {}; }
    [[nodiscard]] static constexpr FieldUse massOnly() noexcept { return {false, true, false, false, false}; }
    [[nodiscard]] static constexpr FieldUse recoveryOnly() noexcept { return {false, false, false, true, false}; }
    // A stiffness that is not a section property of the mass (a spring rate, a torsion constant, a shear factor).
    [[nodiscard]] static constexpr FieldUse stiffnessAndRecovery() noexcept
    {
        return {true, false, false, true, false};
    }
    // An expansion coefficient or a reference temperature: the thermal loads, and the thermal strain the recovery
    // subtracts.
    [[nodiscard]] static constexpr FieldUse loadsAndRecovery() noexcept { return {false, false, true, true, false}; }
    // The same use without the pressure and line loads: a field of the section or of a material, which the loads an
    // element distributes over its surface or length do not read (its geometry, its offset and the load values do).
    [[nodiscard]] constexpr FieldUse withoutDistributedLoads() const noexcept
    {
        FieldUse use = *this;
        use.changesDistributedLoads = false;
        return use;
    }

    [[nodiscard]] constexpr bool isRead() const noexcept
    {
        return changesStiffness || changesMass || changesLoads || changesRecovery || changesDistributedLoads;
    }
};

// One rule of a family's Derivatives facet: what the column of a card changes.
struct FieldUseRule
{
    std::string_view card;
    std::string_view column;
    FieldUse use;
};

// The use of a field by a table of rules: the rule of its card and column; otherwise commonFieldUse(field).
[[nodiscard]] FieldUse findFieldUse(const DesignedField& field, std::initializer_list<FieldUseRule> rules) noexcept;

// The uses every family shares, the fields it knows nothing more about getting FieldUse::all() (conservative):
// - a material card (MAT1, AnisoMat2D, OrthoMat2D, AnisoMat3D): RHO the mass only; GE nothing (structural damping: no
// static effect);
//   TREF and the expansion coefficients (A, A1 ... A6) the thermal loads and the recovery; the stress and strain
//   allowables (ST, SC, SS, XT, XC, YT, YC, S, F12, STRN) the recovery only (margins, failure indices); every other
//   field all but the pressure and line loads (no element reads a material to distribute them);
// - NSM of any property the mass only (a non-structural mass per unit length or area);
// - GE of any property or element (PELAS, BushProp, SpringDofInline, SpringScalarInline) nothing.
[[nodiscard]] FieldUse commonFieldUse(const DesignedField& field) noexcept;

} // namespace oos::elements
