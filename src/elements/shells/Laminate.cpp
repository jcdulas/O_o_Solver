#include "oos/elements/shells/Laminate.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/MaterialMatrices.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <optional>
#include <string>
#include <utility>

namespace oos::elements
{

namespace
{

constexpr double RADIANS_PER_DEGREE = std::numbers::pi / 180.0;

// The codes of O_o_Reader's CompPropGlobal `failureTheory` and `laminateOption` columns (its FailureTheory and
// LaminateOption enumerations, in the order of the card fields).
enum class FailureTheoryCode : std::uint8_t
{
    None = 0,
    Hill = 1,
    Hoffman = 2,
    TsaiWu = 3,
    MaxStrain = 4
};

// The rows [first, last) of laminatePliesTable holding the plies of `propId`: the table is sorted by propId, then ply.
struct PlyRows
{
    std::int64_t first = 0;
    std::int64_t last = 0;
};

// The transverse shear modulus of a ply in direction d (0: x, 1: y) of the material frame of the element: the tensor
// (G1Z, G2Z) of the ply axes rotated by the ply angle θ, G_xz = G1Z cos²θ + G2Z sin²θ, G_yz = G1Z sin²θ + G2Z cos²θ.
[[nodiscard]] double plyShearModulus(const LaminatePly& ply, std::size_t d)
{
    const double c2 = std::cos(ply.angle) * std::cos(ply.angle);
    const double s2 = 1.0 - c2;
    const std::array<double, 2>& g = ply.material.transverseShear;
    return d == 0 ? g[0] * c2 + g[1] * s2 : g[0] * s2 + g[1] * c2;
}

// The transverse shear of a laminate in direction d (x with Q̄11, y with Q̄22) from the bending stress
// distribution of cylindrical bending about the neutral axis zn = B/A (Whitney, "Structural Analysis of Laminated
// Anisotropic Plates", §4.6). Per unit shear force, the equilibrium ∂σ/∂x + ∂τ/∂z = 0 gives
//   τ(z) = −S(z) / D*,  S(z) = ∫_bottom^z Q̄dd (ζ − zn) dζ,  D* = D − B²/A,
// and the shear energy gives the stiffness 1/H = ∫ τ² / G dz (κ G t, κ = 5/6, for one material). τ is quadratic in
// each ply: three Gauss points integrate τ² exactly. Sets H and the τ at the top of each ply.
void setTransverseShear(Laminate& laminate, std::size_t d)
{
    double a = 0.0;
    double b = 0.0;
    double dd = 0.0;
    for (const LaminatePly& ply : laminate.plies)
    {
        const double e = ply.stiffness(d, d);
        a += e * (ply.zTop - ply.zBottom);
        b += e * 0.5 * (ply.zTop * ply.zTop - ply.zBottom * ply.zBottom);
        dd += e * (ply.zTop * ply.zTop * ply.zTop - ply.zBottom * ply.zBottom * ply.zBottom) / 3.0;
    }
    const double neutralAxis = b / a;
    const double reducedRigidity = dd - b * b / a;
    // S(z) − S(z0) inside a ply of modulus e starting at z0.
    const auto firstMoment = [neutralAxis](double e, double z0, double z)
    { return 0.5 * e * ((z - neutralAxis) * (z - neutralAxis) - (z0 - neutralAxis) * (z0 - neutralAxis)); };
    constexpr std::array<double, 3> GAUSS_POINTS{-0.7745966692414834, 0.0, 0.7745966692414834};
    constexpr std::array<double, 3> GAUSS_WEIGHTS{5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
    double below = 0.0;
    double compliance = 0.0;
    for (LaminatePly& ply : laminate.plies)
    {
        const double e = ply.stiffness(d, d);
        const double g = plyShearModulus(ply, d);
        const double half = 0.5 * (ply.zTop - ply.zBottom);
        const double middle = 0.5 * (ply.zTop + ply.zBottom);
        for (std::size_t k = 0; k < GAUSS_POINTS.size(); ++k)
        {
            const double tau = (below + firstMoment(e, ply.zBottom, middle + half * GAUSS_POINTS[k])) / reducedRigidity;
            compliance += GAUSS_WEIGHTS[k] * half * tau * tau / g;
        }
        below += firstMoment(e, ply.zBottom, ply.zTop);
        ply.shearAtTopPerForce[d] = -below / reducedRigidity;
    }
    laminate.transverseShear[d] = 1.0 / compliance;
}

// A stack whose plies all have the same material and the same angle (one material through the thickness).
[[nodiscard]] bool isHomogeneous(const Laminate& laminate)
{
    const LaminatePly& first = laminate.plies.front();
    return std::all_of(laminate.plies.begin(), laminate.plies.end(), [&first](const LaminatePly& ply)
                       { return ply.material.matId == first.material.matId && ply.angle == first.angle; });
}

[[nodiscard]] PlyRows plyRowsOf(const base::ContractReader& plies, std::int32_t propId)
{
    const std::span<const std::int32_t> propIds = plies.values<std::int32_t>("propId");
    const auto range = std::equal_range(propIds.begin(), propIds.end(), propId);
    if (range.first != range.second)
        return {range.first - propIds.begin(), range.second - propIds.begin()};
    // A table out of order: the rows of the PID by a scan (they are contiguous in O_o_Reader's layout).
    const auto first = std::find(propIds.begin(), propIds.end(), propId);
    if (first == propIds.end())
        return {};
    auto last = first;
    while (last != propIds.end() && *last == propId)
        ++last;
    return {first - propIds.begin(), last - propIds.begin()};
}

// The lamination option of `propId` from laminateSummary ("" for blank, a missing summary or PID).
[[nodiscard]] std::string_view laminationOptionOf(const base::ContractReader* summary, std::int32_t propId)
{
    if (summary == nullptr)
        return "";
    const std::span<const std::int32_t> propIds = summary->values<std::int32_t>("propId");
    const base::DictionaryView options = summary->dictionaryTexts("laminateOption", base::NullPolicy::Allow);
    const base::ValidityView isGiven = summary->validity("laminateOption");
    for (std::size_t row = 0; row < propIds.size(); ++row)
        if (propIds[row] == propId)
            return isGiven.isValid(static_cast<std::int64_t>(row)) ? options[static_cast<std::int64_t>(row)]
                                                                   : std::string_view{};
    return "";
}

// The failure theory of the property row: the `failureTheory` column of a CompPropGlobal (O_o_Reader's codes) or of a
// CompProp, as a code or a text. Nothing when the code is a theory the solver does not compute (the Hashin criteria).
struct TheoryReading
{
    bool isSupported = true;
    FailureTheory theory = FailureTheory::None;
    std::string text; // what the card gives, for the message
};

[[nodiscard]] TheoryReading failureTheoryOf(const base::ContractReader& property, std::int64_t row)
{
    constexpr const char* column = "failureTheory";
    if (!property.hasColumn(column) || !property.validity(column).isValid(row))
        return {};
    if (property.columnType(column) == "uint8")
    {
        const std::uint8_t code =
            property.values<std::uint8_t>(column, base::NullPolicy::Allow)[static_cast<std::size_t>(row)];
        switch (static_cast<FailureTheoryCode>(code))
        {
        case FailureTheoryCode::None:
            return {};
        case FailureTheoryCode::Hill:
            return {true, FailureTheory::Hill, "Hill"};
        case FailureTheoryCode::Hoffman:
            return {true, FailureTheory::Hoffman, "Hoffman"};
        case FailureTheoryCode::TsaiWu:
            return {true, FailureTheory::TsaiWu, "TsaiWu"};
        case FailureTheoryCode::MaxStrain:
            return {true, FailureTheory::MaxStrain, "StrainTheory"};
        }
        return {false, FailureTheory::None, std::string("code ").append(std::to_string(code))};
    }
    const std::string_view text = property.dictionaryTexts(column, base::NullPolicy::Allow)[row];
    if (text.empty())
        return {};
    if (text == "Hill")
        return {true, FailureTheory::Hill, "Hill"};
    if (text == "Hoffman")
        return {true, FailureTheory::Hoffman, "Hoffman"};
    if (text == "TsaiWu")
        return {true, FailureTheory::TsaiWu, "TsaiWu"};
    if (text == "StrainTheory")
        return {true, FailureTheory::MaxStrain, "StrainTheory"};
    return {false, FailureTheory::None, std::string(text)};
}

[[nodiscard]] double realOr(const base::ContractReader& reader, const char* column, std::int64_t row, double fallback)
{
    if (!reader.hasColumn(column) || !reader.validity(column).isValid(row))
        return fallback;
    return reader.values<double>(column, base::NullPolicy::Allow)[static_cast<std::size_t>(row)];
}

// The vector T(θ)⁻¹·v = T(−θ)·v: a strain-like vector given in the ply frame, written in the material frame.
[[nodiscard]] std::array<double, 3> strainToMaterialFrame(const std::array<double, 3>& v, double angle)
{
    return strainRotation(-angle) * v;
}

} // namespace

LaminateColumns::LaminateColumns(const ElementBatchView& view) : materials(view)
{
    const input::ModelSnapshot& snapshot = view.snapshot();
    if (snapshot.hasTable(input::TableRole::LaminatePlies))
        plies = &snapshot.laminatePlies();
    if (snapshot.hasTable(input::TableRole::LaminateSummary))
        summary = &snapshot.laminateSummary();
}

Resolution<Laminate> resolveLaminate(const ElementBatchView& view, const LaminateColumns& columns, std::int64_t row,
                                     const input::CardTable& property, std::int64_t propertyRow)
{
    using Result = Resolution<Laminate>;
    const base::ContractReader& card = property.reader();
    Laminate laminate;
    laminate.propId = property.ids()[static_cast<std::size_t>(propertyRow)];
    laminate.isCompPropGlobal = property.card() == "CompPropGlobal";
    const std::string propId = std::to_string(laminate.propId);
    const std::string_view type = property.card();

    const PlyRows rows = columns.plies != nullptr ? plyRowsOf(*columns.plies, laminate.propId) : PlyRows{};
    if (rows.first == rows.last)
        return Result::failure(elementProblem(
            view, row, problem_code::MISSING_PROPERTY, "propId",
            base::concatenate({"expected the ply stack of ", type, " ", propId,
                               " in laminatePliesTable, found none; take the snapshot with "
                               "O_o_Solver.snapshot.take_snapshot, which gathers the laminate tables"})));

    const std::string_view option = laminationOptionOf(columns.summary, laminate.propId);
    if (option == "MembraneOnly")
        laminate.option = LaminateOption::Membrane;
    else if (option == "BendingOnly")
        laminate.option = LaminateOption::Bending;
    else if (option == "SmearedLaminate")
        laminate.option = LaminateOption::Smeared;
    else if (!option.empty() && option != "Symmetry")
        return Result::failure(elementProblem(
            view, row, problem_code::UNSUPPORTED_FIELD, base::concatenate({"", type, ".laminateOption"}),
            base::concatenate(
                {"expected ", type, ".laminateOption blank, Symmetry, MembraneOnly, BendingOnly or SmearedLaminate on ",
                 type, " ", propId, ", found ", option,
                 "; the sandwich option SandwichCore is not supported: model the face sheets and the core as "
                 "plies"})));

    const TheoryReading theory = failureTheoryOf(card, propertyRow);
    if (!theory.isSupported)
        return Result::failure(
            elementProblem(view, row, problem_code::UNSUPPORTED_FIELD, base::concatenate({"", type, ".failureTheory"}),
                           base::concatenate({"expected ", type, ".failureTheory blank, HILL, HOFF, TSAI or STRN on ",
                                              type, " ", propId, ", found ", theory.text,
                                              "; the Hashin criteria are not supported: choose another theory"})));
    laminate.failureTheory = theory.theory;
    laminate.bondAllowable = realOr(card, "bondAllowable", propertyRow, 0.0);
    laminate.referenceTemperature = realOr(card, "referenceTemperature", propertyRow, 0.0);
    laminate.nonStructuralMass = realOr(card, "nonStructMass", propertyRow, 0.0);

    const base::ContractReader& plies = *columns.plies;
    const std::span<const std::int32_t> numbers = plies.values<std::int32_t>("ply");
    const std::span<const std::int32_t> plyIds = plies.values<std::int32_t>("plyId", base::NullPolicy::Allow);
    const base::ValidityView isPlyIdGiven = plies.validity("plyId");
    const std::span<const std::int32_t> matIds = plies.values<std::int32_t>("matId");
    const std::span<const double> thicknesses = plies.values<double>("thickness");
    const std::span<const double> thetas = plies.values<double>("theta");
    const std::span<const double> bottoms = plies.values<double>("zBottom");
    const std::span<const double> tops = plies.values<double>("zTop");
    const base::BooleanView outputs = plies.booleans("sout");
    const base::BooleanView mirrored = plies.booleans("isMirrored");
    // The card's plies: the stack's rows outside the mirrored half of LAM = SYM.
    std::int32_t cardPlies = 0;
    for (std::int64_t p = rows.first; p < rows.last; ++p)
        cardPlies += mirrored[p] ? 0 : 1;
    bool isRigidInX = false;
    bool isRigidInY = false;
    for (std::int64_t p = rows.first; p < rows.last; ++p)
    {
        const auto r = static_cast<std::size_t>(p);
        LaminatePly ply;
        ply.number = numbers[r];
        ply.cardPly = mirrored[p] ? 2 * cardPlies - ply.number : ply.number - 1;
        ply.plyId = isPlyIdGiven.isValid(p) ? plyIds[r] : 0;
        // The ply fields carry their number after the keyword of the first ply's field: MID3 is
        // "CompProp.plyMatId3".
        const std::string field = base::concatenate({"", type, ".plyMatId", std::to_string(ply.number)});
        Resolution<ShellMaterial> material = resolveShellMaterial(view, columns.materials, row, matIds[r], field);
        if (!material)
            return Result::failure(std::move(material.problem));
        ply.material = *material.value;
        ply.thickness = thicknesses[r];
        ply.angle = thetas[r] * RADIANS_PER_DEGREE;
        ply.zBottom = bottoms[r];
        ply.zTop = tops[r];
        ply.hasOutput = outputs[p];
        if (!(ply.thickness > 0.0))
            return Result::failure(elementProblem(
                view, row, problem_code::INVALID_FIELD,
                base::concatenate({"", type, ".plyThickness", std::to_string(ply.number)}),
                base::concatenate({"expected a positive thickness for ply ", std::to_string(ply.number), " of ", type,
                                   " ", propId, ", found ", std::to_string(ply.thickness), "; correct the property"})));
        ply.stiffness = rotateToElementFrame(ply.material.planeStress, ply.angle);
        ply.expansion = strainToMaterialFrame(ply.material.expansion, ply.angle);

        // Classical lamination theory (Jones §4.2) and the thermal resultants (Jones §4.5).
        const double z0 = ply.zBottom;
        const double z1 = ply.zTop;
        const double first = z1 - z0;
        const double second = 0.5 * (z1 * z1 - z0 * z0);
        const double third = (z1 * z1 * z1 - z0 * z0 * z0) / 3.0;
        laminate.a += first * ply.stiffness;
        laminate.b += second * ply.stiffness;
        laminate.d += third * ply.stiffness;
        const std::array<double, 3> stress = ply.stiffness * ply.expansion;
        for (std::size_t k = 0; k < 3; ++k)
        {
            laminate.forcePerDegree[k] += first * stress[k];
            laminate.momentPerDegree[k] += second * stress[k];
            laminate.forcePerGradient[k] += second * stress[k];
            laminate.momentPerGradient[k] += third * stress[k];
        }
        laminate.thickness += ply.thickness;
        laminate.massPerArea += ply.material.density * ply.thickness;
        // A ply rigid in transverse shear (G1Z or G2Z blank) makes the laminate rigid in that direction.
        isRigidInX = isRigidInX || !(ply.material.transverseShear[0] > 0.0);
        isRigidInY = isRigidInY || !(ply.material.transverseShear[1] > 0.0);
        laminate.plies.push_back(std::move(ply));
    }
    // A blank Z0 is −T/2 and follows the plies' thicknesses (PCOMP); the card says whether it was given from data
    // contract 54 (z0_given), before it a Z0 equal to −T/2 to the bit is taken as blank (as
    // design::DerivedProperties, whose overlays move the stack's z with the same rule).
    if (card.hasColumn("bottomOffsetGiven") && card.columnType("bottomOffsetGiven") == "uint8")
        laminate.isZ0Blank =
            card.values<std::uint8_t>("bottomOffsetGiven",
                                      base::NullPolicy::Allow)[static_cast<std::size_t>(propertyRow)] == 0;
    else
        laminate.isZ0Blank = !laminate.plies.empty() && laminate.plies.front().zBottom == -0.5 * laminate.thickness;
    // SOUT defaults to NO, but a laminate whose plies all leave SOUT blank or NO writes every ply, so that a ply
    // output asked for is never empty; with a YES on some plies, only those.
    const bool isAnyPlyAsked =
        std::any_of(laminate.plies.begin(), laminate.plies.end(), [](const LaminatePly& ply) { return ply.hasOutput; });
    if (!isAnyPlyAsked)
        for (LaminatePly& ply : laminate.plies)
            ply.hasOutput = true;
    const std::array<bool, 2> isRigid{isRigidInX, isRigidInY};
    for (std::size_t d = 0; d < 2; ++d)
    {
        if (isRigid[d])
            continue;
        setTransverseShear(laminate, d);
        // Every ply of a homogeneous stack (one material, one angle) takes the mean shear; a stack of several materials
        // or angles (a sandwich) the equilibrium distribution at the top of each ply.
        laminate.hasShearDistribution[d] = !isHomogeneous(laminate);
    }

    switch (laminate.option)
    {
    case LaminateOption::Full:
        break;
    case LaminateOption::Membrane:
        // CompProp LAM = MEM: only the membrane terms (MID1 of the equivalent ShellProp).
        laminate.b = {};
        laminate.d = {};
        laminate.momentPerDegree = {};
        laminate.momentPerGradient = {};
        break;
    case LaminateOption::Bending:
        // LAM = BEND: only the bending terms (MID2).
        laminate.a = {};
        laminate.b = {};
        laminate.forcePerDegree = {};
        laminate.forcePerGradient = {};
        break;
    case LaminateOption::Smeared:
        // LAM = SMEAR: MID1 = MID2 of the equivalent ShellProp with 12I/T³ = 1, MID3 and MID4 blank: D = A t²/12, no
        // coupling, rigid transverse shear; the thermal moments follow the same smearing (none from a uniform ΔT).
        laminate.d = (laminate.thickness * laminate.thickness / 12.0) * laminate.a;
        laminate.b = {};
        laminate.transverseShear = {};
        // Rigid in transverse shear: every ply takes the mean shear Q / t, as in a rigid direction.
        laminate.hasShearDistribution = {};
        laminate.momentPerDegree = {};
        for (std::size_t k = 0; k < 3; ++k)
            laminate.momentPerGradient[k] = laminate.thickness * laminate.thickness / 12.0 * laminate.forcePerDegree[k];
        laminate.forcePerGradient = {};
        break;
    }
    return Result::success(std::move(laminate));
}

} // namespace oos::elements
