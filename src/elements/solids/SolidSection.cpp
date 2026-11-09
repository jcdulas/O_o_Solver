#include "oos/elements/solids/SolidSection.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementReferences.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace oos::elements
{

namespace
{

// The snapshot table of the AnisoMat3D cards (the material card tables "materials/<CARD>" of the snapshot).
constexpr std::string_view ANISO_MAT3D_TABLE = "materials/AnisoMat3D";
// The number of terms of a AnisoMat3D matrix (the upper triangle of 6 × 6), and of its expansion coefficients A1-A6.
constexpr std::size_t ANISO_MAT3D_TERM_COUNT = 21;
constexpr std::size_t ANISO_MAT3D_EXPANSION_COUNT = 6;
// SolidProp CORDM −1 and −2: the element coordinate systems of the solids.
constexpr std::int32_t ELEMENT_SYSTEM = -1;
// The relative size below which a Cholesky pivot counts as zero (a material matrix that is singular to round-off).
constexpr double PIVOT_TOLERANCE = 1.0e-12;
// The upper bound of NU for a solid: ν = 0.5 makes λ infinite (an incompressible material).
constexpr double POISSON_RATIO_LIMIT = 0.5;

[[nodiscard]] bool isOneOf(std::string_view text, std::initializer_list<std::string_view> words)
{
    return std::find(words.begin(), words.end(), text) != words.end();
}

// Why IN and ISOP are not supported for a shape (empty when they are), and the integration they choose.
struct IntegrationChoice
{
    SolidIntegration integration = SolidIntegration::Standard;
    std::string problem;
    const char* field = "SolidProp.integration";
};

// The integration of a shape from IN and ISOP (SolidSection.hpp: resolveSolidSection lists the accepted pairs).
[[nodiscard]] IntegrationChoice chooseIntegration(SolidShape shape, std::string_view in, std::string_view isop)
{
    IntegrationChoice choice;
    const auto accept = [&](SolidIntegration integration)
    {
        choice.integration = integration;
        return choice;
    };
    std::string_view expected;
    switch (shape)
    {
    case SolidShape::Hexa8:
        if (isOneOf(in, {"", "BubbleFunctions"}) && isOneOf(isop, {"", "Reduced"}))
            return accept(SolidIntegration::Standard);
        if (isop == "Full" && isOneOf(in, {"", "Gauss", "TwoPoints"}))
            return accept(SolidIntegration::FullGauss);
        expected =
            "SolidProp.integration and SolidProp.integrationOrder blank (incompatible modes) or "
            "SolidProp.integrationOrder = Full with "
            "SolidProp.integration blank, Gauss or TwoPoints (the trilinear element with 2 × 2 × 2 points)";
        break;
    case SolidShape::Hexa20:
        if (isOneOf(in, {"", "ThreePoints"}) && isOneOf(isop, {"", "Full"}))
            return accept(SolidIntegration::Standard);
        if ((isop == "Reduced" && isOneOf(in, {"", "TwoPoints"})) || (in == "TwoPoints" && isop.empty()))
            return accept(SolidIntegration::ReducedGauss);
        expected =
            "SolidProp.integration and SolidProp.integrationOrder blank (3 × 3 × 3 points), "
            "SolidProp.integrationOrder = Full or "
            "SolidProp.integration = ThreePoints (the same), or SolidProp.integrationOrder = Reduced "
            "or "
            "SolidProp.integration = TwoPoints "
            "(2 × 2 × 2 points)";
        break;
    case SolidShape::Penta6:
        // The linear wedge has one element: the field values of a reduced or bubble form name it too.
        if ((isOneOf(in, {"", "BubbleFunctions"}) && isOneOf(isop, {"", "Reduced"})) ||
            (isop == "Full" && isOneOf(in, {"", "Gauss", "TwoPoints"})))
            return accept(SolidIntegration::Standard);
        expected = "SolidProp.integration and SolidProp.integrationOrder blank";
        break;
    case SolidShape::Penta15:
        if (isOneOf(in, {"", "ThreePoints"}) && isOneOf(isop, {"", "Full"}))
            return accept(SolidIntegration::Standard);
        expected = "SolidProp.integration and SolidProp.integrationOrder blank";
        break;
    case SolidShape::Tetra4:
        // The 4-node SolidTetra's default is the one-point rule, IN blank or TWO; its strain is constant, so every rule
        // gives the same matrices: ISOP = FULL and REDUCED name that element too.
        if (isOneOf(in, {"", "TwoPoints"}) && isOneOf(isop, {"", "Full", "Reduced"}))
            return accept(SolidIntegration::Standard);
        expected = "SolidProp.integration and SolidProp.integrationOrder blank";
        break;
    case SolidShape::Tetra10:
    case SolidShape::Pyram5:
    case SolidShape::Pyram13:
        if (in.empty() && isOneOf(isop, {"", "Full"}))
            return accept(SolidIntegration::Standard);
        expected = "SolidProp.integration and SolidProp.integrationOrder blank";
        break;
    }
    choice.field = in.empty() ? "SolidProp.integrationOrder" : "SolidProp.integration";
    choice.problem = base::concatenate(
        {"expected ", expected, ", found SolidProp.integration = ", in.empty() ? std::string_view("blank") : in,
         " and SolidProp.integrationOrder = ", isop.empty() ? std::string_view("blank") : isop,
         "; this integration is not supported: leave SolidProp.integration and "
         "SolidProp.integrationOrder blank"});
    return choice;
}

} // namespace

SolidSectionColumns::SolidSectionColumns(const ElementBatchView& view)
    : properties(view, "SolidProp"), isotropic(view), anisotropic(view.snapshot().findTable(ANISO_MAT3D_TABLE))
{
    if (properties.table != nullptr)
    {
        const base::ContractReader& solidProp = properties.table->reader();
        matId = solidProp.values<std::int32_t>("matId");
        matFrameId = solidProp.values<std::int32_t>("matFrameId");
        integrationNetwork = solidProp.texts("integration");
        stressPoints = solidProp.texts("stressLocation");
        integrationScheme = solidProp.texts("integrationOrder");
        function = solidProp.texts("elementFunction");
    }
    if (anisotropic != nullptr)
    {
        anisotropicIds = anisotropic->values<std::int32_t>("id");
        anisotropicTerms = anisotropic->fixedLists<double>("stiffness");
        if (anisotropic->hasColumn("thermalExpansion"))
            anisotropicExpansions = anisotropic->fixedLists<double>("thermalExpansion");
        if (anisotropic->hasColumn("referenceTemperature"))
            anisotropicReferences = anisotropic->values<double>("referenceTemperature");
    }
}

Mat6 anisotropicSolidMatrix(std::span<const double> terms)
{
    Mat6 d;
    std::size_t k = 0;
    for (std::size_t i = 0; i < 6; ++i)
        for (std::size_t j = i; j < 6; ++j)
        {
            d(i, j) = terms[k];
            d(j, i) = terms[k];
            ++k;
        }
    return d;
}

bool isPositiveDefinite(const Mat6& matrix)
{
    Mat6 l;
    double largest = 0.0;
    for (std::size_t i = 0; i < 6; ++i)
        largest = std::max(largest, std::abs(matrix(i, i)));
    if (largest <= 0.0)
        return false;
    for (std::size_t j = 0; j < 6; ++j)
    {
        double pivot = matrix(j, j);
        for (std::size_t k = 0; k < j; ++k)
            pivot -= l(j, k) * l(j, k);
        if (pivot <= PIVOT_TOLERANCE * largest)
            return false;
        l(j, j) = std::sqrt(pivot);
        for (std::size_t i = j + 1; i < 6; ++i)
        {
            double sum = matrix(i, j);
            for (std::size_t k = 0; k < j; ++k)
                sum -= l(i, k) * l(j, k);
            l(i, j) = sum / l(j, j);
        }
    }
    return true;
}

Resolution<SolidSection> resolveSolidSection(const ElementBatchView& view, const SolidSectionColumns& columns,
                                             std::int64_t row, SolidShape shape)
{
    using Result = Resolution<SolidSection>;
    Resolution<PropertyRef> property = resolveProperty(view, columns.properties, row);
    if (!property)
        return Result::failure(std::move(property.problem));
    const auto p = static_cast<std::size_t>(property.value->row);
    const std::int64_t propertyRow = property.value->row;

    const std::string_view function = columns.function[propertyRow];
    if (function != "SolidMechanics")
        return Result::failure(elementProblem(
            view, row, problem_code::UNSUPPORTED_FIELD, "SolidProp.elementFunction",
            base::concatenate({"expected SolidProp.elementFunction = SolidMechanics (a structural solid), found ",
                               function,
                               "; fluid, poroelastic and solid-shell properties are outside a linear structural "
                               "analysis: give a SolidProp of SolidProp.elementFunction SolidMechanics"})));
    // SolidProp: STRESS = GAUSS asks for the stresses at the Gauss points. The 4-node SolidTetra has one, its centroid,
    // and a constant strain: that one value is written at the centre and at the four corners, which is what the GRID
    // output gives. The other shapes' Gauss point output (their order and the grid point naming each) is not
    // implemented.
    if (columns.stressPoints[propertyRow] == "Gauss" && shape != SolidShape::Tetra4)
        return Result::failure(elementProblem(
            view, row, problem_code::UNSUPPORTED_FIELD, "SolidProp.stressLocation",
            "expected SolidProp.stressLocation blank or Node (stresses at the centre and the corners), found "
            "Gauss; the output at the Gauss points is supported for the 4-node SolidTetra only (its one point is its "
            "centroid): leave SolidProp.stressLocation blank"));
    const IntegrationChoice integration =
        chooseIntegration(shape, columns.integrationNetwork[propertyRow], columns.integrationScheme[propertyRow]);
    if (!integration.problem.empty())
        return Result::failure(
            elementProblem(view, row, problem_code::UNSUPPORTED_FIELD, integration.field, integration.problem));

    SolidSection section;
    section.integration = integration.integration;
    const std::int32_t matFrameId = columns.matFrameId[p];
    if (matFrameId <= ELEMENT_SYSTEM)
        return Result::failure(elementProblem(
            view, row, problem_code::UNSUPPORTED_FIELD, "SolidProp.matFrameId",
            base::concatenate({"expected SolidProp.matFrameId 0 (the basic system) or a rectangular coordinate "
                               "system, found ",
                               std::to_string(matFrameId),
                               " (an element coordinate system of the solids), which is not supported yet; give "
                               "SolidProp.matFrameId 0 or the ID of a rectangular system"})));
    if (matFrameId > 0)
    {
        Resolution<input::CoordinateFrame> frame =
            resolveRectangularFrame(view, row, matFrameId, "SolidProp.matFrameId");
        if (!frame)
            return Result::failure(std::move(frame.problem));
        section.axes = frame.value->axes;
        section.origin = frame.value->origin;
    }

    const std::int32_t matId = columns.matId[p];
    const std::optional<std::int32_t> materialRow = view.index().findMaterialRow(matId);
    if (!materialRow)
        return Result::failure(elementProblem(
            view, row, problem_code::MISSING_MATERIAL, "SolidProp.matId",
            base::concatenate(
                {"expected the ID of a material card, found ", std::to_string(matId),
                 ", which the model does not define; define the IsoMat or AnisoMat3D or correct the SolidProp"})));
    const std::string_view type = columns.isotropic.types[*materialRow];
    if (type == "IsoMat")
    {
        Resolution<IsotropicMaterial> material =
            resolveIsotropicMaterial(view, columns.isotropic, row, matId, "SolidProp.matId");
        if (!material)
            return Result::failure(std::move(material.problem));
        const double e = material.value->youngModulus;
        const double nu = material.value->poissonRatio;
        if (!(e > 0.0) || !(nu > -1.0 && nu < POISSON_RATIO_LIMIT))
            return Result::failure(elementProblem(
                view, row, problem_code::INVALID_FIELD, "SolidProp.matId",
                base::concatenate(
                    {"expected IsoMat ", std::to_string(matId),
                     " with IsoMat.youngsModulus > 0 and −1 < IsoMat.poissonRatio < 0.5, found "
                     "IsoMat.youngsModulus ",
                     std::to_string(e), " and IsoMat.poissonRatio ", std::to_string(nu),
                     "; a solid needs both (an incompressible IsoMat.poissonRatio = 0.5 is not supported): "
                     "correct the "
                     "IsoMat"})));
        section.elasticity = isotropicSolidMatrix(e, nu);
        section.density = material.value->density;
        section.expansion = {
            material.value->expansion, material.value->expansion, material.value->expansion, 0.0, 0.0, 0.0};
        section.referenceTemperature = material.value->referenceTemperature;
        return Result::success(section);
    }
    if (type == "AnisoMat3D")
    {
        if (columns.anisotropic == nullptr)
            return Result::failure(elementProblem(
                view, row, problem_code::UNSUPPORTED_MATERIAL, "SolidProp.matId",
                base::concatenate({"expected the constants of AnisoMat3D ", std::to_string(matId),
                                   ", found no AnisoMat3D card table in the snapshot (\"materials/AnisoMat3D\": the "
                                   "snapshot was taken without it); take the snapshot with the AnisoMat3D table"})));
        const auto found = std::find(columns.anisotropicIds.begin(), columns.anisotropicIds.end(), matId);
        if (found == columns.anisotropicIds.end())
            return Result::failure(elementProblem(
                view, row, problem_code::MISSING_MATERIAL, "SolidProp.matId",
                base::concatenate(
                    {"expected AnisoMat3D ", std::to_string(matId),
                     " in the AnisoMat3D card table, found it only in the materials table; the two tables "
                     "come from different snapshots"})));
        const std::span<const double> terms = columns.anisotropicTerms[found - columns.anisotropicIds.begin()];
        if (terms.size() != ANISO_MAT3D_TERM_COUNT)
            return Result::failure(elementProblem(
                view, row, problem_code::INVALID_FIELD, "SolidProp.matId",
                base::concatenate({"expected the 21 terms AnisoMat3D.stiffness11-AnisoMat3D.stiffness66 of "
                                   "AnisoMat3D ",
                                   std::to_string(matId), ", found ", std::to_string(terms.size()),
                                   "; the AnisoMat3D table breaks the data contract"})));
        section.elasticity = anisotropicSolidMatrix(terms);
        if (!isPositiveDefinite(section.elasticity))
            return Result::failure(elementProblem(
                view, row, problem_code::INVALID_FIELD, "SolidProp.matId",
                base::concatenate({"expected the matrix AnisoMat3D.stiffness11-AnisoMat3D.stiffness66 of "
                                   "AnisoMat3D ",
                                   std::to_string(matId),
                                   " to be positive definite, found it is not (a material that stores no energy in "
                                   "some strain state); correct the AnisoMat3D"})));
        const input::MaterialTable& materials = view.snapshot().materials();
        const auto density = materials.reader().values<double>("density", base::NullPolicy::Allow);
        section.density = materials.reader().validity("density").isValid(*materialRow)
                              ? density[static_cast<std::size_t>(*materialRow)]
                              : 0.0;
        const auto anisoMat3DRow = found - columns.anisotropicIds.begin();
        if (columns.anisotropicExpansions.size() > 0)
        {
            const std::span<const double> expansions = columns.anisotropicExpansions[anisoMat3DRow];
            if (expansions.size() != ANISO_MAT3D_EXPANSION_COUNT)
                return Result::failure(elementProblem(
                    view, row, problem_code::INVALID_FIELD, "SolidProp.matId",
                    base::concatenate({"expected the 6 expansion coefficients "
                                       "AnisoMat3D.thermalExpansion1-AnisoMat3D.thermalExpansion6 of "
                                       "AnisoMat3D ",
                                       std::to_string(matId), ", found ", std::to_string(expansions.size()),
                                       "; the AnisoMat3D table breaks the data contract"})));
            std::copy(expansions.begin(), expansions.end(), section.expansion.begin());
        }
        if (!columns.anisotropicReferences.empty())
            section.referenceTemperature = columns.anisotropicReferences[static_cast<std::size_t>(anisoMat3DRow)];
        return Result::success(section);
    }
    return Result::failure(elementProblem(
        view, row, problem_code::UNSUPPORTED_MATERIAL, "SolidProp.matId",
        base::concatenate({"expected a IsoMat or a AnisoMat3D for material ", std::to_string(matId), ", found a ", type,
                           "; ", view.card(), " takes an isotropic (IsoMat) or anisotropic (AnisoMat3D) material"})));
}

} // namespace oos::elements
