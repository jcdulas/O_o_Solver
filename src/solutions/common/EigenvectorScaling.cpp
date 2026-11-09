#include "oos/solutions/common/EigenvectorScaling.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>

namespace oos::solutions
{

namespace
{

// The row of largest magnitude of a column among the compared rows (every row when isCompared is empty; the first on a
// tie), or nothing for a zero column.
std::optional<std::size_t> largestRow(const linalg::DenseMatrix& vectors, std::size_t column,
                                      std::span<const std::uint8_t> isCompared)
{
    std::optional<std::size_t> found;
    double largest = 0.0;
    for (std::size_t row = 0; row < vectors.rows(); ++row)
    {
        if (!isCompared.empty() && isCompared[row] == 0)
            continue;
        const double magnitude = std::abs(vectors(row, column));
        if (magnitude > largest)
        {
            largest = magnitude;
            found = row;
        }
    }
    return found;
}

base::Diagnostic pointProblem(const EigenMethod& method, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = base::Severity::Warning;
    diagnostic.code = "norm-point-fallback";
    diagnostic.card = method.card;
    diagnostic.id = method.setId;
    diagnostic.field = "normNode";
    diagnostic.message = std::move(message);
    return diagnostic;
}

// The qualified neutral name of a method field in a diagnostic.
std::string qualifiedField(const EigenMethod& method, const char* field)
{
    return base::concatenate({method.card, ".", field});
}

// The normalisation field of the method card: its neutral name differs from card to card.
std::string normalizationField(const EigenMethod& method)
{
    return qualifiedField(method, method.card == "RealEigenMethod" ? "normalization" : "norm");
}

void reportPointOutside(const EigenMethod& method, base::DiagnosticSink& sink)
{
    sink.report(pointProblem(
        method, base::concatenate(
                    {normalizationField(method), " = Point names point ", std::to_string(method.normalizationPoint),
                     " component ", std::to_string(method.normalizationComponent),
                     ", which is not in the analysis set (unknown or "
                     "constrained): the vectors are normalised by MAX; "
                     "name a free DOF in ",
                     qualifiedField(method, "normNode"), " and ", qualifiedField(method, "normComponent")})));
}

// The scaling of every column by the method, the pivot of Point at pointRow (nothing: Maximum), the largest component
// sought among the compared rows.
std::vector<double> scaleColumns(linalg::DenseMatrix& vectors, const EigenMethod& method,
                                 std::optional<std::size_t> pointRow, std::span<const std::uint8_t> isCompared,
                                 base::DiagnosticSink& sink)
{
    std::vector<double> scales(vectors.columns(), 1.0); // bounded by the modes
    const bool isPointKnown = pointRow.has_value();
    for (std::size_t column = 0; column < vectors.columns(); ++column)
    {
        const std::optional<std::size_t> largest = largestRow(vectors, column, isCompared);
        if (!largest)
            continue;
        // The vector is divided by its pivot (Maximum, Point), so that the pivot becomes exactly +1, or multiplied by
        // ±1 (MASS).
        double pivot = vectors(*largest, column) < 0.0 ? -1.0 : 1.0;
        switch (method.normalization)
        {
        case EigenNormalization::Mass:
            break;
        case EigenNormalization::Max:
            pivot = vectors(*largest, column);
            break;
        case EigenNormalization::Point:
            if (isPointKnown && vectors(*pointRow, column) != 0.0)
                pivot = vectors(*pointRow, column);
            else
            {
                if (isPointKnown)
                    sink.report(pointProblem(
                        method, base::concatenate(
                                    {normalizationField(method), " = Point: mode ", std::to_string(column + 1),
                                     " does not move point ", std::to_string(method.normalizationPoint), " component ",
                                     std::to_string(method.normalizationComponent), ": it is normalised by MAX"})));
                pivot = vectors(*largest, column);
            }
            break;
        }
        for (std::size_t row = 0; row < vectors.rows(); ++row)
            vectors(row, column) /= pivot;
        const double scale = 1.0 / pivot;
        scales[column] = scale;
    }
    return scales;
}

} // namespace

std::vector<double> scaleEigenvectors(linalg::DenseMatrix& vectors, const EigenMethod& method,
                                      const assembly::DofMap& dofMap, const linalg::DofSelection& selection,
                                      base::DiagnosticSink& sink)
{
    std::optional<std::size_t> pointRow;
    if (method.normalization == EigenNormalization::Point)
    {
        const std::optional<base::DofIndex> dof =
            dofMap.findDof(method.normalizationPoint, method.normalizationComponent);
        const std::int32_t equation = dof ? selection.equationOfDof(*dof) : linalg::DofSelection::ELIMINATED;
        if (equation == linalg::DofSelection::ELIMINATED)
            reportPointOutside(method, sink);
        else
            pointRow = static_cast<std::size_t>(equation);
    }
    return scaleColumns(vectors, method, pointRow, {}, sink);
}

std::vector<double> scaleEigenvectorsOnSet(linalg::DenseMatrix& vectorsG, std::span<const std::uint8_t> isAnalysisSet,
                                           const EigenMethod& method, const assembly::DofMap& dofMap,
                                           base::DiagnosticSink& sink)
{
    if (isAnalysisSet.size() != vectorsG.rows())
        throw std::invalid_argument(base::concatenate({"scaleEigenvectorsOnSet: ", std::to_string(isAnalysisSet.size()),
                                                       " flags for vectors of ", std::to_string(vectorsG.rows()),
                                                       " rows; give one flag per g-set DOF"}));
    std::optional<std::size_t> pointRow;
    if (method.normalization == EigenNormalization::Point)
    {
        const std::optional<base::DofIndex> dof =
            dofMap.findDof(method.normalizationPoint, method.normalizationComponent);
        if (!dof || isAnalysisSet[static_cast<std::size_t>(*dof)] == 0)
            reportPointOutside(method, sink);
        else
            pointRow = static_cast<std::size_t>(*dof);
    }
    return scaleColumns(vectorsG, method, pointRow, isAnalysisSet, sink);
}

} // namespace oos::solutions
