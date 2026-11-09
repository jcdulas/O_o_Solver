#include "oos/elements/core/ScalarConnection.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <string>

namespace oos::elements
{

namespace
{

constexpr int FIRST_GRID_COMPONENT = 1;
constexpr int LAST_GRID_COMPONENT = 6;
constexpr double SIGN_OF_END[2] = {1.0, -1.0}; // δ = u1 − u2

} // namespace

double ScalarConnection::relativeMotion(const NodalDisplacements& displacements) const
{
    double motion = 0.0;
    for (std::size_t i = 0; i < count; ++i)
        motion += signs[i] * displacements.valueAt(dofs[i]);
    return motion;
}

Resolution<ScalarConnection> resolveScalarConnection(const ElementBatchView& view, std::int64_t row, std::int32_t g1,
                                                     std::int32_t c1, std::int32_t g2, std::int32_t c2,
                                                     bool isScalarPointCard)
{
    using Result = Resolution<ScalarConnection>;
    const char* pointName = isScalarPointCard ? "scalarPoint" : "node";
    const std::array<std::int32_t, 2> ids{g1, g2};
    const std::array<std::int32_t, 2> components{c1, c2};
    ScalarConnection connection;
    for (std::size_t end = 0; end < 2; ++end)
    {
        if (ids[end] == 0)
            continue; // a grounded end (a blank Gi grounds that end)
        const std::string field = std::string(pointName).append(std::to_string(end + 1));
        const std::optional<input::PointRef> point = view.index().points().find(ids[end]);
        if (!point || (isScalarPointCard && point->kind != input::PointKind::Scalar))
            return Result::failure(elementProblem(
                view, row, problem_code::MISSING_POINT, field,
                base::concatenate({"expected the ID of ",
                                   isScalarPointCard ? "a scalar point" : "a grid or scalar point", ", found ",
                                   std::to_string(ids[end]),
                                   ", which the model does not define as such; define it or correct the card"})));
        const std::int32_t component = components[end];
        ElementDof dof;
        if (point->kind == input::PointKind::Scalar)
        {
            if (component != 0)
                return Result::failure(elementProblem(
                    view, row, problem_code::INVALID_COMPONENT,
                    std::string("components").append(std::to_string(end + 1)),
                    base::concatenate({"expected 0 or blank for the scalar point ", std::to_string(ids[end]),
                                       " (a scalar point has one DOF), found ", std::to_string(component)})));
            dof = ElementDof::scalar(point->row);
        }
        else
        {
            if (component < FIRST_GRID_COMPONENT || component > LAST_GRID_COMPONENT)
                return Result::failure(elementProblem(
                    view, row, problem_code::INVALID_COMPONENT,
                    std::string("components").append(std::to_string(end + 1)),
                    base::concatenate({"expected a component 1 to 6 of the grid point ", std::to_string(ids[end]),
                                       ", found ", std::to_string(component)})));
            dof = ElementDof::grid(point->row, component);
        }
        connection.dofs[connection.count] = dof;
        connection.signs[connection.count] = SIGN_OF_END[end];
        ++connection.count;
    }
    if (connection.count == 0)
        return Result::failure(elementProblem(view, row, problem_code::INVALID_FIELD,
                                              std::string(pointName).append("1"),
                                              "expected at least one connected point, found both ends blank; give "
                                              "SpringDofProperty.node1 or SpringDofProperty.node2 "
                                              "(SpringScalarProperty.scalarPoint1 or "
                                              "SpringScalarProperty.scalarPoint2)"));
    if (connection.count == 2 && connection.dofs[0] == connection.dofs[1])
        return Result::failure(elementProblem(
            view, row, problem_code::DUPLICATE_DOF, std::string(pointName).append("2"),
            base::concatenate({"expected two distinct DOFs, found point ", std::to_string(g1), " component ",
                               std::to_string(c1), " at both ends (the two connection points must be distinct)"})));
    return Result::success(connection);
}

void appendScalarMatrix(ElementMatrixBlock& block, std::int64_t row, const ScalarConnection& connection, double value)
{
    std::array<double, 4> matrix{};
    const std::size_t n = connection.count;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            matrix[i * n + j] = value * connection.signs[i] * connection.signs[j];
    block.append(row, connection.presentDofs(), std::span<const double>(matrix.data(), n * n));
}

} // namespace oos::elements
