#include "oos/elements/shells/ShellConnectivity.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementReferences.hpp"

#include <string>

namespace oos::elements
{

ShellConnectivity::ShellConnectivity(const ElementBatchView& view)
    : rows(view.batch().nodeRows()), ids(view.batch().nodes()), valid(view.columns().validity("nodesRows"))
{
}

template <std::size_t N>
Resolution<ShellCorners<N>> resolveShellCorners(const ElementBatchView& view, const ShellConnectivity& connectivity,
                                                std::int64_t row)
{
    using Result = Resolution<ShellCorners<N>>;
    const std::int64_t nodeCount = view.snapshot().nodes().rowCount();
    const std::span<const std::int32_t> rows = connectivity.rows[row];
    const std::span<const std::int32_t> ids = connectivity.ids[row];
    const bool isValid = connectivity.valid.isValid(row) && rows.size() == N;
    ShellCorners<N> corners;
    for (std::size_t i = 0; i < N; ++i)
    {
        const std::string field = std::string("node").append(std::to_string(i + 1));
        // A null row of nodesRows does not say which grid point is unknown: the IDs tell it.
        const bool isKnown = isValid ? rows[i] >= 0 && rows[i] < nodeCount
                                     : i < ids.size() && view.index().points().findGrid(ids[i]).has_value();
        if (!isKnown)
        {
            const std::string id = i < ids.size() ? std::to_string(ids[i]) : std::string("blank");
            return Result::failure(elementProblem(
                view, row, problem_code::MISSING_POINT, field,
                base::concatenate({"expected the ID of a grid point, found ", id,
                                   ", which the model does not define; define it or correct the card"})));
        }
        for (std::size_t j = 0; j < i; ++j)
            if (ids[j] == ids[i])
                return Result::failure(elementProblem(
                    view, row, problem_code::DEGENERATE_GEOMETRY, field,
                    base::concatenate({"expected distinct grid points, found ", std::to_string(ids[i]),
                                       " twice; a shell needs its corners distinct: correct the card"})));
    }
    if (!isValid)
        throw base::ContractError(base::concatenate(
            {view.card(), " ", std::to_string(view.elementId(row)),
             ": nodesRows is null although every grid point exists; the element table and the node table come from "
             "different snapshots: take every table of the snapshot from one model"}));
    for (std::size_t i = 0; i < N; ++i)
    {
        const std::int32_t gridRow = checkedGridRow(view, rows[i], ids[i]);
        corners.rows[i] = gridRow;
        corners.ids[i] = ids[i];
        corners.positions[i] = view.positionInBasic(gridRow);
        corners.cdFrames[i] = view.displacementFrame(gridRow);
    }
    return Result::success(corners);
}

template Resolution<ShellCorners<3>> resolveShellCorners<3>(const ElementBatchView&, const ShellConnectivity&,
                                                            std::int64_t);
template Resolution<ShellCorners<4>> resolveShellCorners<4>(const ElementBatchView&, const ShellConnectivity&,
                                                            std::int64_t);
template Resolution<ShellCorners<6>> resolveShellCorners<6>(const ElementBatchView&, const ShellConnectivity&,
                                                            std::int64_t);
template Resolution<ShellCorners<8>> resolveShellCorners<8>(const ElementBatchView&, const ShellConnectivity&,
                                                            std::int64_t);

} // namespace oos::elements
