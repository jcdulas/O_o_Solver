#pragma once

#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementFamily.hpp"

#include <optional>
#include <span>
#include <string>

namespace oos::elements
{

// The direction of a pressure on a shell (PLOAD4 fields CID, N1-N3: the pressure acts along N in CID when N is given,
// else along the element normal): a normal pressure (nothing), or the unit vector N
// of the rectangular system CID written in the element frame (columns of `elementAxes`: x, y, n in basic). The
// pressure p then acts as a traction p·N/|N| per unit area of the element's mean surface.
struct PressureDirection
{
    std::optional<base::Vec3> inElementFrame;
    std::string problem; // why the direction cannot be used (empty when it can)
};

[[nodiscard]] PressureDirection pressureDirectionOf(const ElementBatchView& view, const ElementPressure& pressure,
                                                    const base::Mat3& elementAxes);

// Reports the pressures of a shell batch that the loads facet cannot apply: a direction in an unknown or curvilinear
// system or of zero length (field CID). `card` names the element in the messages.
void checkShellPressures(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                         base::DiagnosticSink& sink);

} // namespace oos::elements
