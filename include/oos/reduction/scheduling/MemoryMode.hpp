#pragma once

#include <cstdint>
#include <string_view>

namespace oos::reduction
{

// How a condensed static analysis spends its memory budget: the caller's option `memory_mode` (or the environment
// variable OOS_MEMORY_MODE). Every mode gives the same results to the bit; they differ in what the analysis keeps in
// memory, and so in its time and in the smallest budget it fits (CondensedMemoryPlan):
// - Speed: everything kept while it is needed (each unit's equations to the end, every recovered unit's recovery for
//   the whole constraint set), refused when it does not fit;
// - Auto (the default): Speed when it fits; otherwise the units recovered in sequence (each unit's recovery made,
//   used for the cases of a batch and released), then, when that does not fit either, the equations of as few units
//   as needed released after their condensation and built again from the model at their recovery; refused only when
//   even every release does not fit;
// - LowMemory: the units recovered in sequence and every unit's equations released after its condensation, always:
//   the smallest and most predictable memory, at the price of building each recovered unit's equations again.
enum class MemoryMode : std::uint8_t
{
    Auto,
    Speed,
    LowMemory
};

// "auto", "speed", "low_memory": the names of the option.
[[nodiscard]] const char* nameOf(MemoryMode mode) noexcept;

// The mode of a name of the option. Throws std::invalid_argument for another text, naming the three.
[[nodiscard]] MemoryMode memoryModeOf(std::string_view name);

} // namespace oos::reduction
