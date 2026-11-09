// The family of ShellTria3 (ShellTria3Element.hpp): EdgeShearShellFamily instantiated once on the NASTRAN-95 TRIA3
// kernel.

#include "oos/elements/shells/shellTria3/ShellTria3Element.hpp"

#include "../EdgeShearShellFamily.ipp"
#include "../EdgeShearShellRecovery.ipp"
#include "../EdgeShearShellDerivatives.ipp"

namespace oos::elements
{

template class EdgeShearShellFamily<ShellTria3>;

} // namespace oos::elements
