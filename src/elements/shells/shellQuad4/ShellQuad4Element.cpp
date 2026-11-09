// The family of ShellQuad4 (ShellQuad4Element.hpp): EdgeShearShellFamily instantiated once on the NASTRAN-95 QUAD4
// kernel.

#include "oos/elements/shells/shellQuad4/ShellQuad4Element.hpp"

#include "../EdgeShearShellFamily.ipp"
#include "../EdgeShearShellRecovery.ipp"
#include "../EdgeShearShellDerivatives.ipp"

namespace oos::elements
{

template class EdgeShearShellFamily<ShellQuad4>;

} // namespace oos::elements
