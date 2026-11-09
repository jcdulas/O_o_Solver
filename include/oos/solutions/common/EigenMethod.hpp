#pragma once

#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/linalg/eigen/EigenRequest.hpp"
#include "oos/solutions/common/RunCatalog.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace oos::solutions
{

// How the eigenvectors of a request are scaled (field NORM of LanczosEigenMethod, RealEigenMethod,
// BucklingEigenMethod).
enum class EigenNormalization : std::uint8_t
{
    Mass, // unit generalised mass φᵀ·M·φ = 1 (the default of a vibration analysis)
    Max,  // the component of largest magnitude in the analysis set is +1 (the default of a buckling analysis)
    Point // the component C of point G is +1 (EIGR and BucklingEigenMethod)
};

// The eigenvalue method a subcase selects with METHOD (EIGRL, RealEigenMethod, BucklingEigenMethod), as the solver uses
// it.
//
// The range is in the units of the card: cycles per unit time for a vibration analysis (EIGRL V1, V2 and
// RealEigenMethod F1, F2), load factors for a buckling analysis (EIGRL V1, V2 and BucklingEigenMethod L1, L2). An empty
// bound is a blank field.
struct EigenMethod
{
    std::string card; // "LanczosEigenMethod", "RealEigenMethod" or "BucklingEigenMethod"
    base::EntityId setId = 0;
    std::optional<double> lower;
    std::optional<double> upper;
    std::size_t modeCount = 0;     // ND (EIGB: NDP and NDN of the range); 0 for a blank field
    std::size_t blockSize = 0;     // LanczosEigenMethod MAXSET; 0 for a blank field (the solver's default)
    std::int32_t messageLevel = 0; // LanczosEigenMethod MSGLVL: above 0 the Lanczos shifts are reported
    EigenNormalization normalization = EigenNormalization::Mass;
    base::EntityId normalizationPoint = 0;   // NORM = POINT: G
    std::int32_t normalizationComponent = 0; // NORM = POINT: C (0 for a scalar point)
};

// The method of a subcase: the LanczosEigenMethod, RealEigenMethod (vibration) or BucklingEigenMethod (buckling) card
// of set ID `setId` (an LanczosEigenMethod wins when both an LanczosEigenMethod and the other card have the ID). The
// fields the solver cannot honour are diagnostics: an RealEigenMethod method other than LAN (AHOU, INV, SINV, GIV,
// MGIV, HOU, MHOU, AGIV) is refused naming it; BucklingEigenMethod's inverse power methods (INV, SINV) are replaced by
// the block Lanczos solver over the same range (information); NORM = MASS in a buckling analysis is replaced by MAX
// (MASS is not available for buckling; warning). No card of the ID is an error, and so is a range the solver cannot
// take (a bound not finite, V1 not below V2, V2 not above 0 when the roots are sought from the lowest), naming its
// fields.
[[nodiscard]] std::optional<EigenMethod> readEigenMethod(const input::ModelSnapshot& snapshot, base::EntityId setId,
                                                         AnalysisKind analysis, base::EntityId subcase,
                                                         base::DiagnosticSink& sink);

// The request of the block Lanczos solver (linalg::EigenRequest) for a method (the LanczosEigenMethod rule on V1, V2,
// ND):
//
//   V1   V2   ND     the roots computed
//   V1   V2   ND     the lowest ND in [V1, V2], or all of them when fewer
//   V1   V2   blank  all in [V1, V2]
//   V1   blank ND    the lowest ND above V1
//   V1   blank blank the lowest root above V1
//   blank blank ND   the lowest ND
//   blank blank blank the lowest root
//   blank V2   ND    the lowest ND below V2
//   blank V2   blank all below V2
//
// For a vibration analysis a frequency f is the eigenvalue λ = sign(f)·(2π·f)², and a lower bound at or below 0 means
// "from the lowest root" (the rigid-body modes of a free structure, rounded zeros of either sign, are all found). For a
// buckling analysis the bounds are load factors and the roots are taken by increasing magnitude
// (linalg::EigenRequest).
[[nodiscard]] linalg::EigenRequest eigenRequestOf(const EigenMethod& method, AnalysisKind analysis);

} // namespace oos::solutions
