#pragma once

#include "oos/base/errors/SingularityError.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace oos::linalg
{

// The name of an equation as the user reads it (a grid or scalar point and a component), given by the caller that
// knows the model; linalg only numbers equations. An empty namer names them by number ("equation 12").
using EquationNamer = std::function<base::GridComponent(std::int32_t equation)>;

// The factor diagonal ratio test of PARAM MaxDiagonalRatio and PARAM SingularityBailOut: for each pivot of a
// positive definite factorisation, the ratio of the matrix diagonal to the factor diagonal, a_jj / d_j. A ratio above
// maxRatio flags a near mechanism (the pivot lost almost all of its diagonal to the coupled DOFs); a negative ratio
// (a negative factor diagonal) flags a matrix that is not positive definite. Every such DOF is listed with its ratio,
// then the run stops unless SingularityBailOut is negative.
//
// The caller may give the test another diagonal than the matrix's for some equations (a reference diagonal:
// FactorOptions::referenceDiagonal, ratioTestDiagonal below). The matrix diagonal measures the stiffness a DOF has
// before the elimination only when its terms are stiffnesses at that DOF. The relations of an element formulation
// eliminated before the factorisation (the rotation of a grid point of shells written as a combination Σ c_j·u_j of
// the unknowns of the shells around it) put the stiffness k of what is attached to the eliminated DOF on the diagonal
// of every unknown of the relation, k·c_j², although it holds one combination of them only: the pivots of the others
// come back to the stiffness of their own elements, and the ratio to the inflated diagonal would read as a mechanism
// that is not there. The reference is then the diagonal of the stiffness of the elements at the DOF.
struct PivotPolicy
{
    static constexpr double DEFAULT_MAX_RATIO = 1.0e7; // default of PARAM MaxDiagonalRatio
    static constexpr int DEFAULT_BAILOUT = 0;          // default of PARAM SingularityBailOut

    double maxRatio = DEFAULT_MAX_RATIO;
    // SingularityBailOut < 0: continue with the factor despite the flagged pivots; otherwise the run stops.
    int bailout = DEFAULT_BAILOUT;

    [[nodiscard]] bool doesStopOnIssues() const noexcept { return bailout >= 0; }
};

enum class PivotIssueKind : std::uint8_t
{
    RatioAboveMaximum, // a_jj / d_j > MaxDiagonalRatio: a near mechanism
    NegativePivot      // a_jj / d_j < 0: the matrix is not positive definite
};

// One pivot that fails the ratio test.
struct PivotIssue
{
    std::int32_t equation = 0;   // in the numbering of the factorised matrix (the caller's SymmetricCsc)
    double matrixDiagonal = 0.0; // the diagonal of the test: the matrix's, or the reference given for the equation
    double factorDiagonal = 0.0;
    double ratio = 0.0;
    PivotIssueKind kind = PivotIssueKind::RatioAboveMaximum;
};

// What the ratio test found in a factorisation: the failing pivots in the order the factorisation eliminated them,
// and the largest ratio of all pivots.
//
// Thread-safety: an immutable value. Ownership: owns its list.
class PivotDiagnostics
{
public:
    // How many issues a message lists by name; the count of the others follows.
    static constexpr std::size_t LISTED_ISSUE_COUNT = 20;
    // How many issues are recorded with their values (and how many DOFs a SingularityError carries): more than any
    // report is read for, and a bound that does not grow with the model (a badly scaled model may fail the test at
    // every pivot). The count of all issues is kept.
    static constexpr std::size_t RECORDED_ISSUE_COUNT = 1000;

    PivotDiagnostics() = default;
    // `issues` holds at most RECORDED_ISSUE_COUNT issues, the first ones met; issueCount counts them all.
    PivotDiagnostics(std::vector<PivotIssue> issues, std::int64_t issueCount, double largestRatio,
                     std::int32_t equationOfLargestRatio, double maxRatio);

    // The first issues met (at most RECORDED_ISSUE_COUNT), in elimination order.
    [[nodiscard]] const std::vector<PivotIssue>& issues() const noexcept { return m_issues; }
    // Every issue, recorded or not.
    [[nodiscard]] std::int64_t issueCount() const noexcept { return m_issueCount; }
    [[nodiscard]] bool hasIssues() const noexcept { return m_issueCount > 0; }
    // The largest a_jj / d_j over all pivots (0 without pivots) and its equation (-1 without pivots).
    [[nodiscard]] double largestRatio() const noexcept { return m_largestRatio; }
    [[nodiscard]] std::int32_t equationOfLargestRatio() const noexcept { return m_equationOfLargestRatio; }
    // The MaxDiagonalRatio the test used.
    [[nodiscard]] double maxRatio() const noexcept { return m_maxRatio; }

    // Throws SingularityError naming the DOFs (through `namer`) when there are issues and the policy stops the run
    // (SingularityBailOut >= 0); does nothing otherwise.
    void enforce(const PivotPolicy& policy, const EquationNamer& namer) const;

    // One line per listed issue (at most LISTED_ISSUE_COUNT): the DOF and its ratio.
    [[nodiscard]] std::string describe(const EquationNamer& namer) const;

private:
    std::vector<PivotIssue> m_issues; // at most RECORDED_ISSUE_COUNT
    std::int64_t m_issueCount = 0;
    double m_largestRatio = 0.0;
    std::int32_t m_equationOfLargestRatio = -1;
    double m_maxRatio = PivotPolicy::DEFAULT_MAX_RATIO;
};

// The value of a reference diagonal at an equation it gives no reference for (any negative value is read so).
inline constexpr double NO_REFERENCE_DIAGONAL = -1.0;

// The diagonal the ratio test divides by the factor diagonal of `equation`: the matrix diagonal a_jj, or the reference
// of the equation when `referenceDiagonal` (one value per equation, or empty: no reference at all) gives one, a
// value that is not negative. Without a reference the value and the bits of the test are those of the matrix diagonal.
[[nodiscard]] inline double ratioTestDiagonal(std::span<const double> referenceDiagonal, std::int32_t equation,
                                              double matrixDiagonal) noexcept
{
    if (referenceDiagonal.empty())
        return matrixDiagonal;
    const double reference = referenceDiagonal[static_cast<std::size_t>(equation)];
    return reference >= 0.0 ? reference : matrixDiagonal;
}
// Throws std::invalid_argument when a reference diagonal that is not empty has not one value per equation.
void requireReferenceDiagonal(std::span<const double> referenceDiagonal, std::int64_t equationCount);

// The name of an equation for messages: "grid 1001 component 3", "scalar point 7", the namer's own name of a DOF
// that is no component of a point (base::GridComponent::name), or "equation 12" without a namer.
[[nodiscard]] std::string describeEquation(const EquationNamer& namer, std::int32_t equation);
// The DOF of an equation for SingularityError: the namer's, or {equation, 0} without one.
[[nodiscard]] base::GridComponent gridComponentOf(const EquationNamer& namer, std::int32_t equation);

} // namespace oos::linalg
