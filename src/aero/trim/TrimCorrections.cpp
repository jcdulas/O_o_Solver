#include "oos/aero/trim/TrimCorrections.hpp"

#include "oos/aero/trim/TrimCase.hpp"
#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/base/errors/UnsupportedCardError.hpp"
#include "oos/input/contract/TableRole.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace oos::aero
{

namespace
{

// The fixed names a static aeroelastic solution reads its direct matrix input by (DirectMatrix, DirectMatrixAeroJ,
// DirectMatrixAeroK: the correction factors), O_o_Reader's `role` column.
constexpr std::string_view NORMALWASH_ROLE = "DownwashMatrix";
constexpr std::string_view PRESSURE_ROLE = "AeroPressureMatrix";
constexpr std::string_view WEIGHT_ROLE = "AeroWeightingMatrix";
constexpr std::string_view WEIGHT_ALIAS_ROLE = "AeroWeightingFactor";
// The cards of the direct matrix input.
constexpr std::string_view DIRECT_MATRIX_CARD = "DirectMatrix";
constexpr std::string_view DIRECT_MATRIX_AERO_J_CARD = "DirectMatrixAeroJ";
constexpr std::string_view DIRECT_MATRIX_AERO_J_INTERFERENCE_CARD = "DirectMatrixAeroJInterference";
constexpr std::string_view DIRECT_MATRIX_AERO_K_CARD = "DirectMatrixAeroK";
// On a DirectMatrixAeroJ, a AeroLiftingPanel1 box is a point of the js set with the component 3 (its normal motion).
constexpr std::int32_t BOX_COMPONENT = 3;
// The k set of a AeroLiftingPanel1 box: its normal force and its pitching moment (T3 and R5 of the box's aerodynamic
// grid point).
constexpr std::size_t K_SET_ROWS_PER_BOX = 2;
// The weight of an identity AeroWeightingMatrix: 1 exactly (the deck gives 1.0; any other value is a weighting, not
// applied).
constexpr double IDENTITY_WEIGHT = 1.0;

// One matrix of directMatrixShapes that the trim reads.
struct MatrixShape
{
    std::string name;
    std::string card;
    std::string role;
    bool isComplex = false;
    std::int64_t rowCount = 0;
    std::int64_t columnCount = 0;
};

std::string labelOf(const MatrixShape& matrix)
{
    return base::concatenate({matrix.card, " ", matrix.name});
}

// The terms of one matrix: its stored rows (the snapshot's directMatricesTable rows of its name and card).
struct MatrixTerms
{
    std::vector<std::int64_t> rows; // the table's rows
};

MatrixTerms termsOf(const base::ContractReader* terms, const MatrixShape& matrix)
{
    MatrixTerms found;
    if (terms == nullptr)
        return found;
    const base::DictionaryView names = terms->dictionaryTexts("name");
    const base::DictionaryView cards = terms->dictionaryTexts("card");
    for (std::int64_t row = 0; row < terms->rowCount(); ++row)
        if (names[row] == matrix.name && cards[row] == matrix.card)
            found.rows.push_back(row);
    return found;
}

// The checks every DownwashMatrix and AeroPressureMatrix takes: a single real column (one vector that applies to every
// subcase), on a card this version reads for the role.
void checkVector(const MatrixShape& matrix)
{
    if (matrix.card == DIRECT_MATRIX_AERO_J_INTERFERENCE_CARD)
        throw base::UnsupportedCardError(base::concatenate(
            {labelOf(matrix),
             ": the direct matrix input of the interference elements of aerodynamic bodies (AeroBody2) is "
             "not supported, as bodies are not; give the matrix as a DirectMatrix of the j set or a DirectMatrixAeroJ "
             "of "
             "the AeroLiftingPanel1 boxes"}));
    if (matrix.card != DIRECT_MATRIX_CARD && matrix.card != DIRECT_MATRIX_AERO_J_CARD)
        throw base::UnsupportedCardError(base::concatenate(
            {labelOf(matrix),
             ": expected a DirectMatrix of the j set or a DirectMatrixAeroJ of the AeroLiftingPanel1 boxes for ",
             matrix.role, ", found a ", matrix.card, "; give it as a DirectMatrix or a DirectMatrixAeroJ"}));
    if (matrix.isComplex)
        throw base::SolverError(
            base::concatenate({labelOf(matrix),
                               ": expected a real matrix (a static aeroelastic trim is real), found a complex one "
                               "(DirectMatrix.inputType 3 or 4); give it with DirectMatrix.inputType 1 or 2"}));
    if (matrix.columnCount != 1)
        throw base::SolverError(
            base::concatenate({labelOf(matrix),
                               ": expected one column, a single vector that applies to every subcase (a correction "
                               "factor does not depend on the subcase), found ",
                               std::to_string(matrix.columnCount), " columns; give one column"}));
}

// The values of a DownwashMatrix or AeroPressureMatrix on the boxes, in their order, times `sign`.
void readVector(const MatrixShape& matrix, const base::ContractReader* terms, const AeroBoxes& boxes, double sign,
                std::span<double> values)
{
    checkVector(matrix);
    const MatrixTerms found = termsOf(terms, matrix);
    if (matrix.card == DIRECT_MATRIX_CARD && matrix.rowCount != static_cast<std::int64_t>(boxes.size()))
        throw base::SolverError(base::concatenate(
            {labelOf(matrix), ": expected one row per box of the j set (", std::to_string(boxes.size()),
             " AeroLiftingPanel1 boxes, in the order of the panels and their boxes), found ",
             std::to_string(matrix.rowCount), " rows; give a row for every box, or a DirectMatrixAeroJ by box ID"}));
    if (found.rows.empty())
        return;
    const std::span<const std::int32_t> rows = terms->values<std::int32_t>("row");
    const std::span<const std::int32_t> rowIds = terms->values<std::int32_t>("rowId");
    const std::span<const std::int32_t> components = terms->values<std::int32_t>("rowComponent");
    const std::span<const double> reals = terms->values<double>("real");
    std::vector<bool> isGiven(boxes.size(), false); // bounded by the boxes
    for (const std::int64_t row : found.rows)
    {
        const auto index = static_cast<std::size_t>(row);
        std::size_t box = static_cast<std::size_t>(rows[index]);
        if (matrix.card == DIRECT_MATRIX_AERO_J_CARD)
        {
            const std::optional<std::size_t> position = boxes.find(rowIds[index]);
            if (!position)
                throw base::SolverError(base::concatenate(
                    {labelOf(matrix), ": a term on the point ", std::to_string(rowIds[index]),
                     ", which is no AeroLiftingPanel1 box of the model; give the terms on the boxes' IDs"}));
            if (components[index] != BOX_COMPONENT)
                throw base::SolverError(base::concatenate(
                    {labelOf(matrix), ": the box ", std::to_string(rowIds[index]), " is given the component ",
                     std::to_string(components[index]),
                     ", expected 3 (a AeroLiftingPanel1 box in the js set is its normal motion); give "
                     "component 3"}));
            box = *position;
        }
        if (box >= boxes.size())
            throw base::SolverError(
                base::concatenate({labelOf(matrix), ": a term on the row ", std::to_string(box + 1), " beyond the ",
                                   std::to_string(boxes.size()), " boxes of the j set; give a row for every box"}));
        if (isGiven[box])
            throw base::SolverError(base::concatenate(
                {labelOf(matrix), ": the box ", std::to_string(boxes[box].id), " is given twice; give each box once"}));
        isGiven[box] = true;
        values[box] = sign * reals[index];
    }
}

// A AeroWeightingMatrix or AeroWeightingFactor: accepted only as the identity of the k set (two rows a box, every term
// 1 on the diagonal), which leaves the theoretical forces as they are; any other weighting is refused.
void checkIdentityWeight(const MatrixShape& matrix, const base::ContractReader* terms, const AeroBoxes& boxes)
{
    const std::size_t size = K_SET_ROWS_PER_BOX * boxes.size();
    const MatrixTerms found = termsOf(terms, matrix);
    bool isIdentity = !matrix.isComplex && matrix.rowCount == static_cast<std::int64_t>(size) &&
                      matrix.columnCount == static_cast<std::int64_t>(size) && found.rows.size() == size;
    if (isIdentity)
    {
        const std::span<const std::int32_t> rows = terms->values<std::int32_t>("row");
        const std::span<const std::int32_t> columns = terms->values<std::int32_t>("column");
        const std::span<const double> reals = terms->values<double>("real");
        const std::span<const double> imaginaries = terms->values<double>("imag");
        std::vector<bool> isOnDiagonal(size, false); // bounded by the k set
        for (const std::int64_t row : found.rows)
        {
            const auto index = static_cast<std::size_t>(row);
            const auto diagonal = static_cast<std::size_t>(rows[index]);
            isIdentity = isIdentity && rows[index] == columns[index] && diagonal < size && !isOnDiagonal[diagonal] &&
                         reals[index] == IDENTITY_WEIGHT && imaginaries[index] == 0.0;
            if (diagonal < size)
                isOnDiagonal[diagonal] = true;
        }
    }
    if (!isIdentity)
        throw base::UnsupportedCardError(base::concatenate(
            {labelOf(matrix), ": the weighting of the theoretical aerodynamic forces (", matrix.role,
             ") is not applied by this version, which takes it only as the identity of the k set (",
             std::to_string(size), " ones on the diagonal, two a box); remove it, or give the identity"}));
}

} // namespace

TrimCorrections::TrimCorrections(const input::ModelSnapshot& snapshot, const AeroBoxes& boxes,
                                 base::MemoryBudget& budget)
    : m_normalwash(budget, boxes.size(), "DownwashMatrix normalwash"),
      m_pressures(budget, boxes.size(), "AeroPressureMatrix pressures")
{
    if (!snapshot.hasTable(input::TableRole::DirectMatrixShapes))
        return;
    const base::ContractReader& shapes = snapshot.aeroTable(input::TableRole::DirectMatrixShapes);
    const base::ContractReader* terms = snapshot.hasTable(input::TableRole::DirectMatrices)
                                            ? &snapshot.aeroTable(input::TableRole::DirectMatrices)
                                            : nullptr;
    const base::DictionaryView names = shapes.dictionaryTexts("name");
    const base::DictionaryView cards = shapes.dictionaryTexts("card");
    const base::DictionaryView roles = shapes.dictionaryTexts("role", base::NullPolicy::Allow);
    const base::ValidityView hasRole = shapes.validity("role");
    const base::BooleanView isComplex = shapes.booleans("isComplex");
    const base::BooleanView isInEffect = shapes.booleans("inEffect");
    const std::span<const std::int64_t> rowCounts = shapes.values<std::int64_t>("rowCount");
    const std::span<const std::int64_t> columnCounts = shapes.values<std::int64_t>("columnCount");
    for (std::int64_t row = 0; row < shapes.rowCount(); ++row)
    {
        if (!isInEffect[row])
            continue; // a DirectMatrixAeroJ, DirectMatrixAeroJInterference or DirectMatrixAeroK of a name a
                      // DirectMatrix also has: the DirectMatrix takes precedence
        const auto index = static_cast<std::size_t>(row);
        MatrixShape matrix{std::string(names[row]),
                           std::string(cards[row]),
                           hasRole.isValid(row) ? std::string(roles[row]) : std::string(),
                           isComplex[row],
                           rowCounts[index],
                           columnCounts[index]};
        // O_o_Reader gives AeroPressureMatrix no role on a DirectMatrixAeroK: a k-set AeroPressureMatrix is ambiguous
        // (the pressure equations act on the j set), so it is named, not guessed.
        if (matrix.role.empty() && matrix.card == DIRECT_MATRIX_AERO_K_CARD && matrix.name == PRESSURE_ROLE)
            throw base::UnsupportedCardError(base::concatenate(
                {labelOf(matrix),
                 ": AeroPressureMatrix as a DirectMatrixAeroK of the k set is not read by this version (the pressure "
                 "correction acts on the j set of the boxes); give AeroPressureMatrix as a DirectMatrix of the j "
                 "set or a DirectMatrixAeroJ of the AeroLiftingPanel1 boxes"}));
        if (matrix.role.empty())
            continue; // a matrix of another name: no solution of O_o_Solver reads it
        if (terms == nullptr)
            throw base::ContractError(
                base::concatenate({labelOf(matrix),
                                   ": the snapshot has the shapes of the direct matrix input but not its terms "
                                   "(directMatricesTable); take the snapshot with both"}));
        if (matrix.role == NORMALWASH_ROLE)
        {
            if (hasNormalwash())
                throw base::SolverError(
                    base::concatenate({labelOf(matrix), ": a second DownwashMatrix in effect beside ",
                                       m_normalwashSource, "; give one"}));
            // DownwashMatrix is a downwash, plus the normal onset flow; O_o_Solver's normalwash is minus it
            // (TrimVariables).
            readVector(matrix, terms, boxes, -1.0, m_normalwash.span());
            m_normalwashSource = labelOf(matrix);
        }
        else if (matrix.role == PRESSURE_ROLE)
        {
            if (hasPressures())
                throw base::SolverError(
                    base::concatenate({labelOf(matrix), ": a second AeroPressureMatrix in effect beside ",
                                       m_pressureSource, "; give one"}));
            readVector(matrix, terms, boxes, 1.0, m_pressures.span());
            m_pressureSource = labelOf(matrix);
        }
        else if (matrix.role == WEIGHT_ROLE || matrix.role == WEIGHT_ALIAS_ROLE)
            checkIdentityWeight(matrix, terms, boxes);
    }
}

} // namespace oos::aero
