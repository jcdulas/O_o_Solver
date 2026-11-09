#include "oos/solutions/common/ElementPencil.hpp"

#include "ElementPasses.hpp"
#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace oos::solutions
{
namespace
{

using Combination = ElementPencil::Combination;

// The pieces of the products and diagonals: runs of consecutive rows of one batch (StreamedFront's refinement's).
constexpr std::int64_t PIECE_ROWS = 1024;
// The most equation terms an element carries, for the memory bound of a wave (a CHEXA20's 60 DOFs and the relations
// of its points).
constexpr std::uint64_t TERMS_PER_ELEMENT = 128;
// A lumped mass is held on the equations while its terms off the diagonal are at most this many a DOF (the relations'
// couplings: a PointMassInertia on an RigidInterpolation's reference point, the legs of an RigidBodyStar); a coupled mass
// (COUPMASS) has tens.
constexpr std::size_t MASS_OFF_DIAGONAL_TERMS = 4;

// The facets a combination needs.
std::vector<elements::Facet> facetsOf(Combination combination)
{
    std::vector<elements::Facet> facets; // at most two
    if (combination.stiffness != 0.0 || combination.differential != 0.0)
        facets.push_back(elements::Facet::Stiffness);
    if (combination.mass != 0.0)
        facets.push_back(elements::Facet::Mass);
    return facets;
}

// The matrices of a run of consecutive rows of one batch that a combination needs.
struct RunMatrices
{
    elements::ElementMatrixBlock stiffness;
    elements::ElementMatrixBlock mass;
    elements::ElementMatrixBlock differential;
};

void computeRun(const ElementPencil::Inputs& inputs, const elements::NodalDisplacements& preload,
                std::size_t batchIndex, elements::ElementRange range, Combination combination, RunMatrices& work)
{
    const assembly::AssemblyBatch& batch = inputs.model->elementSet().batches()[batchIndex];
    const elements::FacetSet facets = batch.family->facets();
    work.stiffness.clear();
    work.mass.clear();
    work.differential.clear();
    if (combination.stiffness != 0.0 && facets.has(elements::Facet::Stiffness))
        batch.family->computeStiffness(batch.view, range, work.stiffness);
    if (combination.mass != 0.0 && facets.has(elements::Facet::Mass))
    {
        elements::MassOptions options;
        options.formulation = inputs.massOptions.formulation;
        options.addedNonStructuralMass = inputs.shares->sharesOf(batchIndex);
        batch.family->computeMass(batch.view, range, options, work.mass);
    }
    if (combination.differential != 0.0 && facets.has(elements::Facet::Stiffness))
        batch.family->computeDifferentialStiffness(batch.view, range, preload, work.differential);
}

// Calls take(dofs, matrix, scale) for each matrix of the run in element order (each element's stiffness, mass, then
// differential stiffness), the scale the combination's coefficient (PARAM WTMASS on the mass).
template <typename Take>
void forEachMatrix(const RunMatrices& work, elements::ElementRange range, Combination combination, double weightToMass,
                   Take&& take)
{
    std::size_t k = 0;
    std::size_t m = 0;
    std::size_t d = 0;
    for (std::int64_t row = range.first; row < range.last; ++row)
    {
        for (; k < work.stiffness.size() && work.stiffness.elementRow(k) == row; ++k)
            take(work.stiffness.dofs(k), work.stiffness.matrix(k), combination.stiffness);
        for (; m < work.mass.size() && work.mass.elementRow(m) == row; ++m)
            take(work.mass.dofs(m), work.mass.matrix(m), combination.mass * weightToMass);
        for (; d < work.differential.size() && work.differential.elementRow(d) == row; ++d)
            take(work.differential.dofs(d), work.differential.matrix(d), combination.differential);
    }
}

// T_eᵀ·A_e·T_e on the reduction's equations (m × m, row-major).
void reduceMatrix(const StreamedReduction& reduction, std::size_t n, std::span<const double> matrix,
                  std::vector<double>& reduced)
{
    const std::size_t m = reduction.equations.size();
    reduced.assign(m * m, 0.0);
    for (const StreamedReduction::Term& a : reduction.terms)
        for (const StreamedReduction::Term& b : reduction.terms)
            reduced[static_cast<std::size_t>(a.equation) * m + static_cast<std::size_t>(b.equation)] +=
                a.coefficient * b.coefficient *
                matrix[static_cast<std::size_t>(a.dof) * n + static_cast<std::size_t>(b.dof)];
}

} // namespace

ElementPencil::ElementPencil(const Inputs& inputs, base::MemoryBudget& budget)
    : m_inputs(inputs),
      m_rightHandCombination(inputs.transform == linalg::SpectralTransform::Buckling ? Combination{0.0, 0.0, -1.0}
                                                                                     : Combination{0.0, 1.0, 0.0}),
      m_stiffness(*this, Combination{1.0, 0.0, 0.0}),
      m_rightHand(*this, m_rightHandCombination),
      m_mass(*this, Combination{0.0, 1.0, 0.0}),
      m_shifted(*this, Combination{1.0, 0.0, 0.0})
{
    const assembly::DofMap& dofMap = inputs.model->dofMap();
    if (inputs.transform == linalg::SpectralTransform::Buckling)
    {
        if (inputs.preloadDisplacements.size() != static_cast<std::size_t>(dofMap.gSetSize()))
            throw std::invalid_argument(
                base::concatenate({"ElementPencil: expected the preload's displacements on the g set (",
                                   std::to_string(dofMap.gSetSize()), " values), found ",
                                   std::to_string(inputs.preloadDisplacements.size()),
                                   "; give the displacements of the static preload subcase"}));
        // The g-set order of DofMap is that of NodalDisplacements: six values per grid point row, then the scalar
        // points (DifferentialStiffnessAssembler's).
        const std::size_t gridValues = 6 * static_cast<std::size_t>(dofMap.gridCount());
        m_preload =
            elements::NodalDisplacements{inputs.preloadDisplacements.first(gridValues),
                                         inputs.preloadDisplacements.subspan(gridValues), inputs.preloadTemperatures};
        if (inputs.hasMassSource)
        {
            // A model without mass on the analysis set has no vibration basis (multilevel substructuring keeps its
            // buckling modes).
            base::BudgetedBuffer<double> diagonal(budget, static_cast<std::size_t>(dimension()),
                                                  "pencil from the elements: mass diagonal");
            sumDiagonal(Combination{0.0, 1.0, 0.0}, diagonal.span());
            const std::span<const double> values = diagonal.span();
            m_inputs.hasMassSource =
                std::any_of(values.begin(), values.end(), [](double value) { return value != 0.0; });
        }
        return;
    }
    // A lumped mass on the equations is held: its diagonal and its few terms off it (those the relations' rows couple),
    // the products then n multiplications a column and those terms; a coupled mass (more than MASS_OFF_DIAGONAL_TERMS
    // a DOF off its diagonal) is multiplied element by element.
    const auto n = static_cast<std::size_t>(dimension());
    base::BudgetedBuffer<double> diagonal(budget, n, "pencil from the elements: mass diagonal");
    std::vector<OffDiagonalTerm> offDiagonal; // bounded: MASS_OFF_DIAGONAL_TERMS a DOF, then abandoned
    bool isHeld = true;
    sumDiagonal(Combination{0.0, 1.0, 0.0}, diagonal.span(), &offDiagonal, &isHeld, MASS_OFF_DIAGONAL_TERMS * n);
    if (!isHeld)
        return;
    m_massDiagonal = std::move(diagonal);
    // The terms off the diagonal summed in the order they came (pieces, then elements), by column then row.
    std::stable_sort(offDiagonal.begin(), offDiagonal.end(), [](const OffDiagonalTerm& a, const OffDiagonalTerm& b)
                     { return a.column < b.column || (a.column == b.column && a.row < b.row); });
    std::size_t kept = 0;
    for (std::size_t k = 0; k < offDiagonal.size(); ++k)
    {
        if (kept > 0 && offDiagonal[kept - 1].row == offDiagonal[k].row &&
            offDiagonal[kept - 1].column == offDiagonal[k].column)
            offDiagonal[kept - 1].value += offDiagonal[k].value;
        else
            offDiagonal[kept++] = offDiagonal[k];
    }
    offDiagonal.resize(kept);
    m_massOffDiagonal =
        base::BudgetedBuffer<OffDiagonalTerm>(budget, kept, "pencil from the elements: mass off its diagonal");
    std::copy(offDiagonal.begin(), offDiagonal.end(), m_massOffDiagonal.data());
}

std::int32_t ElementPencil::dimension() const noexcept
{
    return m_inputs.elements->symbolic().dofCount();
}

linalg::FrontSource* ElementPencil::massSource()
{
    if (m_inputs.transform == linalg::SpectralTransform::Vibration)
        return &m_rightHand;
    return m_inputs.hasMassSource ? &m_mass : nullptr;
}

linalg::FrontSource& ElementPencil::shiftedSource(double shift)
{
    // K − σB: K − σM (vibration), K + σK_d (buckling, B = −K_d).
    m_shifted.setCombination(
        Combination{1.0, -shift * m_rightHandCombination.mass, -shift * m_rightHandCombination.differential});
    return m_shifted;
}

void ElementPencil::Source::addTerms(linalg::FrontTerms& front)
{
    const StreamedElements& elements = *m_pencil.m_inputs.elements;
    const assembly::DofMap& dofMap = m_pencil.m_inputs.model->dofMap();
    const double weightToMass = m_pencil.m_inputs.massOptions.weightToMass;
    const std::span<const std::int32_t> placed = elements.elementsAt(front.supernode);
    RunMatrices work;
    StreamedReduction reduction;
    std::vector<base::DofIndex> gDofs;
    std::vector<std::int32_t> positions;
    std::vector<double> reduced;
    // The terms of an element's matrix on its DOFs added to the front: scale·T_eᵀ·A_e·T_e.
    const auto addMatrix = [&](std::span<const elements::ElementDof> dofs, std::span<const double> matrix, double scale)
    {
        passes::gDofsOf(dofMap, dofs, gDofs);
        elements.reduce(gDofs, reduction);
        passes::positionsOf(front, reduction, positions);
        reduceMatrix(reduction, dofs.size(), matrix, reduced);
        const std::size_t m = reduction.equations.size();
        for (std::size_t p = 0; p < m; ++p)
            for (std::size_t q = 0; q <= p; ++q)
                front.at(positions[p], positions[q]) += scale * reduced[p * m + q];
    };
    for (std::size_t k = 0; k < placed.size();)
    {
        // A run of consecutive rows of one batch.
        const StreamedElements::Place place = elements.placeOf(placed[k]);
        std::size_t end = k + 1;
        const std::int64_t batchEnd = elements.batchFirst(place.batch + 1);
        while (end < placed.size() && placed[end] == placed[end - 1] + 1 && placed[end] < batchEnd)
            ++end;
        const elements::ElementRange range{place.row, place.row + static_cast<std::int64_t>(end - k)};
        computeRun(m_pencil.m_inputs, m_pencil.m_preload, place.batch, range, m_combination, work);
        forEachMatrix(work, range, m_combination, weightToMass, addMatrix);
        k = end;
    }
}

void ElementPencil::Source::diagonal(std::int32_t, std::span<double> values) const
{
    std::fill(values.begin(), values.end(), 0.0);
}

std::uint64_t ElementPencil::workBytes(std::size_t threads, std::size_t columns) noexcept
{
    const std::uint64_t wave = 4 * std::max<std::uint64_t>(threads, 1);
    return wave * static_cast<std::uint64_t>(PIECE_ROWS) * TERMS_PER_ELEMENT *
           (static_cast<std::uint64_t>(columns) * sizeof(double) + sizeof(std::int32_t));
}

void ElementPencil::multiply(Combination combination, linalg::ConstDenseMatrixView x, linalg::DenseMatrixView y) const
{
    const assembly::DofMap& dofMap = m_inputs.model->dofMap();
    const StreamedElements& elements = *m_inputs.elements;
    const std::size_t columns = x.columns;
    if (x.rows != y.rows || x.columns != y.columns || x.rows != static_cast<std::size_t>(dimension()))
        throw std::invalid_argument(base::concatenate(
            {"ElementPencil: expected blocks of ", std::to_string(dimension()), " rows with the same columns, found ",
             std::to_string(x.rows), " x ", std::to_string(x.columns), " and ", std::to_string(y.rows), " x ",
             std::to_string(y.columns)}));
    for (std::size_t b = 0; b < columns; ++b)
        std::fill_n(&y(0, b), y.rows, 0.0);
    const std::vector<elements::Facet> facets = facetsOf(combination);
    const std::vector<passes::ElementPiece> pieces = passes::elementPieces(*m_inputs.model, facets, PIECE_ROWS);
    // Per slot of a wave: the piece's terms (an equation and its value per column), in element order.
    struct Terms
    {
        std::vector<std::int32_t> equations;
        std::vector<double> values; // equations × columns, row by row
    };
    std::vector<Terms> partial(passes::waveOf(m_inputs.pool));
    const double weightToMass = m_inputs.massOptions.weightToMass;
    const auto work = [&](std::size_t p, std::size_t slot)
    {
        Terms& into = partial[slot];
        into.equations.clear();
        into.values.clear();
        RunMatrices matrices;
        computeRun(m_inputs, m_preload, pieces[p].batch, pieces[p].range, combination, matrices);
        StreamedReduction reduction;
        std::vector<base::DofIndex> gDofs;
        std::vector<double> values;
        std::vector<double> forces;
        forEachMatrix(matrices, pieces[p].range, combination, weightToMass,
                      [&](std::span<const elements::ElementDof> dofs, std::span<const double> matrix, double scale)
                      {
                          const std::size_t n = dofs.size();
                          passes::gDofsOf(dofMap, dofs, gDofs);
                          elements.reduce(gDofs, reduction);
                          if (reduction.equations.empty())
                              return;
                          passes::elementValues(reduction, n, x, values);
                          forces.assign(n * columns, 0.0);
                          for (std::size_t b = 0; b < columns; ++b)
                              for (std::size_t i = 0; i < n; ++i)
                              {
                                  double sum = 0.0;
                                  for (std::size_t j = 0; j < n; ++j)
                                      sum += matrix[i * n + j] * values[j + b * n];
                                  forces[i + b * n] = scale * sum;
                              }
                          const std::size_t m = reduction.equations.size();
                          const std::size_t base = into.equations.size();
                          into.equations.insert(into.equations.end(), reduction.equations.begin(),
                                                reduction.equations.end());
                          into.values.resize((base + m) * columns, 0.0);
                          for (const StreamedReduction::Term& term : reduction.terms)
                              for (std::size_t b = 0; b < columns; ++b)
                                  into.values[(base + static_cast<std::size_t>(term.equation)) * columns + b] +=
                                      term.coefficient * forces[static_cast<std::size_t>(term.dof) + b * n];
                      });
    };
    const auto merge = [&](std::size_t, std::size_t slot)
    {
        const Terms& from = partial[slot];
        for (std::size_t t = 0; t < from.equations.size(); ++t)
            for (std::size_t b = 0; b < columns; ++b)
                y(static_cast<std::size_t>(from.equations[t]), b) += from.values[t * columns + b];
    };
    passes::inWaves(pieces.size(), m_inputs.pool, m_inputs.cancellation, work, merge);
}

void ElementPencil::sumDiagonal(Combination combination, std::span<double> diagonal,
                                std::vector<OffDiagonalTerm>* offDiagonal, bool* isHeld, std::size_t limit) const
{
    const assembly::DofMap& dofMap = m_inputs.model->dofMap();
    const StreamedElements& elements = *m_inputs.elements;
    std::fill(diagonal.begin(), diagonal.end(), 0.0);
    const std::vector<elements::Facet> facets = facetsOf(combination);
    const std::vector<passes::ElementPiece> pieces = passes::elementPieces(*m_inputs.model, facets, PIECE_ROWS);
    struct Terms
    {
        std::vector<std::int32_t> equations;
        std::vector<double> values;
        std::vector<OffDiagonalTerm> offDiagonal;
    };
    std::vector<Terms> partial(passes::waveOf(m_inputs.pool));
    const double weightToMass = m_inputs.massOptions.weightToMass;
    const auto work = [&](std::size_t p, std::size_t slot)
    {
        Terms& into = partial[slot];
        into.equations.clear();
        into.values.clear();
        into.offDiagonal.clear();
        RunMatrices matrices;
        computeRun(m_inputs, m_preload, pieces[p].batch, pieces[p].range, combination, matrices);
        StreamedReduction reduction;
        std::vector<base::DofIndex> gDofs;
        std::vector<double> reduced;
        forEachMatrix(matrices, pieces[p].range, combination, weightToMass,
                      [&](std::span<const elements::ElementDof> dofs, std::span<const double> matrix, double scale)
                      {
                          passes::gDofsOf(dofMap, dofs, gDofs);
                          elements.reduce(gDofs, reduction);
                          reduceMatrix(reduction, dofs.size(), matrix, reduced);
                          const std::size_t m = reduction.equations.size();
                          for (std::size_t i = 0; i < m; ++i)
                          {
                              into.equations.push_back(reduction.equations[i]);
                              into.values.push_back(scale * reduced[i * m + i]);
                              if (offDiagonal == nullptr)
                                  continue;
                              for (std::size_t j = 0; j < m; ++j)
                                  if (reduction.equations[i] > reduction.equations[j] && reduced[i * m + j] != 0.0)
                                      into.offDiagonal.push_back(
                                          {reduction.equations[i], reduction.equations[j], scale * reduced[i * m + j]});
                          }
                      });
    };
    const auto merge = [&](std::size_t, std::size_t slot)
    {
        const Terms& from = partial[slot];
        for (std::size_t t = 0; t < from.equations.size(); ++t)
            diagonal[static_cast<std::size_t>(from.equations[t])] += from.values[t];
        if (offDiagonal == nullptr || !*isHeld)
            return;
        if (offDiagonal->size() + from.offDiagonal.size() > limit)
        {
            *isHeld = false;
            *offDiagonal = std::vector<OffDiagonalTerm>();
            return;
        }
        offDiagonal->insert(offDiagonal->end(), from.offDiagonal.begin(), from.offDiagonal.end());
    };
    passes::inWaves(pieces.size(), m_inputs.pool, m_inputs.cancellation, work, merge);
}

void ElementPencil::multiplyStiffness(linalg::ConstDenseMatrixView x, linalg::DenseMatrixView y) const
{
    multiply(Combination{1.0, 0.0, 0.0}, x, y);
}

void ElementPencil::multiplyRightHand(linalg::ConstDenseMatrixView x, linalg::DenseMatrixView y) const
{
    if (m_massDiagonal.empty())
    {
        multiply(m_rightHandCombination, x, y);
        return;
    }
    // PERFORMANCE: a lumped mass on the equations multiplied by its diagonal and its terms off it (SparsePencil's
    // lumped mass), not element by element: the same values to round-off (each term summed once, then times x), the
    // terms in a fixed order whatever the threads.
    for (std::size_t b = 0; b < x.columns; ++b)
    {
        for (std::size_t i = 0; i < x.rows; ++i)
            y(i, b) = m_massDiagonal[i] * x(i, b);
        for (const OffDiagonalTerm& term : m_massOffDiagonal.span())
        {
            const auto row = static_cast<std::size_t>(term.row);
            const auto column = static_cast<std::size_t>(term.column);
            y(row, b) += term.value * x(column, b);
            y(column, b) += term.value * x(row, b);
        }
    }
}

void ElementPencil::diagonals(std::span<double> stiffness, std::span<double> rightHand) const
{
    sumDiagonal(Combination{1.0, 0.0, 0.0}, stiffness);
    if (!m_massDiagonal.empty())
        std::copy_n(m_massDiagonal.data(), rightHand.size(), rightHand.begin());
    else
        sumDiagonal(m_rightHandCombination, rightHand);
}

} // namespace oos::solutions
