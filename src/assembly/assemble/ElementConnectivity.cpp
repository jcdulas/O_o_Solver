#include "oos/assembly/assemble/ElementConnectivity.hpp"

#include "BudgetedGrowingArray.hpp"
#include "ElementMatrixSource.hpp"

#include "oos/base/threading/Stopwatch.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace oos::assembly
{

namespace
{

// Pieces formed in parallel before they are appended: bounds the memory of a wave to
// PIECES_PER_WAVE × ELEMENTS_PER_PIECE element groups.
constexpr std::size_t PIECES_PER_WAVE = 64;

// The elements formed to judge the cost of the matrices of a set, and the work from which its pieces are cut fine:
// handing pieces to the threads has a price of its own (a third of a second for a run of the WebAssembly package
// with threads), which half a second of matrices pays.
constexpr std::size_t COST_SAMPLES = 4;
constexpr double COSTLY_WORK_SECONDS = 0.5;

// The elements of one piece.
// PERFORMANCE: a mixed hybrid element condenses its own unknowns, tens of milliseconds for a hexahedron of degree 4: a
// piece of ELEMENTS_PER_PIECE was 12 s of work on one thread, and a model of fewer elements ran on one thread whatever
// the pool. The elements of a formulation with entities are cut in pieces of COSTLY_ELEMENTS_PER_PIECE when their
// matrices are worth it: COST_SAMPLES of them, spread over the list, are formed once to be timed (those whose kernel
// their formulation keeps cost little again, and then the pass itself is light). The size of the pieces changes no
// bit: the groups and the matrices are appended in element order.
// Measured on twelve threads: the preparation of a block of 2 744 hexahedra of degree 4 from 43 s to 39 s (130 s on
// one thread).
std::size_t pieceSizeOf(const ElementSet& set, const ElementMatrixSource& source, const base::ThreadPool* pool)
{
    const auto usual = static_cast<std::size_t>(ElementConnectivity::ELEMENTS_PER_PIECE);
    const auto costly = static_cast<std::size_t>(ElementConnectivity::COSTLY_ELEMENTS_PER_PIECE);
    const std::size_t count = source.elementCount();
    if (!set.hasEntities() || pool == nullptr || pool->threadCount() <= 1 || count <= costly)
        return usual;
    base::Stopwatch watch;
    elements::ElementMatrixBlock block;
    for (std::size_t sample = 0; sample < COST_SAMPLES; ++sample)
    {
        const std::size_t element = sample * count / COST_SAMPLES;
        source.compute(element, element + 1, block, [](std::size_t, const elements::ElementMatrixBlock&) {});
    }
    const double whole = watch.lap() / static_cast<double>(COST_SAMPLES) * static_cast<double>(count);
    return whole >= COSTLY_WORK_SECONDS ? costly : usual;
}

// The node groups of the elements of one piece, in element order, and the stiffness matrices to keep.
struct PieceGroups
{
    std::vector<std::int64_t> sizes; // bounded by a piece
    std::vector<std::int32_t> nodes;
    std::vector<std::int32_t> elementIds; // per group
    std::vector<ElementConnectivity::PointElement> pointElements;
    linalg::ContributionBlock stiffness;                           // the matrices kept, on the node DOFs
    std::vector<std::int64_t> stiffnessElements;                   // per kept matrix: its element in the stiffness list
    std::vector<std::pair<std::int32_t, std::uint8_t>> components; // (node, component bit) of the stiffness DOFs
    // (node, component bit) of the DOFs a matrix of an element without stiffness acts on (the mass of a point mass)
    std::vector<std::pair<std::int32_t, std::uint8_t>> massComponents;
};

} // namespace

ElementConnectivity::ElementConnectivity(const ElementSet& set, const DofMap& dofMap, base::MemoryBudget& budget,
                                         base::ThreadPool* pool, const base::CancellationToken* cancellation,
                                         ElementMatrixStore* stiffnessStore)
{
    // Each batch counts once, by the facet that describes its couplings: the stiffness, else the mass.
    std::vector<std::size_t> batches;
    for (std::size_t b = 0; b < set.batches().size(); ++b)
    {
        const elements::FacetSet facets = set.batches()[b].family->facets();
        if (facets.has(elements::Facet::Stiffness) || facets.has(elements::Facet::Mass))
            batches.push_back(b);
    }
    // The first element of each stiffness batch in the list of StiffnessAssembler (the stiffness batches in order).
    std::vector<std::int64_t> firstStiffnessElement(set.batches().size(), -1); // bounded by the card types
    std::int64_t stiffnessElements = 0;
    for (const std::size_t b : set.batchesWith(elements::Facet::Stiffness))
    {
        firstStiffnessElement[b] = stiffnessElements;
        stiffnessElements += set.batches()[b].view.rowCount();
    }
    const ElementMatrixSource source(
        set, batches, dofMap,
        [](std::size_t, const AssemblyBatch& batch, elements::ElementRange range, elements::ElementMatrixBlock& block)
        {
            if (batch.family->facets().has(elements::Facet::Stiffness))
                batch.family->computeStiffness(batch.view, range, block);
            else
                batch.family->computeMass(batch.view, range, elements::MassOptions{}, block);
        });

    const std::size_t elementCount = source.elementCount();
    const std::size_t pieceSize = pieceSizeOf(set, source, pool);
    const std::size_t pieceCount = (elementCount + pieceSize - 1) / pieceSize;
    BudgetedGrowingArray<std::int64_t> offsets(budget, "element node groups");
    BudgetedGrowingArray<std::int32_t> nodes(budget, "element node groups");
    BudgetedGrowingArray<std::int32_t> elementIds(budget, "element node groups: element IDs");
    BudgetedGrowingArray<PointElement> pointElements(budget, "element node groups: one-node elements");
    offsets.push(0);
    m_stiffenedComponents = base::BudgetedBuffer<std::uint8_t>(budget, static_cast<std::size_t>(dofMap.nodeCount()),
                                                               "element stiffened components");
    m_usedComponents = base::BudgetedBuffer<std::uint8_t>(budget, static_cast<std::size_t>(dofMap.nodeCount()),
                                                          "element used components");
    std::vector<PieceGroups> wave(PIECES_PER_WAVE);
    for (std::size_t firstPiece = 0; firstPiece < pieceCount; firstPiece += PIECES_PER_WAVE)
    {
        if (cancellation != nullptr)
            cancellation->throwIfCancelled();
        const std::size_t pieces = std::min(PIECES_PER_WAVE, pieceCount - firstPiece);
        const auto formPiece = [&](std::size_t p)
        {
            PieceGroups& groups = wave[p];
            groups.sizes.clear();
            groups.nodes.clear();
            groups.elementIds.clear();
            groups.pointElements.clear();
            groups.components.clear();
            groups.massComponents.clear();
            // Fresh containers: those of the last wave were moved into the store.
            groups.stiffness = linalg::ContributionBlock{};
            groups.stiffnessElements = {};
            const bool keepsStiffness = stiffnessStore != nullptr && stiffnessStore->isKept();
            std::vector<linalg::NodeDof> nodeDofs; // bounded by an element
            const std::size_t begin = (firstPiece + p) * pieceSize;
            const std::size_t end = std::min(elementCount, begin + pieceSize);
            elements::ElementMatrixBlock block;
            std::vector<std::int32_t> elementNodes; // bounded by an element
            source.compute(
                begin, end, block,
                [&](std::size_t batchIndex, const elements::ElementMatrixBlock& matrices)
                {
                    const std::int64_t firstKept = keepsStiffness ? firstStiffnessElement[batchIndex] : -1;
                    const bool isStiffness = set.batches()[batchIndex].family->facets().has(elements::Facet::Stiffness);
                    for (std::size_t i = 0; i < matrices.size(); ++i)
                    {
                        for (const elements::ElementDof& dof : matrices.dofs(i))
                            if (isStiffness)
                                groups.components.emplace_back(
                                    source.nodeOf(dof),
                                    static_cast<std::uint8_t>(
                                        1U << static_cast<unsigned>(source.nodeDofOf(dof).component)));
                        if (!isStiffness)
                        {
                            // The DOFs the matrix of an element without stiffness acts on: the rows that are not
                            // zero (a point mass without inertia nor offset leaves the rotations alone).
                            const std::span<const elements::ElementDof> dofs = matrices.dofs(i);
                            const std::span<const double> matrix = matrices.matrix(i);
                            const std::size_t n = dofs.size();
                            for (std::size_t r = 0; r < n && matrix.size() == n * n; ++r)
                            {
                                const std::span<const double> row = matrix.subspan(r * n, n);
                                const bool isUsed =
                                    std::any_of(row.begin(), row.end(), [](double value) { return value != 0.0; });
                                if (isUsed)
                                    groups.massComponents.emplace_back(
                                        source.nodeOf(dofs[r]),
                                        static_cast<std::uint8_t>(
                                            1U << static_cast<unsigned>(source.nodeDofOf(dofs[r]).component)));
                            }
                        }
                        if (firstKept >= 0)
                        {
                            nodeDofs.clear();
                            for (const elements::ElementDof& dof : matrices.dofs(i))
                                nodeDofs.push_back(source.nodeDofOf(dof));
                            groups.stiffness.append(nodeDofs, matrices.matrix(i));
                            groups.stiffnessElements.push_back(firstKept + matrices.elementRow(i));
                        }
                        elementNodes.clear();
                        for (const elements::ElementDof& dof : matrices.dofs(i))
                            elementNodes.push_back(source.nodeOf(dof));
                        std::sort(elementNodes.begin(), elementNodes.end());
                        elementNodes.erase(std::unique(elementNodes.begin(), elementNodes.end()), elementNodes.end());
                        const std::int32_t elementId = set.batches()[batchIndex].view.elementId(matrices.elementRow(i));
                        // One node couples nothing: the diagonal block of every node exists anyway.
                        if (elementNodes.size() < 2)
                        {
                            if (!elementNodes.empty())
                                groups.pointElements.push_back({elementId, elementNodes.front()});
                            continue;
                        }
                        groups.elementIds.push_back(elementId);
                        groups.sizes.push_back(static_cast<std::int64_t>(elementNodes.size()));
                        groups.nodes.insert(groups.nodes.end(), elementNodes.begin(), elementNodes.end());
                    }
                });
        };
        if (pool != nullptr && pieces > 1)
            pool->runBatch(pieces, formPiece);
        else
            for (std::size_t p = 0; p < pieces; ++p)
                formPiece(p);
        for (std::size_t p = 0; p < pieces; ++p)
        {
            nodes.append(wave[p].nodes);
            elementIds.append(wave[p].elementIds);
            pointElements.append(wave[p].pointElements);
            for (const auto& [node, bit] : wave[p].components)
            {
                m_stiffenedComponents[static_cast<std::size_t>(node)] |= bit;
                m_usedComponents[static_cast<std::size_t>(node)] |= bit;
            }
            for (const auto& [node, bit] : wave[p].massComponents)
                m_usedComponents[static_cast<std::size_t>(node)] |= bit;
            for (const std::int64_t size : wave[p].sizes)
                offsets.push(offsets.span().back() + size);
            if (stiffnessStore != nullptr && stiffnessStore->isKept())
                stiffnessStore->appendPiece(std::move(wave[p].stiffnessElements), std::move(wave[p].stiffness));
        }
    }
    m_offsets = base::BudgetedBuffer<std::int64_t>(budget, offsets.size(), "element node groups");
    std::copy(offsets.span().begin(), offsets.span().end(), m_offsets.data());
    m_nodes = base::BudgetedBuffer<std::int32_t>(budget, nodes.size(), "element node groups");
    std::copy(nodes.span().begin(), nodes.span().end(), m_nodes.data());
    m_groupElementIds =
        base::BudgetedBuffer<std::int32_t>(budget, elementIds.size(), "element node groups: element IDs");
    std::copy(elementIds.span().begin(), elementIds.span().end(), m_groupElementIds.data());
    m_pointElements =
        base::BudgetedBuffer<PointElement>(budget, pointElements.size(), "element node groups: one-node elements");
    std::copy(pointElements.span().begin(), pointElements.span().end(), m_pointElements.data());
}

} // namespace oos::assembly
