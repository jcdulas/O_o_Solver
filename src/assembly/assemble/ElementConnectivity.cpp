#include "oos/assembly/assemble/ElementConnectivity.hpp"

#include "BudgetedGrowingArray.hpp"
#include "ElementMatrixSource.hpp"

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
    const auto pieceSize = static_cast<std::size_t>(ELEMENTS_PER_PIECE);
    const std::size_t pieceCount = (elementCount + pieceSize - 1) / pieceSize;
    BudgetedGrowingArray<std::int64_t> offsets(budget, "element node groups");
    BudgetedGrowingArray<std::int32_t> nodes(budget, "element node groups");
    BudgetedGrowingArray<std::int32_t> elementIds(budget, "element node groups: element IDs");
    BudgetedGrowingArray<PointElement> pointElements(budget, "element node groups: one-node elements");
    offsets.push(0);
    m_stiffenedComponents = base::BudgetedBuffer<std::uint8_t>(budget, static_cast<std::size_t>(dofMap.nodeCount()),
                                                               "element stiffened components");
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
                                    static_cast<std::uint8_t>(1U << (dof.kind == input::PointKind::Scalar
                                                                         ? 0
                                                                         : static_cast<unsigned>(dof.component - 1))));
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
                m_stiffenedComponents[static_cast<std::size_t>(node)] |= bit;
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
