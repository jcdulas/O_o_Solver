// The geometry declarations of the model (docs/mixed-hybrid-formulation-spec.md section 4.4): where they come from
// (the program and the cards of the deck), what is refused before anything is built (no declaration, two states for
// one edge), how the declarations reach the reading of the shells and of the skin of the solids, and the
// declarations that name nothing of the model.
//
// A smooth or sharp edge and a declared mesh line apply to every surface that has the edges they name: the shells,
// the skin of the solids, or both when a shell lies on the skin. So each is offered to both parts, and it is unknown
// only when neither takes it.

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementFormulation.hpp"
#include "oos/elements/mixedHybrid/model/MixedHybridModel.hpp"
#include "oos/elements/shells/ShellDirectors.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <numbers>
#include <optional>
#include <set>
#include <span>
#include <vector>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace oos::elements::mixedHybrid
{

namespace
{

// The cards whose geometry the formulation reads from the mesh: the shells (their surface) and the solids (their
// skin). A model with one of them needs the declarations; the line elements alone are straight and need none.
constexpr std::array<std::string_view, 10> CARDS_READ_FROM_THE_MESH = {
    "ShellQuad4", "ShellTria3", "ShellQuad4Drilling", "ShellTria3Drilling", "ShellQuad8",
    "ShellTria6", "SolidTetra", "SolidPyramid",       "SolidPenta",         "SolidHexa"};

// The model parameter that decides the state of the edges nothing declares, in degrees (the angle within which the
// classic shells share a normal at a grid point, and its default).
constexpr std::string_view FOLD_ANGLE_PARAMETER = "ShellNormalAngle";

// A problem of the declarations, with its code (defined below).
base::Diagnostic declarationProblem(std::string_view code, std::string message);

// The model parameter by which a deck without any geometry card says that it was prepared, and the value that says
// so: the declaration of the deck is then empty, and the fold angle decides every edge.
constexpr std::string_view DECLARATIONS_PARAMETER = "GeometryDeclarations";
constexpr std::string_view DECLARATIONS_GIVEN = "Given";
// The states of the SurfaceEdge card.
constexpr std::string_view SMOOTH_STATE = "Smooth";
constexpr std::string_view SHARP_STATE = "Sharp";
constexpr std::string_view RAW_STATE = "Raw";

bool isDeclaredByParameter(const input::ModelSnapshot& snapshot)
{
    if (!snapshot.hasTable(input::TableRole::Params))
        return false;
    const std::optional<input::ParamValue> value = snapshot.params().find(DECLARATIONS_PARAMETER);
    return value && value->kind == input::ParamValue::Kind::Text && value->text == DECLARATIONS_GIVEN;
}

input::GridEdge sortedEdge(std::int32_t first, std::int32_t second)
{
    return first < second ? input::GridEdge{first, second} : input::GridEdge{second, first};
}

// What the declarations of the program name: the edges, and the mesh lines as (grid point the line goes through or
// ends at, far grid point of the edge). The deck says nothing more of these.
struct NamedByProgram
{
    std::set<input::GridEdge> edges;
    std::set<std::pair<std::int32_t, std::int32_t>> lines;

    explicit NamedByProgram(const input::FormulationGeometry& geometry)
    {
        for (const auto* sides : {&geometry.smoothEdges, &geometry.sharpEdges})
            for (const input::EdgeSide& side : *sides)
                edges.insert(side.grids);
        edges.insert(geometry.rawEdges.begin(), geometry.rawEdges.end());
        for (const input::LineContinuation& line : geometry.lineContinuations)
        {
            lines.emplace(line[1], line[0]);
            lines.emplace(line[1], line[2]);
        }
        for (const input::LineEnd& end : geometry.lineEnds)
            lines.emplace(end[1], end[0]);
    }
};

// Adds the SurfaceEdge cards of the deck to `geometry`; a state that is none of the three is a problem.
void addDeckEdges(const base::ContractReader& table, const NamedByProgram& program,
                  input::FormulationGeometry& geometry, std::vector<base::Diagnostic>& problems)
{
    const std::span<const std::int32_t> ids = table.values<std::int32_t>("id");
    const std::span<const std::int32_t> first = table.values<std::int32_t>("node1Id");
    const std::span<const std::int32_t> second = table.values<std::int32_t>("node2Id");
    const std::span<const std::int32_t> elements = table.values<std::int32_t>("elementId");
    const base::Utf8View states = table.texts("state");
    for (std::size_t row = 0; row < ids.size(); ++row)
    {
        const input::GridEdge edge = sortedEdge(first[row], second[row]);
        if (program.edges.contains(edge))
            continue;
        const std::string_view state = states[static_cast<std::int64_t>(row)];
        if (state == SMOOTH_STATE)
            geometry.smoothEdges.push_back({edge, elements[row]});
        else if (state == SHARP_STATE)
            geometry.sharpEdges.push_back({edge, elements[row]});
        else if (state == RAW_STATE)
            geometry.rawEdges.push_back(edge);
        else
        {
            base::Diagnostic problem = declarationProblem(
                "geometry-declaration-unknown",
                base::concatenate({"expected the state of a SurfaceEdge to be Smooth, Sharp or Raw, found '", state,
                                   "'; correct the card"}));
            problem.card = "SurfaceEdge";
            problem.id = ids[row];
            problems.push_back(std::move(problem));
        }
    }
}

// Adds the SurfaceLine cards of the deck to `geometry`: a continuation, or an end when the card names no third grid
// point.
void addDeckLines(const base::ContractReader& table, const NamedByProgram& program,
                  input::FormulationGeometry& geometry)
{
    const std::span<const std::int32_t> from = table.values<std::int32_t>("node1Id");
    const std::span<const std::int32_t> through = table.values<std::int32_t>("throughNodeId");
    const std::span<const std::int32_t> to = table.values<std::int32_t>("node2Id");
    for (std::size_t row = 0; row < from.size(); ++row)
    {
        if (program.lines.contains({through[row], from[row]}))
            continue;
        if (to[row] == 0)
            geometry.lineEnds.push_back({from[row], through[row]});
        else if (!program.lines.contains({through[row], to[row]}))
            geometry.lineContinuations.push_back({from[row], through[row], to[row]});
    }
}

// The fold angle of the model, in radians (zero or negative: every edge that nothing declares is sharp).
double foldAngleOf(const input::ModelSnapshot& snapshot)
{
    double degrees = DEFAULT_SHELL_NORMAL_ANGLE_DEGREES;
    if (snapshot.hasTable(input::TableRole::Params))
        if (const std::optional<input::ParamValue> value = snapshot.params().find(FOLD_ANGLE_PARAMETER))
        {
            if (value->kind == input::ParamValue::Kind::Real)
                degrees = value->real;
            else if (value->kind == input::ParamValue::Kind::Integer)
                degrees = static_cast<double>(value->integer);
        }
    return degrees * std::numbers::pi / 180.0;
}

std::string edgeName(const input::GridEdge& edge)
{
    return base::concatenate({"grid points ", std::to_string(edge[0]), " and ", std::to_string(edge[1])});
}

base::Diagnostic declarationProblem(std::string_view code, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.code = std::string(code);
    diagnostic.message = std::move(message);
    return diagnostic;
}

// The declarations that give one edge side two states, or one mesh line an end and a continuation.
std::vector<base::Diagnostic> conflictsOf(const input::FormulationGeometry& geometry)
{
    constexpr std::string_view CODE = "geometry-declaration-conflict";
    std::vector<base::Diagnostic> out;
    for (const input::EdgeSide& side : geometry.smoothEdges)
        if (std::binary_search(geometry.sharpEdges.begin(), geometry.sharpEdges.end(), side))
            out.push_back(declarationProblem(
                CODE, base::concatenate({"expected one state for each edge side of the geometry declarations, found "
                                         "the edge between ",
                                         edgeName(side.grids), " declared both smooth and sharp",
                                         side.elementId != 0 ? " for element " + std::to_string(side.elementId) : "",
                                         "; keep one of the two declarations"})));
    const auto isRaw = [&](const input::GridEdge& edge)
    { return std::binary_search(geometry.rawEdges.begin(), geometry.rawEdges.end(), edge); };
    for (const auto* sides : {&geometry.smoothEdges, &geometry.sharpEdges})
        for (const input::EdgeSide& side : *sides)
            if (isRaw(side.grids))
                out.push_back(declarationProblem(
                    CODE, base::concatenate({"expected one state for each edge of the geometry declarations, found the "
                                             "edge between ",
                                             edgeName(side.grids), " declared raw and also smooth or sharp; a raw edge "
                                                                   "is straight and folds, keep one declaration"})));
    for (const input::LineEnd& end : geometry.lineEnds)
        for (const input::LineContinuation& line : geometry.lineContinuations)
            if (line[1] == end[1] && (line[0] == end[0] || line[2] == end[0]))
                out.push_back(declarationProblem(
                    CODE, base::concatenate({"expected a mesh line either ended or continued at a grid point, found "
                                             "the line of the edge between grid points ",
                                             std::to_string(end[0]), " and ", std::to_string(end[1]),
                                             " declared both ended and continued at grid point ",
                                             std::to_string(end[1]), "; keep one of the two declarations"})));
    return out;
}

// What the reading of the mesh made of the edges the declarations name: the shells of each edge and their smooth
// partners (the topology of the ShellPart), and the edges of the solids, on their skin or inside them. It says which
// declarations changed nothing though they name edges of the model (reportDeclarationsNotApplied).
class DeclarationReading
{
public:
    DeclarationReading(std::span<const std::int32_t> gridIds, const ShellPart* shells, const SolidPart* solids,
                       const std::map<std::int32_t, std::size_t>& shellOfId)
        : m_shells(shells != nullptr && shells->usable() ? shells : nullptr),
          m_isShellReadingUnknown(shells != nullptr && !shells->usable()), m_shellOfId(shellOfId)
    {
        for (std::size_t grid = 0; grid < gridIds.size(); ++grid)
            m_gridOfId.emplace(gridIds[grid], grid);
        if (solids != nullptr)
            for (const SolidPartEdge& edge : solids->edges())
                (edge.onSkin ? m_skinEdges : m_innerEdges).insert(edge.grids);
        if (m_shells != nullptr)
            m_rawShells.insert(m_shells->input().rawElements.begin(), m_shells->input().rawElements.end());
    }

    // A smooth (`isSmooth`) or sharp edge side changes something of the reading:
    // - on the skin of the solids, an edge declared whole;
    // - an edge of a raw shell, which the declaration keeps out of the raw edges;
    // - smooth, an edge whose shell (or one of whose shells, declared whole) is smooth with the shell that faces
    //   it: on the outline, and on a side of a junction that no shell faces, there is none;
    // - sharp, an edge of two shells or more (a side of a junction that no shell faces ends there already, which is
    //   what the declaration says: it is not reported).
    [[nodiscard]] bool isSideApplied(const input::EdgeSide& side, bool isSmooth) const
    {
        const std::optional<EdgeKey> edge = edgeOf(side.grids[0], side.grids[1]);
        if (!edge || m_isShellReadingUnknown)
            return true; // reported as unknown, or not read: nothing to say
        if (side.elementId == 0 && m_skinEdges.contains(*edge))
            return true;
        const std::optional<std::size_t> shellEdge = shellEdgeOf(*edge);
        if (!shellEdge)
            return !m_innerEdges.contains(*edge);
        const Topology& topology = m_shells->topology();
        const std::vector<std::size_t>& owners = topology.edgeElements[*shellEdge];
        const auto isRaw = [&](std::size_t shell) { return m_rawShells.contains(shell); };
        if (std::any_of(owners.begin(), owners.end(), isRaw))
            return true;
        if (!isSmooth)
            return owners.size() >= 2;
        const std::map<std::size_t, std::size_t>& partners = topology.partners[*shellEdge];
        if (side.elementId == 0)
            return !partners.empty();
        const auto shell = m_shellOfId.find(side.elementId);
        if (shell == m_shellOfId.end() || std::find(owners.begin(), owners.end(), shell->second) == owners.end())
            return true; // an element that is no shell of the edge: reported as unknown
        return partners.contains(shell->second);
    }

    // The mesh line of a continuation is read: its two edges are edges of the shells, or of the skin of the solids.
    [[nodiscard]] bool isContinuationApplied(const input::LineContinuation& line) const
    {
        return m_isShellReadingUnknown || (isShellEdge(line[1], line[0]) && isShellEdge(line[1], line[2])) ||
               (isSkinEdge(line[1], line[0]) && isSkinEdge(line[1], line[2]));
    }

    [[nodiscard]] bool isEndApplied(const input::LineEnd& end) const
    {
        return m_isShellReadingUnknown || isShellEdge(end[1], end[0]) || isSkinEdge(end[1], end[0]);
    }

private:
    [[nodiscard]] std::optional<EdgeKey> edgeOf(std::int32_t firstId, std::int32_t secondId) const
    {
        const auto first = m_gridOfId.find(firstId);
        const auto second = m_gridOfId.find(secondId);
        if (first == m_gridOfId.end() || second == m_gridOfId.end())
            return std::nullopt;
        return EdgeKey::of(first->second, second->second);
    }

    // The edge of the topology of the shells, when shells have it.
    [[nodiscard]] std::optional<std::size_t> shellEdgeOf(const EdgeKey& edge) const
    {
        if (m_shells == nullptr)
            return std::nullopt;
        const Topology& topology = m_shells->topology();
        const auto found = topology.edgeIndex.find(edge);
        if (found == topology.edgeIndex.end() || topology.edgeElements[found->second].empty())
            return std::nullopt;
        return found->second;
    }

    [[nodiscard]] bool isShellEdge(std::int32_t firstId, std::int32_t secondId) const
    {
        const std::optional<EdgeKey> edge = edgeOf(firstId, secondId);
        return edge && shellEdgeOf(*edge).has_value();
    }

    [[nodiscard]] bool isSkinEdge(std::int32_t firstId, std::int32_t secondId) const
    {
        const std::optional<EdgeKey> edge = edgeOf(firstId, secondId);
        return edge && m_skinEdges.contains(*edge);
    }

    const ShellPart* m_shells = nullptr;
    // The shells could not be read (their problems say why): nothing is said of the declarations then.
    bool m_isShellReadingUnknown = false;
    const std::map<std::int32_t, std::size_t>& m_shellOfId;
    std::map<std::int32_t, std::size_t> m_gridOfId;
    std::set<EdgeKey> m_skinEdges;
    std::set<EdgeKey> m_innerEdges;
    std::set<std::size_t> m_rawShells;
};

// One warning for the declarations that name edges of the model and change nothing of its reading (`...Used`: the
// declarations that name edges of the model, the others being reported as unknown): their count and the first.
void reportDeclarationsNotApplied(const input::FormulationGeometry& geometry, const std::vector<char>& smoothUsed,
                                  const std::vector<char>& sharpUsed, const std::vector<char>& continuationsUsed,
                                  const std::vector<char>& endsUsed, const DeclarationReading& reading,
                                  std::vector<base::Diagnostic>& problems)
{
    std::size_t count = 0;
    std::string first;
    const auto add = [&](std::string name)
    {
        if (count == 0)
            first = std::move(name);
        ++count;
    };
    const auto addSides = [&](const std::vector<input::EdgeSide>& sides, const std::vector<char>& used, bool isSmooth)
    {
        for (std::size_t i = 0; i < sides.size(); ++i)
            if (used[i] != 0 && !reading.isSideApplied(sides[i], isSmooth))
                add(base::concatenate(
                    {isSmooth ? "the smooth edge between " : "the sharp edge between ", edgeName(sides[i].grids),
                     sides[i].elementId != 0 ? " for element " + std::to_string(sides[i].elementId) : ""}));
    };
    addSides(geometry.smoothEdges, smoothUsed, true);
    addSides(geometry.sharpEdges, sharpUsed, false);
    for (std::size_t i = 0; i < geometry.lineContinuations.size(); ++i)
        if (continuationsUsed[i] != 0 && !reading.isContinuationApplied(geometry.lineContinuations[i]))
            add(base::concatenate({"the line continuation at grid point ",
                                   std::to_string(geometry.lineContinuations[i][1]), " between grid points ",
                                   std::to_string(geometry.lineContinuations[i][0]), " and ",
                                   std::to_string(geometry.lineContinuations[i][2])}));
    for (std::size_t i = 0; i < geometry.lineEnds.size(); ++i)
        if (endsUsed[i] != 0 && !reading.isEndApplied(geometry.lineEnds[i]))
            add(base::concatenate({"the line end at grid point ", std::to_string(geometry.lineEnds[i][1]),
                                   " of the edge from grid point ", std::to_string(geometry.lineEnds[i][0])}));
    if (count == 0)
        return;
    base::Diagnostic warning = declarationProblem(
        "geometry-declaration-not-applied",
        base::concatenate(
            {std::to_string(count), " geometry declarations name edges of the model and change nothing of its reading "
                                    "(a smooth side that no shell faces across its edge, an edge of the outline of the "
                                    "shells declared smooth or sharp, an edge or a mesh line inside the solids): the "
                                    "first is ",
             first, "; remove them, or declare what was meant"}));
    warning.severity = base::Severity::Warning;
    problems.push_back(std::move(warning));
}

} // namespace

void MixedHybridModel::readDeclarations(const SnapshotIndex& index)
{
    const input::ModelSnapshot& snapshot = index.snapshot();
    m_declarations = snapshot.formulationGeometry();
    const base::ContractReader* edges = snapshot.surfaceEdges();
    const base::ContractReader* facets = snapshot.surfaceFacets();
    const base::ContractReader* lines = snapshot.surfaceLines();
    if (edges == nullptr && facets == nullptr && lines == nullptr && !isDeclaredByParameter(snapshot))
        return;
    // Where the program and the deck name the same edge or the same mesh line, the program wins, as it does for the
    // choice of the formulation.
    input::FormulationGeometry merged = m_declarations.value_or(input::FormulationGeometry{});
    const NamedByProgram program(merged);
    if (edges != nullptr)
        addDeckEdges(*edges, program, merged, m_problems);
    if (facets != nullptr)
    {
        const std::span<const std::int32_t> elements = facets->values<std::int32_t>("elementId");
        merged.rawElements.insert(merged.rawElements.end(), elements.begin(), elements.end());
    }
    if (lines != nullptr)
        addDeckLines(*lines, program, merged);
    try
    {
        merged.normalise();
    }
    catch (const std::invalid_argument& error)
    {
        // the reader refuses a card that names a grid point twice; a table made otherwise is reported here
        m_problems.push_back(declarationProblem("geometry-declaration-unknown", error.what()));
        return;
    }
    m_declarations = std::move(merged);
}

void MixedHybridModel::checkDeclarations(const SnapshotIndex& index)
{
    const input::ModelSnapshot& snapshot = index.snapshot();
    m_foldAngle = foldAngleOf(snapshot);
    if (m_declarations)
    {
        m_declarationUse.smoothEdges.assign(m_declarations->smoothEdges.size(), 0);
        m_declarationUse.sharpEdges.assign(m_declarations->sharpEdges.size(), 0);
        m_declarationUse.lineContinuations.assign(m_declarations->lineContinuations.size(), 0);
        m_declarationUse.lineEnds.assign(m_declarations->lineEnds.size(), 0);
        for (base::Diagnostic& conflict : conflictsOf(*m_declarations))
            m_problems.push_back(std::move(conflict));
        return;
    }
    for (const std::string_view card : CARDS_READ_FROM_THE_MESH)
    {
        const input::ElementBatch* batch = snapshot.findElementBatch(card);
        if (batch == nullptr || batch->rowCount() == 0)
            continue;
        base::Diagnostic diagnostic;
        diagnostic.code = "geometry-declarations-missing";
        diagnostic.card = std::string(card);
        diagnostic.message = base::concatenate(
            {"expected the geometry declarations of a model whose shells or solids are in ", MIXED_HYBRID_FORMULATION,
             " (where its surface is smooth, where it folds, which mesh line goes on into which), found none and ",
             std::to_string(batch->rowCount()), " ", card,
             " elements; give them with the formulationGeometry option or with the SurfaceEdge, SurfaceFacet and "
             "SurfaceLine cards of the deck (an empty option, or the model parameter ",
             DECLARATIONS_PARAMETER, " with the value ", DECLARATIONS_GIVEN, ", says that the model's ",
             FOLD_ANGLE_PARAMETER, " decides every edge)"});
        m_problems.push_back(std::move(diagnostic));
        return;
    }
}

std::int32_t MixedHybridModel::gridIndexOfId(const SnapshotIndex& index, std::int32_t id) const
{
    const std::optional<std::int32_t> row = index.points().findGrid(id);
    return row ? gridIndexOfRow(*row) : -1;
}

void MixedHybridModel::declareShellGeometry(const SnapshotIndex& index, ShellPartInput& input)
{
    input.foldAngle = m_foldAngle;
    if (!m_declarations)
        return;
    // The shells of each edge, by part index and element ID.
    std::map<EdgeKey, std::vector<std::pair<std::size_t, std::int32_t>>> shellsOfEdge;
    const auto gridOfRow = [&](std::int32_t row) { return static_cast<std::size_t>(m_gridIndexOfRow.at(row)); };
    for (const MixedShell& shell : m_shells)
        for (std::size_t k = 0; k < shell.cornerCount; ++k)
        {
            const std::size_t next = (k + 1) % shell.cornerCount;
            shellsOfEdge[EdgeKey::of(gridOfRow(shell.gridRows[k]), gridOfRow(shell.gridRows[next]))].emplace_back(
                shell.partIndex, shell.elementId);
        }
    const auto edgeOf = [&](std::int32_t firstId, std::int32_t secondId) -> std::optional<EdgeKey>
    {
        const std::int32_t a = gridIndexOfId(index, firstId);
        const std::int32_t b = gridIndexOfId(index, secondId);
        if (a < 0 || b < 0)
            return std::nullopt;
        const EdgeKey edge = EdgeKey::of(static_cast<std::size_t>(a), static_cast<std::size_t>(b));
        return shellsOfEdge.contains(edge) ? std::optional<EdgeKey>(edge) : std::nullopt;
    };

    const auto declareSides = [&](const std::vector<input::EdgeSide>& sides, std::vector<char>& used,
                                  std::map<EdgeKey, std::set<std::size_t>>& out)
    {
        for (std::size_t i = 0; i < sides.size(); ++i)
        {
            const input::EdgeSide& side = sides[i];
            const std::optional<EdgeKey> edge = edgeOf(side.grids[0], side.grids[1]);
            if (!edge)
                continue; // perhaps an edge of the skin of the solids
            used[i] = 1;
            if (side.elementId == 0)
            {
                out[*edge].insert(MeshReadingInput::WHOLE_EDGE);
                continue;
            }
            const auto& shells = shellsOfEdge.at(*edge);
            const auto shell = std::find_if(shells.begin(), shells.end(),
                                            [&](const auto& entry) { return entry.second == side.elementId; });
            if (shell == shells.end())
            {
                m_problems.push_back(declarationProblem(
                    "geometry-declaration-unknown",
                    base::concatenate({"expected the element of an edge side of the geometry declarations to be a "
                                       "shell of ",
                                       MIXED_HYBRID_FORMULATION, " on that edge, found element ",
                                       std::to_string(side.elementId), " for the edge between ", edgeName(side.grids),
                                       ", which is not one of the shells of the edge; correct the declaration"})));
                continue;
            }
            out[*edge].insert(shell->first);
        }
    };
    declareSides(m_declarations->smoothEdges, m_declarationUse.smoothEdges, input.smoothSides);
    declareSides(m_declarations->sharpEdges, m_declarationUse.sharpEdges, input.sharpSides);

    // The mesh lines: by the grid the line goes through or ends at, then the far grid of the edge.
    const auto gridOf = [&](std::int32_t id) { return static_cast<std::size_t>(gridIndexOfId(index, id)); };
    for (std::size_t i = 0; i < m_declarations->lineContinuations.size(); ++i)
    {
        const input::LineContinuation& line = m_declarations->lineContinuations[i];
        if (!edgeOf(line[1], line[0]) || !edgeOf(line[1], line[2]))
            continue;
        input.declaredLines[{gridOf(line[1]), gridOf(line[0])}].insert(gridOf(line[2]));
        input.declaredLines[{gridOf(line[1]), gridOf(line[2])}].insert(gridOf(line[0]));
        m_declarationUse.lineContinuations[i] = 1;
    }
    for (std::size_t i = 0; i < m_declarations->lineEnds.size(); ++i)
    {
        const input::LineEnd& end = m_declarations->lineEnds[i];
        if (!edgeOf(end[1], end[0]))
            continue;
        input.declaredLines.try_emplace({gridOf(end[1]), gridOf(end[0])});
        m_declarationUse.lineEnds[i] = 1;
    }
}

void MixedHybridModel::declareSkinGeometry(const SnapshotIndex& index, SolidPartInput& input) const
{
    input.foldAngle = m_foldAngle;
    if (!m_declarations)
        return;
    // Every declaration between grids of the model is offered to the skin: one that names no edge of it changes
    // nothing there (skinDeclarationsUsed says which were edges of the solids).
    const auto edgeOf = [&](std::int32_t firstId, std::int32_t secondId) -> std::optional<EdgeKey>
    {
        const std::int32_t a = gridIndexOfId(index, firstId);
        const std::int32_t b = gridIndexOfId(index, secondId);
        if (a < 0 || b < 0)
            return std::nullopt;
        return EdgeKey::of(static_cast<std::size_t>(a), static_cast<std::size_t>(b));
    };
    const auto declareSides = [&](const std::vector<input::EdgeSide>& sides, std::set<EdgeKey>& out)
    {
        for (const input::EdgeSide& side : sides)
            if (const std::optional<EdgeKey> edge = edgeOf(side.grids[0], side.grids[1]); edge && side.elementId == 0)
                out.insert(*edge);
    };
    declareSides(m_declarations->smoothEdges, input.smoothEdges);
    declareSides(m_declarations->sharpEdges, input.sharpEdges);
    const auto gridOf = [&](std::int32_t id) { return static_cast<std::size_t>(gridIndexOfId(index, id)); };
    for (const input::LineContinuation& line : m_declarations->lineContinuations)
        if (edgeOf(line[1], line[0]) && edgeOf(line[1], line[2]))
        {
            input.declaredLines[{gridOf(line[1]), gridOf(line[0])}].insert(gridOf(line[2]));
            input.declaredLines[{gridOf(line[1]), gridOf(line[2])}].insert(gridOf(line[0]));
        }
    for (const input::LineEnd& end : m_declarations->lineEnds)
        if (edgeOf(end[1], end[0]))
            input.declaredLines.try_emplace({gridOf(end[1]), gridOf(end[0])});
}

void MixedHybridModel::markSkinDeclarationsUsed(const SnapshotIndex& index)
{
    if (!m_declarations || m_solidPart == nullptr)
        return;
    std::set<EdgeKey> solidEdges;
    for (const SolidPartEdge& edge : m_solidPart->edges())
        solidEdges.insert(edge.grids);
    const auto isSolidEdge = [&](std::int32_t firstId, std::int32_t secondId)
    {
        const std::int32_t a = gridIndexOfId(index, firstId);
        const std::int32_t b = gridIndexOfId(index, secondId);
        return a >= 0 && b >= 0 &&
               solidEdges.contains(EdgeKey::of(static_cast<std::size_t>(a), static_cast<std::size_t>(b)));
    };
    const auto markSides = [&](const std::vector<input::EdgeSide>& sides, std::vector<char>& used)
    {
        for (std::size_t i = 0; i < sides.size(); ++i)
            if (sides[i].elementId == 0 && isSolidEdge(sides[i].grids[0], sides[i].grids[1]))
                used[i] = 1;
    };
    markSides(m_declarations->smoothEdges, m_declarationUse.smoothEdges);
    markSides(m_declarations->sharpEdges, m_declarationUse.sharpEdges);
    for (std::size_t i = 0; i < m_declarations->lineContinuations.size(); ++i)
    {
        const input::LineContinuation& line = m_declarations->lineContinuations[i];
        if (isSolidEdge(line[1], line[0]) && isSolidEdge(line[1], line[2]))
            m_declarationUse.lineContinuations[i] = 1;
    }
    for (std::size_t i = 0; i < m_declarations->lineEnds.size(); ++i)
        if (isSolidEdge(m_declarations->lineEnds[i][1], m_declarations->lineEnds[i][0]))
            m_declarationUse.lineEnds[i] = 1;
}

void MixedHybridModel::reportUnknownDeclarations()
{
    if (!m_declarations)
        return;
    constexpr std::string_view CODE = "geometry-declaration-unknown";
    const auto reportSides = [&](const std::vector<input::EdgeSide>& sides, const std::vector<char>& used,
                                 std::string_view state)
    {
        for (std::size_t i = 0; i < sides.size(); ++i)
            if (used[i] == 0)
                m_problems.push_back(declarationProblem(
                    CODE, base::concatenate(
                              {"expected the ", state, " edges of the geometry declarations to be edges of the shells "
                                                       "or of the solids of ",
                               MIXED_HYBRID_FORMULATION,
                               " (two grid points that one of them joins along an edge), found ",
                               edgeName(sides[i].grids),
                               sides[i].elementId != 0
                                   ? " with element " + std::to_string(sides[i].elementId) + ", which no shell joins"
                                   : std::string(", which no edge of a shell or of a solid joins"),
                               "; correct the declaration"})));
    };
    reportSides(m_declarations->smoothEdges, m_declarationUse.smoothEdges, "smooth");
    reportSides(m_declarations->sharpEdges, m_declarationUse.sharpEdges, "sharp");
    for (std::size_t i = 0; i < m_declarations->lineContinuations.size(); ++i)
        if (m_declarationUse.lineContinuations[i] == 0)
        {
            const input::LineContinuation& line = m_declarations->lineContinuations[i];
            m_problems.push_back(declarationProblem(
                CODE, base::concatenate({"expected each line continuation of the geometry declarations to name two "
                                         "edges of one surface of ",
                                         MIXED_HYBRID_FORMULATION, " that meet at its middle grid point, found grid "
                                                                   "points ",
                                         std::to_string(line[0]), ", ", std::to_string(line[1]), " and ",
                                         std::to_string(line[2]),
                                         ", which the shells or the solids do not join by two edges at grid point ",
                                         std::to_string(line[1]), "; correct the declaration"})));
        }
    for (std::size_t i = 0; i < m_declarations->lineEnds.size(); ++i)
        if (m_declarationUse.lineEnds[i] == 0)
        {
            const input::LineEnd& end = m_declarations->lineEnds[i];
            m_problems.push_back(declarationProblem(
                CODE, base::concatenate({"expected each line end of the geometry declarations to name an edge of the "
                                         "shells or of the solids of ",
                                         MIXED_HYBRID_FORMULATION, ", found ", edgeName({end[0], end[1]}),
                                         ", which no edge of a shell or of a solid joins; correct the declaration"})));
        }
    reportDeclarationsNotApplied(*m_declarations, m_declarationUse.smoothEdges, m_declarationUse.sharpEdges,
                                 m_declarationUse.lineContinuations, m_declarationUse.lineEnds,
                                 DeclarationReading(m_gridIds, m_shellPart.get(), m_solidPart.get(), m_shellOfId),
                                 m_problems);
}

} // namespace oos::elements::mixedHybrid
