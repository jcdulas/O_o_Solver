// Checks of the ties of the solids with the shells and the lines alone (SolidPart::ties, SolidPartTies.cpp; spec
// section 3.4, ties 1 to 4): what the edges and faces of the solids follow, the consistency of every tie with the rigid
// motions and the uniform strains of the whole assembly (each dependent equals its row evaluated on the values of the
// field), the functions of a line alone equal to those of the solid edge at degree 4, and the energy of a block under
// shells with the ties substituted (rigid motions without energy, a uniform stretch reproduced exactly).

#include "CheckRunner.hpp"
#include "mixedHybrid/ShellPartCheckSupport.hpp"
#include "mixedHybrid/SolidPartCheckSupport.hpp"
#include "oos/elements/mixedHybrid/line/LineAnalysis.hpp"
#include "oos/elements/mixedHybrid/model/ShellPart.hpp"
#include "oos/elements/mixedHybrid/model/SolidPart.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace
{

namespace mh = oos::elements::mixedHybrid;
namespace sp = oos::checks::solidpart;
namespace hp = oos::checks::shellpart;
using oos::checks::CheckRunner;
using Kind = oos::elements::EntityLayout::Kind;
using TieKind = mh::TieUnknown::Kind;
using Gradient = std::array<std::array<double, 3>, 3>;

// The section of the shells: steel, 2 cm thick.
constexpr double SHELL_THICKNESS = 0.02;
// The kernels the solid parts keep (a hexahedron of degree 4 takes about 3 MB).
constexpr std::size_t KERNEL_CACHE_BYTES = 64u << 20;
// The degree of the lines alone (MIXED_HYBRID_DEFAULT_DEGREE).
constexpr std::size_t LINE_DEGREE = 4;
// A tie holds an affine field to round-off: the values are of order one.
constexpr double TIE_TOLERANCE = 1e-9;
// The faces of the solids that follow the shells of the dome lie on them to 2e-7 (their surfaces are rational, those
// of the solids polynomial); a skin read from the mesh of the solids instead is 2e-4 away on the tetrahedra.
constexpr double SURFACE_TOLERANCE = 1e-6;

// A model of solids, shells and edges of lines alone on the same grids (the dense indices of the solid mesh).
struct TiedModel
{
    sp::Mesh mesh;
    std::vector<mh::ShellCorners> shells;
    std::vector<mh::SolidPartAloneEdge> aloneEdges;
    std::unique_ptr<mh::ShellPart> shellPart;
    std::unique_ptr<mh::SolidPart> solidPart;
};

// Builds the shells (before the solids, which point to them) and the solids.
void build(TiedModel& model)
{
    const std::vector<mh::SolidPoint>& x = model.mesh.input.gridPositions;
    if (!model.shells.empty())
    {
        mh::ShellPartInput input;
        for (std::size_t g = 0; g < x.size(); ++g)
            input.grids.push_back({x[g], model.mesh.input.gridIds[g]});
        std::int32_t id = 1;
        for (const mh::ShellCorners& corners : model.shells)
        {
            mh::ShellPartShell shell;
            shell.elementId = id++;
            shell.corners = corners;
            shell.section = hp::isotropicSection(sp::YOUNG, sp::POISSON, SHELL_THICKNESS);
            input.shells.push_back(shell);
        }
        input.checkElements = false;
        model.shellPart = std::make_unique<mh::ShellPart>(std::move(input));
    }
    mh::SolidPartInput input = model.mesh.input;
    input.shells = model.shellPart.get();
    input.aloneEdges = model.aloneEdges;
    input.kernelCacheBytes = KERNEL_CACHE_BYTES;
    model.solidPart = std::make_unique<mh::SolidPart>(std::move(input));
}

// The grid of lattice point (i, j, k).
std::size_t gridOf(const TiedModel& model, std::size_t i, std::size_t j, std::size_t k)
{
    return model.mesh.lattice.id(i, j, k);
}

// Appends a grid (ID index + 1); returns its index.
std::size_t addGrid(TiedModel& model, const mh::SolidPoint& position)
{
    model.mesh.input.gridPositions.push_back(position);
    model.mesh.input.gridIds.push_back(static_cast<std::int32_t>(model.mesh.input.gridIds.size() + 1));
    return model.mesh.input.gridPositions.size() - 1;
}

// The top of the dome of mesh E of the solid checks: z = (0.4 + 0.3 (0.5 − x² − 0.6 y²)) k on [−0.5, 0.5]².
double domeTop(double x, double y)
{
    return 0.4 + 0.3 * (0.5 - x * x - 0.6 * y * y);
}

mh::SolidPoint domePoint(std::size_t i, std::size_t j, std::size_t k)
{
    const double x = 0.5 * static_cast<double>(i) - 0.5;
    const double y = 0.5 * static_cast<double>(j) - 0.5;
    return {x, y, domeTop(x, y) * static_cast<double>(k)};
}

// A straight block [0, 0.5 ni] × [0, 0.4 nj] × [0, 0.6 nk].
mh::SolidPoint boxPoint(std::size_t i, std::size_t j, std::size_t k)
{
    return {0.5 * static_cast<double>(i), 0.4 * static_cast<double>(j), 0.6 * static_cast<double>(k)};
}

// ---- affine fields

struct AffineField
{
    Gradient a{};
    mh::SolidPoint b{};
};

// The skew gradient of the rotation ω: A x = ω × x.
Gradient rotationGradient(const mh::SolidPoint& omega)
{
    return {{{0.0, -omega[2], omega[1]}, {omega[2], 0.0, -omega[0]}, {-omega[1], omega[0], 0.0}}};
}

// A translation, a rotation, and a general uniform gradient (a uniform strain with a rotation).
std::vector<AffineField> testFields()
{
    return {{{}, {0.3, -0.2, 0.5}},
            {rotationGradient({0.4, -0.7, 0.2}), {0.1, 0.2, -0.3}},
            {{{{1.1, 0.3, -0.2}, {0.25, -0.4, 0.15}, {0.05, 0.35, 0.6}}}, {0.1, -0.1, 0.2}}};
}

mh::SolidPoint rotationOf(const Gradient& a)
{
    return {0.5 * (a[2][1] - a[1][2]), 0.5 * (a[0][2] - a[2][0]), 0.5 * (a[1][0] - a[0][1])};
}

// The values of an affine field on the unknowns of the ties: the grids, the entities of the solids
// (SolidPart::affineField), the unknowns of the shells (ShellPart::affineField) and of the edges alone (straight:
// their functions take nothing, their twist coefficient 0 is ω · (X_upper − X_lower)).
class AffineValues
{
public:
    AffineValues(const TiedModel& model, const AffineField& field) : m_model(model), m_field(field)
    {
        oos::base::Mat3 gradient;
        for (std::size_t r = 0; r < 3; ++r)
            for (std::size_t c = 0; c < 3; ++c)
                gradient(r, c) = field.a[r][c];
        m_solid = model.solidPart->affineField(gradient, Kind::Static);
        if (model.shellPart)
            m_shell = hp::affineVector(*model.shellPart, field.a, field.b);
    }

    [[nodiscard]] double operator()(const mh::TieUnknown& u) const
    {
        switch (u.kind)
        {
        case TieKind::Grid:
        {
            const mh::SolidPoint& x = m_model.mesh.input.gridPositions[u.index];
            const std::array<double, 3>& row = m_field.a[u.dof];
            return m_field.b[u.dof] + row[0] * x[0] + row[1] * x[1] + row[2] * x[2];
        }
        case TieKind::SolidEntity:
            return m_solid[m_model.solidPart->dofOffset(u.index) + u.dof];
        case TieKind::ShellUnknown:
            return m_shell[u.index];
        case TieKind::AloneEdge:
            break;
        }
        for (const mh::SolidPartAloneEdge& edge : m_model.aloneEdges)
        {
            if (edge.edge != u.index)
                continue;
            const std::size_t firstTwist = 3 * (edge.alone.degree - 1);
            if (u.dof != firstTwist)
                return 0.0;
            const mh::SolidPoint chord = mh::sub3(edge.alone.upperPoint, edge.alone.lowerPoint);
            return mh::dot3(rotationOf(m_field.a), chord);
        }
        throw std::runtime_error("no edge alone of that number");
    }

private:
    const TiedModel& m_model;
    AffineField m_field;
    std::vector<double> m_solid;
    mh::Vector m_shell;
};

// The largest |dependent − Σ c term| over the ties.
double tieResidual(const std::vector<mh::SolidTie>& ties, const AffineValues& values)
{
    double worst = 0.0;
    for (const mh::SolidTie& tie : ties)
    {
        double sum = 0.0;
        for (const mh::TieTerm& term : tie.terms)
            sum += term.coefficient * values(term.unknown);
        worst = std::max(worst, std::fabs(values(tie.dependent) - sum));
    }
    return worst;
}

void expectConsistent(CheckRunner& runner, const TiedModel& model, const std::vector<mh::SolidTie>& ties)
{
    for (const AffineField& field : testFields())
    {
        const double residual = tieResidual(ties, AffineValues(model, field));
        runner.expect(residual < TIE_TOLERANCE, "an affine field satisfies every tie: residual " + hp::sci(residual));
    }
}

std::size_t countOf(const std::vector<mh::SolidTie>& ties, TieKind kind)
{
    return static_cast<std::size_t>(
        std::count_if(ties.begin(), ties.end(), [&](const mh::SolidTie& tie) { return tie.dependent.kind == kind; }));
}

std::size_t followingEdges(const mh::SolidPart& part, mh::SolidEdgeTie::Kind kind)
{
    const std::span<const mh::SolidPartEdge> edges = part.edges();
    return static_cast<std::size_t>(std::count_if(
        edges.begin(), edges.end(), [&](const mh::SolidPartEdge& e) { return e.tie.kind == kind && e.follows; }));
}

std::size_t followingFaces(const mh::SolidPart& part)
{
    const std::span<const mh::SolidPartFace> faces = part.faces();
    return static_cast<std::size_t>(
        std::count_if(faces.begin(), faces.end(), [](const mh::SolidPartFace& f) { return f.follows; }));
}

// The height above (x, y) of the surface of shell `shell`, which is a graph over the plane (x, y): Newton's method
// on its reference coordinates, the Jacobian by differences.
double shellHeight(const mh::ShellPart& shells, std::size_t shell, double x, double y)
{
    constexpr int ITERATIONS = 30;
    constexpr double STEP = 1e-6;
    const auto none = [](const mh::FunctionKind&) { return false; };
    const auto at = [&](double xi, double eta)
    {
        const std::array<mh::ReferencePoint, 1> point{{{xi, eta}}};
        return shells.displacementRowsAt(shell, point, none).front().point;
    };
    mh::ReferencePoint r = mh::referenceCentre(shells.input().shells[shell].corners.count);
    for (int iteration = 0; iteration < ITERATIONS; ++iteration)
    {
        const mh::SpaceVector p = at(r[0], r[1]);
        const mh::SpaceVector alongXi = at(r[0] + STEP, r[1]);
        const mh::SpaceVector alongEta = at(r[0], r[1] + STEP);
        const double a = (alongXi[0] - p[0]) / STEP;
        const double b = (alongEta[0] - p[0]) / STEP;
        const double c = (alongXi[1] - p[1]) / STEP;
        const double d = (alongEta[1] - p[1]) / STEP;
        const double determinant = a * d - b * c;
        const double dx = x - p[0];
        const double dy = y - p[1];
        r[0] += (d * dx - b * dy) / determinant;
        r[1] += (a * dy - c * dx) / determinant;
    }
    return at(r[0], r[1])[2];
}

// The largest |z_solid − z_shell| at the points of the kernels of the solids on the faces that follow the shells.
double gapToShells(const TiedModel& model)
{
    constexpr std::size_t POINTS = 6;
    const mh::SolidPart& part = *model.solidPart;
    double out = 0.0;
    for (const mh::SolidPartFace& face : part.faces())
    {
        if (!face.follows)
            continue;
        const std::size_t solid = face.solids.front();
        const std::optional<std::size_t> local = part.localFaceOf(solid, face.view());
        const std::variant<mh::SolidPart::SharedKernel, std::string> kernel = part.kernelOf(solid, Kind::Static, false);
        if (!local || !std::holds_alternative<mh::SolidPart::SharedKernel>(kernel))
            throw std::runtime_error("no kernel for a face under a shell");
        const mh::SolidPart::SharedKernel& k = std::get<mh::SolidPart::SharedKernel>(kernel);
        for (const mh::VolumeFacePoint& point : k->facePointsOf(*local, POINTS))
        {
            const mh::SolidPoint& x = point.position;
            out = std::max(out, std::fabs(x[2] - shellHeight(*model.shellPart, face.tie.shell, x[0], x[1])));
        }
    }
    return out;
}

void expectAccepted(CheckRunner& runner, const mh::SolidPart& part)
{
    for (std::size_t s = 0; s < part.solidCount(); ++s)
    {
        const std::optional<mh::SolidPartProblem> problem = part.validate(s);
        runner.expect(!problem,
                      "solid " + std::to_string(s) + " is accepted" + (problem ? ": " + problem->message : ""));
    }
}

// ---- the models

// 2 × 2 × 1 hexahedra under the dome, four quadrilateral shells on their curved top.
TiedModel domeUnderQuadrilaterals()
{
    TiedModel model;
    model.mesh = sp::latticeMesh({2, 2, 1}, domePoint, 4);
    sp::addHexahedra(model.mesh);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            model.shells.push_back(mh::ShellCorners::quadrilateral(gridOf(model, i, j, 1), gridOf(model, i + 1, j, 1),
                                                                   gridOf(model, i + 1, j + 1, 1),
                                                                   gridOf(model, i, j + 1, 1)));
    build(model);
    return model;
}

// 2 × 2 × 1 cells of six tetrahedra about their diagonal from corner 0 to corner 6 under the dome; on the top of cell
// (0, 0) one quadrilateral shell over its two triangles (their common edge is the diagonal of the quadrilateral), on
// the others two triangular shells on the triangles.
TiedModel domeTetrahedraUnderShells()
{
    TiedModel model;
    model.mesh = sp::latticeMesh({2, 2, 1}, domePoint, 4);
    const std::array<std::array<std::size_t, 4>, 6> tetrahedra{
        {{0, 1, 2, 6}, {0, 2, 3, 6}, {0, 3, 7, 6}, {0, 7, 4, 6}, {0, 4, 5, 6}, {0, 5, 1, 6}}};
    std::int32_t id = 1;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
        {
            const std::array<std::size_t, 8> c = model.mesh.lattice.cell(i, j, 0);
            for (const auto& t : tetrahedra)
                model.mesh.input.solids.push_back(sp::elementOf(id++, {c[t[0]], c[t[1]], c[t[2]], c[t[3]]}));
            // the top of the cell: corners 4, 5, 6, 7, cut along 4-6 by the tetrahedra
            if (i == 0 && j == 0)
            {
                model.shells.push_back(mh::ShellCorners::quadrilateral(c[4], c[5], c[6], c[7]));
                continue;
            }
            model.shells.push_back(mh::ShellCorners::triangle(c[4], c[5], c[6]));
            model.shells.push_back(mh::ShellCorners::triangle(c[4], c[6], c[7]));
        }
    build(model);
    return model;
}

// 2 × 2 × 1 straight hexahedra and a fin of two quadrilateral shells standing on the middle line y = 0.4 of their
// top, leaning towards +y.
TiedModel blockWithFin()
{
    TiedModel model;
    model.mesh = sp::latticeMesh({2, 2, 1}, boxPoint, 4);
    sp::addHexahedra(model.mesh);
    std::vector<std::size_t> top;
    for (std::size_t i = 0; i <= 2; ++i)
        top.push_back(addGrid(model, {0.5 * static_cast<double>(i), 0.5, 1.1}));
    for (std::size_t i = 0; i < 2; ++i)
        model.shells.push_back(
            mh::ShellCorners::quadrilateral(gridOf(model, i, 1, 1), gridOf(model, i + 1, 1, 1), top[i + 1], top[i]));
    build(model);
    return model;
}

// The same block with the fin upright: its normal lies in the top face of the block, which holds its fibre.
TiedModel blockWithUprightFin()
{
    TiedModel model;
    model.mesh = sp::latticeMesh({2, 2, 1}, boxPoint, 4);
    sp::addHexahedra(model.mesh);
    std::vector<std::size_t> top;
    for (std::size_t i = 0; i <= 2; ++i)
        top.push_back(addGrid(model, {0.5 * static_cast<double>(i), 0.4, 1.1}));
    for (std::size_t i = 0; i < 2; ++i)
        model.shells.push_back(
            mh::ShellCorners::quadrilateral(gridOf(model, i, 1, 1), gridOf(model, i + 1, 1, 1), top[i + 1], top[i]));
    build(model);
    return model;
}

// 2 × 1 × 1 straight hexahedra and two bars alone along their top edge y = 0, numbered 7 and 8 by the integrator.
TiedModel blockWithLines()
{
    TiedModel model;
    model.mesh = sp::latticeMesh({2, 1, 1}, boxPoint, 4);
    sp::addHexahedra(model.mesh);
    const mh::LineSection section = mh::LineSection::bar(sp::YOUNG, 8.0e10, 1.0e-3, 2.0e-7, 1.0e-7, 0.0, 3.0e-7, 0.0);
    for (std::size_t i = 0; i < 2; ++i)
    {
        const std::size_t a = gridOf(model, i, 0, 1);
        const std::size_t b = gridOf(model, i + 1, 0, 1);
        const mh::LineElement line = mh::LineElement::between(a, b, 0).oriented({0.0, 0.0, 1.0});
        const mh::EdgeKey key = mh::EdgeKey::of(a, b);
        const std::vector<mh::SolidPoint>& x = model.mesh.input.gridPositions;
        std::variant<mh::AloneEdge, std::string> alone =
            mh::aloneEdgeOfLine(line, section, {key.lower, key.higher}, x[key.lower], x[key.higher], LINE_DEGREE, 0);
        if (const std::string* problem = std::get_if<std::string>(&alone))
            throw std::runtime_error("the edge of the line: " + *problem);
        model.aloneEdges.push_back({key, 7 + i, std::get<mh::AloneEdge>(std::move(alone))});
    }
    build(model);
    return model;
}

// ---- a dense system of a model on the numbering: the ShellPart unknowns, then the entity DOFs of the solids

struct DenseTiedSystem
{
    const TiedModel* model = nullptr;
    std::size_t shellCount = 0;
    std::size_t size = 0;
    mh::Matrix stiffness;
    std::vector<std::optional<mh::VolumeKernel>> kernels;

    [[nodiscard]] std::size_t indexOf(const mh::TieUnknown& u) const
    {
        switch (u.kind)
        {
        case TieKind::Grid:
            return 6 * u.index + u.dof;
        case TieKind::ShellUnknown:
            return u.index;
        case TieKind::SolidEntity:
            return shellCount + model->solidPart->dofOffset(u.index) + u.dof;
        case TieKind::AloneEdge:
            break;
        }
        throw std::runtime_error("no edge alone in the dense system");
    }
    [[nodiscard]] std::size_t indexOf(const mh::SolidLocalUnknown& u) const
    {
        return indexOf(u.onGrid ? mh::TieUnknown{TieKind::Grid, u.index, u.component}
                                : mh::TieUnknown{TieKind::SolidEntity, u.index, u.component});
    }
};

DenseTiedSystem denseSystem(const TiedModel& model)
{
    DenseTiedSystem system;
    system.model = &model;
    const mh::SolidPart& solids = *model.solidPart;
    system.shellCount = model.shellPart->unknownCount();
    system.size = system.shellCount + solids.dofOffset(solids.entities(Kind::Static).size());
    system.stiffness = mh::Matrix(system.size, system.size);
    const hp::DenseAnalysis shells(*model.shellPart);
    for (std::size_t i = 0; i < system.shellCount; ++i)
        for (std::size_t j = 0; j < system.shellCount; ++j)
            system.stiffness(i, j) = shells.stiffness()(i, j);
    system.kernels.resize(solids.solidCount());
    for (std::size_t s = 0; s < solids.solidCount(); ++s)
    {
        mh::SolidResult<mh::VolumeKernel> kernel = solids.buildKernel(s, Kind::Static, false);
        if (!kernel.ok())
            throw std::runtime_error("solid refused: " + kernel.refusal());
        std::vector<std::size_t> dofs;
        for (const mh::SolidLocalUnknown& u : solids.localUnknowns(s, Kind::Static))
            dofs.push_back(system.indexOf(u));
        const mh::Matrix k = solids.localStiffness(s, kernel.value(), Kind::Static);
        for (std::size_t i = 0; i < dofs.size(); ++i)
            for (std::size_t j = 0; j < dofs.size(); ++j)
                system.stiffness(dofs[i], dofs[j]) += k(i, j);
        system.kernels[s] = std::move(kernel.value());
    }
    return system;
}

// u = P u_free + q: the held unknowns at their values, the dependents of the ties written on the free unknowns
// (through the chains of ties), the others free.
struct Reduction
{
    std::vector<std::optional<std::size_t>> freeIndex;
    std::size_t freeCount = 0;
    mh::Matrix p;
    mh::Vector q;
};

Reduction reduce(const DenseTiedSystem& system, const std::vector<mh::SolidTie>& ties,
                 const std::map<std::size_t, double>& held)
{
    const std::size_t n = system.size;
    std::map<std::size_t, std::vector<std::pair<std::size_t, double>>> rows;
    for (const mh::SolidTie& tie : ties)
    {
        std::vector<std::pair<std::size_t, double>> row;
        for (const mh::TieTerm& term : tie.terms)
            row.emplace_back(system.indexOf(term.unknown), term.coefficient);
        rows.emplace(system.indexOf(tie.dependent), std::move(row));
    }
    Reduction out;
    out.freeIndex.assign(n, std::nullopt);
    for (std::size_t i = 0; i < n; ++i)
        if (!held.contains(i) && !rows.contains(i))
            out.freeIndex[i] = out.freeCount++;
    out.p = mh::Matrix(n, out.freeCount);
    out.q = mh::Vector(n, 0.0);
    for (const auto& [i, value] : held)
    {
        if (rows.contains(i))
            throw std::runtime_error("a dependent of a tie is held");
        out.q[i] = value;
    }
    for (std::size_t i = 0; i < n; ++i)
        if (out.freeIndex[i])
            out.p(i, *out.freeIndex[i]) = 1.0;
    // the dependents, each once its terms are known (0 not yet, 1 under way, 2 done)
    std::vector<int> state(n, 0);
    const std::function<void(std::size_t)> resolve = [&](std::size_t i)
    {
        if (state[i] == 2)
            return;
        if (state[i] == 1)
            throw std::runtime_error("a cycle of ties");
        state[i] = 1;
        for (const auto& [j, c] : rows.at(i))
        {
            if (rows.contains(j))
                resolve(j);
            for (std::size_t f = 0; f < out.freeCount; ++f)
                out.p(i, f) += c * out.p(j, f);
            out.q[i] += c * out.q[j];
        }
        state[i] = 2;
    };
    for (const auto& [i, row] : rows)
        resolve(i);
    return out;
}

// The vector of an affine field on the dense numbering (the rotation slots of the grids at ω).
mh::Vector affineVectorOf(const DenseTiedSystem& system, const AffineField& field)
{
    const TiedModel& model = *system.model;
    const AffineValues values(model, field);
    mh::Vector out = hp::affineVector(*model.shellPart, field.a, field.b);
    out.resize(system.size, 0.0);
    const std::span<const mh::SolidPartEntity> entities = model.solidPart->entities(Kind::Static);
    for (std::size_t e = 0; e < entities.size(); ++e)
        for (std::size_t d = 0; d < entities[e].dofCount; ++d)
        {
            const mh::TieUnknown u{TieKind::SolidEntity, e, d};
            out[system.indexOf(u)] = values(u);
        }
    return out;
}

// ---- checks

CHECK_CASE("solidtie: hexahedra under quadrilateral shells follow their curved top")
{
    const TiedModel model = domeUnderQuadrilaterals();
    const mh::SolidPart& part = *model.solidPart;
    // the four faces of the top and their twelve edges follow the shells, and nothing else
    runner.expect(followingFaces(part) == 4, "four faces follow: " + std::to_string(followingFaces(part)));
    runner.expect(
        followingEdges(part, mh::SolidEdgeTie::Kind::Shell) == 12,
        "twelve edges follow the shells: " + std::to_string(followingEdges(part, mh::SolidEdgeTie::Kind::Shell)));
    std::size_t following = 0;
    for (const mh::SolidPartEdge& edge : part.edges())
        following += edge.follows ? 1 : 0;
    runner.expect(following == 12, "no other edge follows");
    // the faces that follow the shells are curved like them
    double curve = 0.0;
    for (const mh::SolidPartFace& face : part.faces())
        if (face.follows)
            for (const mh::SolidPoint& g : face.geometry)
                curve = std::max(curve, mh::norm3(g));
    runner.expect(curve > 1e-3, "the faces under the shells are curved: " + hp::sci(curve));

    // ties 1 and 2: 12 edges of 9 functions, 4 faces of 27; no rotation tie, every shell lies on a face
    const std::vector<mh::SolidTie> ties = part.ties();
    runner.expect(ties.size() == 12 * 9 + 4 * 27, "216 ties: " + std::to_string(ties.size()));
    runner.expect(countOf(ties, TieKind::SolidEntity) == ties.size(), "every dependent is a function of the solids");
    expectConsistent(runner, model, ties);
    expectAccepted(runner, part);
    // the faces under the shells lie on them, to the projection of their rational surface on the functions of degree 4
    const double gap = gapToShells(model);
    runner.expect(gap < SURFACE_TOLERANCE, "the faces under the shells lie on them: " + hp::sci(gap));

    // a held dependent is left out
    const std::size_t entity = *part.faceEntity(*part.faceIndex(
        std::array<std::size_t, 3>{gridOf(model, 0, 0, 1), gridOf(model, 1, 0, 1), gridOf(model, 1, 1, 1)}));
    const std::vector<mh::SolidTie> held =
        part.ties([&](const mh::TieUnknown& u) { return u.kind == TieKind::SolidEntity && u.index == entity; });
    runner.expect(held.size() == ties.size() - 27, "the 27 held functions of a face are left out");

    // without the shells, no tie
    mh::SolidPartInput alone = model.mesh.input;
    const mh::SolidPart bare(alone);
    runner.expect(bare.ties().empty() && followingFaces(bare) == 0, "no tie without the shells");
}

CHECK_CASE("solidtie: tetrahedra under triangles and a quadrilateral follow them, its diagonal too")
{
    const TiedModel model = domeTetrahedraUnderShells();
    const mh::SolidPart& part = *model.solidPart;
    // the eight triangles of the top; their edges: 12 edges of the cells and 3 diagonals of the triangular shells
    // follow shell edges, the diagonal under the quadrilateral follows it as a diagonal
    runner.expect(followingFaces(part) == 8, "eight faces follow: " + std::to_string(followingFaces(part)));
    runner.expect(
        followingEdges(part, mh::SolidEdgeTie::Kind::Shell) == 15,
        "fifteen edges follow shell edges: " + std::to_string(followingEdges(part, mh::SolidEdgeTie::Kind::Shell)));
    runner.expect(followingEdges(part, mh::SolidEdgeTie::Kind::Diagonal) == 1, "one edge follows a diagonal");
    const std::optional<std::size_t> diagonal =
        part.edgeIndex(mh::EdgeKey::of(gridOf(model, 0, 0, 1), gridOf(model, 1, 1, 1)));
    runner.expect(diagonal && part.edges()[*diagonal].tie.kind == mh::SolidEdgeTie::Kind::Diagonal,
                  "the diagonal of the quadrilateral follows it");

    const std::vector<mh::SolidTie> ties = part.ties();
    runner.expect(ties.size() == 16 * 9 + 8 * 9, "216 ties: " + std::to_string(ties.size()));
    expectConsistent(runner, model, ties);
    expectAccepted(runner, part);
    // the faces under the shells lie on them, to the projection of their rational surface on the functions of degree 4
    const double gap = gapToShells(model);
    runner.expect(gap < SURFACE_TOLERANCE, "the faces under the shells lie on them: " + hp::sci(gap));
}

CHECK_CASE("solidtie: a shell fin standing on a block follows it and turns with it")
{
    const TiedModel model = blockWithFin();
    const mh::SolidPart& part = *model.solidPart;
    runner.expect(followingEdges(part, mh::SolidEdgeTie::Kind::Shell) == 2, "the two edges under the fin follow it");
    runner.expect(followingFaces(part) == 0, "no face follows");
    // tie 1 on the two edges, tie 4: the four rotation unknowns of each edge of the fin follow the solids
    const std::vector<mh::SolidTie> ties = part.ties();
    runner.expect(countOf(ties, TieKind::SolidEntity) == 2 * 9, "the functions of the two edges are tied");
    const std::size_t rotations = countOf(ties, TieKind::ShellUnknown);
    runner.expect(rotations == 2 * 4, "the rotations about the two edges are tied: " + std::to_string(rotations));
    for (const mh::SolidTie& tie : ties)
        if (tie.dependent.kind == TieKind::ShellUnknown)
        {
            const mh::ShellPartUnknown u = model.shellPart->describe(tie.dependent.index);
            runner.expect(
                u.kind == mh::ShellPartUnknown::Kind::EntityDof && u.component >= model.shellPart->edgeRotationDof(0),
                "a rotation tie writes a rotation unknown of an edge");
        }
    expectConsistent(runner, model, ties);
}

CHECK_CASE("solidtie: an upright shell fin is tied to the face under it over its thickness")
{
    // The fibre of the fin at its foot lies in the top face of the block: the ties read the face over the thickness of
    // the fin (spec section 3.4). The functions of the two edges under the fin are written from the displacement
    // moments of the edges of the fin and from the unknowns of the faces on both sides, and so are the rotations
    // about them. The ties of the displacement hold any affine field; those of the rotation hold a rigid motion and
    // read, in a uniform strain, the rotation of the fibre: with the foot along x and the fibre along y, the slope
    // ∂u_z/∂y of the top face across the foot, where the field of the shells has the mean rotation of the strain.
    const TiedModel model = blockWithUprightFin();
    const mh::SolidPart& part = *model.solidPart;
    const std::vector<mh::SolidTie> ties = part.ties();
    runner.expect(countOf(ties, TieKind::SolidEntity) == 2 * 9, "the functions of the two edges are tied");
    runner.expect(countOf(ties, TieKind::ShellUnknown) == 2 * 4, "the rotations about the two edges are tied");
    // the faces of the skin of the block by entity, to tell an unknown of a face from one of an edge
    std::set<std::size_t> faceEntities;
    for (std::size_t f = 0; f < part.faces().size(); ++f)
        if (const std::optional<std::size_t> entity = part.faceEntity(f))
            faceEntities.insert(*entity);
    std::vector<mh::SolidTie> displacementTies;
    std::vector<mh::SolidTie> rotationTies;
    for (const mh::SolidTie& tie : ties)
    {
        std::size_t onFaces = 0;
        std::size_t onShells = 0;
        for (const mh::TieTerm& term : tie.terms)
        {
            onFaces += term.unknown.kind == TieKind::SolidEntity && faceEntities.contains(term.unknown.index) ? 1 : 0;
            onShells += term.unknown.kind == TieKind::ShellUnknown ? 1 : 0;
        }
        runner.expect(onFaces > 0, "a tie of the fin reads the faces beside its foot");
        if (tie.dependent.kind == TieKind::SolidEntity)
            runner.expect(onShells == 3, "a function of an edge under the fin is written from the three moments of "
                                         "the edge of the fin along its component: " +
                                             std::to_string(onShells));
        (tie.dependent.kind == TieKind::SolidEntity ? displacementTies : rotationTies).push_back(tie);
    }
    expectConsistent(runner, model, displacementTies);
    // The edges of the foot are 0.5 long along x (boxPoint): the rotation unknown 0 of an edge is its length times
    // the rotation about x, the others vanish for a uniform rotation.
    constexpr double EDGE_LENGTH = 0.5;
    for (const AffineField& field : testFields())
    {
        const AffineValues values(model, field);
        double worst = 0.0;
        for (const mh::SolidTie& tie : rotationTies)
        {
            double sum = 0.0;
            for (const mh::TieTerm& term : tie.terms)
                sum += term.coefficient * values(term.unknown);
            const mh::ShellPartUnknown u = model.shellPart->describe(tie.dependent.index);
            const bool isFirst = u.component == model.shellPart->edgeRotationDof(0);
            worst = std::max(worst, std::fabs(sum - (isFirst ? EDGE_LENGTH * field.a[2][1] : 0.0)));
        }
        runner.expect(worst < TIE_TOLERANCE,
                      "the rotation about the foot is the slope of the top face across it: " + hp::sci(worst));
    }
}

CHECK_CASE("solidtie: lines alone along an edge of a block follow it and turn with it")
{
    const TiedModel model = blockWithLines();
    const mh::SolidPart& part = *model.solidPart;
    runner.expect(followingEdges(part, mh::SolidEdgeTie::Kind::Alone) == 2, "the two edges of the lines are tied");
    for (const mh::SolidPartEdge& edge : part.edges())
        if (edge.tie.kind == mh::SolidEdgeTie::Kind::Alone)
            for (const mh::SolidPoint& g : edge.geometry)
                runner.expect(mh::norm3(g) == 0.0, "an edge of lines alone is straight");

    // tie 3: 9 functions per edge, tie 4: 4 twist coefficients per edge
    const std::vector<mh::SolidTie> ties = part.ties();
    const std::size_t lineTies = countOf(ties, TieKind::AloneEdge);
    runner.expect(lineTies == 2 * (9 + 4), "26 ties of the lines: " + std::to_string(lineTies));
    expectConsistent(runner, model, ties);

    // at degree 4 the functions of the line are those of the solid edge: function l along axis a of the frame of the
    // line is a · (function l of the solid edge)
    double worst = 0.0;
    for (const mh::SolidTie& tie : ties)
    {
        if (tie.dependent.kind != TieKind::AloneEdge || tie.dependent.dof >= 3 * (LINE_DEGREE - 1))
            continue;
        const mh::SolidPartAloneEdge& alone = model.aloneEdges[tie.dependent.index - 7];
        const std::size_t l = tie.dependent.dof / 3;
        const std::size_t axis = tie.dependent.dof % 3;
        const std::size_t entity = *part.edgeEntity(*part.edgeIndex(alone.grids));
        std::array<double, 9> expected{};
        for (std::size_t c = 0; c < 3; ++c)
            expected[3 * l + c] = alone.alone.frame[axis][c];
        std::array<double, 9> found{};
        for (const mh::TieTerm& term : tie.terms)
        {
            runner.expect(term.unknown.kind == TieKind::SolidEntity && term.unknown.index == entity,
                          "a line follows the functions of its solid edge only");
            found[term.unknown.dof] += term.coefficient;
        }
        for (std::size_t i = 0; i < 9; ++i)
            worst = std::max(worst, std::fabs(found[i] - expected[i]));
    }
    runner.expect(worst < 1e-12, "the projection is the identity at degree 4: " + hp::sci(worst));

    // the supports do not hold an edge of lines alone: it is held through its lines
    std::vector<mh::SolidHeldComponent> held;
    for (std::size_t g = 0; g < model.mesh.input.gridPositions.size(); ++g)
        for (std::size_t c = 0; c < 3; ++c)
            held.push_back({g, c, 0.0});
    for (const mh::SolidDerivedSupport& support : part.deriveSupports(held))
    {
        const mh::SolidPartEntity& entity = part.entities(Kind::Static)[support.entity];
        runner.expect(entity.kind != mh::SolidPartEntity::Kind::Edge || !part.edges()[entity.index].follows,
                      "no support on an edge that follows");
    }
}

CHECK_CASE("solidtie: a block under shells, the ties substituted, keeps its rigid motions and a uniform stretch")
{
    TiedModel model;
    model.mesh = sp::latticeMesh({2, 1, 1}, boxPoint, 4);
    sp::addHexahedra(model.mesh);
    for (std::size_t i = 0; i < 2; ++i)
        model.shells.push_back(mh::ShellCorners::quadrilateral(gridOf(model, i, 0, 1), gridOf(model, i + 1, 0, 1),
                                                               gridOf(model, i + 1, 1, 1), gridOf(model, i, 1, 1)));
    build(model);
    const mh::SolidPart& part = *model.solidPart;
    const DenseTiedSystem system = denseSystem(model);
    const std::vector<mh::SolidTie> ties = part.ties();
    double diagonal = 0.0;
    for (std::size_t i = 0; i < system.size; ++i)
        diagonal = std::max(diagonal, system.stiffness(i, i));

    // the six rigid motions: written on the free unknowns, they give back every unknown and have no energy
    const Reduction unheld = reduce(system, ties, {});
    for (std::size_t m = 0; m < 6; ++m)
    {
        AffineField field;
        mh::SolidPoint axis{};
        axis[m % 3] = 1.0;
        if (m < 3)
            field.b = axis;
        else
            field.a = rotationGradient(axis);
        const mh::Vector r = affineVectorOf(system, field);
        mh::Vector rFree(unheld.freeCount, 0.0);
        for (std::size_t i = 0; i < system.size; ++i)
            if (unheld.freeIndex[i])
                rFree[*unheld.freeIndex[i]] = r[i];
        const mh::Vector v = unheld.p * rFree;
        double gap = 0.0;
        for (std::size_t i = 0; i < system.size; ++i)
            gap = std::max(gap, std::fabs(v[i] - r[i]));
        const double force = hp::maxAbsOf(system.stiffness * v) / (diagonal * hp::maxAbsOf(v));
        runner.expect(gap < TIE_TOLERANCE, "rigid motion " + std::to_string(m) + " through the ties: " + hp::sci(gap));
        runner.expect(force < 1e-10, "rigid motion " + std::to_string(m) + " has no energy: " + hp::sci(force));
    }

    // a uniform stretch ε along x (σ_xx = E ε, the top free of traction): the grids and the supports derived from
    // them held, the rest free; the solution is the affine field and the stress of the solids uniform
    constexpr double STRAIN = 1e-3;
    AffineField stretch;
    stretch.a[0][0] = STRAIN;
    stretch.a[1][1] = -sp::POISSON * STRAIN;
    stretch.a[2][2] = -sp::POISSON * STRAIN;
    const mh::Vector exact = affineVectorOf(system, stretch);
    const AffineValues values(model, stretch);
    std::map<std::size_t, double> held;
    std::vector<mh::SolidHeldComponent> gridHeld;
    const std::size_t gridCount = model.mesh.input.gridPositions.size();
    for (std::size_t g = 0; g < gridCount; ++g)
        for (std::size_t c = 0; c < 6; ++c)
        {
            // the rotations of the grids carry no stiffness without lines
            held[6 * g + c] = c < 3 ? values({TieKind::Grid, g, c}) : 0.0;
            if (c < 3)
                gridHeld.push_back({g, c, held[6 * g + c]});
        }
    for (const mh::SolidDerivedSupport& support : part.deriveSupports(gridHeld))
        held[system.indexOf(mh::TieUnknown{TieKind::SolidEntity, support.entity, support.dof})] = support.value;
    std::vector<std::size_t> grids(gridCount);
    for (std::size_t g = 0; g < gridCount; ++g)
        grids[g] = g;
    const std::vector<mh::ShellGridSupport> shellGrids =
        hp::gridSupports(grids, {0, 1, 2}, [&](std::size_t g, std::size_t c) { return values({TieKind::Grid, g, c}); });
    for (const mh::ShellDerivedSupport& support : model.shellPart->supportsOf(shellGrids).held)
        held[support.unknown] = support.value;
    const Reduction reduction = reduce(system, ties, held);
    // K_free = Pᵀ K P, f = −Pᵀ K q
    const mh::Matrix kp = system.stiffness * reduction.p;
    const mh::Matrix kFree = reduction.p.transposed() * kp;
    const mh::Vector kq = system.stiffness * reduction.q;
    mh::Vector rhs(reduction.freeCount, 0.0);
    for (std::size_t f = 0; f < reduction.freeCount; ++f)
        for (std::size_t i = 0; i < system.size; ++i)
            rhs[f] -= reduction.p(i, f) * kq[i];
    const std::optional<mh::Vector> solution = mh::LuDecomposition(kFree).solve(rhs);
    runner.expect(solution.has_value(), "the system with the ties is regular");
    if (!solution)
        return;
    mh::Vector u = reduction.p * *solution;
    for (std::size_t i = 0; i < system.size; ++i)
        u[i] += reduction.q[i];
    double gap = 0.0;
    for (std::size_t i = 0; i < system.size; ++i)
        gap = std::max(gap, std::fabs(u[i] - exact[i]));
    runner.expect(gap < 1e-9 * STRAIN, "the solution is the uniform stretch: " + hp::sci(gap / STRAIN));
    const double stress = sp::YOUNG * STRAIN;
    for (std::size_t s = 0; s < part.solidCount(); ++s)
    {
        const mh::VolumeKernel& kernel = *system.kernels[s];
        mh::Vector local;
        for (const mh::SolidLocalUnknown& unknown : part.localUnknowns(s, Kind::Static))
            local.push_back(u[system.indexOf(unknown)]);
        const mh::Vector beta = mh::SolidPart::stressCoefficients(kernel, local, {});
        const mh::VoigtVector centre = mh::SolidPart::pointStresses(kernel, beta).centre;
        double error = std::fabs(centre[0] - stress);
        for (std::size_t c = 1; c < 6; ++c)
            error = std::max(error, std::fabs(centre[c]));
        runner.expect(error < 1e-8 * stress, "the stress of solid " + std::to_string(s) +
                                                 " is the uniform stress: " + hp::sci(error / stress));
    }
}

} // namespace
