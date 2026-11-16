#pragma once

#include "oos/elements/mixedHybrid/core/DenseDecompositions.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/line/LineAnalysis.hpp"
#include "oos/elements/mixedHybrid/line/LineCore.hpp"
#include "oos/elements/mixedHybrid/model/ShellPart.hpp"
#include "oos/elements/mixedHybrid/shell/ShellSection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// Small models of shells for the checks of ShellPart, and a dense analysis of them on the part numbering: the
// stiffness of the shells and of the lines on their edges assembled with the signs of their local unknowns, the grid
// supports and the supports derived from them held, the grid rotations without line held at zero (they carry no
// stiffness), the unknowns that no element stiffens held at zero, nothing else eliminated (the interiors of the
// shells stay unknowns), solved by LU.
namespace oos::checks::shellpart
{

namespace mh = oos::elements::mixedHybrid;

// The grids and quadrilaterals of a mapped mesh of nx × ny elements: grid (i, j) at f(i / nx, j / ny), numbered
// i (ny + 1) + j; quadrilateral (i, j) of corners (i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1) (the reference's
// mapped_mesh).
struct MappedMesh
{
    std::vector<mh::SpaceVector> grids;
    std::vector<std::array<std::size_t, 4>> quads;
};

inline MappedMesh mappedMesh(std::size_t nx, std::size_t ny, const std::function<mh::SpaceVector(double, double)>& f)
{
    MappedMesh mesh;
    for (std::size_t i = 0; i <= nx; ++i)
        for (std::size_t j = 0; j <= ny; ++j)
            mesh.grids.push_back(
                f(static_cast<double>(i) / static_cast<double>(nx), static_cast<double>(j) / static_cast<double>(ny)));
    const auto id = [&](std::size_t i, std::size_t j) { return i * (ny + 1) + j; };
    for (std::size_t i = 0; i < nx; ++i)
        for (std::size_t j = 0; j < ny; ++j)
            mesh.quads.push_back({id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)});
    return mesh;
}

// The isotropic section of the reference's checks: shear factor 5/6.
inline mh::Section isotropicSection(double e, double nu, double t, double density = 1.0, double expansion = 0.0)
{
    return mh::Section::isotropic(e, nu, t, 5.0 / 6.0, density, expansion);
}

// The input of the shells of a mesh: grid IDs index + 1 (or `ids`), quadrilaterals or each split in two triangles
// (corners 0, 1, 2 and 0, 2, 3).
inline mh::ShellPartInput shellInput(const MappedMesh& mesh, const mh::Section& section, bool triangles,
                                     const std::vector<std::int32_t>& ids = {})
{
    mh::ShellPartInput input;
    for (std::size_t g = 0; g < mesh.grids.size(); ++g)
        input.grids.push_back({mesh.grids[g], ids.empty() ? static_cast<std::int32_t>(g + 1) : ids[g]});
    std::int32_t id = 1;
    const auto add = [&](mh::ShellCorners corners)
    {
        mh::ShellPartShell shell;
        shell.elementId = id++;
        shell.corners = corners;
        shell.section = section;
        shell.kirchhoff = !section.h.has_value();
        input.shells.push_back(shell);
    };
    for (const auto& q : mesh.quads)
    {
        if (triangles)
        {
            add(mh::ShellCorners::triangle(q[0], q[1], q[2]));
            add(mh::ShellCorners::triangle(q[0], q[2], q[3]));
        }
        else
        {
            add(mh::ShellCorners::quadrilateral(q[0], q[1], q[2], q[3]));
        }
    }
    return input;
}

// The supports of the grids `grids`, every component of `components` held, with the values of `value(g, c)`.
inline std::vector<mh::ShellGridSupport> gridSupports(const std::vector<std::size_t>& grids,
                                                      const std::vector<std::size_t>& components,
                                                      const std::function<double(std::size_t, std::size_t)>& value = {})
{
    std::vector<mh::ShellGridSupport> out;
    for (const std::size_t g : grids)
    {
        mh::ShellGridSupport support;
        support.grid = g;
        for (const std::size_t c : components)
        {
            support.held[c] = true;
            support.values[c] = value ? value(g, c) : 0.0;
        }
        out.push_back(support);
    }
    return out;
}

// The held translations of supports, for ShellPartInput::heldTranslations.
inline std::set<std::pair<std::size_t, int>> heldTranslationsOf(const std::vector<mh::ShellGridSupport>& supports)
{
    std::set<std::pair<std::size_t, int>> held;
    for (const mh::ShellGridSupport& s : supports)
        for (int c = 0; c < 3; ++c)
            if (s.held[static_cast<std::size_t>(c)])
                held.insert({s.grid, c});
    return held;
}

// The dense analysis of a part.
class DenseAnalysis
{
public:
    explicit DenseAnalysis(const mh::ShellPart& part) : m_part(part), m_n(part.unknownCount())
    {
        if (!part.usable())
            throw std::runtime_error("the shell part is not usable");
        m_k = mh::Matrix(m_n, m_n);
        m_f.assign(m_n, 0.0);
        for (std::size_t ie = 0; ie < part.input().shells.size(); ++ie)
        {
            std::variant<mh::ShellElementData, std::string> built = part.elementWithMatrices(ie);
            if (const std::string* message = std::get_if<std::string>(&built))
                throw std::runtime_error("shell refused: " + *message);
            m_shells.push_back(std::get<mh::ShellElementData>(std::move(built)));
            const mh::ShellElementData& el = m_shells.back();
            scatter(el.dofs, el.signs, el.matrices.stiffness);
        }
        for (std::size_t il = 0; il < part.lineCount(); ++il)
        {
            std::variant<mh::LineData, std::string> data = part.lineData(il);
            if (const std::string* message = std::get_if<std::string>(&data))
                throw std::runtime_error("line refused: " + *message);
            m_lines.push_back(std::get<mh::LineData>(std::move(data)));
            const mh::LineData& line = m_lines.back();
            scatter(line.dofs, line.signs, line.stiffness);
            m_lineGrids.insert(part.input().lines[il].element.grids.begin(),
                               part.input().lines[il].element.grids.end());
        }
    }

    [[nodiscard]] const mh::Matrix& stiffness() const { return m_k; }
    [[nodiscard]] const mh::ShellElementData& shell(std::size_t ie) const { return m_shells[ie]; }
    [[nodiscard]] const mh::LineData& line(std::size_t il) const { return m_lines[il]; }
    [[nodiscard]] std::size_t size() const { return m_n; }

    // Adds the stiffness `k` of local unknowns (dofs, signs) to a matrix of the part numbering.
    void scatterInto(mh::Matrix& target, const std::vector<std::size_t>& dofs, const std::vector<double>& signs,
                     const mh::Matrix& k) const
    {
        for (std::size_t i = 0; i < dofs.size(); ++i)
            for (std::size_t j = 0; j < dofs.size(); ++j)
                target(dofs[i], dofs[j]) += signs[i] * signs[j] * k(i, j);
    }

    // Loads.
    void clearLoads() { m_f.assign(m_n, 0.0); }
    void addLocal(const std::vector<std::size_t>& dofs, const std::vector<double>& signs, const mh::Vector& local)
    {
        for (std::size_t i = 0; i < dofs.size(); ++i)
            m_f[dofs[i]] += signs[i] * local[i];
    }
    void addShellLoad(std::size_t ie, const mh::Vector& local)
    {
        addLocal(m_shells[ie].dofs, m_shells[ie].signs, local);
    }
    void addLineLoad(std::size_t il, const mh::Vector& local) { addLocal(m_lines[il].dofs, m_lines[il].signs, local); }
    void addGridForce(std::size_t g, std::size_t c, double v) { m_f[6 * g + c] += v; }

    // Solves with the grid supports and those derived from them; the notes of the derivation are kept.
    mh::Vector solve(const std::vector<mh::ShellGridSupport>& grids)
    {
        std::map<std::size_t, double> fixed;
        // the rotations of the grids without lines carry no stiffness
        for (std::size_t g = 0; g < m_part.gridCount(); ++g)
            if (!m_lineGrids.contains(g))
                for (std::size_t c = 3; c < 6; ++c)
                    fixed[6 * g + c] = 0.0;
        for (const mh::ShellGridSupport& s : grids)
            for (std::size_t c = 0; c < 6; ++c)
                if (s.held[c] && (c < 3 || m_lineGrids.contains(s.grid)))
                    fixed[6 * s.grid + c] = s.values[c];
        const mh::ShellSupports derived = m_part.supportsOf(grids);
        notes = derived.notes;
        for (const mh::ShellDerivedSupport& d : derived.held)
            fixed[d.unknown] = d.value;
        // the unknowns no element stiffens
        for (std::size_t i = 0; i < m_n; ++i)
        {
            bool touched = false;
            for (std::size_t j = 0; j < m_n && !touched; ++j)
                touched = m_k(i, j) != 0.0;
            if (!touched)
                fixed.try_emplace(i, 0.0);
        }
        std::vector<std::size_t> free;
        for (std::size_t i = 0; i < m_n; ++i)
            if (!fixed.contains(i))
                free.push_back(i);
        mh::Vector u(m_n, 0.0);
        for (const auto& [i, v] : fixed)
            u[i] = v;
        mh::Matrix a(free.size(), free.size());
        mh::Vector rhs(free.size(), 0.0);
        for (std::size_t p = 0; p < free.size(); ++p)
        {
            rhs[p] = m_f[free[p]];
            for (const auto& [j, v] : fixed)
                rhs[p] -= m_k(free[p], j) * v;
            for (std::size_t q = 0; q < free.size(); ++q)
                a(p, q) = m_k(free[p], free[q]);
        }
        m_free = free;
        const std::optional<mh::Vector> x = mh::LuDecomposition(a).solve(rhs);
        if (!x)
            throw std::runtime_error("singular stiffness");
        for (std::size_t p = 0; p < free.size(); ++p)
            u[free[p]] = (*x)[p];
        return u;
    }

    // The unknowns free in the last solution.
    [[nodiscard]] const std::vector<std::size_t>& freeUnknowns() const { return m_free; }

    // The mean rotation of the shells at the grids without lines put in the rotation slots (the reference's
    // fill_rotations).
    void fillRotations(mh::Vector& u) const
    {
        for (std::size_t g = 0; g < m_part.gridCount(); ++g)
        {
            if (m_lineGrids.contains(g) || m_part.shellsAt(g).empty())
                continue;
            const std::array<std::vector<mh::ShellPartTerm>, 3> rows = m_part.meanRotationRows(g);
            for (std::size_t c = 0; c < 3; ++c)
            {
                double v = 0.0;
                for (const mh::ShellPartTerm& term : rows[c])
                    v += term.coefficient * u[term.unknown];
                u[6 * g + 3 + c] = v;
            }
        }
    }

    std::vector<std::string> notes;

private:
    void scatter(const std::vector<std::size_t>& dofs, const std::vector<double>& signs, const mh::Matrix& k)
    {
        scatterInto(m_k, dofs, signs, k);
    }

    const mh::ShellPart& m_part;
    std::size_t m_n = 0;
    mh::Matrix m_k;
    mh::Vector m_f;
    std::vector<mh::ShellElementData> m_shells;
    std::vector<mh::LineData> m_lines;
    std::set<std::size_t> m_lineGrids;
    std::vector<std::size_t> m_free;
};

// The vector of the part numbering of the affine field u = b + A x: the grid translations, the rotation slots ω (the
// skew part of A) at every grid, the entity DOFs of affineField.
inline mh::Vector affineVector(const mh::ShellPart& part, const std::array<mh::SpaceVector, 3>& a,
                               const mh::SpaceVector& b)
{
    mh::Vector u(part.unknownCount(), 0.0);
    const mh::SpaceVector omega{0.5 * (a[2][1] - a[1][2]), 0.5 * (a[0][2] - a[2][0]), 0.5 * (a[1][0] - a[0][1])};
    for (std::size_t g = 0; g < part.gridCount(); ++g)
    {
        const mh::SpaceVector& x = part.input().grids[g].position;
        for (std::size_t c = 0; c < 3; ++c)
        {
            u[6 * g + c] = b[c] + a[c][0] * x[0] + a[c][1] * x[1] + a[c][2] * x[2];
            u[6 * g + 3 + c] = omega[c];
        }
    }
    part.affineField(a, b, std::span<double>(u).subspan(part.firstEntityUnknown()));
    return u;
}

// The skew gradient of the rotation ω: A x = ω × x.
inline std::array<mh::SpaceVector, 3> rotationGradient(const mh::SpaceVector& omega)
{
    return {{{0.0, -omega[2], omega[1]}, {omega[2], 0.0, -omega[0]}, {-omega[1], omega[0], 0.0}}};
}

// max |v_i|.
inline double maxAbsOf(const mh::Vector& v)
{
    double m = 0.0;
    for (const double x : v)
        m = std::max(m, std::fabs(x));
    return m;
}

// A number for a message, in scientific notation.
inline std::string sci(double v)
{
    char text[32];
    std::snprintf(text, sizeof text, "%.3e", v);
    return text;
}

// The relative difference |a − b| / max(|b|, scale).
inline double relativeDifference(double a, double b, double scale = 0.0)
{
    return std::fabs(a - b) / std::max(std::fabs(b), scale);
}

} // namespace oos::checks::shellpart
