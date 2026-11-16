#pragma once

#include "oos/base/errors/InternalError.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/solid/SolidData.hpp"
#include "oos/elements/mixedHybrid/solid/VolumeShape.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace oos::elements::mixedHybrid
{

// The functions of a solid element and its geometry (document 35 of the reference, sections 3.2 and 3.4).
//
// The functions of an element are the functions of the space of degree p of its shape whose traces on its faces are
// the face functions of VolumeShape.hpp. They are not unique: the space has functions that vanish on the boundary.
// Each is taken in the smallest space that holds its traces (the space of the level of its traces, volumeLevels), and
// of least norm there, by the minimum-norm inverse of the matrix of the traces (minimumNormInverse, from cyclic Jacobi
// rotations). The functions of the corners are then those of level 1 (barycentric, trilinear, prismatic, and the
// rational functions of the five-corner pyramid), so that a straight element is the map of its corners. What is left
// inside belongs to the bubbles of the kernel, which its weak equilibrium eliminates: the stiffness does not depend
// on that choice. Each construction checks that the functions reproduce their traces (largest residual below 1e-8)
// and refuses the element otherwise.
//
// The functions, and so the local unknowns of the element (three per function, VolumeKernel.hpp), are numbered:
//   j = 0 … nv − 1                 the corners, in the order of the input format (nv = shapeCornerCount);
//   j = nv + e·ec + m              edge e (the order of shapeEdges), function m = 0 … ec − 1 of the edge,
//                                  s (1 − s) P_m(2s − 1), s measured from the corner of the lower number;
//                                  ec = edgeFunctionCount(p);
//   j = firstFace(f) + i           face f (the order of shapeFaces), function i = 0 … faceFunctionCount(pf, n) − 1 of
//                                  its inside in the order of faceFunctions (written from the numbers of its corners);
//                                  firstFace(f) = nv + ne·ec + the counts of the faces before f.
// An edge or face function depends only on the numbers of the grid points of its edge or face, never on the element:
// unknown m of an edge (or i of a face) is the same unknown for every element that has it.
//
// The geometry X(r) = Σ_j N_j(r) X_j is carried by the same functions: the corners, and a coefficient per edge and
// face function (zero for a straight edge or a flat face). A rigid motion and a uniform strain are then fields of
// every element, straight or curved.
//
// Thread-safety: elementFunctions may be called from several threads at once (it keeps its results behind a mutex); a
// VolumeGeometry is a value, immutable once built for the kernels that read it.

// The coefficients of the functions of an element of shape `shape` whose corners have the numbers `order` (in the
// order of the shape), for the trace degree p and the face degree pf, on the basis of volumeSpace(shape, p): one row
// per function of the basis, one column per function of the element in the order of the head of this file. Refused
// when the functions do not reproduce their traces (largest residual 1e-8 or more). The result depends on `order` only
// through the comparisons of its numbers: elements whose numbers are in the same relative order share it, and it is
// kept once built for that relative order (up to a bound: a mesh has few orientations of its elements).
// Reference: volume::element_functions
[[nodiscard]] SolidResult<Matrix> elementFunctions(VolumeShape shape, std::span<const base::EntityId> order,
                                                   std::size_t p, std::size_t pf);

// The geometry of an element: its corners and the coefficients of its edges and faces on its functions, with the
// functions themselves.
// Reference: volume::VolumeGeometry
struct VolumeGeometry
{
    VolumeShape shape = VolumeShape::Tetrahedron;
    // `x`: the positions of the corners (basic axes), in the order of the shape.
    std::vector<SolidPoint> corners;
    // `edge`: per edge of the shape (order of shapeEdges), the coefficients of its edgeFunctionCount(p) functions.
    std::vector<std::vector<SolidPoint>> edges;
    // `face`: per face of the shape (order of shapeFaces), the coefficients of its functions.
    std::vector<std::vector<SolidPoint>> faces;
    // `p`, `pf`: the degrees the functions were built for.
    std::size_t traceDegree = 0;
    std::size_t faceDegree = 0;
    // `coef`: the coefficients of the functions on the basis of volumeSpace(shape, traceDegree) (elementFunctions).
    Matrix functionCoefficients;

    // An element of corners at `cornerPositions`, of numbers `order`: straight edges and flat faces.
    // Reference: volume::VolumeGeometry::straight
    [[nodiscard]] static SolidResult<VolumeGeometry> straight(VolumeShape shape,
                                                              std::vector<SolidPoint> cornerPositions,
                                                              std::span<const base::EntityId> order, std::size_t p,
                                                              std::size_t pf);

    // True when an edge or a face has a non-zero coefficient.
    // Reference: volume::VolumeGeometry::curved
    [[nodiscard]] bool curved() const;

    // The number of functions of the element.
    [[nodiscard]] std::size_t functionCount() const { return functionCoefficients.columns(); }

    // The coefficient of the geometry on function j (numbered as in the head of this file), component m.
    // Reference: volume::VolumeGeometry::coordinate
    [[nodiscard]] double coordinate(std::size_t j, std::size_t m) const;

    // Edge `edge` as the parabola through the mid-side node at `midSideNode`: the coefficient 4 (X_m − (X_a + X_b) / 2)
    // on its first function s (1 − s) (which is 1/4 at s = 1/2), the other coefficients of the edge zero. A face all of
    // whose edges have one then keeps no curve of its own (its coefficients stay zero): its surface is that of the
    // quadratic element of the cards (serendipity on a quadrilateral, the six-node triangle). No effect for p < 2 (an
    // edge without functions is straight).
    // Reference: tdnns::solid::build (the geometry of the mid-side grids)
    void setMidSideNode(std::size_t edge, const SolidPoint& midSideNode);

    // The functions of the element at the reference point r.
    // Reference: volume::VolumeGeometry::functions
    template <Scalar T>
    [[nodiscard]] std::vector<T> functions(const std::array<T, 3>& r) const
    {
        const std::vector<T> basis = volumeSpace(shape, traceDegree, r);
        const std::size_t count = functionCoefficients.columns();
        std::vector<T> out(count, constantOf<T>(0.0));
        // Row k of the coefficients is basis function k in every function: the sum for each function runs over k
        // increasing, as a column-by-column product would.
        for (std::size_t k = 0; k < basis.size(); ++k)
        {
            const std::span<const double> row = functionCoefficients.rowSpan(k);
            for (std::size_t j = 0; j < count; ++j)
                if (row[j] != 0.0)
                    out[j] = out[j] + scale(basis[k], row[j]);
        }
        return out;
    }

    // The point of the element at the reference point r: Σ_j N_j(r) X_j.
    // Reference: volume::VolumeGeometry::at
    template <Scalar T>
    [[nodiscard]] std::array<T, 3> at(const std::array<T, 3>& r) const
    {
        const std::vector<T> values = functions(r);
        std::array<T, 3> out{constantOf<T>(0.0), constantOf<T>(0.0), constantOf<T>(0.0)};
        for (std::size_t j = 0; j < values.size(); ++j)
        {
            for (std::size_t m = 0; m < 3; ++m)
            {
                const double c = coordinate(j, m);
                if (c != 0.0)
                    out[m] = out[m] + scale(values[j], c);
            }
        }
        return out;
    }

    // The point of a tetrahedron at the barycentric coordinates l of its corners: r = (2 l1 − 1, 2 l2 − 1, 2 l3 − 1).
    // Reference: volume::VolumeGeometry::position
    template <Scalar T>
    [[nodiscard]] std::array<T, 3> positionAtBarycentric(const std::array<T, 4>& l) const
    {
        OOS_ASSERT(shape == VolumeShape::Tetrahedron);
        const T one = constantOf<T>(1.0);
        return at(std::array<T, 3>{scale(l[1], 2.0) - one, scale(l[2], 2.0) - one, scale(l[3], 2.0) - one});
    }
};

} // namespace oos::elements::mixedHybrid
