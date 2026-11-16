#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace oos::elements::mixedHybrid
{

// Quadrature rules of the mixed hybrid formulation: Gauss-Legendre and Gauss-Jacobi rules on [−1, 1], and the rules
// of the reference triangle, square, tetrahedron, pyramid, pentahedron and hexahedron built from them. The rules of
// the simplices and of the pyramid are tensor rules carried by the collapsed coordinates (M. G. Duffy, Quadrature
// over a pyramid or cube of integrands with a singularity at a vertex, SIAM Journal on Numerical Analysis 19 (1982)
// 1260-1262) of Dubiner (1991) and Sherwin and Karniadakis (1995), cited in PolynomialBases.hpp: the factor of the
// Jacobian of the collapse that is a power of (1 − x) is taken into the Gauss-Jacobi weight of that direction, so
// that a rule of n points per direction stays exact for the polynomials of degree 2n − 1 in that direction. The
// integrands of a curved element are rational; these rules converge on them exponentially with n.
//
// Reference elements: triangle (−1, −1), (1, −1), (−1, 1) (area 2); tetrahedron (−1, −1, −1), (1, −1, −1),
// (−1, 1, −1), (−1, −1, 1) (volume 4/3); pyramid of base [−1, 1]² at z = −1 and apex (0, 0, 1) (volume 8/3);
// pentahedron triangle × [−1, 1] (volume 4); hexahedron [−1, 1]³ (volume 8).
//
// Thread-safety: pure functions.

// An n-point rule on an interval: abscissae increasing, and their weights.
struct LineRule
{
    std::vector<double> points;
    std::vector<double> weights;
};

// A point of a rule on an interval.
struct LineRulePoint
{
    double point = 0.0;
    double weight = 0.0;
};

// A point of a rule on the reference triangle.
struct TriangleRulePoint
{
    double u = 0.0;
    double v = 0.0;
    double weight = 0.0;
};

// A point of a rule on a two-dimensional reference domain.
struct SurfaceRulePoint
{
    std::array<double, 2> point{};
    double weight = 0.0;
};

// A point of a rule on a reference solid.
struct VolumeRulePoint
{
    std::array<double, 3> point{};
    double weight = 0.0;
};

// The n-point Gauss-Legendre rule on [−1, 1], exact for the polynomials of degree 2n − 1. Each root of P_n is found
// by Newton's method from the estimate cos(π (i + 3/4) / (n + 1/2)), with
//   P'_n(z) = n (z P_n(z) − P_{n−1}(z)) / (z² − 1),
// and its weight is w = 2 / ((1 − z²) P'_n(z)²) (M. Abramowitz, I. A. Stegun, Handbook of Mathematical Functions,
// 1964, 25.4.29). The rule is symmetric by construction, and the middle point of an odd rule is 0 exactly.
[[nodiscard]] LineRule gaussLegendre(std::size_t n);

// The n-point Gauss-Jacobi rule for the weight (1 − x)^α on [−1, 1], exact for p (1 − x)^α with p of degree 2n − 1.
// By the method of G. H. Golub, J. H. Welsch (Calculation of Gauss quadrature rules, Mathematics of Computation 23
// (1969) 221-230): the nodes are the eigenvalues of the symmetric tridiagonal Jacobi matrix of the recurrence of the
// orthonormal polynomials P_k^(α,0),
//   J_kk = −α² / ((2k + α)(2k + α + 2)),   J_{k,k+1} = 2m (m + α) / (c √(c² − 1)),  m = k + 1, c = 2m + α,
// and the weight of a node is μ0 v_0², v the unit eigenvector and μ0 = ∫ (1 − x)^α dx = 2^(α+1) / (α + 1). The
// eigenvalue problem is solved by jacobiEigen; the nodes are returned increasing.
[[nodiscard]] LineRule gaussJacobi(std::size_t n, double alpha);

// The n × n rule on the reference triangle: Gauss-Legendre on the square [−1, 1]² carried by the collapse
// u = (1 + a)(1 − b) / 2 − 1, v = b, of Jacobian (1 − b) / 2. Exact for the polynomials of total degree 2n − 2; the
// weights sum to the area 2. Points with a outer, b inner.
[[nodiscard]] std::vector<TriangleRulePoint> triangleRule(std::size_t n);

// The n × n × n rule on the reference tetrahedron: Gauss-Legendre in a, Gauss-Jacobi of weight (1 − b) in b and of
// weight (1 − c)² in c, carried by the collapse of the cube
//   r = (1 + a)(1 − b)(1 − c) / 4 − 1,   s = (1 + b)(1 − c) / 2 − 1,   t = c,   Jacobian (1 − b)(1 − c)² / 8.
// Exact for the polynomials of total degree 2n − 1; the weights sum to the volume 4/3. a outer, then b, then c.
[[nodiscard]] std::vector<VolumeRulePoint> tetrahedronRule(std::size_t n);

// The n × n × n Gauss-Legendre rule on the hexahedron [−1, 1]³ (x outer, then y, then z).
[[nodiscard]] std::vector<VolumeRulePoint> hexahedronRule(std::size_t n);

// The rule of the pentahedron: the triangle rule of n × n points times the n-point Gauss-Legendre rule in z (z
// inner).
[[nodiscard]] std::vector<VolumeRulePoint> pentahedronRule(std::size_t n);

// The n × n × n rule of the pyramid: x = a q, y = b q, q = (1 − z) / 2, so that dx dy dz = q² da db dz and
// q² = (1 − z)² / 4 goes into the Gauss-Jacobi weight of z (α = 2). a outer, then b, then z.
[[nodiscard]] std::vector<VolumeRulePoint> pyramidRule(std::size_t n);

// A rule on a face of `cornerCount` corners in the coordinates of the face, the weights those of the area of the
// parameter domain: a triangle (3) in (L1, L2), the barycentric coordinates of its corners 1 and 2 (the triangle rule
// of n × n points, area 1/2); a quadrilateral (4) in (s, t) ∈ [0, 1]², s from corner 0 to corner 1, t from corner 0
// to corner 3 (the n × n Gauss-Legendre rule, area 1).
[[nodiscard]] std::vector<SurfaceRulePoint> faceRule(std::size_t cornerCount, std::size_t n);

// The rule on the reference domain of a shell element of `cornerCount` corners: the n × n Gauss-Legendre rule on the
// square [−1, 1]² (4 corners), or the collapsed n × n rule of the reference triangle (3 corners, triangleRule).
[[nodiscard]] std::vector<SurfaceRulePoint> referenceSurfaceRule(std::size_t cornerCount, std::size_t n);

// The n-point Gauss-Legendre rule carried to [0, 1] (the parameter of an edge).
[[nodiscard]] std::vector<LineRulePoint> unitIntervalRule(std::size_t n);

} // namespace oos::elements::mixedHybrid
