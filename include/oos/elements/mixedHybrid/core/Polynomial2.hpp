#pragma once

#include <array>
#include <cstddef>

namespace oos::elements::mixedHybrid
{

// A polynomial of the two reference coordinates (ξ, η) of a shell element, in the monomial basis:
//   p(ξ, η) = Σ_{i,j < DEGREE_BOUND} c_ij ξ^i η^j,   c_ij at i·DEGREE_BOUND + j.
// The reference spaces of the shell (displacement, shear, moments, membrane strains) are built from such polynomials
// once per shape and degree, by sums, products and derivatives that are exact (no quadrature): their dual bases are
// then exact to the round-off. DEGREE_BOUND bounds the degree in each variable (below 12): the spaces of degree k = 4
// and their products stay well within it, and a product beyond it is a bug (InternalError).
//
// Thread-safety: a plain value type (a fixed array, no allocation); the operations are pure.
class Polynomial2
{
public:
    // The degree in each variable is below this bound.
    static constexpr std::size_t DEGREE_BOUND = 12;

    // The zero polynomial.
    Polynomial2() = default;

    [[nodiscard]] static Polynomial2 zero() { return {}; }
    // ξ^i η^j.
    [[nodiscard]] static Polynomial2 monomial(std::size_t i, std::size_t j);
    // The constant c.
    [[nodiscard]] static Polynomial2 constant(double c);
    // a + b ξ.
    [[nodiscard]] static Polynomial2 linearInXi(double a, double b);
    // a + b η.
    [[nodiscard]] static Polynomial2 linearInEta(double a, double b);

    // The coefficient of ξ^i η^j.
    [[nodiscard]] double coefficient(std::size_t i, std::size_t j) const
    {
        return m_coefficients[i * DEGREE_BOUND + j];
    }
    [[nodiscard]] double& coefficient(std::size_t i, std::size_t j) { return m_coefficients[i * DEGREE_BOUND + j]; }

    [[nodiscard]] Polynomial2 scaled(double s) const;
    // ∂p/∂ξ and ∂p/∂η.
    [[nodiscard]] Polynomial2 dXi() const;
    [[nodiscard]] Polynomial2 dEta() const;
    // p(ξ, η) = Σ_i (Σ_j c_ij η^j) ξ^i.
    [[nodiscard]] double evaluate(double xi, double eta) const;
    // The value and the derivatives (∂ξ, ∂η, ∂ξξ, ∂ξη, ∂ηη) at (ξ, η).
    [[nodiscard]] std::array<double, 6> jet(double xi, double eta) const;

    friend Polynomial2 operator+(const Polynomial2& a, const Polynomial2& b);
    friend Polynomial2 operator-(const Polynomial2& a, const Polynomial2& b);
    // The exact product; throws InternalError when a degree of the product reaches DEGREE_BOUND.
    friend Polynomial2 operator*(const Polynomial2& a, const Polynomial2& b);
    friend bool operator==(const Polynomial2&, const Polynomial2&) = default;

private:
    std::array<double, DEGREE_BOUND * DEGREE_BOUND> m_coefficients{};
};

} // namespace oos::elements::mixedHybrid
