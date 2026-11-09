#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/MatN.hpp"

#include <array>

namespace oos::elements
{

// The constitutive matrices of the surface elements, in Voigt order (εx, εy, γxy) with the engineering shear strain
// γxy = 2εxy (MAT1, AnisoMat2D: the stress-strain relation of a plate is written on γxy).

// The plane-stress matrix of an isotropic material (MAT1): E/(1 − ν²)·[[1, ν, 0], [ν, 1, 0], [0, 0, (1 − ν)/2]].
// Precondition: |ν| < 1 (the input format bounds NU to (−1, 0.5]).
[[nodiscard]] base::Mat3 planeStressMatrix(double youngModulus, double poissonRatio);

// The rotation of the strains T(φ) from a frame to the frame turned by φ about the normal:
// T(φ) = [[c², s², cs], [s², c², −cs], [−2cs, 2cs, c² − s²]], c = cos φ, s = sin φ.
[[nodiscard]] base::Mat3 strainRotation(double angleInRadians);

// A constitutive matrix M given in the material frame, at the angle φ from the element x axis, written in the element
// frame: M_e = T(φ)ᵀ·M·T(φ). An isotropic matrix is unchanged.
[[nodiscard]] base::Mat3 rotateToElementFrame(const base::Mat3& materialMatrix, double angleInRadians);

// The constitutive matrices of the solids, in the Voigt order of the solid result tables and of AnisoMat3D (stresses
// σx, σy, σz, τxy, τyz, τzx and engineering strains εx, εy, εz, γxy, γyz, γzx, γ = 2ε). A 6-vector in this order:
using Voigt6 = std::array<double, 6>;
using Mat6 = base::MatN<6>;

// The isotropic matrix of an elastic solid from E and ν (Lamé constants λ = Eν/((1 + ν)(1 − 2ν)), μ = E/2(1 + ν);
// Zienkiewicz & Taylor vol. 1 §6.2): [[λ + 2μ, λ, λ], ...] on the normal strains, μ on the engineering shears.
// Precondition: −1 < ν < 0.5 (the input format bounds NU to (−1, 0.5]; 0.5 is incompressible and has no such matrix).
[[nodiscard]] Mat6 isotropicSolidMatrix(double youngModulus, double poissonRatio);

// The transformation of engineering strains to a rotated frame: ε_local = T·ε_global, where the columns of `axes` are
// the local axes written in the global frame (ε_local,ij = Σ_kl a_ki a_lj ε_global,kl, with the factors 2 of the
// engineering shears). The stresses transform with T⁻ᵀ, and a constitutive matrix given in the local frame is
// D_global = Tᵀ·D_local·T (the strain energy is the same in both frames).
[[nodiscard]] Mat6 solidStrainTransformation(const base::Mat3& axes);

} // namespace oos::elements
