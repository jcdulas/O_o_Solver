// The checks of the derivatives of the shape functions of a face (assembly::faceShapeDerivatives), which the glued
// contact reads the tangents of a face and the rotation of a shell in its plane from.

#include "CheckRunner.hpp"
#include "oos/assembly/constraints/FaceShapeFunctions.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <variant>
#include <vector>

using oos::assembly::FaceShapeDerivatives;
using oos::assembly::FaceShapeValues;

namespace
{

constexpr double STEP = 1.0e-4;

// The values of the shape functions of `face` at (ξ, η).
FaceShapeValues valuesAt(std::span<const std::int32_t> face, double xi, double eta)
{
    return std::get<FaceShapeValues>(oos::assembly::faceShapeFunctions(face, xi, eta));
}

} // namespace

CHECK_CASE("assembly: the derivatives of the shape functions of a face are those of the functions")
{
    // The faces: linear and quadratic, every mid-side point present or some absent (0).
    const std::vector<std::vector<std::int32_t>> faces{{1, 2, 3, 4},       {1, 2, 3, 4, 5, 6, 7, 8},
                                                       {1, 2, 3, 4, 5, 0, 7, 0}, {1, 2, 3},
                                                       {1, 2, 3, 4, 5, 6}, {1, 2, 3, 0, 5, 6}};
    const std::array<std::array<double, 2>, 3> quadPoints{
        std::array<double, 2>{0.3, -0.6}, std::array<double, 2>{-1.0, 1.0}, std::array<double, 2>{0.0, 0.0}};
    const std::array<std::array<double, 2>, 3> trianglePoints{
        std::array<double, 2>{0.2, 0.5}, std::array<double, 2>{0.0, 0.0}, std::array<double, 2>{0.6, 0.4}};
    for (const std::vector<std::int32_t>& face : faces)
    {
        const bool isTriangle = face.size() == 3 || face.size() == 6;
        for (const std::array<double, 2>& at : isTriangle ? trianglePoints : quadPoints)
        {
            const FaceShapeDerivatives d = oos::assembly::faceShapeDerivatives(face, at[0], at[1]);
            // The functions are polynomials of degree two in each coordinate: a central difference is exact to the
            // rounding of the values.
            const FaceShapeValues xiPlus = valuesAt(face, at[0] + STEP, at[1]);
            const FaceShapeValues xiMinus = valuesAt(face, at[0] - STEP, at[1]);
            const FaceShapeValues etaPlus = valuesAt(face, at[0], at[1] + STEP);
            const FaceShapeValues etaMinus = valuesAt(face, at[0], at[1] - STEP);
            double sumXi = 0.0;
            double sumEta = 0.0;
            const std::string what = "a face of " + std::to_string(face.size()) + " points";
            for (std::size_t k = 0; k < face.size(); ++k)
            {
                runner.expectNear(d.dXi[k], (xiPlus.values[k] - xiMinus.values[k]) / (2.0 * STEP), 1.0e-9,
                                  what + ": the derivative with respect to xi");
                runner.expectNear(d.dEta[k], (etaPlus.values[k] - etaMinus.values[k]) / (2.0 * STEP), 1.0e-9,
                                  what + ": the derivative with respect to eta");
                sumXi += d.dXi[k];
                sumEta += d.dEta[k];
            }
            // The functions sum to 1: their derivatives to 0.
            runner.expectNear(sumXi, 0.0, 1.0e-12, what + ": the sum of the derivatives with respect to xi");
            runner.expectNear(sumEta, 0.0, 1.0e-12, what + ": the sum of the derivatives with respect to eta");
        }
    }
}
