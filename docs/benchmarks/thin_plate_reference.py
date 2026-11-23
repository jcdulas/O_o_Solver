"""3D elasticity solution at the centre of the simply supported square plate."""

import numpy as np


def centre_response(last_odd_mode):
    side, thickness, young, poisson, pressure = 1.0, 0.01, 70e9, 0.3, 1000.0
    n = np.arange(1, last_odd_mode + 1, 2)
    deflection = bottom_stress = 0.0
    top_stress = -(1 + 2 * poisson) * pressure / 2
    for m in n:
        # Uniform pressure in the sine basis, evaluated at the centre.
        load = 16 * pressure * (-1.0) ** ((m + n - 2) // 2) / (np.pi**2 * m * n)
        k = np.pi * np.hypot(m, n) / side
        b = k * thickness / 2
        # Even and odd thickness solutions for sigma_zz. The exponentials keep
        # b / (sinh(b) * cosh(b)) finite for the highest wavenumbers.
        decay = np.exp(-2 * b)
        ratio = 4 * b * decay / (1 - decay**2)
        odd = 1 / (1 - ratio)
        even = 1 / (1 + ratio)
        deflection += np.sum(load * (1 - poisson**2) / (young * k) * (1 + decay) / (1 - decay) * odd)
        bottom_stress += np.sum(load * (1 + poisson) / 2 * (odd - even))
        top_stress -= np.sum(load * (1 + poisson) / 2 * (odd + even - 2))
    return deflection, bottom_stress, top_stress


if __name__ == "__main__":
    for last_mode in (401, 1001, 2001):
        w, bottom, top = centre_response(last_mode)
        print(f"Mode {last_mode}: deflection {w * 1000:.9f} mm, surface stresses {bottom:.3f}, {top:.3f} Pa")
