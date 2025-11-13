"""
constants.py — Precompute per-sphere constants for the Fast Spheres algorithm.

Implements Appendix 1 of the 1992 paper, translating the mathematical constants
(B_C, B'_C, ρ, etc.) into numeric form suitable for modern simulation.

All arithmetic uses NumPy to allow vectorisation and portable precision.
"""

import numpy as np


def precompute_constants(light_dir, radius, max_index, p=8):
    """
    Precompute constants needed for shading and differencing.

    Parameters
    ----------
    light_dir : tuple or array-like of shape (3,)
        Light source direction vector (lx, ly, lz).
    radius : float or int
        Sphere radius (pixels or arbitrary units).
    max_index : int
        Maximum colour index (i.e. the size of the colour table).
    p : int, optional
        Fixed-point precision in bits (default = 8).

    Returns
    -------
    constants : dict
        Dictionary of all precomputed constants, corresponding to Appendix 1.
    """

    lx, ly, lz = light_dir
    r = radius
    
    # Normalise light direction
    Lnorm = np.sqrt(lx*lx + ly*ly + lz*lz)
    if Lnorm == 0:
        raise ValueError("light_dir cannot be zero vector")
    lx /= Lnorm
    ly /= Lnorm
    lz /= Lnorm

    # Vector magnitude
    B_C = np.sqrt(lx**2 + ly**2 + lz**2)

    # Reciprocal term
    Bp_C = 1.0 / (B_C * r**2)

    # Extract mantissa and exponent of the reciprocal
    mantissa, exponent = np.frexp(Bp_C)  # mantissa in [0.5, 1), base-2 exponent

    rho = np.round((2**p) * mantissa).astype(int)
    C_C = int(exponent - p)

    # Core constants
    D_C = max_index
    Dp_C = D_C * rho
    Dpp_C = Dp_C * r

    # Direction-scaled coefficients
    E_C = Dp_C * lz
    F_C = Dpp_C * lx
    G_C = Dpp_C * ly
    H_C = -Dpp_C * lz * r

    # Difference coefficients
    K_C = 2 * E_C
    L_C = E_C + F_C
    M_C = E_C - F_C
    Q_C = E_C - G_C
    R_C = E_C + G_C
    W_C = 2 * K_C

    constants = {
        "E_C": E_C, "F_C": F_C, "G_C": G_C, "H_C": H_C,
        "K_C": K_C, "L_C": L_C, "M_C": M_C, "Q_C": Q_C, "R_C": R_C, "W_C": W_C,
        "C_C": C_C, "rho": rho, "B_C": B_C, "Dp_C": Dp_C, "Dpp_C": Dpp_C,
        "radius": radius, "max_index": max_index,
    }

    return constants


if __name__ == "__main__":
    # Quick sanity test
    consts = precompute_constants((0, 0, -1), radius=50, max_index=255)
    for k, v in consts.items():
        print(f"{k:4s} = {v}")
