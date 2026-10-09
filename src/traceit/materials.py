import numpy as np
import traceit.units as units
import traceit.freq as freq


class Dielectric:
    """
    A class to represent a dielectric.

    Attributes
    ---------
    name: str
        name of the dielectric
    dk: float
        relative permittivity (eps' at f_ref)
    df: float
        dissipative factor (eps''/eps' at f_ref)
    f_ref: float
        reference frequency from datasheet (optional, default: 1e9)
    m1: float
        lower pole-spread bound (optional, default: 4.0)
    m2: float
        upper bound (optional, default: 12.0)
    eps_inf: float
    delta_eps: float
    """
