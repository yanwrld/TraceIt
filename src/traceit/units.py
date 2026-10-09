"""
This file defines physical constants and unit conversions.

All values in TraceIt are expressed in SI units.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy import constants as _c

# physical constants (from scipy/CODATA)
C0: float = _c.c # speed of light in vaccum [m/s]
MU0: float = _c.mu_0 # permeability of free space [H/m]
EPS0: float = _c.epsilon_0 # permittivity of free space [F/m]
ETA0: float = float(np.sqrt(MU0 / EPS0)) # wave impedance in free space [ohm]

# length conversion factors
MM: float = 1e-3
UM: float = 1e-6
INCH: float = 25.4e-3
MIL: float = INCH / 1000
OZ_CU: float = 34.79e-6 # copper thickness of 1oz/ft2

# frequency conversion factors
MHZ: float = 1e6
GHZ: float = 1e9
PS: float = 1e-12
NS: float = 1e-9
GBPS: float = 1e9

# conductivity
SIGMA_CU: float = 5.8e7 # annealed copper [S/m]
NP_TO_DB: float = 20.0 / np.log(10.0) # 1 neper

_TINY = 1e-300 # to avoid singularity at log(0)

def db20(x: ArrayLike) -> np.ndarray:
    # magnitude of field qty in dB
    return 20.0 * np.log10(np.maximum(np.abs(x), _TINY))

def db10(x: ArrayLike) -> np.ndarray:
    # magnitude of power qty in dB
    return 10.0 * np.log10(np.maximum(np.abs(x), _TINY))

"""
log prefactors:

use db20 for amplitude and db10 for log

for any given wave: P is proportional to A^2
if dB conversion is defined as 10 * log10(A_lin / A_0)
db10(P) is proportional to 10 * log10(A_lin^2 / A_0^2) = 20 * log10(...)

using prefactor of 20 for power and 10 for amplitude allows same
decibel relationship to be used for both values.
"""

def from_db20(db: ArrayLike) -> np.ndarray:
    # field qty from dB
    return 10.0 ** (np.asarray(db) / 20.0)

def np_per_m_to_db_per_inch(alpha: ArrayLike) -> np.ndarray:
    # np/m to db/in
    return np.asarray(alpha) * NP_TO_DB * INCH
