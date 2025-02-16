import numpy as np
from enum import Enum


class RobustMethod(Enum):
    NONE = 0
    HUBER = 1
    CAUCHY = 2
    TUKEY = 3

def huber_loss(e, delta):
    if np.abs(e) <= delta:
        return 0.5 * e**2
    else:
        return delta * (np.abs(e) - 0.5 * delta)

def cauchy_loss(e, c):
    return (c**2 / 2) * np.log(1 + (e**2 / c**2))

def tukey_loss(e, c):
    if np.abs(e) <= c:
        return (c**2 / 6) * (1 - (1 - (e / c) ** 2) ** 3)
    else:
        return (c**2 / 6)

def robust_weight(e, method=RobustMethod.NONE, param=1.0):
    if method == RobustMethod.HUBER:
        return 1 if np.abs(e) <= param else param / np.abs(e)
    elif method == RobustMethod.CAUCHY:
        return 1 / (1 + (e / param) ** 2)
    elif method == RobustMethod.TUKEY:
        return (1 - (e / param) ** 2) ** 2 if np.abs(e) <= param else 0
    else:
        # no robustness
        return 1