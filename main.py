### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

import numpy as np

nInst = 50
currentPos = np.zeros(nInst)


def getMyPosition(prcSoFar):
    global currentPos
    nInst, nt = prcSoFar.shape

    # Limit to training period only (first 250 days)
    if nt > 250:
        return np.zeros(nInst, dtype=int)

    # Not enough data for return computation
    if nt < 2:
        return np.zeros(nInst, dtype=int)

    # Log return from previous day
    lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])

    # Normalize returns (L2 norm)
    lNorm = np.linalg.norm(lastRet)
    if lNorm > 0:
        lastRet /= lNorm

    # Convert to share positions based on price and return
    rpos = (5000 * lastRet / prcSoFar[:, -1]).astype(int)

    # Update cumulative position
    currentPos = (currentPos + rpos).astype(int)

    return currentPos