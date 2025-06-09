import numpy as np

nInst = 50
currentPos = np.zeros(nInst)
lastTradeDay = -1000
holdingWindow = 10
lastSignal = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global currentPos, lastTradeDay, lastSignal
    nInst, nt = prcSoFar.shape

    # Trade only in a defined window
    if nt < 100 or nt > 500:
        return currentPos

    # Trade only every 15 days
    if (nt - lastTradeDay) < holdingWindow:
        return currentPos

    # --- Parameters ---
    ma_window = 50
    zscore_threshold = 1
    top_k = 15
    max_dollar_position = 10000

    price = prcSoFar[:, -1]
    ma = np.mean(prcSoFar[:, -ma_window:], axis=1)
    std = np.std(prcSoFar[:, -ma_window:], axis=1) + 1e-6
    slope = ma - np.mean(prcSoFar[:, -ma_window-5:-5], axis=1)

    zscore = (price - ma) / std
    momentum = np.sign(slope)

    # Breakout signal with trend confirmation
    raw_signal = np.zeros(nInst)
    raw_signal[(zscore >= zscore_threshold) & (momentum > 0)] = 1
    raw_signal[(zscore <= -zscore_threshold) & (momentum < 0)] = -1

    # Select top-k strongest signals
    ranked_indices = np.argsort(-np.abs(zscore * raw_signal))
    signal = np.zeros(nInst)
    count = 0
    for idx in ranked_indices:
        if raw_signal[idx] != 0 and count < top_k:
            signal[idx] = raw_signal[idx]
            count += 1

    # Convert signal to position
    target_dollar = max_dollar_position * signal
    proposedPos = (target_dollar / price).astype(int)

    currentPos = proposedPos
    lastSignal = signal
    lastTradeDay = nt

    return currentPos
