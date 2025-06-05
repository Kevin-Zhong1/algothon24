import numpy as np

nInst = 50
currentPos = np.zeros(nInst)
entryPrice = np.zeros(nInst)  # track avg entry price per stock
lastTradeDay = -3
lastSignal = np.zeros(nInst)
a = 0
b = 0

def getMyPosition(prcSoFar):
    global currentPos, entryPrice, lastTradeDay, lastSignal, a, b
    nInst, nt = prcSoFar.shape

    if nt > 500 or nt < 100:
        return currentPos

    # Throttle trading frequency
    if (nt - lastTradeDay) < 3:
        return currentPos

    # --- PARAMETERS ---
    ma_window = 100
    k = 2.0  # breakout band std multiplier
    stop_loss_pct = 0.03  # 3% stop-loss threshold
    mean_reversion_band = 0.5  # exit band around MA
    max_dollar_change = 1500
    min_signal_strength = 0.02

    price = prcSoFar[:, -1]
    ma = np.mean(prcSoFar[:, -ma_window:], axis=1)
    std = np.std(prcSoFar[:, -ma_window:], axis=1) + 1e-6
    slope = ma - np.mean(prcSoFar[:, -ma_window-5:-5], axis=1)

    zscore = (price - ma) / std

    # Define breakout signals with momentum confirmation
    signal = np.zeros(nInst)
    signal[zscore > k] = 1
    signal[zscore < -k] = -1
    signal *= np.sign(slope)


    # Filter weak signals
    strong_signal = (np.abs(zscore) > min_signal_strength) * signal

    # Signal persistence filter
    persistent_signal = strong_signal * (strong_signal == lastSignal)

    # Non-linear sizing
    strength = (zscore ** 2) * persistent_signal / std

    norm = np.linalg.norm(strength)
    if norm > 0:
        strength /= norm

    target_dollar = 10000 * strength
    proposedPos = (target_dollar / price).astype(int)

    # --- Stop Loss & Mean-Reversion Exit ---
    a, b = 0, 0
    # For instruments with positions, check stop-loss
    for i in range(nInst):
        if currentPos[i] > 0:
            # Long position stop-loss
            if price[i] < entryPrice[i] * (1 - stop_loss_pct):
                proposedPos[i] = 0  # exit position
                a += 1
        elif currentPos[i] < 0:
            # Short position stop-loss
            if price[i] > entryPrice[i] * (1 + stop_loss_pct):
                proposedPos[i] = 0  # exit position
                b += 1

        # Mean-reversion exit band
        if currentPos[i] != 0:
            if abs(zscore[i]) < mean_reversion_band:
                proposedPos[i] = 0

    # Throttle position change
    curr_dollar = currentPos * price
    prop_dollar = proposedPos * price
    delta_dollar = np.clip(prop_dollar - curr_dollar,
                           -max_dollar_change, max_dollar_change)
    newPos = currentPos + (delta_dollar / price).astype(int)

    # Clean small positions to zero
    newPos[np.abs(zscore) < k] = 0

    # Update entry prices only when position changes
    for i in range(nInst):
        if newPos[i] != currentPos[i]:
            if newPos[i] != 0:
                entryPrice[i] = price[i]  # update entry price
            else:
                entryPrice[i] = 0  # reset if exited

    lastSignal = signal
    currentPos = newPos
    lastTradeDay = nt
    print(a, b)
    return currentPos
