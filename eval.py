#!/usr/bin/env python

import numpy as np
import pandas as pd
from main import getMyPosition as getPosition
import matplotlib.pyplot as plt

nInst = 0
nt = 0
commRate = 0.0010
dlrPosLimit = 10000


def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep="\s+", header=None, index_col=None)
    (nt, nInst) = df.shape
    return (df.values).T


pricesFile = "./prices.txt"
prcAll = loadPrices(pricesFile)
print("Loaded %d instruments for %d days" % (nInst, nt))


def calcPL(prcHist):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    value = 0
    todayPLL = []
    (_, nt) = prcHist.shape

    for t in range(1, 501):  #* change for backtesting
        prcHistSoFar = prcHist[:, :t]
        newPosOrig = getPosition(prcHistSoFar)
        curPrices = prcHistSoFar[:, -1]
        posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
        newPos = np.clip(newPosOrig, -posLimits, posLimits)
        deltaPos = newPos - curPos
        dvolumes = curPrices * np.abs(deltaPos)
        dvolume = np.sum(dvolumes)
        totDVolume += dvolume
        comm = dvolume * commRate
        cash -= curPrices.dot(deltaPos) + comm
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        todayPLL.append(todayPL)
        value = cash + posValue
        ret = 0.0
        if totDVolume > 0:
            ret = value / totDVolume
        print(
            "Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf"
            % (t, value, todayPL, totDVolume, ret)
        )

    pll = np.array(todayPLL)
    (plmu, plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if plstd > 0:
        annSharpe = np.sqrt(250) * plmu / plstd
    score = plmu - 0.1 * plstd

    # === Plotting section ===
    plt.figure(figsize=(12, 6))
    plt.plot(todayPLL, label="Daily P&L", color="royalblue")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)

    stats_text = (
        f"mean(PL): {plmu:.1f}\n"
        f"StdDev(PL): {plstd:.2f}\n"
        f"Score: {score:.2f}\n"
        f"Sharpe: {annSharpe:.2f}\n"
        f"Return: {ret:.5f}\n"
        f"Total Volume: {totDVolume:.0f}"
    )
    plt.text(
        0.01,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.7),
    )

    plt.title("Daily Profit & Loss")
    plt.xlabel("Trading Day Index")
    plt.ylabel("P&L ($)")
    plt.legend()
    plt.grid(True)

    plt.savefig("pnl_plot.png")
    print("Plot saved as pnl_plot.png")
    # ========================

    return (plmu, ret, plstd, annSharpe, totDVolume)


(meanpl, ret, plstd, sharpe, dvol) = calcPL(prcAll)
score = meanpl - 0.1 * plstd
print("=====")
print("mean(PL): %.1lf" % meanpl)
print("return: %.5lf" % ret)
print("StdDev(PL): %.2lf" % plstd)
print("annSharpe(PL): %.2lf " % sharpe)
print("totDvolume: %.0lf " % dvol)
print("Score: %.2lf" % score)