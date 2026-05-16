# -*- coding: utf-8 -*-
import os
import tempfile
import mplfinance as mpf
from stock_analyzer.config import PRIMARY_BG

def generate_chart(ticker, df):
    save_path = os.path.join(
        tempfile.gettempdir(),
        f"{ticker}_chart.png"
    )

    chart_df = df[
        ["Open", "High", "Low", "Close", "Volume"]
    ].copy()

    chart_df.dropna(inplace=True)

    apds = [
        mpf.make_addplot(
            df["EMA20"],
            color="#00E5FF"
        ),
        mpf.make_addplot(
            df["EMA50"],
            color="#FF9800"
        ),
        mpf.make_addplot(
            df["BB_HIGH"],
            color="gray"
        ),
        mpf.make_addplot(
            df["BB_LOW"],
            color="gray"
        ),
    ]

    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        facecolor=PRIMARY_BG,
        figcolor=PRIMARY_BG,
        edgecolor="gray",
        gridstyle="--"
    )

    mpf.plot(
        chart_df,
        type="candle",
        volume=True,
        style=style,
        addplot=apds,
        figsize=(16, 8),
        tight_layout=True,
        savefig=save_path
    )

    return save_path
