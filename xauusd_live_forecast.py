import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
from datetime import timedelta

sys.path.append("./")
from model import Kronos, KronosTokenizer, KronosPredictor


# ── 1. Fetch latest XAUUSD data up to NOW ────────────────────────────────────
print("Fetching latest XAUUSD 1H data...")
raw = yf.download("GC=F", period="60d", interval="1h", progress=False)
raw.columns = [col[0].lower() for col in raw.columns]
raw = raw.reset_index()
ts_col = [c for c in raw.columns if c.lower() == "datetime"][0]
raw = raw.rename(columns={ts_col: "timestamps"})
raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)
df = raw[["timestamps","open","high","low","close","volume"]].dropna().reset_index(drop=True)
df["amount"] = df["volume"] * df["close"]

latest = df["timestamps"].iloc[-1]
print(f"Latest candle: {latest}  |  Total rows: {len(df)}")


# ── 2. Load Kronos ────────────────────────────────────────────────────────────
print("\nLoading Kronos...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model     = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)
print(f"Device: {predictor.device}")


# ── 3. Use ALL available data as context, predict next 48 candles ─────────────
lookback = min(400, len(df))   # use up to 400 bars of history
pred_len = 48                  # predict next 48 hours into the future

x_df        = df.iloc[-lookback:][["open","high","low","close","volume","amount"]].reset_index(drop=True)
x_timestamp = df.iloc[-lookback:]["timestamps"].reset_index(drop=True)

# Generate future timestamps (hourly, skipping nothing — gold trades ~23h/day)
last_ts   = df["timestamps"].iloc[-1]
y_timestamp = pd.Series([last_ts + timedelta(hours=i+1) for i in range(pred_len)])

print(f"\nContext : {x_timestamp.iloc[0]}  →  {x_timestamp.iloc[-1]}  ({lookback} bars)")
print(f"Forecast: {y_timestamp.iloc[0]}  →  {y_timestamp.iloc[-1]}  ({pred_len} bars)")


# ── 4. Predict ────────────────────────────────────────────────────────────────
print("\nRunning inference...")
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=5,   # 5 samples averaged = smoother future path
    verbose=True,
)

print("\nPredicted candles (first 5):")
print(pred_df[["open","high","low","close","volume"]].head())


# ── 5. Candlestick chart ──────────────────────────────────────────────────────
hist_show = 72   # show last 72 historical candles for context

hist = df.iloc[-hist_show:][["timestamps","open","high","low","close","volume"]].reset_index(drop=True)
pred = pred_df[["open","high","low","close","volume"]].copy().reset_index()
pred.columns = ["timestamps","open","high","low","close","volume"]

n_hist = len(hist)
n_pred = len(pred)
combined = pd.concat([hist, pred], ignore_index=True)


def draw_candles(ax, df, offset, color_up, color_down, width=0.6):
    for i, row in df.iterrows():
        x = offset + i
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = color_up if c >= o else color_down
        ax.plot([x, x], [l, h], color=color, lw=1.0, zorder=2)
        body_bot = min(o, c)
        body_h   = max(abs(c - o), (h - l) * 0.015)
        rect = plt.Rectangle((x - width/2, body_bot), width, body_h,
                              color=color, zorder=3)
        ax.add_patch(rect)


fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(22, 11),
    gridspec_kw={"height_ratios": [3, 1]},
    facecolor="#0d1117"
)

fig.suptitle(
    f"XAUUSD (Gold) — Live Edge Forecast  |  1H Candles\n"
    f"History up to: {latest.strftime('%Y-%m-%d %H:%M')} UTC  →  "
    f"Predicting: {y_timestamp.iloc[0].strftime('%H:%M')} – {y_timestamp.iloc[-1].strftime('%Y-%m-%d %H:%M')} UTC",
    color="white", fontsize=12, fontweight="bold", y=0.99
)

for ax in (ax1, ax2):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#30363d")

# Draw candles
draw_candles(ax1, hist, 0,      color_up="#4fc3f7", color_down="#1976D2")
draw_candles(ax1, pred, n_hist, color_up="#FFD700", color_down="#FF8C00")

# Forecast boundary line
ax1.axvline(x=n_hist - 0.5, color="#ff4444", lw=2.0, ls="--", zorder=5)
ymin = combined["low"].min()
ymax = combined["high"].max()
ax1.text(n_hist + 0.3, ymin + (ymax - ymin) * 0.02,
         "▶ FUTURE", color="#ff4444", fontsize=9, fontweight="bold")

# Shaded future region
ax1.axvspan(n_hist - 0.5, n_hist + n_pred, alpha=0.06, color="#FFD700")

# X ticks every 8 bars
tick_pos    = list(range(0, n_hist + n_pred, 8))
tick_labels = [combined.loc[i, "timestamps"].strftime("%m/%d %Hh") for i in tick_pos if i < len(combined)]
ax1.set_xticks(tick_pos[:len(tick_labels)])
ax1.set_xticklabels(tick_labels, rotation=45, ha="right", color="#aaaaaa", fontsize=7)
ax1.set_xlim(-1, n_hist + n_pred)
ax1.set_ylim(ymin * 0.999, ymax * 1.001)
ax1.set_ylabel("Price (USD)", color="white", fontsize=11)
ax1.grid(True, alpha=0.1, color="white")
ax1.autoscale_view()

# Legend
blue_p = mpatches.Patch(color="#4fc3f7", label=f"Historical ({hist_show}h)")
gold_p = mpatches.Patch(color="#FFD700", label=f"Kronos Forecast (next {pred_len}h)")
ax1.legend(handles=[blue_p, gold_p], loc="upper left",
           facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=10)

# Volume
for i, row in combined.iterrows():
    color = "#4fc3f7" if i < n_hist else "#FFD700"
    ax2.bar(i, row["volume"], color=color, alpha=0.75, width=0.8)

ax2.axvline(x=n_hist - 0.5, color="#ff4444", lw=2.0, ls="--")
ax2.axvspan(n_hist - 0.5, n_hist + n_pred, alpha=0.06, color="#FFD700")
ax2.set_xticks(tick_pos[:len(tick_labels)])
ax2.set_xticklabels(tick_labels, rotation=45, ha="right", color="#aaaaaa", fontsize=7)
ax2.set_xlim(-1, n_hist + n_pred)
ax2.set_ylabel("Volume", color="white", fontsize=9)
ax2.grid(True, alpha=0.1, color="white")

plt.tight_layout()
out = "./xauusd_live_forecast.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
print(f"\nChart saved → {out}")
plt.show()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Forecast Summary ──")
print(f"  Current close : ${df['close'].iloc[-1]:.2f}")
print(f"  Pred close +1h: ${pred_df['close'].iloc[0]:.2f}")
print(f"  Pred close +24h: ${pred_df['close'].iloc[23]:.2f}")
print(f"  Pred close +48h: ${pred_df['close'].iloc[-1]:.2f}")
print(f"  Pred high (48h): ${pred_df['high'].max():.2f}")
print(f"  Pred low  (48h): ${pred_df['low'].min():.2f}")
