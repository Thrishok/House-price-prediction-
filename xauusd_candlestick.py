import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf

sys.path.append("./")
from model import Kronos, KronosTokenizer, KronosPredictor


# ── 1. Fetch XAUUSD data ─────────────────────────────────────────────────────
print("Fetching XAUUSD data...")
raw = yf.download("GC=F", period="2y", interval="1h", progress=False)
raw.columns = [col[0].lower() for col in raw.columns]
raw = raw.reset_index()
ts_col = [c for c in raw.columns if c.lower() == "datetime"][0]
raw = raw.rename(columns={ts_col: "timestamps"})
raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)
df = raw[["timestamps", "open", "high", "low", "close", "volume"]].dropna().reset_index(drop=True)
df["amount"] = df["volume"] * df["close"]
print(f"Rows: {len(df)}  |  {df['timestamps'].iloc[0]} → {df['timestamps'].iloc[-1]}")


# ── 2. Load Kronos ────────────────────────────────────────────────────────────
print("\nLoading Kronos...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model     = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)
print(f"Device: {predictor.device}")


# ── 3. Prepare windows ────────────────────────────────────────────────────────
lookback  = 400
pred_len  = 48
start_idx = len(df) - (lookback + pred_len)
segment   = df.iloc[start_idx:].reset_index(drop=True)

x_df        = segment.loc[:lookback-1, ["open","high","low","close","volume","amount"]]
x_timestamp = segment.loc[:lookback-1, "timestamps"]
y_timestamp = segment.loc[lookback:lookback+pred_len-1, "timestamps"]


# ── 4. Predict ────────────────────────────────────────────────────────────────
print("\nRunning inference...")
pred_df = predictor.predict(
    df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
    pred_len=pred_len, T=1.0, top_p=0.9, sample_count=3, verbose=True,
)


# ── 5. Build combined OHLCV ───────────────────────────────────────────────────
hist_show = 72   # last 72 historical bars to display

hist = segment.iloc[lookback - hist_show: lookback][["timestamps","open","high","low","close","volume"]].copy().reset_index(drop=True)
pred = pred_df[["open","high","low","close","volume"]].copy().reset_index()
pred.columns = ["timestamps","open","high","low","close","volume"]

n_hist = len(hist)
n_pred = len(pred)
combined = pd.concat([hist, pred], ignore_index=True)


# ── 6. Draw candlesticks ──────────────────────────────────────────────────────
def draw_candles(ax, df, start_i, color_up, color_down, width=0.6):
    for i, row in df.iterrows():
        x    = start_i + i
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = color_up if c >= o else color_down
        ax.plot([x, x], [l, h], color=color, lw=1.0, zorder=2)
        body_bot = min(o, c)
        body_h   = max(abs(c - o), (h - l) * 0.01)
        rect = plt.Rectangle((x - width/2, body_bot), width, body_h,
                              color=color, zorder=3)
        ax.add_patch(rect)


fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(20, 10),
    gridspec_kw={"height_ratios": [3, 1]},
    facecolor="#0d1117"
)
fig.suptitle(
    "XAUUSD (Gold Futures) — Kronos Hourly Forecast\n"
    "🔵 Blue = Historical  |  🟡 Gold = Predicted next 48h",
    color="white", fontsize=13, fontweight="bold", y=0.99
)

for ax in (ax1, ax2):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#30363d")

# Historical candles — blue
draw_candles(ax1, hist, 0, color_up="#4fc3f7", color_down="#1565C0")

# Predicted candles — gold
draw_candles(ax1, pred, n_hist, color_up="#FFD700", color_down="#FF8C00")

# Forecast divider
ax1.axvline(x=n_hist - 0.5, color="#ff4444", lw=1.5, ls="--", zorder=5)
ax1.text(n_hist - 0.3, ax1.get_ylim()[0] if ax1.get_ylim()[0] != 0 else combined["low"].min(),
         "  ◀ Forecast start", color="#ff4444", fontsize=8, va="bottom")

# X ticks every 12 bars
tick_pos    = list(range(0, n_hist + n_pred, 12))
tick_labels = [combined.loc[i, "timestamps"].strftime("%m/%d %Hh") for i in tick_pos]
for ax in (ax1, ax2):
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", color="#aaaaaa", fontsize=7)
    ax.set_xlim(-1, n_hist + n_pred)
    ax.grid(True, alpha=0.12, color="white")

ax1.set_ylabel("Price (USD)", color="white", fontsize=11)
ax1.autoscale_view()

# Legend
blue_p = mpatches.Patch(color="#4fc3f7", label="Historical candles")
gold_p = mpatches.Patch(color="#FFD700", label="Predicted candles (Kronos)")
ax1.legend(handles=[blue_p, gold_p], loc="upper left",
           facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=10)

# Volume bars
for i, row in combined.iterrows():
    color = "#4fc3f7" if i < n_hist else "#FFD700"
    ax2.bar(i, row["volume"], color=color, alpha=0.75, width=0.8)

ax2.axvline(x=n_hist - 0.5, color="#ff4444", lw=1.5, ls="--")
ax2.set_ylabel("Volume", color="white", fontsize=9)

plt.tight_layout()
out = "./xauusd_candlestick_forecast.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
print(f"\nChart saved → {out}")
plt.show()

# ── Metrics ───────────────────────────────────────────────────────────────────
gt_close   = segment.loc[lookback:lookback+pred_len-1, "close"].values
pred_close = pred_df["close"].values
mae  = np.mean(np.abs(gt_close - pred_close))
mape = np.mean(np.abs((gt_close - pred_close) / gt_close)) * 100
rmse = np.sqrt(np.mean((gt_close - pred_close) ** 2))
print(f"\n── Metrics ──\n  MAE: {mae:.2f} USD  |  RMSE: {rmse:.2f} USD  |  MAPE: {mape:.4f}%")
