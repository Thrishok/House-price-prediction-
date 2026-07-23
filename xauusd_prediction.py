import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf

sys.path.append("./")
from model import Kronos, KronosTokenizer, KronosPredictor


# ── 1. Fetch XAUUSD (Gold Futures) hourly data ──────────────────────────────
print("Fetching XAUUSD data from Yahoo Finance...")
raw = yf.download("GC=F", period="2y", interval="1h", progress=False)

# Flatten multi-level columns (Price level + Ticker level → just price name)
raw.columns = [col[0].lower() for col in raw.columns]
raw = raw.reset_index()  # brings Datetime into a column
# index name may be 'Datetime' or 'datetime'
ts_col = [c for c in raw.columns if c.lower() == "datetime"][0]
raw = raw.rename(columns={ts_col: "timestamps"})
raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)

# Keep required columns, drop rows with NaN
df = raw[["timestamps", "open", "high", "low", "close", "volume"]].dropna().reset_index(drop=True)
df["amount"] = df["volume"] * df["close"]   # synthetic amount

print(f"Total rows: {len(df)}  |  Date range: {df['timestamps'].iloc[0]} → {df['timestamps'].iloc[-1]}")


# ── 2. Load Kronos model ─────────────────────────────────────────────────────
print("\nLoading Kronos tokenizer and model from HuggingFace...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model     = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)
print(f"Model loaded on device: {predictor.device}")


# ── 3. Prepare context window & prediction horizon ──────────────────────────
lookback = 400   # ~400 hours of history fed to the model
pred_len = 48    # predict next 48 hours

# Use the last (lookback + pred_len) rows so we have ground truth to compare
total_needed = lookback + pred_len
assert len(df) >= total_needed, f"Not enough data: need {total_needed}, got {len(df)}"

start_idx = len(df) - total_needed
segment   = df.iloc[start_idx:].reset_index(drop=True)

x_df        = segment.loc[:lookback-1, ["open", "high", "low", "close", "volume", "amount"]]
x_timestamp = segment.loc[:lookback-1, "timestamps"]
y_timestamp = segment.loc[lookback:lookback+pred_len-1, "timestamps"]

print(f"\nContext window : {x_timestamp.iloc[0]}  →  {x_timestamp.iloc[-1]}  ({lookback} bars)")
print(f"Prediction span: {y_timestamp.iloc[0]}  →  {y_timestamp.iloc[-1]}  ({pred_len} bars)")


# ── 4. Run prediction ────────────────────────────────────────────────────────
print("\nRunning autoregressive inference...")
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=3,   # average 3 samples for smoother output
    verbose=True,
)

print("\nPredicted OHLCV (first 5 rows):")
print(pred_df[["open", "high", "low", "close", "volume"]].head())


# ── 5. Visualise ─────────────────────────────────────────────────────────────
gt_close  = segment.loc[:lookback+pred_len-1, "close"]
gt_ts     = segment.loc[:lookback+pred_len-1, "timestamps"]

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
fig.suptitle("Kronos Forecast — XAUUSD (Gold) Hourly", fontsize=15, fontweight="bold")

# — Close price —
ax = axes[0]
ax.plot(gt_ts.values, gt_close.values, color="steelblue", lw=1.5, label="Ground Truth")
ax.plot(y_timestamp.values, pred_df["close"].values, color="tomato", lw=1.8, ls="--", label="Prediction")
ax.axvline(x=x_timestamp.iloc[-1], color="gray", ls=":", lw=1, label="Forecast start")
ax.set_ylabel("Close Price (USD)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_title("Close Price")

# — High / Low band —
ax = axes[1]
ax.fill_between(y_timestamp.values, pred_df["low"].values, pred_df["high"].values,
                alpha=0.3, color="orange", label="Predicted H/L band")
ax.plot(gt_ts.values, segment.loc[:lookback+pred_len-1, "high"].values,
        color="steelblue", lw=1, alpha=0.6, label="GT High")
ax.plot(gt_ts.values, segment.loc[:lookback+pred_len-1, "low"].values,
        color="steelblue", lw=1, alpha=0.6, ls="--", label="GT Low")
ax.axvline(x=x_timestamp.iloc[-1], color="gray", ls=":", lw=1)
ax.set_ylabel("Price (USD)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_title("High / Low Band")

# — Volume —
ax = axes[2]
gt_vol   = segment.loc[:lookback+pred_len-1, "volume"].values
ax.bar(gt_ts.values, gt_vol, color="steelblue", alpha=0.5, width=0.03, label="GT Volume")
ax.bar(y_timestamp.values, pred_df["volume"].values, color="tomato", alpha=0.6, width=0.03, label="Predicted Volume")
ax.axvline(x=x_timestamp.iloc[-1], color="gray", ls=":", lw=1)
ax.set_ylabel("Volume", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_title("Volume")

plt.tight_layout()
out_path = "./xauusd_kronos_forecast.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {out_path}")
plt.show()


# ── 6. Simple accuracy metrics ───────────────────────────────────────────────
gt_pred_close = segment.loc[lookback:lookback+pred_len-1, "close"].values
pred_close    = pred_df["close"].values

mae  = np.mean(np.abs(gt_pred_close - pred_close))
mape = np.mean(np.abs((gt_pred_close - pred_close) / gt_pred_close)) * 100
rmse = np.sqrt(np.mean((gt_pred_close - pred_close) ** 2))

print("\n── Forecast Metrics (Close Price) ──")
print(f"  MAE  : {mae:.4f} USD")
print(f"  RMSE : {rmse:.4f} USD")
print(f"  MAPE : {mape:.4f} %")
