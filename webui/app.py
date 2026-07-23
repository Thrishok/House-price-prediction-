import os, sys, io, base64, warnings, datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# ── globals ──────────────────────────────────────────────────────────────────
_tokenizer = None
_model     = None
_predictor = None

MODELS = {
    'kronos-mini':  {'id': 'NeoQuasar/Kronos-mini',  'tok': 'NeoQuasar/Kronos-Tokenizer-2k',   'ctx': 2048, 'params': '4.1M'},
    'kronos-small': {'id': 'NeoQuasar/Kronos-small', 'tok': 'NeoQuasar/Kronos-Tokenizer-base', 'ctx': 512,  'params': '24.7M'},
    'kronos-base':  {'id': 'NeoQuasar/Kronos-base',  'tok': 'NeoQuasar/Kronos-Tokenizer-base', 'ctx': 512,  'params': '102.3M'},
}

TIMEFRAME_MAP = {
    '1min': '1', '5min': '5', '15min': '15', '30min': '30',
    '60min': '60', 'daily': 'daily', 'weekly': 'weekly', 'monthly': 'monthly',
}

# ── data fetching ─────────────────────────────────────────────────────────────

def _rename(df):
    mapping = {
        '日期': 'timestamps', '时间': 'timestamps',
        '开盘': 'open', '开盘价': 'open',
        '收盘': 'close', '收盘价': 'close',
        '最高': 'high', '最高价': 'high',
        '最低': 'low', '最低价': 'low',
        '成交量': 'volume', '成交额': 'amount',
    }
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    return df


def fetch_cn_stock(symbol, period, adjust='qfq'):
    if period in ('daily', 'weekly', 'monthly'):
        df = ak.stock_zh_a_hist(symbol=symbol, period=period, adjust=adjust)
    else:
        df = ak.stock_zh_a_hist_min_em(symbol=symbol, period=period.replace('min', ''), adjust=adjust)
    df = _rename(df)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    return df.sort_values('timestamps').reset_index(drop=True)


def fetch_hk_stock(symbol, period):
    df = ak.stock_hk_hist(symbol=symbol, period='daily' if period not in ('daily',) else period, adjust='qfq')
    df = _rename(df)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    return df.sort_values('timestamps').reset_index(drop=True)


def fetch_us_stock(symbol, period):
    df = ak.stock_us_hist(symbol=symbol, period='daily', adjust='qfq')
    df = _rename(df)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    return df.sort_values('timestamps').reset_index(drop=True)


def fetch_crypto(symbol, period):
    # symbol like BTC, ETH; period like daily/60min
    coin_map = {'BTC': 'bitcoin', 'ETH': 'ethereum', 'BNB': 'binancecoin',
                'SOL': 'solana', 'XRP': 'ripple', 'ADA': 'cardano', 'DOGE': 'dogecoin'}
    coin_id = coin_map.get(symbol.upper(), symbol.lower())
    df = ak.crypto_hist(symbol=coin_id, period='daily', start_date='20200101',
                        end_date=datetime.datetime.now().strftime('%Y%m%d'))
    df = _rename(df)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    return df.sort_values('timestamps').reset_index(drop=True)


def fetch_forex(symbol, period):
    # symbol like EURUSD, GBPUSD
    df = ak.forex_hist(symbol=symbol)
    df = _rename(df)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    return df.sort_values('timestamps').reset_index(drop=True)


def fetch_index(symbol, period):
    df = ak.stock_zh_index_hist(symbol=symbol, period='daily')
    df = _rename(df)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    return df.sort_values('timestamps').reset_index(drop=True)


def fetch_commodity(symbol, period):
    # Gold, Silver, Oil via akshare futures
    futures_map = {'GOLD': 'AU0', 'SILVER': 'AG0', 'OIL': 'SC0', 'COPPER': 'CU0'}
    code = futures_map.get(symbol.upper(), symbol)
    df = ak.futures_zh_daily_sina(symbol=code)
    df = _rename(df)
    if 'timestamps' not in df.columns and '日期' in df.columns:
        df = df.rename(columns={'日期': 'timestamps'})
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    return df.sort_values('timestamps').reset_index(drop=True)


def fetch_yahoo(symbol, period):
    """Fetch data from Yahoo Finance via yfinance or direct chart API fallback."""
    if not YFINANCE_AVAILABLE:
        raise RuntimeError('yfinance not installed')

    mapping = {
        '1min': '1m', '5min': '5m', '15min': '15m', '30min': '30m', '60min': '60m',
        'daily': '1d', 'weekly': '1wk', 'monthly': '1mo'
    }
    interval = mapping.get(period, period)

    if interval.endswith('m'):
        yf_period = '7d' if interval == '1m' else '60d'
    else:
        yf_period = 'max'

    df = None
    try:
        df = yf.download(tickers=symbol, period=yf_period, interval=interval,
                         progress=False, auto_adjust=False, threads=False)
    except Exception as exc:
        # allow fallback to direct API if yfinance itself fails
        df = None
        fallback_error = f'yfinance download failed: {exc}'

    if df is None or df.empty:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=yf_period, interval=interval, auto_adjust=False)
        except Exception as exc:
            fallback_error = f'yfinance history failed: {exc}'
            df = None

    if df is None or df.empty:
        # try direct Yahoo chart API as a more robust fallback
        try:
            import requests
            import urllib.parse

            if interval.endswith('m'):
                days = 7
            elif interval == '1d':
                days = 365 * 2
            elif interval == '1wk':
                days = 365 * 5
            else:
                days = 365 * 10

            start = int((datetime.datetime.now() - datetime.timedelta(days=days)).timestamp())
            end = int(datetime.datetime.now().timestamp())
            url = (
                f'https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}'
                f'?period1={start}&period2={end}&interval={interval}&includePrePost=false&events=history'
            )
            resp = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
            resp.raise_for_status()
            payload = resp.json()
            result = payload.get('chart', {}).get('result')
            if not result:
                raise RuntimeError(payload.get('chart', {}).get('error') or 'No chart result data')

            result = result[0]
            timestamps = result.get('timestamp') or []
            quote = result.get('indicators', {}).get('quote', [{}])[0]
            if not timestamps or not quote or not quote.get('open'):
                raise RuntimeError('Yahoo chart API returned no usable quote data')

            df = pd.DataFrame({
                'timestamps': pd.to_datetime(timestamps, unit='s'),
                'open': quote.get('open', []),
                'high': quote.get('high', []),
                'low': quote.get('low', []),
                'close': quote.get('close', []),
                'volume': quote.get('volume', []),
            })
        except Exception as exc:
            raise RuntimeError(f'yfinance returned no data and Yahoo chart API fallback failed: {exc}')

    if df is None or df.empty:
        raise RuntimeError(f'yfinance returned no data for {symbol} with interval {interval}; {fallback_error if "fallback_error" in locals() else "no additional info"}')

    df = df.reset_index(drop=True)
    df = df.rename(columns={
        'Date': 'timestamps', 'Datetime': 'timestamps',
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
        'Adj Close': 'adj_close', 'Volume': 'volume'
    })
    if 'timestamps' in df.columns:
        df['timestamps'] = pd.to_datetime(df['timestamps'])

    missing = [c for c in ['open', 'high', 'low', 'close'] if c not in df.columns]
    if missing:
        raise RuntimeError(f'Yahoo data missing required OHLC columns: {missing} from {list(df.columns)}')

    return df.sort_values('timestamps').reset_index(drop=True)


def get_market_data(market, symbol, period, lookback_bars):
    """Unified data fetcher. Returns a clean DataFrame."""
    try:
        # support multiple market backends including yfinance
        if market == 'yahoo':
            df = fetch_yahoo(symbol, period)
        elif market == 'cn_stock':
            df = fetch_cn_stock(symbol, period)
        elif market == 'hk_stock':
            df = fetch_hk_stock(symbol, period)
        elif market == 'us_stock':
            df = fetch_us_stock(symbol, period)
        elif market == 'crypto':
            df = fetch_crypto(symbol, period)
        elif market == 'forex':
            df = fetch_forex(symbol, period)
        elif market == 'cn_index':
            df = fetch_index(symbol, period)
        elif market == 'commodity':
            df = fetch_commodity(symbol, period)
        else:
            return None, f"Unknown market: {market}"

        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None, f"Missing columns: {missing}"

        for c in required:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        else:
            df['volume'] = 0.0
        if 'amount' not in df.columns:
            df['amount'] = 0.0

        df = df.dropna(subset=required).reset_index(drop=True)

        if len(df) < 50:
            return None, f"Not enough data: only {len(df)} bars"

        return df, None
    except Exception as e:
        return None, str(e)


# ── chart generation ──────────────────────────────────────────────────────────

def _candlestick(ax, df, color_up='#26a69a', color_down='#ef5350', alpha=1.0):
    """Draw candlesticks on ax using numeric x positions."""
    for i, (_, row) in enumerate(df.iterrows()):
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        color = color_up if c >= o else color_down
        ax.plot([i, i], [l, h], color=color, linewidth=0.8, alpha=alpha)
        height = abs(c - o) or (h - l) * 0.01
        rect = Rectangle((i - 0.4, min(o, c)), 0.8, height,
                          facecolor=color, edgecolor=color, alpha=alpha)
        ax.add_patch(rect)


def make_chart(hist_df, pred_df, symbol, market, period, pred_len,
               show_volume=True, show_ma=True, show_bollinger=True,
               chart_style='dark', ma_periods=(20, 50)):
    style_bg   = '#131722' if chart_style == 'dark' else '#ffffff'
    style_fg   = '#d1d4dc' if chart_style == 'dark' else '#333333'
    style_grid = '#1e222d' if chart_style == 'dark' else '#e0e0e0'
    style_pred_bg = '#1a2332' if chart_style == 'dark' else '#f0f8ff'

    rows = 2 if show_volume else 1
    height_ratios = [3, 1] if show_volume else [1]

    fig, axes = plt.subplots(rows, 1, figsize=(16, 9 if show_volume else 6),
                             gridspec_kw={'height_ratios': height_ratios},
                             facecolor=style_bg)
    ax = axes[0] if show_volume else axes
    ax_vol = axes[1] if show_volume else None

    ax.set_facecolor(style_bg)
    if ax_vol:
        ax_vol.set_facecolor(style_bg)

    n_hist = len(hist_df)
    n_pred = len(pred_df)
    total  = n_hist + n_pred

    # shade prediction zone
    ax.axvspan(n_hist - 0.5, total - 0.5, alpha=0.08,
               color='#2196f3' if chart_style == 'dark' else '#bbdefb')

    # historical candles
    _candlestick(ax, hist_df)

    # prediction candles (offset x by n_hist)
    pred_df_shifted = pred_df.copy()
    for i, (idx, row) in enumerate(pred_df.iterrows()):
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        xi = n_hist + i
        color = '#66bb6a' if c >= o else '#ff7043'
        ax.plot([xi, xi], [l, h], color=color, linewidth=0.8, alpha=0.9)
        height = abs(c - o) or (h - l) * 0.01
        rect = Rectangle((xi - 0.4, min(o, c)), 0.8, height,
                          facecolor=color, edgecolor=color, alpha=0.9)
        ax.add_patch(rect)

    # moving averages on history
    if show_ma:
        close = hist_df['close'].values
        for ma_p, ma_color in zip(ma_periods, ['#ff9800', '#2196f3', '#9c27b0']):
            if len(close) >= ma_p:
                ma = pd.Series(close).rolling(ma_p).mean().values
                ax.plot(range(n_hist), ma, color=ma_color, linewidth=1.2,
                        label=f'MA{ma_p}', alpha=0.85)

    # Bollinger Bands
    if show_bollinger and len(hist_df) >= 20:
        close = hist_df['close'].values
        mid = pd.Series(close).rolling(20).mean().values
        std = pd.Series(close).rolling(20).std().values
        upper, lower = mid + 2 * std, mid - 2 * std
        x = range(n_hist)
        ax.plot(x, upper, color='#7e57c2', linewidth=0.8, linestyle='--', alpha=0.6, label='BB Upper')
        ax.plot(x, lower, color='#7e57c2', linewidth=0.8, linestyle='--', alpha=0.6, label='BB Lower')
        ax.fill_between(x, lower, upper, alpha=0.04, color='#7e57c2')

    # divider line
    ax.axvline(x=n_hist - 0.5, color='#ffd700', linewidth=1.5, linestyle='--', alpha=0.8)
    ax.text(n_hist, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1,
            ' Forecast →', color='#ffd700', fontsize=9, va='top', alpha=0.9)

    # x-axis ticks
    all_ts = list(hist_df['timestamps']) + list(pred_df.index if hasattr(pred_df.index, '__iter__') else [])
    tick_step = max(1, total // 12)
    tick_pos   = list(range(0, total, tick_step))
    tick_labels = []
    for tp in tick_pos:
        if tp < len(hist_df):
            tick_labels.append(hist_df['timestamps'].iloc[tp].strftime('%m/%d'))
        else:
            pi = tp - n_hist
            if pi < len(pred_df) and 'timestamps' in pred_df.columns:
                tick_labels.append(pred_df['timestamps'].iloc[pi].strftime('%m/%d'))
            else:
                tick_labels.append(f'+{pi}')

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, color=style_fg, fontsize=8, rotation=30)
    ax.set_xlim(-1, total + 1)
    ax.tick_params(colors=style_fg)
    ax.yaxis.tick_right()
    ax.yaxis.set_tick_params(labelcolor=style_fg, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(style_grid)
    ax.grid(True, color=style_grid, linewidth=0.5, alpha=0.6)

    if show_ma or show_bollinger:
        legend = ax.legend(loc='upper left', fontsize=8, framealpha=0.3,
                           labelcolor=style_fg, facecolor=style_bg)

    # title
    last_close = hist_df['close'].iloc[-1]
    pred_close = pred_df['close'].iloc[-1]
    chg = (pred_close / last_close - 1) * 100
    chg_str = f'+{chg:.2f}%' if chg >= 0 else f'{chg:.2f}%'
    chg_color = '#26a69a' if chg >= 0 else '#ef5350'
    ax.set_title(
        f'{symbol.upper()}  ·  {period}  ·  Last: {last_close:.4f}  '
        f'Forecast end: {pred_close:.4f}  ({chg_str})',
        color=style_fg, fontsize=11, pad=10,
        fontweight='bold'
    )

    # volume
    if show_volume and ax_vol is not None:
        for i, (_, row) in enumerate(hist_df.iterrows()):
            color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
            ax_vol.bar(i, row.get('volume', 0), color=color, alpha=0.6, width=0.8)
        for i, (_, row) in enumerate(pred_df.iterrows()):
            ax_vol.bar(n_hist + i, row.get('volume', 0), color='#2196f3', alpha=0.5, width=0.8)
        ax_vol.set_facecolor(style_bg)
        ax_vol.set_xlim(-1, total + 1)
        ax_vol.set_xticks([])
        ax_vol.tick_params(colors=style_fg, labelsize=8)
        ax_vol.yaxis.tick_right()
        ax_vol.yaxis.set_tick_params(labelcolor=style_fg, labelsize=8)
        for spine in ax_vol.spines.values():
            spine.set_edgecolor(style_grid)
        ax_vol.grid(True, color=style_grid, linewidth=0.5, alpha=0.4)
        ax_vol.set_ylabel('Volume', color=style_fg, fontsize=8)
        ax_vol.axvline(x=n_hist - 0.5, color='#ffd700', linewidth=1.5, linestyle='--', alpha=0.8)

    fig.text(0.01, 0.01, 'Powered by Kronos · For research purposes only · Not financial advice',
             color=style_fg, fontsize=7, alpha=0.5)

    plt.tight_layout(pad=1.5)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=style_bg)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ── routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    # pass available models to template so dropdown shows without JS
    models = {}
    for k, v in MODELS.items():
        models[k] = {'name': k, 'params': v.get('params', ''), 'description': f"Model id: {v.get('id','')}"}
    return render_template('index.html', models=models)


@app.route('/api/status')
def status():
    return jsonify({
        'model_available': MODEL_AVAILABLE,
        'akshare_available': AKSHARE_AVAILABLE,
        'model_loaded': _predictor is not None,
        'models': {k: {'params': v['params'], 'ctx': v['ctx']} for k, v in MODELS.items()},
    })


@app.route('/api/available-models')
def available_models():
    # Return richer metadata for front-end
    models = {}
    for k, v in MODELS.items():
        models[k] = {
            'name': k,
            'params': v.get('params', ''),
            'description': f"Model id: {v.get('id', '')}",
        }
    return jsonify({'model_available': MODEL_AVAILABLE, 'models': models})


@app.route('/api/model-status')
def model_status():
    if _predictor is None:
        return jsonify({'loaded': False, 'available': MODEL_AVAILABLE})
    # try to return basic info
    try:
        cur = {
            'name': getattr(_predictor, 'name', 'kronos'),
            'device': getattr(_predictor, 'device', 'cpu')
        }
    except Exception:
        cur = {'name': 'kronos', 'device': 'cpu'}
    return jsonify({'loaded': True, 'available': MODEL_AVAILABLE, 'current_model': cur})


@app.route('/api/load-model', methods=['POST'])
def load_model():
    global _tokenizer, _model, _predictor
    if not MODEL_AVAILABLE:
        return jsonify({'error': 'Kronos library not installed'}), 400

    data      = request.get_json()
    # accept either 'model' or 'model_key' from frontend
    model_key = data.get('model') or data.get('model_key') or 'kronos-small'
    device    = data.get('device', 'cpu')

    if model_key not in MODELS:
        return jsonify({'error': f'Unknown model: {model_key}'}), 400

    cfg = MODELS[model_key]
    try:
        _tokenizer = KronosTokenizer.from_pretrained(cfg['tok'])
        _model     = Kronos.from_pretrained(cfg['id'])
        _predictor = KronosPredictor(_model, _tokenizer, device=device, max_context=cfg['ctx'])
        return jsonify({'success': True, 'message': f'{model_key} loaded on {device}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/data-files')
def data_files():
    # list CSV files in common data folders
    roots = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'finetune_csv', 'data'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results'),
    ]
    files = []
    for r in roots:
        if os.path.isdir(r):
            for fn in os.listdir(r):
                if fn.lower().endswith(('.csv', '.json', '.parquet')):
                    fp = os.path.join(r, fn)
                    try:
                        size = os.path.getsize(fp)
                        files.append({'name': fn, 'path': os.path.relpath(fp, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'size': f"{size} bytes"})
                    except Exception:
                        files.append({'name': fn, 'path': os.path.relpath(fp, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'size': 'unknown'})
    return jsonify(files)


@app.route('/api/load-data', methods=['POST'])
def load_data():
    d = request.get_json()
    file_path = d.get('file_path')
    if not file_path:
        return jsonify({'error': 'file_path required'}), 400

    # resolve relative path inside project
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_path = os.path.abspath(os.path.join(base, file_path))
    if not abs_path.startswith(base) or not os.path.exists(abs_path):
        return jsonify({'error': 'file not found or invalid path'}), 400

    try:
        if abs_path.lower().endswith('.csv'):
            df = pd.read_csv(abs_path)
        elif abs_path.lower().endswith('.parquet'):
            df = pd.read_parquet(abs_path)
        elif abs_path.lower().endswith('.json'):
            df = pd.read_json(abs_path)
        else:
            return jsonify({'error': 'unsupported file type'}), 400

        df = _rename(df)
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])

        info = {
            'rows': len(df),
            'columns': list(df.columns),
            'start_date': str(df['timestamps'].iloc[0]) if 'timestamps' in df.columns and len(df) else '',
            'end_date': str(df['timestamps'].iloc[-1]) if 'timestamps' in df.columns and len(df) else '',
            'price_range': {'min': float(df['close'].min()) if 'close' in df.columns else 0.0, 'max': float(df['close'].max()) if 'close' in df.columns else 0.0},
            'timeframe': d.get('timeframe', 'unknown')
        }

        return jsonify({'success': True, 'message': f'Loaded {os.path.basename(abs_path)}', 'data_info': info})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/preview-data', methods=['POST'])
def preview_data():
    """Return basic info about the instrument before prediction."""
    if not AKSHARE_AVAILABLE:
        return jsonify({'error': 'akshare not installed'}), 400

    d = request.get_json()
    df, err = get_market_data(d['market'], d['symbol'], d['period'], 600)
    if err:
        return jsonify({'error': err}), 400

    return jsonify({
        'rows': len(df),
        'start': str(df['timestamps'].iloc[0]),
        'end':   str(df['timestamps'].iloc[-1]),
        'last_close': float(df['close'].iloc[-1]),
        'last_open':  float(df['open'].iloc[-1]),
        'last_high':  float(df['high'].iloc[-1]),
        'last_low':   float(df['low'].iloc[-1]),
        'has_volume': bool(df['volume'].sum() > 0),
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    if not MODEL_AVAILABLE:
        return jsonify({'error': 'Kronos library not installed'}), 400
    if _predictor is None:
        return jsonify({'error': 'Model not loaded — click Load Model first'}), 400
    d = request.get_json()

    # allow predictions from local files or yfinance; prefer yahoo when requested
    file_path = d.get('file_path')
    market = d.get('market', 'cn_stock')

    if not file_path:
        # if using Yahoo, require yfinance
        if market == 'yahoo':
            if not YFINANCE_AVAILABLE:
                return jsonify({'error': 'yfinance not installed'}), 400
        else:
            # non-yahoo backends require akshare
            if not AKSHARE_AVAILABLE:
                return jsonify({'error': 'akshare not installed and no file_path provided'}), 400
    symbol      = d.get('symbol', '600580')
    period      = d.get('period', 'daily')
    lookback    = int(d.get('lookback', 300))
    pred_len    = int(d.get('pred_len', 60))
    temperature = float(d.get('temperature', 1.0))
    top_p       = float(d.get('top_p', 0.9))
    sample_count= int(d.get('sample_count', 1))
    show_volume = bool(d.get('show_volume', True))
    show_ma     = bool(d.get('show_ma', True))
    show_bollinger = bool(d.get('show_bollinger', True))
    chart_style = d.get('chart_style', 'dark')
    ma_p1       = int(d.get('ma_period1', 20))
    ma_p2       = int(d.get('ma_period2', 50))

    # fetch data (from file_path or remote)
    if file_path:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abs_path = os.path.abspath(os.path.join(base, file_path))
        if not abs_path.startswith(base) or not os.path.exists(abs_path):
            return jsonify({'error': 'file not found or invalid path'}), 400
        try:
            if abs_path.lower().endswith('.csv'):
                df = pd.read_csv(abs_path)
            elif abs_path.lower().endswith('.parquet'):
                df = pd.read_parquet(abs_path)
            elif abs_path.lower().endswith('.json'):
                df = pd.read_json(abs_path)
            else:
                return jsonify({'error': 'unsupported file type'}), 400
            df = _rename(df)
            if 'timestamps' in df.columns:
                df['timestamps'] = pd.to_datetime(df['timestamps'])
            err = None
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        df, err = get_market_data(market, symbol, period, lookback + pred_len + 100)
    if err:
        return jsonify({'error': f'Data fetch failed: {err}'}), 400

    if len(df) < lookback + 10:
        return jsonify({'error': f'Not enough data ({len(df)} bars), reduce lookback'}), 400

    lookback = min(lookback, len(df) - 10)

    hist_df = df.iloc[-lookback:].copy().reset_index(drop=True)
    x_df    = hist_df[['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_ts    = hist_df['timestamps']

    # generate future timestamps
    last_ts   = hist_df['timestamps'].iloc[-1]
    freq_map  = {'1min': '1min', '5min': '5min', '15min': '15min', '30min': '30min',
                 '60min': '60min', 'daily': 'B', 'weekly': 'W-FRI', 'monthly': 'MS'}
    freq      = freq_map.get(period, 'B')
    y_ts      = pd.date_range(start=last_ts, periods=pred_len + 1, freq=freq)[1:]
    y_ts      = pd.Series(y_ts)

    try:
        pred_df = _predictor.predict(
            df=x_df,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=pred_len,
            T=temperature,
            top_p=top_p,
            sample_count=sample_count,
        )
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

    pred_df['timestamps'] = y_ts.values[:len(pred_df)]

    # build chart
    img_b64 = make_chart(
        hist_df, pred_df, symbol, market, period, pred_len,
        show_volume=show_volume, show_ma=show_ma, show_bollinger=show_bollinger,
        chart_style=chart_style, ma_periods=(ma_p1, ma_p2),
    )

    last_close = float(hist_df['close'].iloc[-1])
    pred_close = float(pred_df['close'].iloc[-1])
    chg = (pred_close / last_close - 1) * 100

    return jsonify({
        'success': True,
        'image': img_b64,
        'summary': {
            'symbol': symbol,
            'period': period,
            'lookback_bars': lookback,
            'pred_bars': len(pred_df),
            'last_close': round(last_close, 6),
            'pred_open':  round(float(pred_df['open'].iloc[0]),  6),
            'pred_high':  round(float(pred_df['high'].max()),    6),
            'pred_low':   round(float(pred_df['low'].min()),     6),
            'pred_close': round(pred_close, 6),
            'change_pct': round(chg, 2),
            'pred_start': str(pred_df['timestamps'].iloc[0])[:10],
            'pred_end':   str(pred_df['timestamps'].iloc[-1])[:10],
        }
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7070)
