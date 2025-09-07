import pandas as pd
import numpy as np
import os

# Try to import akshare; fall back to local cache when unavailable
try:
    import akshare as ak  # type: ignore
except Exception:
    ak = None

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Handle both Chinese and English column names; ensure required columns exist
    rename_map = {
        '日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume',
        'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
    }
    df = df.rename(columns=rename_map)
    # If there are extra columns, keep only needed for downstream
    needed = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df[needed]


def _fetch_from_network(stock_code: str) -> pd.DataFrame:
    if ak is None:
        raise RuntimeError("akshare 不可用，无法联网获取数据")
    df = ak.stock_zh_a_hist(symbol=stock_code, start_date="20150101", period="daily", adjust="qfq")
    return _normalize_columns(df)


def _fetch_from_local_cache(stock_code: str) -> pd.DataFrame:
    # 1) Per-code pickle (e.g., 600030.pkl)
    for path in (f"{stock_code}.pkl", os.path.join("data", f"{stock_code}.pkl")):
        if os.path.exists(path):
            df = pd.read_pickle(path)
            return _normalize_columns(df)

    # 2) Shared pickle with multiple codes
    shared_paths = ["stock_data.pkl", os.path.join("data", "stock_data.pkl")]
    for sp in shared_paths:
        if os.path.exists(sp):
            obj = pd.read_pickle(sp)
            # If it's a dict-like cache: {code: DataFrame}
            if isinstance(obj, dict):
                if stock_code in obj:
                    return _normalize_columns(obj[stock_code])
                # Try without leading exchange prefix or other variants
                for k in obj.keys():
                    if str(k).endswith(str(stock_code)):
                        return _normalize_columns(obj[k])
                raise ValueError(f"本地缓存未找到股票 {stock_code}")
            # If it's a DataFrame, try filtering by common code columns
            if isinstance(obj, pd.DataFrame):
                df_all = obj
                for code_col in ["code", "symbol", "ts_code", "stock_code"]:
                    if code_col in df_all.columns:
                        sub = df_all[df_all[code_col].astype(str).str.endswith(str(stock_code))]
                        if not sub.empty:
                            return _normalize_columns(sub)
                # Maybe it's already a single-code DataFrame
                try:
                    return _normalize_columns(df_all)
                except Exception:
                    pass
            raise ValueError(f"无法从 {sp} 解析数据")

    raise FileNotFoundError("未找到本地数据缓存，请检查 stock_data.pkl 或 {code}.pkl")


def fetch_stock_data(stock_code: str) -> pd.DataFrame:
    """
    获取股票日线数据，优先联网，失败则读取本地缓存。
    返回列: date, open, high, low, close, volume
    """
    try:
        return _fetch_from_network(stock_code)
    except Exception:
        return _fetch_from_local_cache(stock_code)

def save_stock_data(stock_code):
    try:
        df = fetch_stock_data(stock_code)
        df.to_pickle(f'{stock_code}.pkl')
    except Exception as e:
        print(f"save fail: {e}")
  

def compute_kdj(df: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    low_min = df['low'].rolling(window=n, min_periods=1).min()
    high_max = df['high'].rolling(window=n, min_periods=1).max()
    rsv = (df['close'] - low_min) / (high_max - low_min + 1e-9) * 100

    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d
    df['K'] = k
    df['D'] = d
    df['J'] = j
    # Simple buy signal: K crosses above D and K < 30 (oversold)
    cross_up = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
    df['kdj_buy_signal'] = (cross_up & (df['K'] < 30)).astype(int)
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df['return'] = df['close'].pct_change()
    df['ma5'] = df['close'].rolling(5, min_periods=1).mean()
    df['ma10'] = df['close'].rolling(10, min_periods=1).mean()
    df['ma_diff'] = df['ma5'] - df['ma10']
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(5, min_periods=1).mean()
    df['high_close_diff'] = df['high'] - df['close']
    df['close_low_diff'] = df['close'] - df['low']
    df['range'] = df['high'] - df['low']
    df['return_sign'] = np.sign(df['return'])
    # label: whether price rises within next 2 days
    df['label'] = (df['close'].shift(-2) > df['close']).astype(int)

    df = compute_kdj(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df

def prepare_data_for_model(df: pd.DataFrame):
    features = [
        'return', 'ma_diff', 'vol_ratio', 'high_close_diff',
        'close_low_diff', 'range', 'return_sign', 'K', 'D', 'J', 'kdj_buy_signal'
    ]
    # Ensure all features exist
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"缺少特征列: {missing}")
    X = df[features]
    y = df['label']
    return X, y

if __name__ == "__main__":
    df = fetch_stock_data("600030")
    df = feature_engineering(df)
    print(df.tail())
    
