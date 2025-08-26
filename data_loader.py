import akshare as ak
import pandas as pd
import numpy as np

def fetch_stock_data(stock_code):
    df = ak.stock_zh_a_hist(symbol=stock_code, start_date="20150101", period="daily", adjust="qfq")
    df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量']]
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def compute_kdj(df, n=9):
    low_min = df['low'].rolling(window=n).min()
    high_max = df['high'].rolling(window=n).max()
    rsv = (df['close'] - low_min) / (high_max - low_min + 1e-9) * 100

    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()
    j = 3 * k - 2 * d
    df['J'] = j
    # print(type(df))
    print(df['J'])
    return df

def feature_engineering(df):
    # df['return'] = df['close'].pct_change()
    # df['ma5'] = df['close'].rolling(5).mean()
    # df['ma10'] = df['close'].rolling(10).mean()
    # df['ma_diff'] = df['ma5'] - df['ma10']
    # df['vol_ratio'] = df['volume'] / df['volume'].rolling(5).mean()
    # df['high_close_diff'] = df['high'] - df['close']
    # df['close_low_diff'] = df['close'] - df['low']
    # df['range'] = df['high'] - df['low']
    # df['return_sign'] = np.sign(df['return'])
    df['label'] = (df['close'].shift(-2) > df['close']).astype(int)  # 改进标签定义

    df = compute_kdj(df)
    df = df.dropna().reset_index(drop=True)
    return df

def prepare_data_for_model(df):
    # features = [
    #     'return', 'ma_diff', 'vol_ratio', 'high_close_diff',
    #     'close_low_diff', 'range', 'return_sign', 'J', 'kdj_buy_signal'
    # ]
    features = ['J']
    X = df[features]
    y = df['label']
    return X, y

if __name__ == "__main__":
  df = fetch_stock_data("600030")
  df = compute_kdj(df)
  print(df)
  