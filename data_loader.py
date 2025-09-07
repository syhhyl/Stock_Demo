import akshare as ak
import pandas as pd
import numpy as np

def fetch_stock_data(stock_code):
  '''
  fetch_stock_data get a stock infomation include:
  date open high low close volume 
  '''
  df = ak.stock_zh_a_hist(symbol=stock_code, start_date="20150101", period="daily", adjust="qfq")
  df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量']]
  df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
  df['date'] = pd.to_datetime(df['date'])
  df = df.sort_values('date').reset_index(drop=True)
  return df

def save_stock_data(stock_code):
  try:
    df = fetch_stock_data(stock_code)
    df.to_pickle(f'{stock_code}.pkl')
  except:
    print("save fail")
  

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

if __name__ == "__main__":
  df = fetch_stock_data("600030")
  df.to_pickle("stock_data.pkl")
  print(df)
  print("read data.pkl")
  df = pd.read_pickle("stock_data.pkl")
  print(df)
  # print(df.iloc[20])
  