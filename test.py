import akshare as ak

def get_600_stocks():
    # 获取沪市的股票代码和简称
    df = ak.stock_info_sh_name_code()
    # 过滤600开头
    df_600 = df[df["证券代码"].str.startswith("600")]
    return df_600["证券代码"].tolist()

if __name__ == "__main__":
    stock_list = get_600_stocks()
    print(stock_list[:10])  # 打印前10个