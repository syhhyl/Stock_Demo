import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from data_loader import fetch_stock_data, feature_engineering, prepare_data_for_model

def predict(stock_code):
    df = fetch_stock_data(stock_code)
    df = feature_engineering(df)
    X, y = prepare_data_for_model(df)

    if len(X) < 3:
        raise ValueError("数据太少")

    model = joblib.load('stock_model.pkl')
    preds = model.predict(X, num_iteration=model.best_iteration)

    # 策略增强
    kdj_boost = 0.1 * X['kdj_buy_signal'].values
    final_preds = np.clip(preds + kdj_boost, 0, 1)

    try:
        auc = roc_auc_score(y, final_preds)
        print(f"AUC = {auc:.4f}")
    except Exception as e:
        print(f"AUC 计算失败: {e}")

    last = final_preds[-1]
    print(f"\n明日上涨概率 ≈ {last:.4f}")
    if last > 0.6:
        print("建议：可以买入")
    elif last > 0.5:
        print("建议：谨慎买入")
    else:
        print("建议：不建议买入")

if __name__ == "__main__":
    while True:
        code = input("输入股票代码 (q退出): ").strip().upper()
        if code == "Q":
            break
        try:
            predict(code)
        except Exception as e:
            print(e)