import os
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from data_loader import fetch_stock_data, feature_engineering, prepare_data_for_model

def predict(stock_code):
    df = fetch_stock_data(stock_code)
    df = feature_engineering(df)
    X_full, y = prepare_data_for_model(df)

    if len(X_full) < 3:
        raise ValueError("数据太少")

    if not os.path.exists('stock_model.pkl'):
        raise FileNotFoundError("未找到模型文件 stock_model.pkl，请先运行训练脚本")
    model = joblib.load('stock_model.pkl')
    best_iter = getattr(model, 'best_iteration', None)
    # Align features to model if possible
    X = X_full.copy()
    try:
        feat_names = model.feature_name()
        if isinstance(feat_names, (list, tuple)) and len(feat_names) > 0:
            if all(name in X.columns for name in feat_names):
                X = X[feat_names]
    except Exception:
        pass
    preds = model.predict(X, num_iteration=best_iter)

    # 策略增强
    kdj_boost = 0.1 * X_full['kdj_buy_signal'].values if 'kdj_buy_signal' in X_full.columns else 0.0
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

def info(stock_code):
  df = fetch_stock_data(stock_code) 
  df = feature_engineering(df)
  print(df)
    
if __name__ == "__main__":
    while True:
        code = input("输入股票代码 (q退出): ").strip().upper()
        if code == "Q":
            break
        try:
            predict(code)
            # info(code)
        except Exception as e:
            print(e)
