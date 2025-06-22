import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from data_loader import fetch_stock_data, feature_engineering, prepare_data_for_model

def load_stock_list(file_path="stock_code.txt"):
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def train_model():
    all_X, all_y = [], []
    stock_list = load_stock_list()
    for code in stock_list:
        try:
            df = fetch_stock_data(code)
            df = feature_engineering(df)
            X, y = prepare_data_for_model(df)
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"❌ {code} error: {e}")

    X_all = pd.concat(all_X, ignore_index=True)
    y_all = pd.concat(all_y, ignore_index=True)

    X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, test_size=0.2, shuffle=False)

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 42
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        num_boost_round=300,
        callbacks=[
            lgb.early_stopping(20),
            lgb.log_evaluation(10)
        ]
    )

    joblib.dump(model, 'stock_model.pkl')
    print("✅ 模型已保存为 stock_model.pkl")

if __name__ == "__main__":
    train_model()