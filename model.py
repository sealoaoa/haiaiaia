import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
from config import Config
from database import get_last_n_sessions
from features import prepare_features_for_training, prepare_features_for_prediction

class TimeSeriesModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_cols = None

    def train(self, df):
        """Huấn luyện mô hình trên toàn bộ df (đã có đủ số phiên)"""
        X, y, self.feature_cols = prepare_features_for_training(df)
        
        # Chuẩn hóa
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Tìm siêu tham số tối ưu bằng GridSearchCV với TimeSeriesSplit
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [50, 100],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        model = xgb.XGBClassifier(objective='binary:logistic', random_state=42, use_label_encoder=False, eval_metric='logloss')
        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_scaled, y)
        
        self.model = grid_search.best_estimator_
        print(f"Best params: {grid_search.best_params_}, Best CV accuracy: {grid_search.best_score_:.4f}")
        
        # Đánh giá trên tập train (có thể dùng cross-validation đã có)
        y_pred = self.model.predict(X_scaled)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        print(f"Train accuracy: {acc:.4f}, F1: {f1:.4f}")
        
        return self.model

    def predict(self, df):
        """Dự đoán cho phiên tiếp theo dựa trên df (gồm WINDOW_SIZE phiên gần nhất)"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model chưa được huấn luyện")
        X = prepare_features_for_prediction(df)
        # Đảm bảo X có đúng các cột
        X = X[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[0]  # [xác suất class 0, xác suất class 1]
        pred = self.model.predict(X_scaled)[0]
        return int(pred), float(proba[1])  # trả về class và xác suất tăng

    def save(self):
        joblib.dump(self.model, Config.MODEL_PATH)
        joblib.dump(self.scaler, Config.SCALER_PATH)
        # Lưu feature_cols vào file riêng hoặc cùng scaler
        joblib.dump(self.feature_cols, 'feature_cols.joblib')

    def load(self):
        self.model = joblib.load(Config.MODEL_PATH)
        self.scaler = joblib.load(Config.SCALER_PATH)
        self.feature_cols = joblib.load('feature_cols.joblib')

# Biến toàn cục để lưu trạng thái model
model_instance = TimeSeriesModel()
