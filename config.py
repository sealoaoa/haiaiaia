import os

class Config:
    DATABASE_PATH = "sessions.db"
    MODEL_PATH = "model.joblib"
    SCALER_PATH = "scaler.joblib"
    WINDOW_SIZE = 50  # Số phiên dùng để train
    MIN_SESSIONS_FOR_PREDICT = 50
    FEATURE_COLS = None  # Sẽ được khởi tạo sau khi feature engineering
