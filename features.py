import pandas as pd
import numpy as np

def add_technical_features(df):
    """Thêm các chỉ báo kỹ thuật vào DataFrame"""
    # Đảm bảo dữ liệu được sắp xếp
    df = df.sort_values('id').reset_index(drop=True)
    
    # Giá đóng cửa
    close = df['close']
    
    # Lags
    for lag in range(1, 11):
        df[f'close_lag{lag}'] = close.shift(lag)
    
    # SMA
    for window in [5, 10, 20]:
        df[f'SMA_{window}'] = close.rolling(window).mean()
    
    # EMA
    for window in [5, 10, 20]:
        df[f'EMA_{window}'] = close.ewm(span=window, adjust=False).mean()
    
    # Rolling std (volatility)
    for window in [5, 10]:
        df[f'std_{window}'] = close.rolling(window).std()
    
    # RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # Bollinger Bands (20, 2)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['BB_upper'] = sma20 + 2 * std20
    df['BB_lower'] = sma20 - 2 * std20
    df['BB_pctB'] = (close - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Return và log return
    df['return'] = close.pct_change()
    df['log_return'] = np.log(close / close.shift(1))
    
    # Volume features
    df['volume_change'] = df['volume'].pct_change()
    df['volume_SMA_5'] = df['volume'].rolling(5).mean()
    
    # Drop NaN (do rolling và shift)
    df = df.dropna().reset_index(drop=True)
    return df

def prepare_features_for_training(df):
    """Tạo feature matrix X và target y"""
    df_feat = add_technical_features(df.copy())
    
    # Target: 1 nếu close tăng so với phiên trước, 0 nếu giảm hoặc bằng
    df_feat['target'] = (df_feat['close'] > df_feat['close'].shift(1)).astype(int)
    # Bỏ dòng đầu tiên vì không có target
    df_feat = df_feat.iloc[1:]  # Dòng đầu tiên target NaN, nhưng đã dropna ở trên? Cẩn thận.
    # Thực tế, add_technical_features đã dropna, nên target sẽ được tính trên dữ liệu còn lại.
    # Cần đảm bảo target được tính sau khi đã có feature.
    # Tốt hơn: tính target trước khi dropna?
    # Ở đây ta tính target trên df gốc và align sau.
    # Cách đơn giản: sau khi có df_feat, target là (df_feat['close'] > df_feat['close'].shift(1)).astype(int) nhưng shift lần nữa.
    # Thay vào đó, ta lấy target từ close gốc trước khi feature engineering?
    # Để đơn giản, ta tính target từ df gốc và sau đó cắt theo index.
    # Nhưng df_feat đã bỏ đi các dòng đầu, nên target cũng phải tương ứng.
    # Ta làm như sau:
    # - Tính target trên df gốc: target = (df['close'].shift(-1) > df['close']).astype(int) (dự đoán cho phiên tiếp theo)
    # - Sau khi tạo feature, target sẽ được lấy từ dòng hiện tại cho phiên tiếp theo? Cần cẩn thận.
    # Với bài toán next-step prediction, ta muốn dùng feature của phiên hiện tại để dự đoán target của phiên sau.
    # Vậy target cần là (close.shift(-1) > close).astype(int), và feature là các giá trị hiện tại.
    # Khi đó, dòng cuối cùng sẽ không có target.
    
    # Cách xử lý đúng:
    df = df.sort_values('id').reset_index(drop=True)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)  # target cho phiên tiếp theo
    # Loại bỏ dòng cuối (không có target)
    df = df.iloc[:-1]
    
    # Tạo features
    df_feat = add_technical_features(df)  # add_technical_features đã dropna, nên còn lại các dòng có đủ feature
    # Lấy X và y
    feature_cols = [col for col in df_feat.columns if col not in ['id', 'date', 'open', 'high', 'low', 'close', 'volume', 'target']]
    X = df_feat[feature_cols]
    y = df_feat['target']
    return X, y, feature_cols

def prepare_features_for_prediction(df):
    """Tạo feature cho dự đoán từ dữ liệu hiện tại (cần ít nhất WINDOW_SIZE phiên)"""
    df_feat = add_technical_features(df.copy())
    # Lấy dòng cuối cùng (phiên hiện tại) làm feature cho dự đoán phiên tiếp theo
    # Nhưng add_technical_features có thể tạo ra NaN ở đầu, nên dòng cuối chắc chắn có đủ feature
    feature_cols = [col for col in df_feat.columns if col not in ['id', 'date', 'open', 'high', 'low', 'close', 'volume']]
    last_row = df_feat.iloc[-1:][feature_cols]
    return last_row
