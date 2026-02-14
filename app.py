from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
from datetime import datetime
import pandas as pd

from database import init_db, insert_session, get_all_sessions, get_last_n_sessions, count_sessions
from model import model_instance, TimeSeriesModel
from config import Config

# Khởi tạo database
init_db()

app = FastAPI(title="Time Series Prediction API")

class SessionData(BaseModel):
    date: Optional[str] = Field(None, description="Ngày (có thể để trống, tự động lấy datetime hiện tại)")
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: float = Field(..., description="Giá đóng cửa (bắt buộc)")
    volume: Optional[float] = None

def train_model_background(window_size=Config.WINDOW_SIZE):
    """Hàm chạy nền để huấn luyện mô hình trên window_size phiên gần nhất"""
    print("Bắt đầu huấn luyện mô hình trong background...")
    df = get_last_n_sessions(window_size)
    if len(df) < window_size:
        print(f"Không đủ dữ liệu: cần {window_size}, có {len(df)}")
        return
    model_instance.train(df)
    model_instance.save()
    print("Huấn luyện hoàn tất và đã lưu model.")

@app.post("/update")
async def update_session(session: SessionData, background_tasks: BackgroundTasks):
    """
    Nhận dữ liệu một phiên mới, lưu vào database.
    Nếu số phiên sau khi lưu >= 50, kích hoạt huấn luyện (background).
    """
    # Xử lý date
    if session.date is None:
        date_str = datetime.now().isoformat()
    else:
        date_str = session.date
    
    # Lưu vào DB
    insert_session(
        date=date_str,
        open=session.open,
        high=session.high,
        low=session.low,
        close=session.close,
        volume=session.volume
    )
    
    # Đếm số phiên hiện tại
    total = count_sessions()
    
    # Nếu đủ 50 phiên, kích hoạt train
    if total >= Config.WINDOW_SIZE:
        # Kiểm tra nếu model chưa tồn tại hoặc muốn train lại mỗi lần (phương án B)
        # Ở đây ta train lại mỗi lần có phiên mới (phương án B)
        background_tasks.add_task(train_model_background, Config.WINDOW_SIZE)
        return {
            "message": f"Đã nhận phiên thứ {total}. Đã kích hoạt huấn luyện lại mô hình trong background.",
            "total_sessions": total
        }
    else:
        return {
            "message": f"Đã nhận phiên thứ {total}. Cần {Config.WINDOW_SIZE - total} phiên nữa để có thể dự đoán.",
            "total_sessions": total
        }

@app.get("/predict")
async def predict():
    """
    Trả về dự đoán cho phiên tiếp theo dựa trên mô hình hiện tại và 50 phiên gần nhất.
    Yêu cầu: đã có ít nhất 50 phiên và model đã được huấn luyện.
    """
    total = count_sessions()
    if total < Config.MIN_SESSIONS_FOR_PREDICT:
        raise HTTPException(status_code=400, detail=f"Cần ít nhất {Config.MIN_SESSIONS_FOR_PREDICT} phiên để dự đoán. Hiện có {total}.")
    
    # Kiểm tra model đã tồn tại chưa? Nếu chưa, có thể load hoặc báo lỗi.
    if model_instance.model is None:
        # Thử load từ file
        try:
            model_instance.load()
            print("Đã load model từ file.")
        except:
            raise HTTPException(status_code=400, detail="Model chưa được huấn luyện. Vui lòng đợi quá trình huấn luyện hoàn tất.")
    
    # Lấy 50 phiên gần nhất
    df = get_last_n_sessions(Config.WINDOW_SIZE)
    try:
        pred, proba = model_instance.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự đoán: {str(e)}")
    
    # Lấy giá close hiện tại (phiên cuối cùng trong df)
    last_close = df.iloc[-1]['close']
    # Giá dự đoán không có vì là classification, chỉ có xu hướng
    direction = "tăng" if pred == 1 else "giảm"
    return {
        "prediction": direction,
        "probability_up": proba,
        "last_close": last_close,
        "total_sessions": total,
        "note": "Dự đoán xu hướng phiên tiếp theo (tăng/giảm so với phiên hiện tại)"
    }

@app.get("/status")
async def status():
    """Trả về trạng thái hiện tại"""
    total = count_sessions()
    model_loaded = model_instance.model is not None
    return {
        "total_sessions": total,
        "model_loaded": model_loaded,
        "window_size": Config.WINDOW_SIZE,
        "ready_to_predict": total >= Config.MIN_SESSIONS_FOR_PREDICT and model_loaded
    }

@app.on_event("startup")
async def startup_event():
    """Khi khởi động, thử load model nếu có"""
    try:
        model_instance.load()
        print("Đã load model từ file khi khởi động.")
    except:
        print("Chưa có model, sẽ huấn luyện khi đủ dữ liệu.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
