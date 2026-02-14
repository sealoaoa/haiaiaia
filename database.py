import sqlite3
import pandas as pd
from config import Config

def init_db():
    conn = sqlite3.connect(Config.DATABASE_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
    ''')
    conn.commit()
    conn.close()

def insert_session(date, open, high, low, close, volume):
    conn = sqlite3.connect(Config.DATABASE_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO sessions (date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (date, open, high, low, close, volume))
    conn.commit()
    conn.close()

def get_all_sessions():
    conn = sqlite3.connect(Config.DATABASE_PATH)
    df = pd.read_sql_query("SELECT * FROM sessions ORDER BY id", conn)
    conn.close()
    return df

def get_last_n_sessions(n):
    conn = sqlite3.connect(Config.DATABASE_PATH)
    df = pd.read_sql_query(f"SELECT * FROM sessions ORDER BY id DESC LIMIT {n}", conn)
    conn.close()
    return df.sort_values('id')  # Trả về theo thứ tự tăng dần

def count_sessions():
    conn = sqlite3.connect(Config.DATABASE_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM sessions")
    count = c.fetchone()[0]
    conn.close()
    return count
