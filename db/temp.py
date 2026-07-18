import db
import pandas as pd
import numpy as np
import sqlite3

data_db = "/home/neuro_linux/data/finish/summary.db"

def _connect():
    # Подключение к базе данных
    conn = sqlite3.connect(data_db)
    query = "SELECT * FROM matches"
    data = pd.read_sql(query, conn)
    conn.close()
    return data

data = _connect()
data['fb_res'] = np.select(
    [np.argmin(data[['b1', 'bX', 'b2']].values, axis=1) == 0,
     np.argmin(data[['b1', 'bX', 'b2']].values, axis=1) == 1,
     np.argmin(data[['b1', 'bX', 'b2']].values, axis=1) == 2],
    [0, 1, 2])
data['fact_res'] = np.select(
    [data['h_res'] > data['g_res'], data['h_res'] == data['g_res'], data['h_res'] < data['g_res']],
    [0, 1, 2])
data['target'] = np.select(
                            [np.argmin(data[['b1', 'bX', 'b2']].values, axis=1) == data['fact_res'],
                                     np.argmax(data[['b1', 'bX', 'b2']].values, axis=1) == data['fact_res']],
                            [0,2],
                                     default=1)
print((data['target'] == 0).sum())
print((data['target'] == 1).sum())
print((data['target'] == 2).sum())