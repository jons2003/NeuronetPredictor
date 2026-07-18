import sqlite3
import pandas as pd

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap, BoundaryNorm
import plotly.graph_objects as go

def graf_4D():
    # Цвета по условию
    colors = []
    for val in test_precision:
        if val < 0.1: colors.append('yellow')
        elif val >= 0.1 and val < 0.15: colors.append('green')
        elif val >= 0.15 and val < 0.2: colors.append('red')
        elif val >= 0.2 and val < 0.25: colors.append('blue')
        else: colors.append('brown')

    # Создание 3D-графика
    fig = go.Figure(data=[go.Scatter3d(
        x=alpha,
        y=gamma,
        z=alpha_fp,
        mode='markers',
        marker=dict(
            size=8,
            color=colors,
            opacity=0.8,
        ),
    )])

    # Настройка осей
    fig.update_layout(
        scene=dict(
            xaxis_title='Alpha',
            yaxis_title='Gamma',
            zaxis_title='Alpha_fp'
        ),
        title='3D Plot with Plotly'
    )
    fig.show()  # Откроется в браузере с полным контролем вращения

def graf_3D(df, columns:list):
    # Создание сетки методом pivot
    grid = df.pivot_table(index=columns[0], columns=columns[1], values=columns[2], aggfunc='mean')

    # Заполнение пропусков (если есть)
    grid = grid.interpolate(method='linear', axis=0).interpolate(method='linear', axis=1)

    # Извлечение координат
    X = grid.columns.values
    Y = grid.index.values
    Z = grid.values

    # Построение  поверхности
    fig = go.Figure(data=[
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale='Viridis',  # Цветовая схема
            opacity=0.8,  # Прозрачность
            contours={  # Контуры осей
                'x': {"show": True, "color": "gray"},
                'y': {"show": True, "color": "gray"},
                'z': {"show": True}
            }
        )
    ])

    # Настройка внешнего вида
    fig.update_layout(
        title='Поверхность из DataFrame',
        scene=dict(
            xaxis_title=columns[0],
            yaxis_title=columns[1],
            zaxis_title=columns[2],
            camera_eye=dict(x=1.5, y=1.5, z=0.6)  # Угол обзора
        ),
        width=800,
        height=600
    )

    fig.show()

df = pd.read_sql("SELECT * FROM journal", sqlite3.connect('/home/neuro_linux/data/models/journal.db'))
alpha = df['alpha'].values
gamma = df['gamma'].values
alpha_fp = df['alpha_fp'].values
test_precision = df['test_precision'].values



# graf_4D()
graf_3D(df, ['gamma', 'alpha_fp', 'test_precision'])