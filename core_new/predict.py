import gc
import tensorflow as tf
import numpy as np
import pandas as pd
from core.get_data import Df_getter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from core.custom.custom_metric_precision1 import PrecisionClass1
from core.custom.custom_focal_loss import focal_precision_loss
import matplotlib.pyplot as plt
from keras import backend as K
# Визуализация
import seaborn as sns
import os
from joblib import dump, load

class Predicate:
    def __init__(self, predata:tuple, model_path:str):
        self.X_processed = predata[0]
        self.X = predata[2]
        if predata[1].empty: self.y = None
        else: self.y = predata[1]
        self.model_path = model_path

    def predicate(self, focal_params:dict):
        model = tf.keras.models.load_model(
            self.model_path,
            custom_objects={"loss": focal_precision_loss(focal_param=focal_params),
                            "PrecisionClass1": PrecisionClass1,
                            'metrics':[PrecisionClass1(),
                                       tf.keras.metrics.Precision(name='keras_precision'),
                                       tf.keras.metrics.Recall()]})

        y_pr = model.predict(self.X_processed)

        self.X['p_pred'] = y_pr
        self.X['y_pred'] = (y_pr > 0.1).astype(int)
        cols = ['b1', 'bX', 'b2', 'target', 'y_pred']
        self.X[cols] = self.X[cols].apply(pd.to_numeric, errors='coerce')
        self.X['amount'] = np.select([((self.X['y_pred'] == 1) & (self.X['target'] == self.X['y_pred'])),
                                             ((self.X['y_pred'] == 1) & (self.X['target'] != self.X['y_pred']))],
                                    [np.max(self.X.loc[:,['b1', 'bX', 'b2']].values, axis=1)-1, -1],
                                            default=0)
        print(self.X[['h_res', 'g_res', 'b1', 'bX', 'b2', 'target', 'y_pred', 'p_pred', 'amount']])
        print(f"{self.X['amount'].sum():.2f}")
        print(self.X.groupby('target').size())
        print(self.X.groupby('y_pred').size())
        all = self.X['y_pred'].sum()
        succses = self.X[(self.X['y_pred']==1) & (self.X['target']==1)]
        if not succses.empty:
            print(f"{succses['y_pred'].sum()}, {all}, {succses['y_pred'].sum()*100/all:2f}")
            out_data = (round(float(succses['y_pred'].sum()), 0),
                    round(float(all), 0),
                    round(succses['y_pred'].sum()*100/all, 2),
                    round(self.X['amount'].sum(), 2))
            K.clear_session()# Очистка графа TensorFlow
            gc.collect()       # Принудительный вызов сборщика мусора
            del model, self.X
            return out_data
        else:
            out_data = (None,
                        round(float(all), 0),
                        None,
                        round(self.X['amount'].sum(), 2))
            print("Прогноз не содержит данные")
            K.clear_session()# Очистка графа TensorFlow
            del model, self.X
            gc.collect()  # Принудительный вызов сборщика мусора
            return out_data
