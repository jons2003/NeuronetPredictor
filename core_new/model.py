import gc
import tensorflow as tf
import numpy as np
import pandas as pd
from pandas.core.config_init import val_mca

from sklearn.model_selection import train_test_split
from keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.ops.metrics_impl import recall

from core_new.custom.custom_metric_precision1 import PrecisionClass1
from core_new.custom.custom_focal_loss import focal_precision_loss
import matplotlib.pyplot as plt
from keras import backend as K
# Визуализация
import seaborn as sns
import os
from joblib import dump, load
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Отключает GPU


class Model:
    def __init__(self, data_proc:dict, focal_params:dict, fit_params:dict,
                 early_stopping_params:dict, model_checkpoint_params:dict):

        self.X_train, self.y_train, self.futures_columns = self._init_datasets(data_proc,'Xy_train')
        self.X_val, self.y_val = self._init_datasets(data_proc, 'Xy_val')
        self.X_test, self.y_test = self._init_datasets(data_proc, 'Xy_test')
        self.focal_params = focal_params
        self.fit_params = fit_params
        self.early_stopping_params = early_stopping_params
        self.model_checkpoint_params = model_checkpoint_params

    def _init_datasets(self, data_proc, name='Xy_train'):
        X = data_proc[name][0].drop(columns=['league', 'host', 'guest', 'target'])
        y = data_proc[name][0][['target']]
        features_inds = data_proc[name][1]
        if name == 'Xy_train':
            return X, y, features_inds
        else: return X, y


    def learning(self):
        # Построение модели
        model = tf.keras.Sequential([
            tf.keras.layers.DenseFeatures(self.futures_columns),
            tf.keras.layers.Dense(512, activation='relu', input_shape=(self.X_train.shape[1],),
                                  # kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-4, l2=1e-4)
                                  kernel_regularizer=None
                                  ),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu',
                                  # kernel_regularizer=tf.keras.regularizers.l2(1e-5)
                                  kernel_regularizer=None
                                  ),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        focal_func_loss = focal_precision_loss(self.focal_params)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            metrics=[PrecisionClass1(),
                     tf.keras.metrics.Precision(name='keras_precision',thresholds=0.5),
                     tf.keras.metrics.Recall()],
            # metrics=[PrecisionClass1(), tf.keras.metrics.Precision(name='keras_precision', thresholds=0.5)],
            loss=focal_func_loss
            # loss='binary_crossentropy',
            # metrics=['AUC', 'Precision', 'Recall']
        )
        callbacks = [
            EarlyStopping(**self.early_stopping_params),
            ModelCheckpoint(**self.model_checkpoint_params)]



        history = model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), callbacks=callbacks, **self.fit_params)
        max_val_precision1 = round(float(max(history.history['val_precision_class1'])), 2)
        print(f"Максимальная точность на валидации: {max_val_precision1}")

        # Вычисление матрицы потерь
        # ---- на валидации ----
        from sklearn.metrics import confusion_matrix
        y_pred_class = (model.predict(self.X_val) > 0.1).astype(int)
        matrix = confusion_matrix(self.y_val, y_pred_class)
        val_TP = int(matrix[1,1])
        val_FP = int(matrix[0,1])
        val_precision = round(float(val_TP/(val_TP+val_FP)), 2)
        val_recall = round(float((val_TP+val_FP)/np.sum(matrix)),2)
        print("Матрица val-данных:",matrix[0,1], matrix[1,0])
        print(matrix, matrix[1,1] / (matrix[0,1] + matrix[1,1]))
        # ----- на тесте ------
        y_pred_class = (model.predict(self.X_test) > 0.1).astype(int)
        matrix = confusion_matrix(self.y_test, y_pred_class)
        test_TP = int(matrix[1, 1])
        test_FP = int(matrix[0, 1])
        test_precision = round(float(test_TP / (test_TP + test_FP)), 2)
        test_recall = round(float((test_TP+test_FP)/np.sum(matrix)),2)
        print("Матрица test-данных:")
        print(matrix, matrix[1,1] / (matrix[0,1] + matrix[1,1]))

        # Метрики на тестовых данных
        # y_test_res = round(float(model.evaluate(self.X_test, self.y_test)[1]),2)

        out_data = (val_TP, val_FP,  val_precision, val_recall,
                    test_TP, test_FP, test_precision, test_recall
                    )
        # Show the learning curves
        history_df = pd.DataFrame(history.history)
        history_df.loc[:, ['val_keras_precision', 'val_loss', 'val_precision_class1','recall']].plot()
        plt.show()

        K.clear_session()  # Очистка графа TensorFlow
        gc.collect()  # Принудительный вызов сборщика мусора
        del model
        return out_data

