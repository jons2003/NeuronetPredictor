import gc
from pyexpat import features

import tensorflow as tf
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from keras import optimizers
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.ops.metrics_impl import recall

from core.custom.custom_metric_precision1 import PrecisionClass1
from core.custom.custom_focal_loss import focal_precision_loss
import matplotlib.pyplot as plt
from keras import backend as K
# Визуализация
import seaborn as sns
import os
from joblib import dump, load
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Отключает GPU


class Model_sport:
    def __init__(self, data_proc:dict, focal_params:dict, fit_params:dict,
                 early_stopping_params:dict, model_checkpoint_params:dict):

        self.X_train, self.y_train, self.layers_combined, self.inputs = self._init_datasets(data_proc, 'Xy_train')
        self.X_val, self.y_val = self._init_datasets(data_proc, 'Xy_val')
        self.X_test, self.y_test = self._init_datasets(data_proc, 'Xy_test')
        self.focal_params = focal_params
        self.fit_params = fit_params
        self.early_stopping_params = early_stopping_params
        self.model_checkpoint_params = model_checkpoint_params

    def _init_datasets(self, data_proc, name:str):
        X = data_proc[name][0].drop(columns=['target'])
        X_tensors = [tf.convert_to_tensor(X[['b1', 'bX', 'b2']].values, dtype=tf.float32)]
        features_  = list(filter(lambda x: 'ind_' not in x,list(X.drop(columns=['b1', 'bX', 'b2']).columns)))
        indicators_ = list(filter(lambda x: 'ind_'  in x, list(X.drop(columns=['b1', 'bX', 'b2']).columns)))
        columns = [column for pair in zip(features_,indicators_) for column in pair]
        for column in columns:
            if 'ind_' in column: X_tensors.append(tf.convert_to_tensor(X[column].values, dtype=tf.int32))
            else: X_tensors.append(tf.convert_to_tensor(X[column].values, dtype=tf.float32))

        y = data_proc[name][0][['target']]
        y_tensor = tf.convert_to_tensor(y.values, dtype=tf.int32)
        layers_combined = data_proc[name][1]
        inputs = data_proc[name][2]
        if name == 'Xy_train':
            return X_tensors, y_tensor, layers_combined,  inputs
        else: return X_tensors, y_tensor

    def learning(self):
        #  Создание слоев модели
        x = Dense(128, activation='relu',
                  # kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-4, l2=1e-4),
                  kernel_regularizer=None
                  ) (self.layers_combined)  # Первый скрытый слой
        x = BatchNormalization()(x)  # Нормализация активаций
        x = Dropout(0.5)(x)  # Регуляризация
        x = Dense(64, activation='relu',
                  # kernel_regularizer = tf.keras.regularizers.l2(1e-5),
                  kernel_regularizer = None
                  )(x) # Второй скрытый слой
        x = BatchNormalization()(x)  # Нормализация активаций
        x = Dropout(0.3)(x)  # Регуляризация
        # x = Dense(64, activation='relu')(x)  # Третий скрытый слой
        # x = Dropout(0.2)(x)  # Регуляризация
        output = Dense(1, activation='sigmoid')(x)  # Бинарная классификация5

        # 5. Создание модели
        model = Model(inputs=self.inputs, outputs=output)
        focal_func_loss = focal_precision_loss(self.focal_params)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            # optimizer=optimizers.RMSprop(0.0001),
            metrics=[PrecisionClass1(),
                     tf.keras.metrics.Precision(name='keras_precision',thresholds=0.5),
                     tf.keras.metrics.Recall()],
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

        from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix
        # Подбор порога предсказания по максимальной  F1-score
            # Предсказание вероятностей на валидации
        y_probs = model.predict(self.X_val).ravel()
            # Поиск порога по максимальному F1
        precision, recall, thresholds = precision_recall_curve(self.y_val, y_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
        best_threshold = thresholds[tf.argmax(f1_scores)] # Оптимальный порог
        print('Оптимальный порог = ', best_threshold)

        # fig, ax = plt.subplots()
        # ax.plot(recall, precision, color='purple')
        # ax.set_title('Precision-Recall Curve')
        # ax.set_ylabel('Precision')
        # ax.set_xlabel('Recall')
        # plt.show()

        y_probs = model.predict(self.X_val).ravel()
        plt.hist(y_probs, bins=50)
        plt.show()

        best_threshold = 0.9
        # Вычисление матрицы потерь с оптимальным порогом
        # ---- на валидации ----

        y_pred_class = (model.predict(self.X_val) > best_threshold).astype(int)
        matrix = confusion_matrix(self.y_val, y_pred_class)
        val_TP = int(matrix[1,1])
        val_FP = int(matrix[0,1])
        val_FN = int(matrix[1,0])
        val_precision = round(float(val_TP/(val_TP+val_FP)), 2)
        val_recall = round(float((val_TP)/(val_TP+val_FN)),2)
        print("Матрица val-данных:")
        print(matrix, val_precision, val_recall)
        # ----- на тесте ------
        y_pred_class = (model.predict(self.X_test) > best_threshold).astype(int)
        matrix = confusion_matrix(self.y_test, y_pred_class)
        test_TP = int(matrix[1, 1])
        test_FP = int(matrix[0, 1])
        test_precision = round(float(test_TP / (test_TP + test_FP)), 2)
        test_recall = round(float((test_TP+test_FP)/np.sum(matrix)),2)
        print("Матрица test-данных:")
        print(matrix, test_precision, test_recall)

        out_data = (val_TP, val_FP,  val_precision, val_recall,
                    test_TP, test_FP, test_precision, test_recall
                    )
        # Show the learning curves
        # history_df = pd.DataFrame(history.history)
        # history_df.loc[:, ['val_keras_precision', 'val_loss', 'val_precision_class1','recall']].plot()
        # plt.show()

        K.clear_session()  # Очистка графа TensorFlow
        gc.collect()  # Принудительный вызов сборщика мусора
        del model
        return out_data

