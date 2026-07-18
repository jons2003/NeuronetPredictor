from pyexpat import features

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump, load
from sklearn.utils import resample
from collections import Counter
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from tensorflow.keras.layers import Input, Concatenate


class Preproccessing:

    def __init__(self,predata:tuple, preproc_file:str, categorical_features:str, balance_method):
        self.data, self.cols_drop, self.cols_res = predata
        self.preprocessor = preproc_file
        self.categorical_features = categorical_features
        self.balance_method = balance_method
        self.random_state = 42

    def _check_distr(self,X,y):
        dist = {0: y[y['target'] == 0].size, 1: y[y['target'] == 1].size}
        print(f'Распределение классов в train-данных до балансировки (после разделения) = {round(100*dist[1]/(dist[0]+dist[1]), 2)}% = {dist}\n'
              f'X: {X.shape}, y: {y.shape}')

    # ормализация только первичных признаков!!! Индикаторы исключены из процесса нормализации
    # Аргумент функции - словарь с DataFrames раделенных данных
    def _normalization(self, row_data:dict)->list:
        #  row_data = {'X_train':X_train, 'X_val':X_val, 'X_test':X_test} - для обучения модели
        #  row_data = {'X_train':X_train} - для предсказания (X_train - весь набор входных данных) - на выходе только [X_train_processed]
        # Разделение на категориальные, числовые и индикаторные признаки
        if self.categorical_features == 'y': categorical_features = ['league', 'host', 'guest']
        else: categorical_features = []
        indicators_features = [column for column in row_data['X_train'] if 'ind_' in column]
        numeric_features = list(row_data['X_train'].columns)
        [numeric_features.remove(feature) for feature in categorical_features+indicators_features if feature in numeric_features]
        #Пайплайн предобработки
        numeric_transformer = Pipeline([
            # ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline([
            # ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(sparse_output=False))  # Отключаем разреженные матрицы
        ])
        preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_features),
                                          ('cat', categorical_transformer, categorical_features)]
                                         )
        # Обучение преобразователя на train-данных c Сохранением предобработчика после обучения
        row_X_train = row_data.pop('X_train')
        X_train_processed = preprocessor.fit_transform(row_X_train[categorical_features+numeric_features])
        dump(preprocessor, self.preprocessor)
        # Получите имена колонок после всех преобразований
        column_names = preprocessor.get_feature_names_out()
        X_train_processed = pd.DataFrame(X_train_processed, columns=categorical_features+numeric_features, index=row_X_train.index)
        X_train_processed = pd.concat([row_X_train[[*categorical_features]], X_train_processed, row_X_train[[*indicators_features]]], axis=1)
        # print(f'\nX_train_processed:\n{X_train_processed}')

        data_processed = [X_train_processed]
        # Преобразование валидационных и тестовых данных(при наличии - в режиме learning)
        if 'X_val' in row_data.keys():
            row_X_val = row_data.pop('X_val')
            X_val_processed = preprocessor.transform(row_X_val[categorical_features+numeric_features])
            X_val_processed = pd.DataFrame(X_val_processed, columns=categorical_features+numeric_features, index=row_X_val.index)
            X_val_processed = pd.concat([row_X_val[[*categorical_features]], X_val_processed, row_X_val[[*indicators_features]]], axis=1)
            # print(f'\nX_val_processed: \n{X_val_processed}')
            data_processed.append(X_val_processed)
        if 'X_test' in  row_data.keys():
            row_X_test = row_data.pop('X_test')
            X_test_processed = preprocessor.transform(row_X_test[categorical_features+numeric_features])
            X_test_processed = pd.DataFrame(X_test_processed, columns=categorical_features+numeric_features, index=row_X_test.index)
            X_test_processed = pd.concat([row_X_test[[*categorical_features]], X_test_processed, row_X_test[[*indicators_features]]], axis=1)
            # print(f'\nX_test_processed: \n{X_test_processed}')
            data_processed.append(X_test_processed)
        return data_processed

    def indicators_setup(self, frames:dict):
        dict_Xy = {}
        for name, frame_ in frames.items():
            # Числовые признаки + их индикаторы
            list_layers_col_ind = []
            col_target = frame_.pop('target')
            frame_['target'] = col_target

            # Для каждого исходного признака
            input_bets = Input(shape=(3,), name='bets_input')
            inputs = [input_bets]
            columns = list(frame_.columns)
            for col in columns:
                if f"ind_{col.split('+')[0]}" in list(map(lambda x: x.split('+')[0], columns)):
                    # Объединяем слои Input  признака и его индикатора пропуск
                    input_ = Input(shape=(1,), name=col)
                    indicator_ = Input(shape=(1,), name=f'ind_{col}')
                    # Добавляем входы Input в список
                    inputs.extend([input_, indicator_])
                    # Добавляем объединенный слой Concatenate() в список
                    col_ind_combined = Concatenate()([input_, indicator_])
                    list_layers_col_ind.append(col_ind_combined)
            layers_combained = Concatenate()([input_bets, *list_layers_col_ind])
            dict_Xy.setdefault(name, (frame_, layers_combained, inputs))
        return dict_Xy

    def balance(self, X,y):
        """
        X, y - тренировочные данные
        Балансировка данных выбранным методом.
        :param method: 'undersampling' или 'oversampling' или 'n' - без балансировки
        :return: Сбалансированные X, y
        """
        if self.balance_method == 'undersample':
            return self._undersample(X,y)
        elif self.balance_method == 'oversample':
            return self._oversample(X,y)
        elif self.balance_method == 'SMOTE':
            return self._smote(X,y)
        elif self.balance_method == 'n':
            print("Данные не балансировались!")
            return X, y.values.ravel()

    def _undersample(self, X, y):
        """
            Уменьшение размера мажоритарного класса
            Определение самого частого и редкого классов
        """

        majority_class = max(self.class_distribution, key=self.class_distribution.get)
        minority_class = min(self.class_distribution, key=self.class_distribution.get)
        # Разделение данных
        self._check_distr(X,y)
        y = np.array(y)
        y = y.ravel()  # Преобразует (n, 1) → (n,)
        X_major = X[y == majority_class]
        y_major = y[y == majority_class]
        X_minor = X[y == minority_class]
        y_minor = y[y == minority_class]

        # Уменьшение мажоритарного класса
        X_major_down = resample(X_major,
                                replace=False,
                                n_samples=len(y_minor),
                                random_state=self.random_state)
        y_major_down = resample(y_major,
                                replace=False,
                                n_samples=len(y_minor),
                                random_state=self.random_state)

        # Объединение данных
        X_balanced = np.vstack([X_major_down, X_minor])
        y_balanced = np.hstack([y_major_down, y_minor])


        return X_balanced, y_balanced

    def _oversample(self,X, y):
        # Определение самого частого и редкого классов
        majority_class = max(self.class_distribution, key=self.class_distribution.get)
        minority_class = min(self.class_distribution, key=self.class_distribution.get)

        self._check_distr(X, y)
        y = np.array(y)
        y = y.ravel()  # Преобразует (n, 1) → (n,)

        # Разделение данных
        X_major = X[y == majority_class]
        y_major = y[y == majority_class]
        X_minor = X[y == minority_class]
        y_minor = y[y == minority_class]

        # Методы увеличения
        # Простое дублирование
        X_minor_up = resample(X_minor,
                              replace=True,
                              n_samples=len(y_major),
                              random_state=self.random_state)
        y_minor_up = resample(y_minor,
                              replace=True,
                              n_samples=len(y_major),
                              random_state=self.random_state)


        # Объединение данных
        X_balanced = np.vstack([X_major, X_minor_up])
        y_balanced = np.hstack([y_major, y_minor_up])

        # Перемешиваем сбалансированные данные
        X_balanced, y_balanced = shuffle(X_balanced, y_balanced, random_state=42)
        print(f"Class 0 after oversampling: {np.sum(y_balanced == 0)}\n"
              f"Class 1 after oversampling: {np.sum(y_balanced == 1)}")

        return X_balanced, y_balanced

    def _smote(self,X,y):
        self._check_distr(X, y)
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        # Перемешиваем сбалансированные данные
        X_balanced, y_balanced = shuffle(X_balanced, y_balanced, random_state=42)
        print(f"Class 0 after SMOTE: {np.sum(y_balanced == 0)}\n"
              f"Class 1 after SMOTE: {np.sum(y_balanced == 1)}")
        y_balanced = np.array(y_balanced)
        y_balanced = y_balanced.ravel()  # Преобразует (n, 1) → (n,)
        return X_balanced, y_balanced

    def learning(self):
        print("---- Этап 2. Разделение данных на выборки, стандартизация, балансировка ---\n")
        y = self.data[self.cols_res]
        X = self.data.drop(columns=self.cols_drop) # Categorical [league, host, guest] - в данных
        if self.categorical_features == 'n':
            X = X.drop(columns=['league', 'host', 'guest'])
        self.class_distribution = {0: y[y['target'] == 0].size, 1: y[y['target']==1].size}

        # 1. Разделение данных
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3,
                                                            random_state=self.random_state,stratify=y) # Стратифицированное разделение
        # Разделение на валидацию и тест:
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5,
                                                          random_state=self.random_state, stratify=y_temp)
        print(f"Размерности раздельных выборок:  train - {X_train.shape}, val - {X_val.shape}, test - {X_test.shape}")

        #  2. Нормализация входных признаков данных
        row_data = {'X_train':X_train, 'X_val':X_val, 'X_test':X_test}
        X_train_processed, X_val_processed, X_test_processed = self._normalization(row_data) # формат данных - pandas.Dataframe

        #  3. Балансировка тренировочных данных
        X_train_balanced, y_train_balanced = self.balance(X_train_processed, y_train)
        counter = dict(Counter(y_train_balanced))
        counter.setdefault(0, counter.pop(0))
        counter.setdefault(1, counter.pop(1))
        print(f"\nМиноритарный класс в TRAIN после балансировки: {round(100 * counter[1]/(counter[0]+counter[1]))}%  - {counter}")

        y_val_temp = np.array(y_val).ravel()
        counter = dict(Counter(y_val_temp))
        counter.setdefault(0, counter.pop(0))
        counter.setdefault(1, counter.pop(1))
        print(f"\nРаспределение классов в TEST:  {round(100 * counter[1]/(counter[0]+counter[1]))}%  - {counter}")

        print(f"Размерность данных после Preproccessing()\n"
              f"X_train_balanced - {X_train_balanced.shape}, X_val_processed - {X_val_processed.shape}, X_test_processed - {X_test_processed.shape}\n"
              f"y_train_balanced - {y_train_balanced.shape}, y_val - {y_val.values.ravel().shape}, y_test - {y_test.values.ravel().shape}")

        # 4.  Установка соответствий между столбцами данных и индикаторов
        columns_X = list(X_train_processed.columns)
        sets_X = [pd.DataFrame(set_X, columns=columns_X) for set_X in [X_train_balanced, X_val_processed, X_test_processed]]
        sets_y = [pd.DataFrame(set_y, columns=['target']) for set_y in [y_train_balanced, y_val, y_test]]
        sets_Xy = list(zip(sets_X, sets_y))
        sets_Xy = list(map(lambda x: pd.concat([x[0],x[1]], axis=1), sets_Xy))
        dict_Xy = {'Xy_train':sets_Xy[0], 'Xy_val':sets_Xy[1], 'Xy_test':sets_Xy[2]}
        dict_Xy = self.indicators_setup(dict_Xy)
        return dict_Xy
        return X_train_balanced, X_val_processed, X_test_processed, y_train_balanced, y_val.values.ravel(), y_test.values.ravel()

    def predicate(self):
        try:
            y = self.data[self.cols_res]
        except: y = None
        X = self.data.drop(columns=self.cols_drop)
        preprocessor = load(self.preprocessor)
        X_processed = preprocessor.transform(X)
        return X_processed,  y, X, self.data

