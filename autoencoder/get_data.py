import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sqlite3
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from joblib import dump, load
import matplotlib.pyplot as plt

'''
Группы столбцов по назначению
Испльзуются:
 - heads: Инфармация о матче 
 - score: категории результата и предсказания бука - фильтрация событий (успешность прогноза бука) 
 - coeffs+stats: кэфы бука и предматчевая статистика -  для обучения автоэнкодера
 - drops_corr+drops_empty: избыточные данные (кореллируют с 'HJ_win', 'HJ_draw', 'HJ_loss') или 
   пустые и предварительные (до синтеза новых)
'''
heads = ['date', 'league', 'host', 'guest']
match = ['league', 'host', 'guest']
res = ['h_res', 'g_res']
coeffs = ['b1', 'bX', 'b2']
stats = ['HA_win', 'HA_draw', 'HA_loss', 'HA_goal_avg', 'HA_loss_avg',
         'HA_yellow', 'HA_red', 'HA_corner', 'HA_penalty', 'HA_offside',
         'GA_win', 'GA_draw', 'GA_loss', 'GA_goal_avg', 'GA_loss_avg',
         'GA_yellow', 'GA_red', 'GA_corner', 'GA_penalty', 'GA_offside',
         'HJ_win', 'HJ_draw', 'HJ_loss', 'HJ_goal_avg',
         'HJ_yellow', 'HJ_red', 'HJ_corner', 'HJ_penalty', 'HJ_offside',
         'GJ_goal_avg',
         'GJ_yellow', 'GJ_red', 'GJ_corner', 'GJ_penalty', 'GJ_offside',
         'h_tourn', 'g_tourn']
stats_empty = ['GJ_win', 'GJ_draw', 'GJ_loss']
stats_any = ['HJ_loss_avg', 'GJ_loss_avg', 'H_place', 'G_place', 'num_place']
score = ['sc_res', 'sc_pred']


def sapiro(data):
    import scipy

    stat, p = scipy.stats.shapiro(data)  # тест Шапиро-Уилк
    print('Statistics=%.3f, p-value=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Принять гипотезу о нормальности')
    else:
        print('Отклонить гипотезу о нормальности')

def hist(df:pd.DataFrame, values:list):
    """Построение гистограммы распределения признаков из списка values
        args:
                df - все данные
                values - список наименований признаков
    """
    data = df
    colors = ['lightblue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']

    for value in values:
        plt.figure(figsize=(10, 6))
        plt.title(f'Распределение значений величин')
        plt.hist(data[value], bins=60, alpha=0.3, label=value, color=colors[1], edgecolor='black')
        plt.xlabel(f'Величина {value}')
        plt.ylabel(f'Частота {value}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

def conf_matrix(df:pd.DataFrame):
    #  Расчет матрицы ошибок прогнозов букмекера
    cm = confusion_matrix(df['res'], df['pred'])
    classes = ['b1', 'bX', 'b2']  # замените на ваши классы
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    # Добавляем названия строк и столбцов
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'

def cut_data(df:pd.DataFrame, columns:list):
    quants = [i/100 for i in range(70, 101, 1)]
    df[columns].quantile(quants)

def correlation(df):
    df=df.drop(heads, axis=1)
    matrix = df.corr()
    # Визуализируем тепловую карту
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Матрица корреляций')
    plt.show()
    # Преобразуем матрицу корреляции в удобный формат
    corr_pairs = matrix.unstack().sort_values(ascending=False)
    # Убираем диагональ (корреляция переменной с самой собой = 1)
    corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
    # Смотрим топ-10 наиболее коррелированных пар
    print("Топ-10 положительных корреляций:")
    print(corr_pairs.head(20))
    # Смотрим топ-10 наиболее отрицательно коррелированных пар
    print("\nТоп-10 отрицательных корреляций:")
    print(corr_pairs.tail(20))
    mask = ((matrix>0.5)&(matrix<1.0)).any(axis=1)
    print(matrix[mask])

class Betting:

    def __init__(self, file_db='/home/neuro_linux/autoencoder/data/summary.db', is_succsess=0):
        """ Arg: is_succsess = 1/0 - удачный/ошибочный прогноз букмекера"""
        self.df = self.data_syntetic(pd.read_sql('SELECT * FROM matches', sqlite3.connect(file_db)), is_succsess)

    def clip_outliers_median(self, df, multiplier=2):
        """
        Обрезает выбросы по значению multiplier * медиана для каждого столбца

        Parameters:
        df - DataFrame
        multiplier - множитель для медианы (по умолчанию 2)
        """
        df_clean = df.copy()
        for column in df_clean.select_dtypes(include=[np.number]).columns:
            median_val = df_clean[column].median()
            upper_limit = median_val * multiplier
            # Обрезаем выбросы сверху
            df_clean[column] = np.where(df_clean[column] > upper_limit, upper_limit, df_clean[column])
        return df_clean



    def data_syntetic(self, df: pd.DataFrame, is_succsess):
        #  Cинтез новых столбцов турнирного положения
        df['h_tourn'] = df['H_place'] / df['num_place']
        df['g_tourn'] = df['G_place'] / df['num_place']

        #  Создание столбцов фактического и предсказанного результата матчей
        min_columns = df[['b1', 'bX', 'b2']].min(axis=1)
        conditions_res = [(df['h_res'] > df['g_res']), (df['h_res'] == df['g_res']), (df['h_res'] < df['g_res'])]
        conditions_pred = [(df.b1 == min_columns), (df.bX == min_columns), (df.b2 == min_columns)]

        select = [0, 1, 2]
        df['sc_res'] = np.select(conditions_res, select)
        df['sc_pred'] = np.select(conditions_pred, select)
        df['succses'] = np.where(df['sc_res'] == df['sc_pred'], 1, 0)
        # Выбираем события с ошибочными прогнозами букмекера
        df_success = df[df['succses'] == is_succsess]
        df_success = df_success.drop(stats_empty + res + stats_any + score + ['succses'], axis=1)
        ''''---Обрезаем выбросы по удвоенному значению медианы---'''
        df_cleaned = self.clip_outliers_median(df_success, multiplier=2)
        return df_cleaned

    def fit_normalization(self, df: pd.DataFrame):

        # Разделение полных данных на train и val ДО масштабирования
        X_train, X_val = train_test_split(df, test_size=0.2, random_state=42)
        numeric_features = coeffs + stats

        # Проверим разные скалеры
        for scaler_name, scaler in [
            ('StandardScaler', StandardScaler()),
            # ('RobustScaler', RobustScaler()),
            # ('MinMaxScaler', MinMaxScaler())
        ]:
            num_transform = Pipeline([
                ('imputer', SimpleImputer(missing_values=np.nan,strategy='median')),
                ('scaler', scaler)
            ])
            cat_transform = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Отключаем разреженные матрицы
            ])
            preprocessor = ColumnTransformer([
                ('num', num_transform, numeric_features),
                # ('cat', cat_transform, match)
            ])
            # # Обучение преобразователя на train-данных c сохранением предобработчика после обучения
            X_train_scaled = preprocessor.fit_transform(X_train)

            dump(preprocessor, 'preprocessor.proc')
            # Возвращаем train из array к виду DataFrame
            # X_train_scaled = pd.DataFrame(X_train_scaled, columns=numeric_features, index=X_train.index.to_list())
            # X_train_scaled = X_train[heads].join(X_train_scaled)

            # Трансформируем val на обученном преобразователе
            X_val_scaled = preprocessor.transform(X_val)

            # X_val_scaled = pd.DataFrame(X_val_scaled, columns=numeric_features, index=X_val.index)
            # X_val_scaled = pd.concat([X_val[heads], X_val_scaled], axis=1)

            # print(X_train_scaled)
            # print(X_val_scaled)
            # print(X_train_scaled.dtypes.equals(X_val_scaled.dtypes))

            return X_train_scaled, X_val_scaled

#
# bet = Betting()
# data = bet.fit_normalization(bet.df)
# print(data)

# print(df.info())

# df_median = df.copy()
# df_median[coeffs+stats] = df[coeffs+stats].apply(lambda x: x.fillna(np.nanmean(x, axis=0)))
# print(df_median['HA_yellow'])
# df.describe([.5,.75,.95]).to_excel('res.xlsx')


# print(df['b1'].max())
# histo(df, ['b1'])
# df_end = normalization(df_res)


