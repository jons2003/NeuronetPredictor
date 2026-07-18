import asyncio
import sqlite3
import numpy as np
import pandas as pd
from pandas import DataFrame
from db import db
from openpyxl.workbook import Workbook
from environment import Env

class Df_getter:

    def __init__(self, data_db:str):
        self.data_db = data_db

    def _connect(self, vol_data):
        # Подключение к базе данных
        conn = sqlite3.connect(self.data_db)
        # Получение списка всех столбцов таблицы
        columns = list(map(lambda x: x[1], asyncio.run(db.connect("PRAGMA table_info(matches)", self.data_db))))
        # Удаление столбца даты и пустых столбцов HJ_loss_avg и HJ_loss_avg
        # columns = columns[1:]
        columns.remove('HJ_loss_avg')
        columns.remove('GJ_loss_avg')

        if vol_data == 'full':
            # Создание динамического запроса на отбор всех записей
            query = f"SELECT "
            [query := query + col.strip('\'') + ', '  for col in columns]
            query = list(query)
            query.pop(-2)
            query = ''.join(query)
            query += "FROM matches"

        elif vol_data == "dense":
            # Создание динамического запроса на отбор записей со всемя заполненными значениями
            query = "SELECT * FROM matches WHERE date IS NOT NULL"
            for column in columns: query = f"{query}\nAND {column} IS NOT NULL"

        data = pd.read_sql(query, conn)
        conn.close()
        return data

    def mask(self,vol_data='full', mask_name='simple', args=None):
        # vol_data = 'full' - все исходные данные, ' dense' - выбор только полностью заполненных записей
        # mask_name - имя метода с необходимыми количеством и параметрами столбцов Dataframe
        # args - аргументы метода 'mask_name'
        print("\n---- Этап 1. Шаблонирование данных под задачу (Df_getter().mask() ---- ")
        data = self._connect(vol_data)
        mask = getattr(Mask(data), mask_name)
        return mask(args=args)

class Mask:
    def __init__(self, data:DataFrame, config_file='/home/neuro_linux/configs/config.yaml'):
        self.data = data
        self.env = Env(config_file=config_file)
        # self.env.env_set(model_name=model_name)  # Инициализация окружения из файла конфигурации



    # Использование исходных данных (без синтеза)
    def _cols_raw(self):
        # Создание столбцов с относительным положением команд в турнирной таблице
        self.data['H_tourn'] = np.select(
            [(self.data['H_place'].notna()) & (self.data['num_place'].notna())],
            [self.data['H_place'] / self.data['num_place']],
            default=np.nan)
        self.data['G_tourn'] = np.select(
            [self.data['G_place'].notna() & self.data['num_place'].notna()],
            [self.data['G_place'] / self.data['num_place']],
            default=np.nan)
        # Список столбцов, исключаемых из исходных данных
        cols_exclude = ['date','h_res','g_res','H_place','G_place', 'num_place', 'HJ_loss_avg', 'GJ_loss_avg']
        return cols_exclude

    # Синтез одного столбца из двух парных
    def _merge_cols_norm(self, col1, col2):
            col_name = f"{col1}+{col2}"
            self.data[col_name] = np.select([(self.data[col1] == 0) & (self.data[col2] == 0),
                                             self.data[col1] == 0,
                                             self.data[col2] == 0,
                                             (self.data[col1].isna() | self.data[col2].isna())],
                                            [0.5, 0.0, 1, np.nan],
                                            default=(self.data[col1]) / (self.data[col1] + self.data[col2] + 10 ** -7))
            self.data = self.data.drop(columns=[col1, col2])

    # Синтетическое обьдинение всех парных столбцов в таблице
    def _cols_synthetic(self):
        columns = list(self.data.columns)
        columns.remove('H_place')
        columns.remove('G_place')
        lenght = len(columns)
        pairs = []
        for i in range(lenght):  # Создания списка парных столбцов для синтеза нового
            for j in range(i + 1, lenght):
                if columns[i][1:] == columns[j][1:]:
                    pairs.append([columns[i], columns[j]])
                    continue
        pairs = pairs[1:]
        pairs_list = []
        for pair in pairs:
            self._merge_cols_norm(pair[0], pair[1])
            pairs_list.append(f"{pair[0]}+{pair[1]}")
        # Создание столбцов с относительным положением команд в турнирной таблице
        self.data['H_tourn'] = np.select(
            [(self.data['H_place'].notna()) & (self.data['num_place'].notna())],
            [self.data['H_place'] / self.data['num_place']],
            default=np.nan)
        self.data['G_tourn'] = np.select(
            [self.data['G_place'].notna() & self.data['num_place'].notna()],
            [self.data['G_place'] / self.data['num_place']],
            default=np.nan)
        self.data = self.data.drop(columns=['H_place', 'G_place', 'num_place','HJ_draw+GJ_draw', 'HJ_loss+GJ_loss'])
        # Список столбцов, исключаемых из исходных данных
        cols_exclude = (['date', 'h_res', 'g_res'])
        return cols_exclude

    # Добавление пропущенных данных и создание столбцов-индикаторов
    def _indicators_sintetic(self):
        columns = list(self.data.columns)[9:]
        lenght = len(columns)
        # Заполнение пропусков и создание индикаторов
        for column in columns:
            self.data[f'ind_{column}'] = self.data[column].isna().astype(int)
            self.data[column] = self.data[column].fillna(self.data[column].median())

    # Создание целевой перемнной target нейросети
    def _targets(self, version='max_loss_fb'):
        if version == 'max_loss_fb':
            # 1 - если победил аутсайдер (по кэфам букмекера)
            # 0 - При победе лидера или ничьей
            self.data['target'] = 0
            for idx, row in self.data.iterrows():
                if ((row['h_res'] > row['g_res'] and row['b1'] == max(row['b1'], row['bX'], row['b2'])) or
                        (row['g_res'] > row['h_res'] and row['b2'] == max(row['b1'], row['bX'], row['b2']))):
                    self.data.at[idx, 'target'] = 1
        elif version == 'fact_res':     # Резерв для других целевых переменных
            pass


    def contr(self, args:dict):
        # 'data_synthetic' - True/False - использование синтезированных столбцов или сырых данных
        # 'target_version' - определление целевой переменной в def _targets()

        if args['data_synthetic'] == 'y':
            cols_exclude = self._cols_synthetic()  # Синтез столбцов данных в DataFrame self.data
        elif args['data_synthetic'] ==  'n':
            cols_exclude = self._cols_raw()
        if self.env('vol_data') == 'full': # Создание индикаторов пропущенных данных в предматчевой статистике
            self._indicators_sintetic()
        self._targets(version=args['target_version'])
        cols_res = ['target']

        cols_drop = cols_exclude + cols_res
        data = self.data
        print('\n****************   Predta на выходе Df_getter *****************\n')
        print(f"{data.info()}\ncols_drop = {cols_drop}\ncols_res = {cols_res}\n")
        # data.to_excel('/home/neuro_linux/core/test.xlsx')
        return data, cols_drop, cols_res













