import re
import sys
from pathlib import Path
# Добавляем путь к папке db в sys.path
sys.path.append(str(Path(__file__).parent))
import aiosqlite
from queries import Queries as q

async def connect(query:str, file:str, param=None):
    async with aiosqlite.connect(file) as db:
        cursor = await db.execute(query, param)
        await db.commit()
        result = await  cursor.fetchall()
        return result

async def create_dynamic_table(table:str, columns:dict, file:str):
    columns_def = ', '.join([f'{col} {dtype}' for col, dtype in columns.items()])
    await connect(f'''CREATE TABLE IF NOT EXISTS {table} ({columns_def})''', file)

async def insert_matches(file: str, table: str, data: dict):
    # Создания словаря номеров столбцов по названию
    col_nums = {}
    [col_nums.setdefault(key, num) for num, key in enumerate(q.columns.keys())]
    # Генерация SQL-запроса
    columns = ', '.join(col_nums.keys())
    placeholders = ', '.join(['?' for _ in col_nums])
    query = f'INSERT INTO {table} ({columns}) VALUES ({placeholders})'
    # Создание списка данных по номерам столбцов
    data_list = [None] * len(col_nums.keys())
    for key, val in data.items():
        if key in col_nums:
            index = col_nums[key]
            data_list.pop(index)
            data_list.insert(index, val)
    await connect(query, file, data_list)

async def update_stat(file: str, table: str, record: dict):
    id = record.pop("id", None)
    tourn = record.pop('tourn', None)
    query = f'''UPDATE matches SET tourn = ? WHERE id= ?'''
    await connect(query, file, (tourn, id))
    print(record)
    query_string = ', '.join([f'{key} = {value}' for key, value in record.items()])
    query = f'UPDATE {table} SET {query_string} WHERE id = {id}'
    print(query)
    await connect(query, file)

async def update_res(file: str, table: str, record: dict):
    query = f'''UPDATE matches SET h_res = ?,  g_res = ?  WHERE host = ? AND guest = ?'''
    await connect(query, file, (record['h_res'], record['g_res'], record['host'],record['guest']))




# async def change_record_db(order_info:list, file):
#     if order_info[10] == 'NEW':
#         num = await connect(queries.COUNT, {'number': order_info[1], 'status': order_info[10]}, file)
#         if num[0][0] == 0: await connect(queries.ADD_REC, order_info, file)
#     elif order_info[-1] == 'FILLED' or order_info[-1] == 'CANCELED' : await connect(queries.DEL_NUM, {'number': order_info[1]}, file)
#
# async  def remove_order(number_order:int, file:str):
#     await connect(queries.DEL_NUM, {'number': number_order}, file)
#
# async def temp_filling(record:list):
#     global  LONG_NUM, LONG_AMOUNT, SHORT_NUM, SHORT_AMOUNT
#     lst = await connect(queries.FIND_ORDER, {'number': record[1]}, file=Env.JOURNAL_DB)
#     if record[10] == "FILLED" or record[10] == 'CANCELED' or (record[10] == 'NEW' and len(lst) <= 1):
#         if record[6] == 'STOP' and record[10] == 'FILLED':
#             if record[7] == 'LONG':
#                 LONG_NUM += float(record[3])
#                 LONG_AMOUNT += float(record[3]) * float(record[9]) if LONG_NUM > 0 else 0
#             elif record[7] == 'SHORT':
#                 SHORT_NUM += float(record[3])
#                 SHORT_AMOUNT += float(record[3]) * float(record[9]) if SHORT_NUM > 0 else 0
#             AVG_LONG = round(LONG_AMOUNT / LONG_NUM, 4) if LONG_NUM > 0 else 0
#             AVG_SHORT = round(SHORT_AMOUNT/SHORT_NUM, 4) if SHORT_NUM > 0  else 0
#             rec = record+[LONG_NUM, round(LONG_AMOUNT, 4), AVG_LONG, SHORT_NUM, round(SHORT_AMOUNT, 4), AVG_SHORT]
#
#         elif (record[6] == 'TAKE_PROFIT_MARKET' or record[6] == 'STOP_MARKET') and record[10] == 'FILLED':
#             if record[7] == 'LONG':
#                 LONG_NUM -= float(record[3])
#                 LONG_AMOUNT -= (float(record[3]) * float(record[9])) if LONG_NUM > 0 else LONG_AMOUNT
#             elif record[7] == 'SHORT':
#                 SHORT_NUM -= float(record[3])
#                 SHORT_AMOUNT -= float(record[3]) * float(record[9]) if SHORT_NUM > 0 else SHORT_AMOUNT
#             AVG_LONG = round(LONG_AMOUNT / LONG_NUM, 4) if LONG_NUM > 0 else 0
#             AVG_SHORT = round(SHORT_AMOUNT / SHORT_NUM, 4) if SHORT_NUM > 0 else 0
#             rec = record + [LONG_NUM, round(LONG_AMOUNT, 4), AVG_LONG, SHORT_NUM, round(SHORT_AMOUNT, 4), AVG_SHORT]
#         else:
#             rec = record + ['','','','','','']
#         await connect(queries.ADD_REC_TEMP, rec, file=Env.JOURNAL_DB)
#         rec = await connect(queries.SELECT_ALL_TEMP, file=Env.JOURNAL_DB)
#         return rec[-1]
#
# async def finish_filling(finish: list):
#     for record in finish:
#         await connect(queries.ADD_REC_FINISH, record, file=Env.JOURNAL_DB)
#     result = await connect(queries.SELECT_ALL_FINISH, file=Env.JOURNAL_DB)
#     return result
#
# async def table_finish(temp_record:list):
#     if temp_record:
#         finish = []
#         active = list(await connect(queries.READ_DATA, file=Env.ACTIVE_DB))
#         finish.append(list(temp_record))
#         for record in active:
#             finish.append(['']+list(record)+['','','','','',''])
#         table_finish = await finish_filling(finish)
#         return table_finish # список записей в формате БД