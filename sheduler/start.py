import sqlite3, time
from multiprocessing import Manager, Pool
import logging
import multiprocessing as mp

from markdown_it.cli.parse import print_heading

from environment import Env
from sheduler.task_manager import TaskManager
# from keras import backend as K
import random



# Глобальные объекты синхронизации (создаются в главном процессе)
manager = Manager()
sem = manager.Semaphore(16)  # Используем менеджер для семафора
lock = manager.Lock()  # И для блокировки
env_dic = {}
#
def connect(query, file_db, param=None):
    """Безопасное подключение к SQLite"""
    try:
        with sqlite3.connect(file_db) as db:
            db.execute("PRAGMA journal_mode=WAL")
            cursor = db.execute(query, param) if param else db.execute(query)
            db.commit()
            return cursor.fetchall()
    except sqlite3.Error as e:
        logging.error(f"SQLite error: {e}")
        raise
    finally:
        # cursor.close()
        db.close()


def init_db(file_db:str):
    """Инициализация таблицы в базе данных"""
    connect("""
    CREATE TABLE IF NOT EXISTS journal
    (id INTEGER PRIMARY KEY AUTOINCREMENT,
     val_TP INTEGER, val_FP INTEGER, val_precision REAL, val_recall REAL,
     test_TP INTEGER, test_FP INTEGER, test_precision REAL, test_recall REAL, 
     model TEXT,
     alpha REAL, gamma REAL, alpha_fp REAL)
    """, file_db=file_db)


def run_proc(args):
    """Функция, выполняемая в каждом процессе"""
    model_name, fit_params, focal_params = args  # Убрали sem и lock из аргументов

    with sem:  # Используем глобальный семафор
        try:
            res = TaskManager(model_name=model_name).create_model(fit_params, focal_params)

            print('res=', res)
            # Подготовка данных
            values = (*res,
                      model_name,
                      focal_params['alpha'],
                      focal_params['gamma'],
                      focal_params['alpha_fp'])

            # Критическая секция с глобальной блокировкой
            with lock:
                connect(
                    """INSERT INTO journal
                    (val_TP, val_FP, val_precision, val_recall,
                     test_TP, test_FP, test_precision, test_recall,
                     model, 
                     alpha, gamma, alpha_fp)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    env_dic['journal_db'],
                    param=values
                )

            return True
        except Exception as e:
            logging.error(f"Error in process: {e}")
            return False


def start():
    global env_dic
    # Получение параметров модели
    config_file = '/home/neuro_linux/configs/config.yaml'
    fit_params = {'epochs': 200, 'batch_size': 32, 'class_weight': {0: 1, 1: 1}}
    env = Env(config_file=config_file)
    env_dic = env.yaml_get()
    env.env_set()
    # Инициализация БД
    init_db(env_dic['journal_db'])

    # Создаем список параметров
    task_args = []
    num = 0
    for n in range(1):
        for i in range(*map(lambda x: int(x*10), env_dic['alpha_range'])):
            alpha = i / 10
            for j in range(*map(lambda x: int(x*10), env_dic['gamma_range'])):
                gamma = j / 10
                for k in range(*map(lambda x: int(x*10), env_dic['alpha_fp_range'])):
                    alpha_fp = k / 10
                    task_args.append((
                        f"model_{num}",
                        fit_params,
                        {'alpha': alpha, 'gamma': gamma, 'alpha_fp': alpha_fp}
                    ))
                    num += 1
    random.shuffle(task_args)

    # single_model(task_args[0])

    # Запуск процессов
    start_time = time.time()

    # Используем фиксированное количество процессов
    cpu_count = mp.cpu_count()
    max_processes = 22

    with Pool(processes=max_processes) as pool:
        results = pool.map(run_proc, task_args)

    logging.info(f"Успешно завершено: {sum(results)} процессов")
    logging.info(f"Время выполнения: {time.time() - start_time:.2f} секунд")

#


def single_model(args):
    model_name, fit_params, focal_params = args
    focal_params = {'alpha': 7, 'gamma':5.0, 'alpha_fp': 50}
    print(fit_params, focal_params)
    res = TaskManager(model_name='model_name').create_model(fit_params, focal_params)

start()
