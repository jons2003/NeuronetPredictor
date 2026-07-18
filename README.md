# Прогнозирование спортивных исходов с использованием нейросетей

Проект представляет собой комплексную систему для анализа и прогнозирования результатов футбольных матчей на основе предматчевых коэффициентов букмекеров и статистических данных. Включает в себя:

- Предобработку данных (нормализацию, создание индикаторов пропусков, балансировку классов)
- Обучение нейросетевых моделей (бинарная классификация – победа аутсайдера или нет)
- Кастомизированную функцию потерь (Focal Loss с дополнительным штрафом за ложноположительные срабатывания)
- Многопроцессорный перебор гиперпараметров (alpha, gamma, alpha_fp) для нахождения оптимальной конфигурации
- Автоэнкодер для выявления аномалий в прогнозах букмекеров

Код написан на Python с использованием TensorFlow/Keras, scikit-learn, SQLite и других библиотек.

---

## Особенности

- **Гибкая конфигурация** через YAML-файлы (пути к БД, параметры модели, диапазоны гиперпараметров)
- **Разделение данных** на тренировочную, валидационную и тестовую выборки со стратификацией
- **Балансировка классов** (undersampling, oversampling, SMOTE) для борьбы с дисбалансом
- **Создание индикаторов пропусков** – модель явно узнает, какие данные отсутствовали
- **Синтез новых признаков** (отношение парных статистик, турнирное положение)
- **Многопоточный запуск** для параллельного перебора гиперпараметров Focal Loss
- **Сохранение результатов** в SQLite-журнал и сохранение лучших моделей (формат `.keras`)

---

## Структура проекта

```
neuro_linux/
├── autoencoder/               # Автоэнкодер для анализа ошибок букмекеров
│   ├── ae_model.py            # Архитектура автоэнкодера
│   ├── get_data.py            # Загрузка и подготовка данных для автоэнкодера
│   └── start.py               # Обучение автоэнкодера и сравнение ошибок
├── configs/                   # Конфигурационные файлы (YAML)
│   ├── config.yaml            # Основной конфиг
│   └── config1.yaml           # Альтернативный конфиг (для тестов)
├── core/                      # Основная логика (первая версия)
│   ├── custom/                # Кастомные метрики и функции потерь
│   │   ├── custom_focal_loss.py
│   │   ├── custom_metric_precision1.py
│   │   └── ...
│   ├── get_data.py            # Извлечение данных из БД, маскирование
│   ├── model.py               # Построение и обучение модели
│   ├── prepocessing.py        # Предобработка, нормализация, балансировка
│   ├── start.py               # Точка входа для перебора гиперпараметров
│   └── task_manager.py        # Оркестрация этапов (данные -> препроцессинг -> модель)
├── core_new/                  # Вторая версия (с использованием DenseFeatures)
│   ├── custom/                # Аналогичные кастомные функции
│   ├── get_data.py
│   ├── model.py               # Использует tf.keras.layers.DenseFeatures
│   ├── prepocessing.py
│   └── predict.py             # Инференс модели на новых данных
├── db/                        # Работа с SQLite (asyncio, запросы)
│   ├── db.py                  # Асинхронные функции подключения
│   ├── queries.py             # SQL-запросы
│   └── temp.py                # Вспомогательные скрипты
├── sheduler/                  # Запуск перебора гиперпараметров
│   ├── start.py               # Многопроцессорный пул
│   ├── task_manager.py        # Обёртка над core_new (или core) для планировщика
│   └── journal_analize.py     # Визуализация результатов (3D-графики)
├── environment.py             # Класс для чтения конфигурации и работы с окружением
└── requirements.txt           # (нужно создать, в коде нет, но можно указать зависимости)
```

---

## Установка и зависимости

Проект требует Python 3.8+.

Основные зависимости:

- tensorflow >= 2.13
- scikit-learn
- pandas
- numpy
- sqlite3 (встроенная)
- matplotlib, seaborn, plotly (для визуализации)
- imbalanced-learn (для SMOTE)
- pyyaml
- joblib
- aiosqlite (для асинхронной работы с БД)

Установите их с помощью pip:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn plotly imbalanced-learn pyyaml joblib aiosqlite
```

---

## Конфигурация

Все настройки хранятся в `configs/config.yaml`. Основные параметры:

```yaml
database:
  data_db:      /path/to/summary.db      # база с историческими матчами
  prematch_db:  /path/to/prematch.db     # база с предматчевыми данными (для прогнозов)
  models_dir:   /path/to/models/         # директория для сохранения моделей
  journal_db:   /path/to/journal.db      # журнал результатов перебора

DF_getter:
  vol_data:     full                     # 'full' или 'dense' (только полные записи)
  mask_name:    contr                    # метод фильтрации данных
  args_mask:
    data_synthetic: y                    # использовать синтезированные признаки
    target_version: max_loss_fb          # способ формирования целевой переменной

Preprocessing:
  balance_method: SMOTE                  # undersample / oversample / SMOTE / n
  categoricl_features: n                 # использовать ли категориальные (лига, команды)

Model:
  compile:
    focal_params:
      alpha:     0.95                    # вес класса 1 в Focal Loss
      gamma:     1.0                     # параметр фокусировки
      alpha_fp:  10.0                    # штраф за ложноположительные
    focal_range:
      alpha_range:  1.0 9.0 0.5          # диапазон перебора (начало, конец, шаг)
      gamma_range:  1.0 5.0 0.5
      alpha_fp_range: 1.0 10.0 0.5
```

---

## Использование

### 1. Обучение модели с заданными гиперпараметрами

Для запуска обучения одной модели используйте `TaskManager`:

```python
from sheduler.task_manager import TaskManager
from environment import Env

env = Env('configs/config.yaml')
env.env_set()

fit_params = {'epochs': 200, 'batch_size': 32, 'class_weight': {0: 1, 1: 1}}
focal_params = {'alpha': 1.5, 'gamma': 3.0, 'alpha_fp': 10.0}

task = TaskManager('my_model')
result = task.create_model(fit_params, focal_params)
print(result)  # (val_TP, val_FP, val_precision, val_recall, test_TP, test_FP, test_precision, test_recall)
```

Модель и препроцессор сохраняются в папку `models_dir` под именем `my_model.keras` и `my_model.proc`.

### 2. Перебор гиперпараметров (поиск оптимальных alpha, gamma, alpha_fp)

Запустите `sheduler/start.py` (или `core/start.py`). Он:

- Прочитает диапазоны из `config.yaml`
- Создаст пул задач для всех комбинаций
- Запустит их параллельно (количество процессов ограничено семафором)
- Сохранит результаты в `journal.db`

```bash
python sheduler/start.py
```

### 3. Визуализация результатов

Используйте `sheduler/journal_analize.py` для построения 3D-графиков зависимости точности от гиперпараметров.

```bash
python sheduler/journal_analize.py
```

### 4. Прогнозирование на новых данных

Для предсказания используйте `core_new/predict.py` (или адаптируйте `TaskManager.predicate_model`). Необходимо загрузить обученный препроцессор и модель.

```python
from core_new.get_data import Df_getter
from core_new.prepocessing import Preproccessing
from core_new.predict import Predicate

# Загрузка данных из prematch_db
df_getter = Df_getter('/path/to/prematch.db')
predata = df_getter.mask(vol_data='full', mask_name='contr',
                         args={'data_synthetic': 'y', 'target_version': 'max_loss_fb'})

# Предобработка (без балансировки и разделения)
preproc = Preproccessing(predata=predata, preproc_file='models/my_model.proc')
X_processed, y, X_raw, _ = preproc.predicate()

# Инференс
predictor = Predicate(predata=(X_processed, y, X_raw, None), model_path='models/my_model.keras')
focal_params = {'alpha': 1.5, 'gamma': 3.0, 'alpha_fp': 10.0}
out = predictor.predicate(focal_params)
print(out)
```

---

## Автоэнкодер (autoencoder)

Отдельный модуль для анализа качества прогнозов букмекеров. Обучает автоэнкодер на матчах, где букмекер ошибся, и затем сравнивает ошибки реконструкции на правильных и неправильных прогнозах. Это позволяет выявить, какие матчи являются "аномальными" с точки зрения букмекерских коэффициентов.

### Запуск автоэнкодера

```bash
cd autoencoder
python start.py
```

Будут построены графики распределения ошибок и выполнены статистические тесты.

---

## Примечания

- В проекте присутствуют две реализации основной модели: `core` (использует `Model` с несколькими входами и `Concatenate`) и `core_new` (использует `DenseFeatures` для удобной работы с категориальными признаками). Рекомендуется использовать `core_new` как более современную.
- Для работы требуется база данных SQLite с таблицей `matches`, содержащей исторические матчи, коэффициенты и статистику. Пример схемы см. в `db/queries.py`.
- При переборе гиперпараметров используется многопроцессорность, но каждый процесс создает свою копию модели – учтите потребление памяти.
- Функция потерь `focal_precision_loss` является ключевой для обучения, она штрафует ложноположительные срабатывания, что критично для задачи предсказания редких событий (победа аутсайдера).

---


