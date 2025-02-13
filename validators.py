import sys
import json
import pandas as pd
import logging

def e_path(num):
    path = ""
    if num == 2:
        path = "data/data.json"
    if num == 1:
        path = "data/conditions.json"
    return path

def e_read(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error("Файл не найден: %s", path)
        raise
    except json.JSONDecodeError:
        logging.error("Файл содержит некорректный JSON: %s", path)
        raise
    except IOError:
        logging.error("Ошибка в e_read при чтении файла: %s", path)
        raise

def e_write(path, data):
    try:
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except IOError:
        logging.error("Ошибка в e_write: %s", path)
        raise

def e_convert(data):
    try:
        if isinstance(data, dict):
            data = [data]
        return pd.DataFrame(data)
    except Exception as e:
        logging.error("Ошибка в e_convert: %s", e)
        raise

def e_file(e):
    logging.error("Ошибка в e_file: %s", e)
    sys.exit(1)

def e_fill_values(df, columns):
    for column in columns:
        if df[column].isnull().any():
            logging.error("Не удалось заполнить пропуски в e_fill_values: %s", column)
            raise

def e_fill(e):
    logging.error("Ошибка в e_fill: %s", e)
    sys.exit(1)

def e_scale_price(df, column):
    try:
        if column not in df:
            logging.error("Столбец отсутствует в данных: %s", column)
            raise
        if not pd.api.types.is_numeric_dtype(df[column]):
            logging.error("Столбец содержит некорректные данные: %s", column)
            raise
        if (df[column] < 0).any():
            logging.error("Столбец содержит отрицательные данные: %s", column)
            raise

    except KeyError as e:
        logging.error("Ошибка KeyError при валидации объёма: %s", e)
        raise
    except TypeError as e:
        logging.error("Ошибка типа данных при валидации объёма: %s", e)
        raise
    except Exception as e:
        logging.error("Ошибка при валидации объёма: %s", e)
        raise

def e_scale_format(df, column):
    try:
        if column not in df:
            logging.error("Столбец отсутствует в данных: %s", column)
            raise
        if not pd.api.types.is_numeric_dtype(df[column]):
            logging.error("Столбец содержит некорректные данные: %s", column)
            raise
        df[column] = df[column].apply(lambda x: x / 1000 if x > 100 else x)
        df[column] = df[column].apply(lambda x: x if 0.1 <= x <= 10 else 0.75)
        if (df[column] < 0).any():
            logging.error("Столбец содержит отрицательные данные: %s", column)
            raise
        return df
    except KeyError as e:
        logging.error("Ошибка KeyError при валидации объёма: %s", e)
        raise
    except TypeError as e:
        logging.error("Ошибка типа данных при валидации объёма: %s", e)
        raise
    except Exception as e:
        logging.error("Ошибка при валидации объёма: %s", e)
        raise

def e_scale(e):
    logging.error("Ошибка в e_scale: %s", e)
    sys.exit(1)

def e_preprocess(e):
    logging.error("Ошибка в e_preprocess: %s", e)
    sys.exit(1)

def e_train(e):
    logging.error("Ошибка в e_train: %s", e)
    sys.exit(1)

def e_learn(e):
    logging.error("Ошибка в e_learn: %s", e)
    sys.exit(1)

def e_save(e):
    logging.error("Ошибка в e_save: %s", e)
    sys.exit(1)

def e_load(e):
    logging.error("Ошибка в e_load: %s", e)
    sys.exit(1)