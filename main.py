import pandas as pd
import os
import numpy as np
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import joblib
import sys
import json
import io
import logging
import shutil
from validators import e_path, e_read, e_write, e_convert, e_file, e_fill_values, e_fill, e_scale_price, \
    e_scale_format, e_scale, e_preprocess, e_train, e_learn, e_save, e_load

def log():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    os.makedirs('logs', exist_ok=True)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    file_handler = logging.FileHandler("logs/main.log", mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def read_file(num):
    try:
        logging.info("Функция read_file вызвана с аргументом: %s", num)

        if num == 1:
            conditions_path = e_path(1)
            conditions = e_read(conditions_path)
            cf = e_convert(conditions)
            logging.debug("Данные из conditions загружены: %s", cf.head())
            return cf

        if num == 2:
            data_path = e_path(2)
            data = e_read(data_path)
            e_write("data/used_data.json", data)
            used_data = e_read("data/used_data.json")
            df = e_convert(used_data)
            conditions = read_file(1)
            type_to_keep = conditions["type"]

            if isinstance(type_to_keep, list):
                type_to_keep = type_to_keep[0]
            elif isinstance(type_to_keep, pd.Series):
                type_to_keep = type_to_keep.iloc[0]

            type_to_keep = int(type_to_keep)
            df = df[df["type"] == type_to_keep]

        logging.info("Отфильтрованные данные: %s", df.head())
        return df

    except Exception as e:
        e_file(e)

def fill(df, num):
    try:
        logging.info("Функция fill начата")
        columns_to_fill = ['sugar', 'varietals', 'region', 'color', 'producer', 'country']
        if num == 1:
            columns_to_fill = ['sugar', 'varietals', 'region', 'color', 'country']
        for column in columns_to_fill:
            df[column] = df[column].fillna('any')
        e_fill_values(df, columns_to_fill)
        logging.debug("Пробелы заполнены для столбцов: %s", columns_to_fill)
        return df
    except Exception as e:
        e_fill(e)

def scale(df):
    try:
        logging.info("Функция scale начата")
        df['price'] = np.log1p(df['price'])
        e_scale_price(df, "price")
        df = e_scale_format(df, "format")
        logging.debug("Масштабирование выполнено: %s", df.head())
        return df
    except Exception as e:
        e_scale(e)

def preprocess(df, num):
    try:
        logging.info("Функция preprocess начата")
        df = fill(df, num)
        df = scale(df)
        logging.debug("Preprocess завершён: %s", df.head())
        logging.debug(f"Данные после препроцессинга: {df.head()}")
        return df
    except Exception as e:
        e_preprocess(e)

def train(df):
    try:
        logging.info("Функция train начата")
        x = df.drop('name', axis=1)
        y = df['name']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        cat_features = ['producer', 'country', 'varietals', 'region', 'color', 'sugar']
        train_pool = Pool(data=x_train, label=y_train, cat_features=cat_features)
        feature_order = train_pool.get_feature_names()
        print("Порядок признаков в модели:", feature_order)
        test_pool = Pool(data=x_test, label=y_test, cat_features=cat_features)
        logging.info("Данные успешно разделены на обучающие и тестовые")
        # sample_weight = calculate_weights(user_prefs, X_train)
        return train_pool, test_pool
    except Exception as e:
        e_train(e)

def learn():
    try:
        logging.info("Программа запущена")
        train_pool, test_pool = train(preprocess(read_file(2), 2))
        logging.info("Функция learn начата")
        model = CatBoostClassifier(iterations=1, learning_rate=0.03, depth=6, l2_leaf_reg=14, border_count=32)
        # model = model.fit(train_pool, sample_weight=sample_weight)
        model = model.fit(train_pool)
        print("Vesa: ", model.feature_importances_)
        y_pred = model.predict(test_pool)
        f1 = f1_score(test_pool.get_label(), y_pred, average='weighted')
        logging.info("Обучение завершено с точностью: %.2f", f1)
        return model
    except Exception as e:
        e_learn(e)

def save(model, df=read_file(2)):
    try:
        model_path = os.path.join("logs", 'model1.pkl')
        joblib.dump(model, model_path)
        logging.info("Модель успешно сохранена")
        timestamp = pd.to_datetime('now').strftime('%d_%m_%H%M%S')
        df_path = os.path.join("logs", f"data_{timestamp}.json")
        df = df.to_dict(orient='records')
        with open(df_path, 'w', encoding='utf-8') as file:
            json.dump(df, file, ensure_ascii=False, indent=4)
        logging.info("Сохранение данных завершено")
    except Exception as e:
        e_save(e)

def load():
    try:
        logging.info("Загрузка модели начата")
        model = joblib.load('logs/model1.pkl')
        cf = read_file(1)
        cf = preprocess(cf, 1)
        cat_features = ['country', 'varietals', 'region', 'color', 'sugar']
        predict_pool = Pool(data=cf, cat_features=cat_features)
        probabilities = model.predict_proba(predict_pool)
        top_5_indices = np.argsort(probabilities, axis=1)[:, -5:]
        class_labels = model.classes_
        top_5_predictions = [
            [(class_labels[idx], probabilities[i][idx]) for idx in reversed(top_indices)]
            for i, top_indices in enumerate(top_5_indices)
        ]
        for i, preds in enumerate(top_5_predictions):
            logging.info(f"Прогноз для объекта {i + 1}: {preds}")
        timestamp = pd.to_datetime('now').strftime('%d_%m_%H%M%S')
        predictions_path = os.path.join("logs", f"top5_predictions_{timestamp}.json")
        predictions_to_save = [
            {
                "predictions": [{"class": label, "probability": prob} for label, prob in preds]
            }
            for i, preds in enumerate(top_5_predictions)
        ]
        with open(predictions_path, 'w', encoding='utf-8') as file:
            json.dump(predictions_to_save, file, ensure_ascii=False, indent=4)
        return predictions_path
    except Exception as e:
        e_load(e)

def copy(predictions_path, data_path="data/data.json"):
    try:
        with open(predictions_path, 'r', encoding='utf-8') as pred_file:
            predictions = json.load(pred_file)

        with open(data_path, 'r', encoding='utf-8') as data_file:
            data = json.load(data_file)

        matched_objects = []

        for prediction in predictions:
            for pred in prediction['predictions']:
                predicted_class = pred['class']
                matching_object = next((obj for obj in data if obj['name'] == predicted_class), None)
                if matching_object:
                    matched_objects.append(matching_object)

        with open("logs/result.json", 'w', encoding='utf-8') as output:
            json.dump(matched_objects, output, ensure_ascii=False, indent=4)

        logging.info("Сохранение данных завершено")
    except Exception as e:
        logging.error("Ошибка в copy: %s", e)

if __name__ == "__main__":
    log()
    save(learn())
    copy(load())
