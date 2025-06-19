from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
import logging

app = Flask(__name__)
CORS(app)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = '93be52922464c8b6d8dc69c14553ab05'
lat, lon = 53.354, 83.763  # Томск, Россия
MODEL_PATH = 'air_quality_model.h5'
SCALER_PATH = 'air_quality_scaler.pkl'

# Проверка наличия файлов модели и скалера
if not os.path.exists(MODEL_PATH):
    logger.error(f"Модель не найдена по пути: {MODEL_PATH}")
    raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")

if not os.path.exists(SCALER_PATH):
    logger.error(f"Скалер не найден по пути: {SCALER_PATH}")
    raise FileNotFoundError(f"Скалер не найден по пути: {SCALER_PATH}")

# Загрузка модели с метриками
try:
    model = load_model(MODEL_PATH, compile=False)
    # Если модель требует компиляции, добавьте нужные метрики/оптимизатор
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    logger.info("✅ Модель загружена и скомпилирована")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    raise

try:
    scaler = joblib.load(SCALER_PATH)
    logger.info("✅ Скалер загружен")
except Exception as e:
    logger.error(f"Ошибка загрузки скалера: {e}")
    raise

# Ожидаемые признаки для предсказания (без PM2.5, т.к. это целевая переменная)
EXPECTED_FEATURES = ['co', 'no', 'no2', 'o3', 'so2', 'pm10', 'nh3']

@app.route('/api/fetch_air_quality', methods=['GET'])
def fetch_air_quality():
    url = f'https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}'
    try:
        response = requests.get(url, timeout=10).json()
        
        if 'list' not in response or not response['list']:
            logger.error("Нет данных в ответе API")
            return jsonify({"error": "Нет данных в ответе API"}), 502

        components = response['list'][0]['components']
        main = response['list'][0]['main']

        # Проверка наличия всех необходимых компонентов 
        missing_components = [comp for comp in EXPECTED_FEATURES + ['pm2_5'] if comp not in components]
        if missing_components:
            logger.warning(f"Отсутствуют компоненты в ответе API: {missing_components}")

        data_to_save = pd.DataFrame([{
            'co': components.get('co', np.nan),
            'no': components.get('no', np.nan),
            'no2': components.get('no2', np.nan),
            'o3': components.get('o3', np.nan),
            'so2': components.get('so2', np.nan),
            'pm2_5': components.get('pm2_5', np.nan),
            'pm10': components.get('pm10', np.nan),
            'nh3': components.get('nh3', np.nan),
            'aqi': main.get('aqi', np.nan)
        }])

        # Сохранение данных с проверкой
        try:
            file_exists = os.path.isfile('air_quality_data.csv')
            data_to_save.to_csv('air_quality_data.csv', mode='a', header=not file_exists, index=False)
            logger.info("Данные успешно сохранены")
        except Exception as e:
            logger.error(f"Ошибка сохранения данных: {e}")

        return jsonify(response)

    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка запроса к API: {e}")
        return jsonify({"error": "Ошибка запроса к API", "details": str(e)}), 503
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict_quality', methods=['POST'])
def predict_quality():
    try:
        data = request.get_json(force=True)
        
        if 'components' not in data:
            return jsonify({"error": "Отсутствует ключ 'components' в JSON"}), 400
            
        features = data['components']
        
        # Проверка наличия всех требуемых признаков
        missing_features = [feat for feat in EXPECTED_FEATURES if feat not in features]
        if missing_features:
            return jsonify({"error": f"Отсутствуют признаки: {missing_features}"}), 400

        # Проверка валидности значений
        try:
            input_values = [float(features[feat]) for feat in EXPECTED_FEATURES]
        except ValueError as e:
            return jsonify({"error": "Все значения должны быть числами", "details": str(e)}), 400

        input_features = np.array(input_values).reshape(1, -1)
        
        # Проверка соответствия размерности скейлера
        if input_features.shape[1] != scaler.n_features_in_:
            logger.warning(f"Размерность входных данных не соответствует ожидаемой: {scaler.n_features_in_}")
            
        scaled_features = scaler.transform(input_features)
        
        # Предсказание с обработкой возможных ошибок
        try:
            prediction = model.predict(scaled_features, verbose=0)[0][0]
        except Exception as e:
            logger.error(f"Ошибка предсказания модели: {e}")
            return jsonify({"error": "Ошибка предсказания модели", "details": str(e)}), 500

        return jsonify({
            "predicted_pm2_5": round(float(prediction), 2),
            "model_input": {f: float(v) for f, v in zip(EXPECTED_FEATURES, input_values)},
            "model_info": {
                "input_shape": model.input_shape,
                "output_shape": model.output_shape
            }
        })

    except Exception as e:
        logger.error(f"Ошибка в обработке запроса: {e}")
        return jsonify({"error": str(e)}), 500

# Добавленный эндпоинт для проверки модели
@app.route('/api/check_model', methods=['GET'])
def check_model():
    """Проверка состояния модели и скалера"""
    try:
        # Пример тестового ввода
        test_input = np.random.rand(1, len(EXPECTED_FEATURES))
        scaled_test = scaler.transform(test_input)
        prediction = model.predict(scaled_test, verbose=0)
        
        return jsonify({
            "status": "ok",
            "model_summary": str(model.to_json()),
            "scaler_features": scaler.n_features_in_,
            "test_prediction": float(prediction[0][0]),
            "input_example": {f: float(v) for f, v in zip(EXPECTED_FEATURES, test_input[0])}
        })
    except Exception as e:
        return jsonify({"status": "error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
