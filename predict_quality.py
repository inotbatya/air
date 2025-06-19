from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Пути к модели и скалеру
MODEL_PATH = 'air_quality_model.keras'
SCALER_PATH = 'scaler.pkl'

# Проверка наличия файлов
if not os.path.exists(MODEL_PATH):
    logger.error(f"Модель не найдена по пути: {MODEL_PATH}")
    raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")

if not os.path.exists(SCALER_PATH):
    logger.error(f"Скалер не найден по пути: {SCALER_PATH}")
    raise FileNotFoundError(f"Скалер не найден по пути: {SCALER_PATH}")

# Загрузка модели и скалера
try:
    model = load_model(MODEL_PATH)
    logger.info("✅ Модель загружена")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    raise

try:
    scaler = joblib.load(SCALER_PATH)
    logger.info("✅ Скалер загружен")
except Exception as e:
    logger.error(f"Ошибка загрузки скалера: {e}")
    raise

# Ожидаемые признаки (включая погодные данные)
EXPECTED_FEATURES = [
    'co', 'no', 'no2', 'o3', 'so2', 'pm10', 'nh3',
    'temp', 'humidity', 'wind_speed', 'pressure', 'clouds'
]

@app.route('/api/predict_quality', methods=['POST'])
def predict_quality():
    """Предсказание уровня PM2.5 на основе входных данных"""
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

        # Преобразование и масштабирование
        input_features = np.array(input_values).reshape(1, -1)
        
        if input_features.shape[1] != scaler.n_features_in_:
            logger.warning(f"Размерность входных данных не соответствует ожидаемой: {scaler.n_features_in_}")
            
        scaled_features = scaler.transform(input_features)
        
        # Предсказание
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

@app.route('/api/check_model', methods=['GET'])
def check_model():
    """Диагностика состояния модели"""
    try:
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
