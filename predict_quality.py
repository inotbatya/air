import numpy as np
from tensorflow.keras.models import load_model
from joblib import load

# Загружаем модель и масштабировщик
model_file = 'air_quality_model.keras'
scaler_file = 'scaler.pkl'

try:
    model = load_model(model_file)
    scaler = load(scaler_file)
    print(f"Модель загружена из '{model_file}', масштабировщик загружен из '{scaler_file}'.")
except Exception as e:
    print(f"Ошибка при загрузке модели или масштабировщика: {e}")
    exit()

# Функция для предсказания качества воздуха
def predict_air_quality(input_data):
    """
    input_data: список или массив с параметрами [co, no, no2, o3, so2, pm10, nh3]
    Возвращает предсказанное значение PM2.5.
    """
    try:
        # Преобразуем данные в нужный формат
        input_data = np.array(input_data).reshape(1, -1)
        # Масштабируем входные данные
        input_data_scaled = scaler.transform(input_data)
        # Делаем предсказание
        prediction = model.predict(input_data_scaled)
        return prediction[0][0]  # Возвращаем предсказание
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")
        return None

if __name__ == "__main__":
    # Пример данных для предсказания
    example_data = [270.37, 0, 3.94, 94.41, 2.41, 8.71, 1.22]  # [co, no, no2, o3, so2, pm10, nh3]
    
    print(f"Входные данные: {example_data}")
    predicted_pm2_5 = predict_air_quality(example_data)
    if predicted_pm2_5 is not None:
        print(f"Предсказанное качество воздуха (PM2.5): {predicted_pm2_5:.2f}")
    else:
        print("Не удалось сделать предсказание.")
