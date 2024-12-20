import requests
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os

# Ваш API ключ для OpenWeatherMap
API_KEY = '93be52922464c8b6d8dc69c14553ab05'

# Координаты для города Barnaul
lat = 53.354
lon = 83.763

# Загрузка обученной модели
model_path = 'air_quality_model.h5'
if not os.path.exists(model_path):
    print("Модель не найдена. Пожалуйста, обучите модель и сохраните её перед использованием.")
    exit()

model = load_model(model_path)

# Получение текущих данных о качестве воздуха
def get_current_air_quality():
    try:
        url = f'http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}'
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return [
            data['list'][0]['components']['co'],
            data['list'][0]['components']['no'],
            data['list'][0]['components']['no2'],
            data['list'][0]['components']['o3'],
            data['list'][0]['components']['so2'],
            data['list'][0]['components']['pm10'],
            data['list'][0]['components']['nh3']
        ]
    except Exception as e:
        print(f"Ошибка при получении данных: {e}")
        return None

# Нормализация данных
def normalize_data(data, scaler_path='scaler.pkl'):
    from joblib import load
    if not os.path.exists(scaler_path):
        print("Файл scaler.pkl не найден. Обучите модель и сохраните масштабировщик.")
        exit()
    scaler = load(scaler_path)
    return scaler.transform([data])

if __name__ == "__main__":
    air_quality_data = get_current_air_quality()
    if air_quality_data is None:
        print("Не удалось получить текущие данные. Завершение работы.")
        exit()

    # Нормализация входных данных
    normalized_data = normalize_data(air_quality_data)

    # Прогноз качества воздуха
    prediction = model.predict(normalized_data)
    print(f"Прогнозируемое значение PM2.5: {prediction[0][0]:.2f}")

# Сохраняем модель
model.save('air_quality_model.h5')

# Сохраняем масштабировщик
from joblib import dump
dump(scaler, 'scaler.pkl')
