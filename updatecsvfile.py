import requests
import pandas as pd
from datetime import datetime
import os
import time

# Ваш API ключ для OpenWeatherMap
API_KEY = '93be52922464c8b6d8dc69c14553ab05'

# Координаты для города Barnaul
lat = 53.354
lon = 83.763

# Функция для получения данных о качестве воздуха
def get_air_quality():
    try:
        url = f'http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}'
        response = requests.get(url)
        response.raise_for_status()  # Проверяет наличие ошибок HTTP
        data = response.json()

        # Формируем структуру данных
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'co': data['list'][0]['components']['co'],
            'no': data['list'][0]['components']['no'],
            'no2': data['list'][0]['components']['no2'],
            'o3': data['list'][0]['components']['o3'],
            'so2': data['list'][0]['components']['so2'],
            'pm2_5': data['list'][0]['components']['pm2_5'],
            'pm10': data['list'][0]['components']['pm10'],
            'nh3': data['list'][0]['components']['nh3']
        }
    except Exception as e:
        print(f"Ошибка при получении данных: {e}")
        return None

# Функция для сбора данных и сохранения их в CSV
def collect_data():
    air_quality_data = get_air_quality()
    # Сохраняем модель
    model.save('air_quality_model.h5')
    # Сохраняем масштабировщик
    from joblib import dump
    dump(scaler, 'scaler.pkl')
    if air_quality_data is None:
        print("Не удалось получить данные. Пропуск итерации.")
        return

    df = pd.DataFrame([air_quality_data])

    # Проверяем, существует ли файл, чтобы дописать данные
    file_name = 'air_quality_data.csv'
    if os.path.exists(file_name):
        df.to_csv(file_name, mode='a', index=False, header=False)
    else:
        df.to_csv(file_name, index=False)

    print(f"Данные успешно добавлены: {air_quality_data}")

if __name__ == "__main__":
    while True:
        print("Сбор данных начат...")
        collect_data()
        print("Ожидание перед следующим запросом...")
        time.sleep(600)  # Запрашивает данные каждые 10 минут
