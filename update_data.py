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
        response.raise_for_status()
        data = response.json()
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'co': data['list'][0]['components']['co'],
            'no': data['list'][0]['components']['no'],
            'no2': data['list'][0]['components']['no2'],
            'o3': data['list'][0]['components']['o3'],
            'so2': data['list'][0]['components']['so2'],
            'pm2_5': data['list'][0]['components']['pm2_5'],
            'pm10': data['list'][0]['components']['pm10'],
            'nh3': data['list'][0]['components']['nh3'],
        }
    except Exception as e:
        print(f"Ошибка при получении данных: {e}")
        return None

# Функция для записи данных в CSV
def update_csv(data, file_name='air_quality_data.csv'):
    df = pd.DataFrame([data])
    if os.path.exists(file_name):
        df.to_csv(file_name, mode='a', index=False, header=False)
    else:
        df.to_csv(file_name, index=False)
    print(f"Данные успешно добавлены: {data}")

if __name__ == "__main__":
    print("Сбор данных начат...")
    while True:
        air_quality_data = get_air_quality()
        if air_quality_data:
            update_csv(air_quality_data)
        else:
            print("Не удалось получить данные. Пропуск...")
        print("Ожидание перед следующим запросом...")
        time.sleep(3600)  # Запрос каждые 60 минут
