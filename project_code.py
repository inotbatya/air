import requests
import pandas as pd
from datetime import datetime
import os

# Ваш API ключ для OpenWeatherMap
API_KEY = '93be52922464c8b6d8dc69c14553ab05'

# Координаты для города Barnaul
lat = 53.354
lon = 83.763

# Функция для получения данных о качестве воздуха
def get_air_quality():
    url = f'http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}'
    response = requests.get(url)
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
        'nh3': data['list'][0]['components']['nh3']
    }

# Функция для сбора данных и сохранения их в CSV
def collect_data():
    air_quality_data = get_air_quality()
    df = pd.DataFrame([air_quality_data])

    # Проверяем, существует ли файл, чтобы дописать данные
    file_name = 'air_quality_data.csv'
    if os.path.exists(file_name):
        # Указываем соответствие заголовков
        df.to_csv(file_name, mode='a', index=False, header=False)
    else:
        df.to_csv(file_name, index=False)

if __name__ == "__main__":
    while True:
        collect_data()
