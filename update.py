import requests
import pandas as pd
from datetime import datetime
import time

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
        'timestamp': datetime.now(),
        'AQI': data['list'][0]['main']['aqi'],
        'PM2.5': data['list'][0]['components']['pm2_5'],
        'PM10': data['list'][0]['components']['pm10'],
        'temperature': weather_data['main']['temp'],  # Температура
        'humidity': weather_data['main']['humidity'],  # Влажность
        'wind_speed': weather_data['wind']['speed']  # Скорость ветра
    }

# Функция для сохранения данных в CSV
def save_to_csv(data):
    df = pd.DataFrame([data])
    df.to_csv('air_quality_data.csv', mode='a', header=not pd.io.common.file_exists('air_quality_data.csv'), index=False)

# Получаем и сохраняем данные
def collect_data():
    air_quality_data = get_air_quality()
    save_to_csv(air_quality_data)

# Настроим программу на выполнение по расписанию
if __name__ == '__main__':
    while True:
        collect_data()
        print(f"Данные собраны и сохранены: {datetime.now()}")
        time.sleep(86400 /  20)  # 86400 секунд в сутки, делим на 10 запросов в день
