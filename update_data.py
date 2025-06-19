import requests
import pandas as pd
import time
import os
import logging
from datetime import datetime, timedelta

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

API_KEY = 'API_KEY'

# Города с координатами
cities = [
    {'city': 'India', 'lat': 25.38374, 'lon': 74.61914},
    {'city': 'Mexico', 'lat': 19.31114, 'lon': 99.09668},
    {'city': 'Bishkek', 'lat': 42.94034, 'lon': 74.53125},
    {'city': 'Poland', 'lat': 52.16045, 'lon': 18.98438},
    {'city': 'Greece', 'lat':39.3003, 'lon': 21.5332}
]

CSV_PATH = 'air_quality_data.csv'
MIN_INTERVAL_MINUTES = 60  # Минимальный интервал между запросами для одного города

def fetch_city_data(city):
    """Получает данные по одному городу с обработкой ошибок"""
    url = f'https://api.openweathermap.org/data/2.5/air_pollution?lat={city["lat"]}&lon={city["lon"]}&appid={API_KEY}'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Выбросит исключение при HTTP ошибках 
        data = response.json()
        
        components = data['list'][0]['components']
        aqi = data['list'][0]['main']['aqi']
        
        return {
            'city': city['city'],
            'co': components['co'],
            'no': components['no'],
            'no2': components['no2'],
            'o3': components['o3'],
            'so2': components['so2'],
            'pm2_5': components['pm2_5'],
            'pm10': components['pm10'],
            'nh3': components['nh3'],
            'aqi': aqi,
            'timestamp': datetime.now().isoformat()
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка запроса для {city['city']}: {e}")
        return None
    except KeyError as e:
        logger.error(f"Неверный формат ответа для {city['city']}: {e}")
        return None

def load_last_record(city_name):
    """Загружает последнюю запись для города из CSV"""
    if not os.path.exists(CSV_PATH):
        return None
        
    try:
        df = pd.read_csv(CSV_PATH)
        if df.empty:
            return None
            
        df_city = df[df['city'] == city_name]
        if df_city.empty:
            return None
            
        return df_city.iloc[-1].to_dict()
    except Exception as e:
        logger.error(f"Ошибка чтения CSV: {e}")
        return None

def should_update(last_record, current_data):
    """Проверяет, нужно ли обновлять данные"""
    if last_record is None:
        return True
        
    last_time = datetime.fromisoformat(last_record['timestamp'])
    current_time = datetime.fromisoformat(current_data['timestamp'])
    
    # Обновляем, если прошло больше MIN_INTERVAL_MINUTES или данные отличаются
    time_diff = (current_time - last_time).total_seconds() / 60
    if time_diff >= MIN_INTERVAL_MINUTES:
        return True
        
    # Проверяем значимые изменения в ключевых параметрах (например, PM2.5)
    pm2_5_diff = abs(current_data['pm2_5'] - last_record['pm2_5'])
    if pm2_5_diff > 5:  # Если изменение больше 5 μg/m³
        logger.info(f"Обнаружено значительное изменение PM2.5 для {current_data['city']}")
        return True
        
    return False

def save_data(data):
    """Сохраняет данные в CSV"""
    df = pd.DataFrame([data])
    try:
        if not os.path.exists(CSV_PATH):
            df.to_csv(CSV_PATH, index=False)
        else:
            df.to_csv(CSV_PATH, mode='a', header=False, index=False)
        logger.info(f"Данные сохранены для {data['city']}")
    except Exception as e:
        logger.error(f"Ошибка сохранения данных: {e}")

def main():
    while True:
        for city in cities:
            try:
                logger.info(f"Получение данных для {city['city']}")
                data = fetch_city_data(city)
                
                if data:
                    last_record = load_last_record(city['city'])
                    if should_update(last_record, data):
                        save_data(data)
                    else:
                        logger.info(f"Данные для {city['city']} актуальны, пропуск")
                
                # Задержка между запросами к API
                time.sleep(5)
            
            except Exception as e:
                logger.error(f"Ошибка обработки данных для {city['city']}: {e}")
        
        # Интервал между циклами (в часах)
        logger.info(f"Ожидание {MIN_INTERVAL_MINUTES} минут перед следующим циклом")
        time.sleep(MIN_INTERVAL_MINUTES * 60)

if __name__ == "__main__":
    main()
