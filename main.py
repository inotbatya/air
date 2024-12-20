import requests
import pandas as pd
from datetime import datetime
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from joblib import dump

# Ваш API ключ для OpenWeatherMap
API_KEY = '93be52922464c8b6d8dc69c14553ab05'

# Координаты для города Barnaul
lat = 53.354
lon = 83.763

# Файл для хранения данных
csv_file = 'air_quality_data.csv'

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
def update_csv(data):
    df = pd.DataFrame([data])
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', index=False, header=False)
    else:
        df.to_csv(csv_file, index=False)
    print(f"Данные успешно добавлены: {data}")

# Функция для обучения и сохранения модели
def train_and_save_model():
    # Проверяем файл данных
    if not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0:
        print("Файл air_quality_data.csv отсутствует или пуст. Пропуск обучения.")
        return

    # Загрузка данных
    column_names = ["timestamp", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
    try:
        df = pd.read_csv(csv_file, names=column_names, header=0)  # Пропускаем заголовки
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return

    # Убираем строки с пропущенными значениями
    df = df.dropna()

    # Разделяем признаки (X) и целевую переменную (y)
    X = df[["co", "no", "no2", "o3", "so2", "pm10", "nh3"]]
    y = df["pm2_5"]

    # Нормализация данных
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Создаём модель
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse')

    # Разбиваем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучаем модель
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Сохраняем модель и масштабировщик
    model.save('air_quality_model.keras')
    dump(scaler, 'scaler.pkl')

    print("Модель успешно обучена и сохранена!")
    print("Файлы: 'air_quality_model.keras' и 'scaler.pkl'.")

if __name__ == "__main__":
    print("Запуск автономного процесса...")
    while True:
        print("\n--- Обновление данных ---")
        air_quality_data = get_air_quality()
        if air_quality_data:
            update_csv(air_quality_data)
        else:
            print("Не удалось получить данные. Пропуск...")
        
        print("\n--- Обучение модели ---")
        train_and_save_model()

        print("\nОжидание перед следующим обновлением...")
        time.sleep(3600)  # Повторение каждые 60 минут
