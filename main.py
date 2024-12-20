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
        'nh3': data['list'][0]['components']['nh3'],
    }

# Функция для сбора данных и сохранения их в CSV
def collect_data():
    air_quality_data = get_air_quality()
    df = pd.DataFrame([air_quality_data])

    # Проверяем, существует ли файл, чтобы дописать данные
    file_name = 'air_quality_data.csv'
    if os.path.exists(file_name):
        df.to_csv(file_name, mode='a', index=False, header=False)
    else:
        df.to_csv(file_name, index=False)

if __name__ == "__main__":
    collect_data()

# Обновлённый main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
column_names = ["timestamp", "PM2.5", "PM10", "temperature", "humidity", "wind_speed", "column7", "column8", "column9"]

# Проверяем, существует ли файл и не пустой ли он
csv_file = 'air_quality_data.csv'
if not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0:
    print("Файл air_quality_data.csv отсутствует или пуст. Проверьте скрипт update.py.")
    exit()

# Загружаем данные из CSV
df = pd.read_csv(csv_file, names=column_names))

# Убираем строки с пропущенными значениями
df = df.dropna()

# Выбираем признаки (X) и целевую переменную (y)
X = df[['PM2.5', 'PM10', 'temperature', 'humidity', 'wind_speed']]
    
y = df['pm2_5']

# Нормализуем данные
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

# Выводим результат
print("Модель успешно обучена!")
