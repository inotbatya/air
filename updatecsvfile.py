import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
from joblib import dump

# Проверяем файл данных
csv_file = 'air_quality_data.csv'
if not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0:
    print("Файл air_quality_data.csv отсутствует или пуст. Проверьте сбор данных.")
    exit()

# Загрузка данных
column_names = ["timestamp", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
try:
    df = pd.read_csv(csv_file, names=column_names, header=0)  # Пропускаем заголовки
except Exception as e:
    print(f"Ошибка при загрузке данных: {e}")
    exit()

# Убираем строки с пропущенными значениями
df = df.dropna()

# Разделяем признаки (X) и целевую переменную (y)
X = df[["co", "no", "no2", "o3", "so2", "pm10", "nh3"]]
y = df["pm2_5"]

# Нормализация данных
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Создаем модель
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
model.save('air_quality_model.h5')
print("Модель успешно сохранена в 'air_quality_model.h5'.")

dump(scaler, 'scaler.pkl')
print("Масштабировщик сохранён в 'scaler.pkl'.")
