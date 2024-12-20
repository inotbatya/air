import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

# Отключение некоторых опций TensorFlow для оптимизации
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Заголовки для обработки данных
column_names = ["timestamp", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]

# Проверяем, существует ли файл и не пустой ли он
csv_file = 'air_quality_data.csv'
if not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0:
    print("Файл air_quality_data.csv отсутствует или пуст. Проверьте скрипт сбора данных.")
    exit()

# Загружаем данные из CSV
try:
    df = pd.read_csv(csv_file, names=column_names, header=0)  # header=0 для пропуска строки с названиями колонок
except Exception as e:
    print(f"Ошибка при чтении файла: {e}")
    exit()

# Убираем строки с пропущенными значениями
df = df.dropna()

# Выбираем признаки (X) и целевую переменную (y)
X = df[["co", "no", "no2", "o3", "so2", "pm10", "nh3"]]
y = df["pm2_5"]

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
