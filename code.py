import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
column_names = ["timestamp", "PM2.5", "PM10", "temperature", "humidity", "wind_speed", "column7", "column8", "column9"]

csv_file = 'air_quality_data.csv'
if not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0:
    print("Файл air_quality_data.csv отсутствует или пуст. Проверьте скрипт update.py.")
    exit()

df = pd.read_csv(csv_file, names=column_names)
df = df.dropna()

X = df[['PM2.5', 'PM10', 'temperature', 'humidity', 'wind_speed']]
y = df['PM2.5']

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

print("Модель успешно обучена!")
