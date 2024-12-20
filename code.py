if __name__ == "__main__":
  import pandas as pd
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense
  from tensorflow import keras
  import os
  import keras
  
  
  os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
  
  # Загружаем данные из CSV
  df = pd.read_csv('air_quality_data.csv')
  
  # Убираем строки с пропущенными значениями
  df = df.dropna()
  
  # Выбираем признаки (X) и целевую переменную (y)
  X = df[['PM2.5', 'PM10', 'temperature', 'humidity', 'wind_speed']]
  y = df['AQI']
  
  # Нормализуем данные
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  
  # Разделяем данные на обучающую и тестовую выборки
  X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
  
  # Создаем модель нейронной сети
  model = Sequential()
  model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
  model.add(Dense(32, activation='relu'))
  model.add(Dense(1, activation='linear'))  # линейная активация для регрессии
  
  # Компилируем модель
  model.compile(loss='mean_squared_error', optimizer='adam')
  
  # Обучаем модель
  model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)
  
  # Оценка модели
  loss = model.evaluate(X_test, y_test)
  print(f"Тестовая ошибка: {loss}")
  
  # Прогнозирование
  predictions = model.predict(X_test)
  print(predictions)
