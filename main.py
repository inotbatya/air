import requests

API_KEY = "ваш_API_ключ"
lat = 53.3498  # Широта для Barnaul
lon = 83.7637  # Долгота для Barnaul
url_air = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"

response = requests.get(url_air)
if response.status_code == 200:
    air_data = response.json()
    print("AQI:", air_data["list"][0]["main"]["aqi"])
    print("PM2.5:", air_data["list"][0]["components"]["pm2_5"])
    print("PM10:", air_data["list"][0]["components"]["pm10"])
else:
    print("Ошибка:", response.status_code)
