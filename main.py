import requests

API_KEY = "93be52922464c8b6d8dc69c14553ab05"
city = "Barnaul"
url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    print("Город:", data["name"])
    print("Температура:", data["main"]["temp"], "°C")
    print("Качество воздуха:", data.get("aqi", "нет данных"))
else:
    print("Ошибка:", response.status_code)
