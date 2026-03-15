import os
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

# Создаем директории
os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

def generate_temperature_data(days=30, base_temp=20, seasonal_amplitude=8, 
                              noise_level=2.0, anomaly_prob=0.1, anomaly_magnitude=10):
  # Создаем временной ряд
    days_array = np.arange(days)
    
    # Сезонная компонента (годовой цикл)
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * days_array / 365)
    
    # Тренд (постепенное потепление)
    trend = 0.05 * days_array
    
    # Случайный шум
    noise = np.random.normal(0, noise_level, days)
    
    # Базовая температура
    temperature = base_temp + seasonal + trend + noise
    
    # Добавляем аномалии (резкие скачки температуры)
    for i in range(days):
        if random.random() < anomaly_prob:
            # Аномалия может быть как вверх, так и вниз
            anomaly = random.choice([-1, 1]) * anomaly_magnitude * random.uniform(0.5, 1.5)
            temperature[i] += anomaly
    
    # Создаем DataFrame
    df = pd.DataFrame({
        'day': days_array,
        'temperature': temperature,
        'is_anomaly': [random.random() < anomaly_prob for _ in range(days)]
    })
    
    return df

    # Генерируем 5 наборов для тренировки
print("Генерация тренировочных данных...")
for i in range(5):
    # Используем разные параметры для каждого набора
    df = generate_temperature_data(
        days=30 + i*5,  # разные длины
        base_temp=15 + i*2,
        noise_level=1.5 + i*0.5,
        anomaly_prob=0.05 + i*0.02
    )
    df.to_csv(f"train/data_{i}.csv", index=False)
    print(f"  Создан train/data_{i}.csv с {len(df)} записями")

# Генерируем 3 набора для тестирования
print("\nГенерация тестовых данных...")
for i in range(3):
    df = generate_temperature_data(
        days=30 + i*3,
        base_temp=20,
        noise_level=2.0,
        anomaly_prob=0.08
    )
    df.to_csv(f"test/data_{i}.csv", index=False)
    print(f"  Создан test/data_{i}.csv с {len(df)} записями")

print("\n Данные успешно созданы!")
print(f"  - train: {len(os.listdir('train'))} файлов")
print(f"  - test: {len(os.listdir('test'))} файлов")