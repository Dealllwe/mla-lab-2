import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


# Собираем все тренировочные данные для обучения scaler
all_temperatures = []
all_days = []

# Проходим по всем файлам в папке train
for filename in os.listdir("train"):
    if filename.endswith(".csv"):
        filepath = os.path.join("train", filename)
        df = pd.read_csv(filepath)
        
        # Сохраняем исходные данные для последующего использования
        all_temperatures.extend(df['temperature'].values)
        all_days.extend(df['day'].values)
        
        print(f"  Загружен {filename}: {len(df)} записей")

# Преобразуем в numpy массивы и изменяем форму для scaler
X_temperatures = np.array(all_temperatures).reshape(-1, 1)

# Создаем и обучаем StandardScaler
scaler = StandardScaler()
scaler.fit(X_temperatures)

print(f"\nStandardScaler обучен:")
print(f"  Среднее (mean): {scaler.mean_[0]:.2f}")
print(f"  Стандартное отклонение (scale): {scaler.scale_[0]:.2f}")

# Применяем масштабирование к каждому тренировочному файлу
print("\nПрименение масштабирования к тренировочным данным...")
for filename in os.listdir("train"):
    if filename.endswith(".csv"):
        filepath = os.path.join("train", filename)
        df = pd.read_csv(filepath)
        
        # Масштабируем температуру
        temperatures_scaled = scaler.transform(df[['temperature']])
        
        # Добавляем новую колонку с масштабированными значениями
        df['temperature_scaled'] = temperatures_scaled
        
        # Сохраняем обратно
        df.to_csv(filepath, index=False)
        print(f"  Обработан {filename}")

# Сохраняем scaler для использования в тестировании
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n Предобработка завершена!")
print("  Scaler сохранен в scaler.pkl")