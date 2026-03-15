import os
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Собираем все тренировочные данные
X_train = []  # признаки (дни)
y_train = []  # целевая переменная (масштабированная температура)

# Загружаем все файлы из train
for filename in os.listdir("train"):
    if filename.endswith(".csv"):
        filepath = os.path.join("train", filename)
        df = pd.read_csv(filepath)
        
        # Проверяем, есть ли масштабированная температура
        if 'temperature_scaled' in df.columns:
            X_train.extend(df['day'].values)
            y_train.extend(df['temperature_scaled'].values)
        else:
            # Если нет, используем исходную
            X_train.extend(df['day'].values)
            y_train.extend(df['temperature'].values)
        
        print(f"  Загружен {filename}")

# Преобразуем в numpy массивы и изменяем форму
X_train = np.array(X_train).reshape(-1, 1)
y_train = np.array(y_train)

print(f"\nДанные для обучения:")
print(f"  Количество образцов: {len(X_train)}")
print(f"  Диапазон дней: от {X_train.min()} до {X_train.max()}")
print(f"  Диапазон температур: от {y_train.min():.2f} до {y_train.max():.2f}")

# Создаем и обучаем модель линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Оцениваем модель на тренировочных данных
y_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

print(f"\nРезультаты обучения:")
print(f"  Коэффициент (slope): {model.coef_[0]:.4f}")
print(f"  Смещение (intercept): {model.intercept_:.4f}")
print(f"  MSE: {mse:.4f}")
print(f"  R2 score: {r2:.4f}")

# Сохраняем модель
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n Модель обучена и сохранена в model.pkl")