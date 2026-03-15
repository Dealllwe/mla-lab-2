import os
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Загружаем модель и scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print("Модель и scaler загружены")

# Для хранения результатов по всем тестовым файлам
all_results = []

# Тестируем на каждом файле из test
print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")

for filename in os.listdir("test"):
    if filename.endswith(".csv"):
        filepath = os.path.join("test", filename)
        df = pd.read_csv(filepath)
        
        # Подготовка данных
        X_test = df[['day']].values
        
        # Масштабируем температуру тем же scaler'ом
        temperatures_scaled = scaler.transform(df[['temperature']])
        y_test_scaled = temperatures_scaled.flatten()
        
        # Предсказание
        y_pred_scaled = model.predict(X_test)
        
        # Обратное масштабирование для получения исходных значений
        y_test_original = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
        y_pred_original = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Вычисление метрик
        mae_scaled = mean_absolute_error(y_test_scaled, y_pred_scaled)
        mae_original = mean_absolute_error(y_test_original, y_pred_original)
        rmse_original = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        r2 = r2_score(y_test_scaled, y_pred_scaled)
        
        # Сохраняем результаты
        result = {
            'file': filename,
            'samples': len(df),
            'mae_scaled': mae_scaled,
            'mae_original': mae_original,
            'rmse_original': rmse_original,
            'r2': r2
        }
        all_results.append(result)
        
        # Выводим результаты для файла
        print(f"\nФайл: {filename}")
        print(f"  Количество записей: {len(df)}")
        print(f"  MAE (масштабированная): {mae_scaled:.4f}")
        print(f"  MAE (исходная температура): {mae_original:.2f}°C")
        print(f"  RMSE (исходная): {rmse_original:.2f}°C")
        print(f"  R2 score: {r2:.4f}")

# Вычисляем средние результаты
print("СРЕДНИЕ РЕЗУЛЬТАТЫ ПО ВСЕМ ТЕСТОВЫМ ФАЙЛАМ")

avg_mae_scaled = np.mean([r['mae_scaled'] for r in all_results])
avg_mae_original = np.mean([r['mae_original'] for r in all_results])
avg_rmse = np.mean([r['rmse_original'] for r in all_results])
avg_r2 = np.mean([r['r2'] for r in all_results])

print(f"  Средняя MAE (масштабированная): {avg_mae_scaled:.4f}")
print(f"  Средняя MAE (исходная): {avg_mae_original:.2f}°C")
print(f"  Средняя RMSE: {avg_rmse:.2f}°C")
print(f"  Средний R2: {avg_r2:.4f}")

# Сохраняем результаты в файл
results_df = pd.DataFrame(all_results)
results_df.to_csv("test_results.csv", index=False)
print("\n Детальные результаты сохранены в test_results.csv")