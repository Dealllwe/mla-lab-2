# Лабораторная работа №2: простой ML-конвейер

Репозиторий реализует все этапы задания:

1. `data_creation.py` — генерирует несколько синтетических наборов данных по температуре:
   - `clean_signal` (чистые данные),
   - `noisy_signal` (данные с повышенным шумом),
   - `anomalous_signal` (данные с аномалиями).

   Каждый набор автоматически делится на `train` и `test` и сохраняется в `data/train/*.csv` и `data/test/*.csv`.

2. `model_preprocessing.py` — обучает и применяет стандартизацию признаков (аналог `StandardScaler`) по train-данным и создает:
   - `data/train_preprocessed/*.csv`,
   - `data/test_preprocessed/*.csv`.

3. `model_preparation.py` — обучает линейную регрессию на предобработанных train-данных и сохраняет модель в `data/model.joblib`.

4. `model_testing.py` — оценивает модель на предобработанных test-данных, считает метрики `MAE`, `RMSE`, `R2`.

5. `pipeline.sh` — запускает все шаги по порядку.

## Запуск

```bash
chmod +x pipeline.sh
./pipeline.sh
```

## Артефакты после запуска

- `data/train/*.csv`, `data/test/*.csv` — исходные датасеты;
- `data/scaler.joblib` — параметры стандартизации;
- `data/train_preprocessed/*.csv`, `data/test_preprocessed/*.csv` — предобработанные данные;
- `data/model.joblib` — обученная модель;
- `logs/*.log` — логи каждого этапа.
