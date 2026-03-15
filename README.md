# Лабораторная работа №2: простой ML-конвейер

Репозиторий реализует полный учебный ML-пайплайн из задания:

1. `data_creation.py`
   - генерирует 3 синтетических набора данных (`clean_signal`, `noisy_signal`, `anomalous_signal`),
   - добавляет шум и аномалии,
   - делит каждый набор на `train/test` и сохраняет в CSV.

2. `model_preprocessing.py`
   - обучает стандартизацию признаков по train-данным,
   - сохраняет параметры scaler в `data/scaler.joblib`,
   - создает `data/train_preprocessed/*.csv` и `data/test_preprocessed/*.csv`.

3. `model_preparation.py`
   - обучает линейную регрессию на предобработанных train-данных,
   - сохраняет модель в `data/model.joblib`.

4. `model_testing.py`
   - оценивает модель на предобработанных test-данных,
   - считает метрики `MAE`, `RMSE`, `R2`.

5. `pipeline.sh`
   - запускает все этапы последовательно,
   - прерывается при ошибке (`set -euo pipefail`).

## Запуск

```bash
chmod +x pipeline.sh
./pipeline.sh
```

## Артефакты после запуска

- `data/train/*.csv`, `data/test/*.csv` — исходные датасеты;
- `data/scaler.joblib` — параметры стандартизации признаков;
- `data/train_preprocessed/*.csv`, `data/test_preprocessed/*.csv` — предобработанные данные;
- `data/model.joblib` — обученная модель;
- `logs/*.log` — логи по этапам.
