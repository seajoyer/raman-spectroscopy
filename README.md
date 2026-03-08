# Raman Spectrum Classifier

Классификация рамановских спектров биологических тканей мозга крыс на три класса: **control**, **endo**, **exo**.

## Использование

### Обучение

```bash
python raman_classifier.py train \
    --data     all_raman_spectra.parquet \
    --model    raman_model.pkl \
    --trials   50 \
    --features 120 \
    --models   xgb lgb cat rf et svm mlp
```

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `--data` | — | Путь к `.parquet` файлу с обучающими данными |
| `--model` | `raman_model.pkl` | Куда сохранить модель |
| `--trials` | `50` | Количество Optuna-итераций на модель |
| `--features` | `120` | Сколько признаков оставить после отбора |
| `--models` | все | Выбор базовых моделей из: `xgb lgb cat rf et svm mlp` |

Обученная модель сохраняется в `.pkl` файл.

### Предсказание

```bash
python raman_classifier.py predict \
    --spectrum  path/to/spectrum.txt \
    --model     raman_model.pkl
```

Скрипт выводит вероятности по каждому классу и итоговую метку.

### Формат входного спектра

```
#Wave       #Intensity
2002.417969 12803.853516
2001.458008 13013.024414
...
```

Поддерживаются разделители Tab и пробел. Диапазон спектра (`1500` или `2900` cm⁻¹) определяется автоматически.

## Пайплайн

```
.parquet / .txt
    │
    ├─ ALS baseline correction
    ├─ Savitzky-Golay smoothing
    └─ SNV normalization
         │
         ├─ Band integrals (биохимические полосы)
         ├─ Производные 1-го и 2-го порядка
         ├─ Вейвлет-признаки (db8)
         ├─ Зональная статистика
         └─ PCA + NMF
              │
              ├─ LightGBM feature selection (top-N)
              ├─ Base models × Optuna tuning (LOAO-CV)
              └─ Logistic Regression stacking
```

**Валидация:** Leave-One-Animal-Out (LOAO) — каждый фолд соответствует одному животному, что предотвращает data leakage между картами одного животного.
