#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

def create_metadata_dataframe(root_dir='../data'):
    """
    Scans a directory structure for text files and extracts metadata into a DataFrame.

    Args:
        root_dir (str): The root directory containing class folders (control, endo, exo).

    Returns:
        pd.DataFrame: A DataFrame containing file paths, labels, and extracted metadata.
    """
    data_info = []
    
    # Define class directories and their corresponding labels
    class_map = {'control': 0, 'endo': 1, 'exo': 2}

    for class_dir, label in class_map.items():
        class_path = os.path.join(root_dir, class_dir)
        
        # Skip if the class directory doesn't exist
        if not os.path.exists(class_path):
            continue

        for subdir in os.listdir(class_path):
            subpath = os.path.join(class_path, subdir)
            
            # Ensure we are iterating over directories
            if not os.path.isdir(subpath):
                continue

            for file in os.listdir(subpath):
                if file.endswith('.txt'):
                    full_path = os.path.join(subpath, file)
                    parts = file.split('_')
                    
                    # Parse filename parts
                    region = parts[0]
                    
                    # Safely extract 'center' and 'place' using list comprehensions
                    center_matches = [p for p in parts if 'center' in p]
                    place_matches = [p for p in parts if 'place' in p]

                    # Only append if the expected parts exist in the filename
                    if center_matches and place_matches:
                        center = center_matches[0].replace('center', '')
                        place = place_matches[0].replace('place', '')

                        data_info.append({
                            'path': full_path, 
                            'label': label, 
                            'subdir': subdir,
                            'region': region, 
                            'center': center, 
                            'place': place
                        })

    return pd.DataFrame(data_info)

def _parse_metadata(file_path: Path) -> dict:
    """Простой и надёжный парсинг метаданных из пути и имени файла."""
    # label: control / endo / exo (всегда на 3 уровне от конца)
    label = file_path.parts[-3].lower()
    
    # group: mk1 → 1, mend2a → 2a, mexo3 → 3 и т.д.
    group_dir = file_path.parts[-2].lower()
    group = group_dir.replace('mk', '').replace('mend', '').replace('mexo', '')
    
    # filename в нижнем регистре для поиска (используется только для парсинга)
    fname = file_path.name.lower()
    
    # center
    center = '1500' if 'center1500' in fname else '2900' if 'center2900' in fname else 'unknown'
    
    # brain: всё, что идёт до _control_, _endo_ или _exo_
    split_key = f'_{label}_'
    brain = fname.split(split_key)[0] if split_key in fname else 'unknown'
    
    return {
        'label': label,
        'group': group,
        'center': center,
        'brain': brain
        # filename исключён
    }


def load_raman_spectra(
    root_dir: str,
    use_float32: bool = True,
    save_to_parquet: Optional[str] = None,
    test_mode: bool = False
) -> pd.DataFrame:
    """
    Загружает ВСЕ спектры (.txt) в один DataFrame.
    
    Особенности (учитывая объём данных ~26 МБ на файл):
    - float32 вместо float64 → экономия ~50% памяти
    - категориальные типы для метаданных
    - пропускает все *_Average.txt
    - опционально сразу сохраняет в parquet (рекомендуется!)
    - test_mode=True — загружает только 5 файлов (для отладки)
    
    Параметры:
        root_dir          : путь к папке, где лежат control/, endo/, exo/
        use_float32       : True (по умолчанию) — экономит память
        save_to_parquet   : путь к файлу .parquet (например "all_spectra.parquet")
                            Если указан — сохранит и вернёт DF
        test_mode         : True → только 5 файлов (быстро проверить)
    
    Возврат:
        pandas DataFrame со столбцами:
        X, Y, Wave, Intensity, label, group, center, brain
    """
    root = Path(root_dir)
    
    # Собираем все .txt (кроме Average)
    txt_files = [
        f for f in root.rglob("*.txt")
        if "average" not in f.name.lower()
    ]
    
    if test_mode:
        txt_files = txt_files[:5]
        print(f"ТЕСТОВЫЙ РЕЖИМ: загружаем только {len(txt_files)} файлов")
    
    print(f"Найдено файлов: {len(txt_files)}")
    
    dfs = []
    dtype = {'X': 'float32', 'Y': 'float32', 'Wave': 'float32', 'Intensity': 'float32'} \
            if use_float32 else None
    
    for file_path in txt_files:
        meta = _parse_metadata(file_path)
        
        try:
            df = pd.read_csv(
                file_path,
                sep='\t',                    # точно табуляция
                names=['X', 'Y', 'Wave', 'Intensity'],
                skiprows=1,                  # пропускаем строку #X	#Y	#Wave	#Intensity
                dtype=dtype,
                engine='c'                   # самый быстрый движок
            )
            
            # Добавляем метаданные
            for k, v in meta.items():
                df[k] = v
            
            # Сразу переводим метаданные в category (экономия памяти)
            # filename исключён из списка
            for col in ['label', 'group', 'center', 'brain']:
                df[col] = df[col].astype('category')
            
            dfs.append(df)
            
        except Exception as e:
            print(f"Ошибка при чтении {file_path.name}: {e}")
    
    if not dfs:
        print("Нет данных для загрузки.")
        return pd.DataFrame()
    
    print("Объединяем все файлы в один DataFrame...")
    result = pd.concat(dfs, ignore_index=True)
    
    # Финальная оптимизация
    print(f"Загружено строк: {len(result):,} | Форма: {result.shape}")
    memory_gb = result.memory_usage(deep=True).sum() / (1024 ** 3)
    print(f"Потребление памяти: {memory_gb:.2f} GB")
    
    # Сохранение в parquet (самый важный совет!)
    if save_to_parquet:
        print(f"Сохраняем в {save_to_parquet} (zstd — максимальное сжатие)...")
        result.to_parquet(
            save_to_parquet,
            compression='zstd',      # лучшее сжатие
            index=False
        )
    
    return result
