#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

def load_single_map(file_path: str) -> pd.DataFrame:
    """Загружает один файл .txt со спектральной картой Raman."""
    df = pd.read_csv(file_path, sep='\t')
    # Убираем символы # из заголовков
    df.columns = [col.strip('# ').strip() for col in df.columns]
    return df


def plot_raman_spectra(
    df: Optional[pd.DataFrame] = None,
    file_path: Optional[str] = None,
    label: Optional[str] = None,
    group: Optional[str] = None,
    center: Optional[str] = None,
    brain: Optional[str] = None,
    place: Optional[str] = None,
    points: Optional[List[Tuple[int, int]]] = None,
    normalize: bool = True,
    alpha: Optional[float] = None,
    figsize: Tuple[int, int] = (14, 8),
    title: Optional[str] = None,
    plot_average: bool = False,
    wave_range: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Универсальная функция для построения графиков Рамановских спектров.

    Параметры:
        df — большой DataFrame со всеми спектрами (126M строк)
        file_path — путь к одному .txt файлу (альтернатива df)
        label, group, center, brain, place — фильтры для df (place обязателен для одной карты!)
        points — список точек карты [(col, row), ...], где col=1..35, row=1..15 (1-based)
                Если None и plot_average=False — строятся ВСЕ спектры карты (с прозрачностью)
        normalize — нормализация каждого спектра по максимуму
        plot_average — если True + points=None: строится средний спектр + std
        wave_range — ограничение диапазона волновых чисел, например (600, 1800)
    """
    # ======================== 1. Загрузка данных ========================
    if file_path is not None:
        map_df = load_single_map(file_path)
        source_name = Path(file_path).name
    elif df is not None:
        conditions = []
        if label is not None: conditions.append(f"label == '{label}'")
        if group is not None: conditions.append(f"group == '{group}'")
        if center is not None: conditions.append(f"center == '{center}'")
        if brain is not None: conditions.append(f"brain == '{brain}'")
        if place is not None: conditions.append(f"place == '{place}'")

        map_df = df.copy()
        if conditions:
            map_df = map_df.query(" and ".join(conditions))
        source_name = f"{label} | {group} | {brain} | place={place} | center={center}"
    else:
        raise ValueError("Укажите либо file_path, либо df + параметры фильтрации")

    if len(map_df) == 0:
        raise ValueError("Нет данных по заданным параметрам")

    # ======================== 2. Координатная сетка карты ========================
    # Точки рассчитываются ДИНАМИЧЕСКИ внутри функции (как и требовалось)
    x_coords = sorted(map_df['X'].unique())
    y_coords = sorted(map_df['Y'].unique())

    print(f"✅ Загружена карта {len(x_coords)} × {len(y_coords)} точек")

    # ======================== 3. Специальный режим «средний спектр» ========================
    if plot_average and points is None:
        grouped = map_df.groupby('Wave')['Intensity'].agg(['mean', 'std']).reset_index()

        x = grouped['Wave'].values
        y_mean = grouped['mean'].values
        y_std = grouped['std'].values

        if normalize:
            max_val = y_mean.max()
            y_mean = y_mean / max_val
            y_std = y_std / max_val

        if wave_range:
            mask = (x >= wave_range[0]) & (x <= wave_range[1])
            x, y_mean, y_std = x[mask], y_mean[mask], y_std[mask]

        plt.figure(figsize=figsize)
        plt.plot(x, y_mean, 'b-', linewidth=2.5, label='Средний спектр')
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.35, color='blue')
        plt.title(title or f'Средний Рамановский спектр — {source_name}')
        plt.xlabel('Волновое число, см⁻¹')
        plt.ylabel('Нормализованная интенсивность' if normalize else 'Интенсивность')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

    # ======================== 4. Выбор точек ========================
    if points is None:
        # Все спектры карты
        points = [(i + 1, j + 1) for i in range(len(x_coords)) for j in range(len(y_coords))]
        alpha = alpha or 0.06
    else:
        alpha = alpha or 0.85

    # ======================== 5. Построение графиков ========================
    plt.figure(figsize=figsize)
    plotted = 0

    for col, row in points:
        if not (1 <= col <= len(x_coords) and 1 <= row <= len(y_coords)):
            print(f"⚠️ Точка ({col}, {row}) вне диапазона карты")
            continue

        target_x = x_coords[col - 1]
        target_y = y_coords[row - 1]

        spectrum = map_df[(map_df['X'] == target_x) & (map_df['Y'] == target_y)].copy()
        if len(spectrum) == 0:
            continue

        spectrum = spectrum.sort_values('Wave')

        wave = spectrum['Wave'].values
        intensity = spectrum['Intensity'].values

        if wave_range:
            mask = (wave >= wave_range[0]) & (wave <= wave_range[1])
            wave = wave[mask]
            intensity = intensity[mask]

        if normalize and len(intensity) > 0:
            intensity = intensity / intensity.max()

        plt.plot(wave, intensity, alpha=alpha, linewidth=1.1, label=f'({col},{row})')
        plotted += 1

    plt.xlabel('Волновое число, см⁻¹')
    plt.ylabel('Нормализованная интенсивность' if normalize else 'Интенсивность')
    plt.title(title or f'Рамановские спектры — {source_name}')
    plt.grid(True, alpha=0.3)

    if plotted <= 12:
        plt.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

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


import pandas as pd
from pathlib import Path
import re
from typing import Optional

def _parse_metadata(file_path: Path) -> dict:
    """Парсинг метаданных из пути и имени файла + place"""
    label = file_path.parts[-3].lower()
    
    group_dir = file_path.parts[-2].lower()
    group = group_dir.replace('mk', '').replace('mend', '').replace('mexo', '')
    
    fname = file_path.name.lower()
    
    # center
    center = '1500' if 'center1500' in fname else '2900' if 'center2900' in fname else 'unknown'
    
    # brain — всё до _control_, _endo_, _exo_
    split_key = f'_{label}_'
    brain_part = fname.split(split_key)[0] if split_key in fname else 'unknown'
    # убираем возможные остатки в конце (например _group если есть)
    brain = re.sub(r'_group.*', '', brain_part).strip('_')
    
    # place — ищем placeX_Y или placeX_Y_...
    place_match = re.search(r'place(\d+)_(\d+)', fname)
    place = f"{place_match.group(1)}_{place_match.group(2)}" if place_match else 'unknown'
    
    return {
        'label': label,
        'group': group,
        'center': center,
        'brain': brain,
        'place': place,
        'filename': file_path.name
    }


import pandas as pd
from pathlib import Path
import re
from typing import Optional

def _parse_metadata(file_path: Path) -> dict:
    """Парсинг метаданных из пути и имени файла"""
    label = file_path.parts[-3].lower()
    
    group_dir = file_path.parts[-2].lower()
    group = group_dir.replace('mk', '').replace('mend', '').replace('mexo', '')
    
    fname = file_path.name.lower()
    
    # center
    center = '1500' if 'center1500' in fname else '2900' if 'center2900' in fname else 'unknown'
    
    # brain — часть до _control_ / _endo_ / _exo_
    split_key = f'_{label}_'
    brain_part = fname.split(split_key)[0] if split_key in fname else 'unknown'
    # убираем возможные лишние окончания
    brain = re.sub(r'(_group.*|_1group.*|_2agroup.*|_2bgroup.*|_3group.*)', '', brain_part).strip('_')
    
    # place — placeX_Y
    place_match = re.search(r'place(\d+)_(\d+)', fname)
    place = f"{place_match.group(1)}_{place_match.group(2)}" if place_match else 'unknown'
    
    return {
        'label': label,
        'group': group,
        'center': center,
        'brain': brain,
        'place': place
    }


import pandas as pd
from pathlib import Path
import re
from typing import Optional

# Явно задаём возможные значения (для экономии памяти и ускорения)
LABEL_CATS = ['control', 'endo', 'exo']
GROUP_CATS = ['1', '2a', '2b', '3']
CENTER_CATS = ['1500', '2900']

# brain — 7 значений, которые вы указали
BRAIN_CATS = [
    'cortex',
    'cortex_left',
    'cortex_right',
    'striatum_left',
    'striatum_right',
    'cerebellum_left',
    'cerebellum_right'
]

def _parse_metadata(file_path: Path) -> dict:
    label = file_path.parts[-3].lower()
    
    group_dir = file_path.parts[-2].lower()
    group = group_dir.replace('mk', '').replace('mend', '').replace('mexo', '')
    
    fname = file_path.name.lower()
    
    center = '1500' if 'center1500' in fname else '2900' if 'center2900' in fname else 'unknown'
    
    split_key = f'_{label}_'
    brain_part = fname.split(split_key)[0] if split_key in fname else 'unknown'
    brain = re.sub(r'(_group.*|_1group.*|_2agroup.*|_2bgroup.*|_3group.*)', '', brain_part).strip('_')
    
    place_match = re.search(r'place(\d+)_(\d+)', fname)
    place = f"{place_match.group(1)}_{place_match.group(2)}" if place_match else 'unknown'
    
    return {
        'label': label,
        'group': group,
        'center': center,
        'brain': brain,
        'place': place
    }


def load_raman_spectra(
    root_dir: str,
    use_float32: bool = True,
    save_to_parquet: Optional[str] = None,
    test_mode: bool = False
) -> pd.DataFrame:
    """
    Загружает спектры с максимально компактными типами для категориальных колонок.
    """
    root = Path(root_dir)
    
    txt_files = [f for f in root.rglob("*.txt") if "average" not in f.name.lower()]
    
    if test_mode:
        txt_files = txt_files[:5]
        print(f"ТЕСТ → только {len(txt_files)} файлов")
    
    print(f"Файлов найдено: {len(txt_files)}")
    
    dfs = []
    float_dtype = {'X': 'float32', 'Y': 'float32', 'Wave': 'float32', 'Intensity': 'float32'} \
                   if use_float32 else None
    
    # Собираем все значения place для создания ordered=False Categorical
    all_places = set()
    
    for file_path in txt_files:
        meta = _parse_metadata(file_path)
        all_places.add(meta['place'])
        
        try:
            df = pd.read_csv(
                file_path,
                sep='\t',
                names=['X', 'Y', 'Wave', 'Intensity'],
                skiprows=1,
                dtype=float_dtype,
                engine='c'
            )
            
            for k, v in meta.items():
                df[k] = v
            
            dfs.append(df)
            
        except Exception as e:
            print(f"Ошибка {file_path.name}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    print("Объединяем...")
    df = pd.concat(dfs, ignore_index=True)
    
    # Применяем CategoricalDtype с фиксированными категориями
    df['label']  = df['label'].astype(pd.CategoricalDtype(categories=LABEL_CATS,  ordered=False))
    df['group']  = df['group'].astype(pd.CategoricalDtype(categories=GROUP_CATS,  ordered=False))
    df['center'] = df['center'].astype(pd.CategoricalDtype(categories=CENTER_CATS, ordered=False))
    df['brain']  = df['brain'].astype(pd.CategoricalDtype(categories=BRAIN_CATS,  ordered=False))
    
    # place — берём все встреченные значения
    place_list = sorted(all_places)  # для воспроизводимости сортируем
    df['place'] = df['place'].astype(pd.CategoricalDtype(categories=place_list, ordered=False))
    
    # Проверка памяти после оптимизации
    mem_gb = df.memory_usage(deep=True).sum() / (1024 ** 3)
    print(f"Строк: {len(df):,}")
    print(f"Память после оптимизации типов ≈ {mem_gb:.2f} GB")
    
    if save_to_parquet:
        print(f"Сохранение → {save_to_parquet}")
        df.to_parquet(save_to_parquet, compression='zstd', index=False)
        print("Сохранено")
    
    return df
