FILES_PATH  = [
    "striatum_left_exo_2Bgroup_633nm_center1500_obj100_power100_1s_5acc_map35x15_step2_place1_3.txt",
    "striatum_left_exo_2Bgroup_633nm_center1500_obj100_power100_1s_5acc_map35x15_step2_place2_3.txt",
    "striatum_left_exo_2Bgroup_633nm_center1500_obj100_power100_1s_5acc_map35x15_step2_place3_3.txt",
    "striatum_left_exo_2Bgroup_633nm_center2900_obj100_power100_1s_5acc_map35x15_step2_place1_3.txt"
]

import pandas as pd
from scipy.signal import find_peaks
from utils.normalize import smooth

for data_path in FILES_PATH:

    base = '/home/noda/Projects/romans-spectors/data/exo/mexo2b/'
    columns = ['X', 'Y', 'Wave', 'Intensity']

    df = pd.read_csv(base + data_path, sep='\s+', names=columns, header=0)

    unique_x, unique_y = list(set(df['X'])), list(set(df['Y']))

    result = []

    for x in unique_x:
        for y in unique_y:
            dot_df = df[(df['X'] == x) & (df['Y'] == y)]
            result.append(dot_df[['Wave', 'Intensity']])

    mean_spectrum = df.groupby('Wave')['Intensity'].mean().reset_index()

    indices, properties = find_peaks(mean_spectrum['Intensity'], prominence=50)

    all_peaks = pd.DataFrame({
        'Wave': mean_spectrum['Wave'].iloc[indices].values,
        'Intensity': mean_spectrum['Intensity'].iloc[indices].values,
        'Prominence': properties['prominences']
    })

    N = 6
    top_peaks = all_peaks.nlargest(N, 'Prominence')

    top_peaks = top_peaks.sort_values('Wave').reset_index(drop=True)

    print(f"Топ {N} пиков по выраженности:")
    print(top_peaks)

    top_peaks['Brain'] = 'striatum'
    top_peaks['Label'] = 'exo'

    top_peaks.to_csv(f'/home/noda/Projects/romans-spectors/data/mean/{data_path.split('/')[-1].split('.')[0]}-mean.csv', index=False, sep=';', encoding='utf-8-sig')
    print()
