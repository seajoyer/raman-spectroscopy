import pandas as pd
import numpy as np
import torch
from scipy.interpolate import interp1d
from torch import nn

file_path = '/home/noda/Projects/romans-spectors/data/cortex_endo_Average.txt'

df = pd.read_csv(file_path, sep='\s+')
df.columns = ['Wave', 'Intensity']

def transform_single_spectrum(df: pd.DataFrame, n_points: int = 466) -> pd.DataFrame:
    group = df.sort_values('Wave').groupby('Wave', as_index=False)['Intensity'].mean()
    group['Intensity'] = group['Intensity'] / group['Intensity'].mean()
    common_grid = np.linspace(group['Wave'].min(), group['Wave'].max(), n_points)
    interp_fn = interp1d(group['Wave'], group['Intensity'],
                         kind='linear', bounds_error=False, fill_value='extrapolate')
    intensities = interp_fn(common_grid)

    return pd.DataFrame([intensities], columns=common_grid)


df = transform_single_spectrum(df)


class RamanMultiNet(nn.Module):
    def __init__(self, input_dim):
        super(RamanMultiNet, self).__init__()
        self.net = nn.Sequential(

            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 16),
            nn.ReLU(),

            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)


model = RamanMultiNet(df.shape[1])

weights = torch.load('best_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(weights)

model.eval()

input_tensor = torch.tensor(df.values, dtype=torch.float32)

with torch.no_grad():
    test_outputs = model(input_tensor)
    _, predicted = torch.max(test_outputs, 1)

print(f"Предсказанный индекс класса: {predicted.item()}")
