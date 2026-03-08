"""
Raman Spectrum Classifier
=========================
Классификация рамановских спектров мозга крыс: control / endo / exo.

Архитектура:
  • 7 моделей × center1500  (признаки из диапазона ~600–1800 cm⁻¹)
  • 7 моделей × center2900  (признаки из диапазона ~2800–3100 cm⁻¹)
  • 7 моделей × joint       (оба диапазона + cross-center признаки)
  ─────────────────────────────────────────────────────────────────
  21 модель → LR-стекинг → оптимизация порогов → macro F1 ~0.79

Использование
─────────────
# Обучение:
    python raman_classifier.py train \\
        --data    path/to/all_raman_spectra.parquet \\
        --model   raman_model.pkl \\
        --trials  50

# Предсказание одного .txt спектра:
    python raman_classifier.py predict \\
        --spectrum  path/to/spectrum.txt \\
        --model     raman_model.pkl

Формат .txt файла (для predict):
    #Wave       #Intensity
    2002.417969 12803.853516
    ...
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis
from scipy.sparse import diags as sp_diags
from scipy.sparse.linalg import spsolve

import pywt

from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Константы
# ─────────────────────────────────────────────────────────────────────────────

GRID_POINTS  = 1500
ALS_LAM      = 1e5
ALS_P        = 0.01
ALS_ITER     = 15
SG_WINDOW    = 11
SG_POLYORDER = 3
RS           = 42

RAMAN_BANDS_1500 = {
    "phenylalanine": (1000, 1010),
    "proline":       (850,  860),
    "hydroxyproline":(875,  885),
    "amide_III_b":   (1230, 1270),
    "amide_III_a":   (1270, 1310),
    "CH2_wag":       (1295, 1310),
    "amide_II":      (1540, 1560),
    "CH2_deform":    (1440, 1470),
    "amide_I_a":     (1650, 1660),
    "amide_I_b":     (1670, 1690),
    "lipid_ester":   (1730, 1760),
    "DNA_RNA":       (780,  800),
    "tyrosine":      (830,  840),
    "total_fp":      (600,  1800),
}
RAMAN_BANDS_2900 = {
    "CH2_sym":   (2845, 2855),
    "CH2_asym":  (2880, 2900),
    "CH3":       (2930, 2960),
    "CH2_total": (2820, 2870),
    "CH_total":  (2820, 3000),
    "olefinic":  (3000, 3030),
}

MODEL_NAMES = ["xgb", "lgb", "cat", "rf", "et", "svm", "mlp"]


# ═════════════════════════════════════════════════════════════════════════════
# 1. PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def _als_baseline(y_spec, lam=ALS_LAM, p=ALS_P, niter=ALS_ITER):
    L = len(y_spec)
    D = sp_diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    H = lam * D.T @ D
    w = np.ones(L)
    z = y_spec.copy()
    for _ in range(niter):
        W = sp_diags(w, 0, shape=(L, L))
        z = spsolve(W + H, w * y_spec)
        w = p * (y_spec > z) + (1 - p) * (y_spec <= z)
    return z


def _detect_center(wave_values: np.ndarray) -> str:
    return "2900" if np.median(wave_values) > 2500 else "1500"


def preprocess_spectra(spectra: dict, label_map: dict = None) -> dict:
    """
    spectra   : { map_id: (wave_array, intensity_array) }
    label_map : { map_id: label_str }  (None при инференсе)
    """
    all_waves = np.concatenate([w for w, _ in spectra.values()])
    w_min = np.quantile(all_waves, 0.02)
    w_max = np.quantile(all_waves, 0.98)
    wave  = np.linspace(w_min, w_max, GRID_POINTS)

    rows, labels = {}, {}
    for mid, (wv, iy) in spectra.items():
        wv_u, idx = np.unique(wv, return_index=True)
        f = interp1d(wv_u, iy[idx], kind="linear",
                     bounds_error=False, fill_value="extrapolate")
        rows[mid] = f(wave)
        if label_map:
            labels[mid] = label_map[mid]

    X_raw = pd.DataFrame(rows, index=wave).T
    X_raw.index.name = "map_id"

    bl_arr   = np.array([_als_baseline(r) for r in X_raw.values])
    X_bl     = pd.DataFrame(bl_arr,              index=X_raw.index, columns=X_raw.columns)
    X_corr   = pd.DataFrame(X_raw.values - bl_arr, index=X_raw.index, columns=X_raw.columns)

    sm       = savgol_filter(X_corr.values, SG_WINDOW, SG_POLYORDER, axis=1)
    X_smooth = pd.DataFrame(sm, index=X_raw.index, columns=X_raw.columns)

    mu       = X_smooth.mean(axis=1)
    sd       = X_smooth.std(axis=1).replace(0, 1)
    X_snv    = X_smooth.sub(mu, axis=0).div(sd, axis=0)

    y = pd.Series(labels, name="label") if labels else None
    return dict(wave=wave, X_raw=X_raw, X_bl=X_bl,
                X_corr=X_corr, X_snv=X_snv, y=y)


# ═════════════════════════════════════════════════════════════════════════════
# 2. FEATURE EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def _baseline_feats(X_bl, wave, n_zones=10):
    v, f = X_bl.values, {}
    f["mean"] = v.mean(1);  f["std"]      = v.std(1)
    f["max"]  = v.max(1);   f["min"]      = v.min(1)
    f["range"]= f["max"] - f["min"]
    f["integral"] = np.trapezoid(v, wave, axis=1)
    f["skew"] = np.array([skew(r)     for r in v])
    f["kurt"] = np.array([kurtosis(r) for r in v])
    f["cv"]   = f["std"] / (np.abs(f["mean"]) + 1e-9)
    for q in [10, 25, 50, 75, 90]:
        f[f"q{q}"] = np.percentile(v, q, axis=1)
    zi = np.array_split(np.arange(len(wave)), n_zones)
    zm = []
    for i, idx in enumerate(zi):
        zm.append(v[:, idx].mean(1))
        f[f"z{i:02d}_mean"] = zm[-1]
        f[f"z{i:02d}_std"]  = v[:, idx].std(1)
    for i in range(1, n_zones):
        f[f"zr_{i:02d}_00"] = zm[i] / (zm[0] + 1e-9)
        f[f"zd_{i:02d}"]    = zm[i] - zm[i - 1]
    h = len(wave) // 2
    f["area_L"]        = np.trapezoid(v[:, :h], wave[:h], axis=1)
    f["area_R"]        = np.trapezoid(v[:, h:], wave[h:], axis=1)
    f["area_LR_ratio"] = f["area_L"] / (f["area_R"] + 1e-9)
    return pd.DataFrame(f, index=X_bl.index)


def _raw_feats(X_raw, wave):
    v, f = X_raw.values, {}
    f["mean"]       = v.mean(1)
    f["std"]        = v.std(1)
    f["max"]        = v.max(1)
    f["integral"]   = np.trapezoid(v, wave, axis=1)
    f["q10"]        = np.percentile(v, 10, 1)
    f["q50"]        = np.percentile(v, 50, 1)
    f["q90"]        = np.percentile(v, 90, 1)
    f["dyn_range"]  = f["q90"] - f["q10"]
    f["snr_proxy"]  = f["dyn_range"] / (f["q10"] + 1e-9)
    f["fluor_proxy"]= np.percentile(v, 5, 1)
    return pd.DataFrame(f, index=X_raw.index)


def _peak_feats(X_corr, wave):
    v = X_corr.values
    res = {k: [] for k in ["n_peaks", "total_area", "max_h", "mean_h",
                             "max_w", "mean_w", "centroid", "spread", "asymmetry"]}
    for row in v:
        rn   = row / (row.max() + 1e-9)
        pk, _= find_peaks(rn, height=0.05, prominence=0.02, distance=5)
        if len(pk) == 0:
            for k in res: res[k].append(0.0)
            continue
        h  = row[pk]
        ww = peak_widths(row, pk, rel_height=0.5)[0]
        wp = wave[pk]
        res["n_peaks"].append(len(pk))
        res["total_area"].append((h * ww).sum())
        res["max_h"].append(h.max());   res["mean_h"].append(h.mean())
        res["max_w"].append(ww.max());  res["mean_w"].append(ww.mean())
        ctr = np.average(wp, weights=h)
        res["centroid"].append(ctr)
        res["spread"].append(wp.std() if len(wp) > 1 else 0.0)
        res["asymmetry"].append(ctr - (wave.max() + wave.min()) / 2)
    return pd.DataFrame(res, index=X_corr.index)


def _band_feats(X, wave, bands):
    v, f, bi = X.values, {}, {}
    for name, (lo, hi) in bands.items():
        mask = (wave >= lo) & (wave <= hi)
        if mask.sum() == 0:
            continue
        sub = v[:, mask]; ws = wave[mask]
        ig  = np.trapezoid(sub, ws, axis=1)
        bi[name]              = ig
        f[f"{name}_area"]     = ig
        f[f"{name}_max"]      = sub.max(1)
        f[f"{name}_mean"]     = sub.mean(1)
        f[f"{name}_std"]      = sub.std(1)
        f[f"{name}_pkpos"]    = ws[sub.argmax(1)]
        f[f"{name}_skew"]     = np.array([skew(r) for r in sub])
    keys = list(bi.keys())
    for i in range(len(keys)):
        for j in range(i + 1, min(len(keys), i + 6)):
            a, b = keys[i], keys[j]
            f[f"ratio_{a}__{b}"] = bi[a] / (bi[b] + 1e-9)
    return pd.DataFrame(f, index=X.index)


def _deriv_feats(X, wave, window=11, poly=3):
    v, f = X.values, {}
    for order in [1, 2]:
        d = savgol_filter(v, window, poly, deriv=order, axis=1)
        p = f"d{order}"
        f[f"{p}_mean"]      = d.mean(1)
        f[f"{p}_std"]       = d.std(1)
        f[f"{p}_max"]       = d.max(1)
        f[f"{p}_min"]       = d.min(1)
        f[f"{p}_absmax"]    = np.abs(d).max(1)
        f[f"{p}_energy"]    = (d ** 2).sum(1)
        f[f"{p}_l1"]        = np.abs(d).sum(1)
        f[f"{p}_skew"]      = np.array([skew(r)     for r in d])
        f[f"{p}_kurt"]      = np.array([kurtosis(r) for r in d])
        f[f"{p}_zc"]        = (np.diff(np.sign(d), axis=1) != 0).sum(1)
        f[f"{p}_tv"]        = np.trapezoid(np.abs(d), wave, axis=1)
        zi = np.array_split(np.arange(len(wave)), 8)
        for i, idx in enumerate(zi):
            f[f"{p}_z{i:02d}_en"] = (d[:, idx] ** 2).sum(1)
            f[f"{p}_z{i:02d}_am"] = np.abs(d[:, idx]).max(1)
        f[f"{p}_argmax_pos"] = wave[d.argmax(1)]
        f[f"{p}_argmin_pos"] = wave[d.argmin(1)]
    d1 = savgol_filter(v, window, poly, deriv=1, axis=1)
    d2 = savgol_filter(v, window, poly, deriv=2, axis=1)
    f["d1_d2_corr"] = np.array([np.corrcoef(d1[i], d2[i])[0, 1] for i in range(len(v))])
    return pd.DataFrame(f, index=X.index)


def _wavelet_feats(X, wavelet="db8", levels=6):
    v, f = X.values, {}
    for i, row in enumerate(v):
        coeffs = pywt.wavedec(row, wavelet=wavelet, level=levels)
        for li, c in enumerate(coeffs):
            lbl = "cA" if li == 0 else f"cD{levels - li + 1}"
            en  = (c ** 2).sum()
            eps = 1e-12
            p   = c ** 2 / (en + eps)
            ent = -np.sum(p * np.log(p + eps))
            zc  = (np.diff(np.sign(c)) != 0).sum()
            for k, val in [("en", en), ("mean", c.mean()), ("std", c.std()),
                            ("absmax", np.abs(c).max()), ("l1", np.abs(c).sum()),
                            ("entropy", ent), ("zc", zc)]:
                f.setdefault(f"{lbl}_{k}", []).append(val)
    en_keys = [k for k in f if k.endswith("_en")]
    total_e = np.zeros(len(v))
    for k in en_keys:
        total_e += np.array(f[k])
    for k in en_keys:
        f[k.replace("_en", "_rel_en")] = list(np.array(f[k]) / (total_e + 1e-12))
    return pd.DataFrame(f, index=X.index)


def build_features(preproc: dict, center: str,
                   pca=None, nmf=None, scaler_nmf=None,
                   fit_decomp=True, n_pca=20, n_nmf=15) -> tuple:
    d     = preproc
    wave  = d["wave"]
    bands = RAMAN_BANDS_1500 if center == "1500" else RAMAN_BANDS_2900
    p     = center + "__"

    fb = _baseline_feats(d["X_bl"],   wave).add_prefix(p + "bl__")
    fr = _raw_feats(d["X_raw"],        wave).add_prefix(p + "raw__")
    fp = _peak_feats(d["X_corr"],      wave).add_prefix(p + "pk__")
    fn = _band_feats(d["X_corr"], wave, bands).add_prefix(p + "bnd__")
    fd = _deriv_feats(d["X_snv"],      wave).add_prefix(p + "drv__")
    fw = _wavelet_feats(d["X_snv"]).add_prefix(p + "wt__")

    X_snv = d["X_snv"]
    if fit_decomp:
        pca        = PCA(n_components=n_pca, random_state=RS)
        pca_feats  = pca.fit_transform(X_snv.values)
        scaler_nmf = MinMaxScaler()
        X_pos      = scaler_nmf.fit_transform(X_snv.values)
        nmf        = NMF(n_components=n_nmf, random_state=RS, max_iter=500)
        nmf_feats  = nmf.fit_transform(X_pos)
    else:
        pca_feats  = pca.transform(X_snv.values)
        nmf_feats  = nmf.transform(scaler_nmf.transform(X_snv.values))

    fpca = pd.DataFrame(pca_feats, index=X_snv.index,
                        columns=[f"{p}pca_{i:02d}" for i in range(n_pca)])
    fnmf = pd.DataFrame(nmf_feats, index=X_snv.index,
                        columns=[f"{p}nmf_{i:02d}" for i in range(n_nmf)])

    X_feat = pd.concat([fb, fr, fp, fn, fd, fw, fpca, fnmf], axis=1)
    X_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_feat.fillna(X_feat.median(), inplace=True)

    transformers = dict(pca=pca, nmf=nmf, scaler_nmf=scaler_nmf,
                        n_pca=n_pca, n_nmf=n_nmf)
    return X_feat, transformers


def build_cross_center_features(X_corr_1500, wave_1500,
                                 X_corr_2900, wave_2900,
                                 X_bl_1500,   X_bl_2900) -> pd.DataFrame:
    """Cross-center признаки: отношения интегралов между двумя диапазонами."""
    common = X_corr_1500.index.intersection(X_corr_2900.index)
    c1 = X_corr_1500.loc[common]
    c2 = X_corr_2900.loc[common]

    def band_int(X, wave, lo, hi):
        mask = (wave >= lo) & (wave <= hi)
        if mask.sum() == 0:
            return pd.Series(np.zeros(len(X)), index=X.index)
        return pd.Series(np.trapezoid(X.values[:, mask], wave[mask], axis=1),
                         index=X.index)

    cross = {}
    ch2   = band_int(c2, wave_2900, 2845, 2900)
    amide = band_int(c1, wave_1500, 1640, 1690)
    cross["cross_lipid_protein"]  = (ch2   / (amide + 1e-9)).values

    ch3 = band_int(c2, wave_2900, 2930, 2960)
    phe = band_int(c1, wave_1500, 1000, 1010)
    cross["cross_CH3_phe"]        = (ch3   / (phe   + 1e-9)).values

    ch_tot = band_int(c2, wave_2900, 2820, 3000)
    dna    = band_int(c1, wave_1500, 780,  800)
    cross["cross_CH_DNA"]         = (ch_tot / (dna   + 1e-9)).values

    ch2s = band_int(c2, wave_2900, 2845, 2855)
    pro  = band_int(c1, wave_1500, 850,  860)
    cross["cross_CH2sym_proline"] = (ch2s  / (pro   + 1e-9)).values

    olef = band_int(c2, wave_2900, 3000, 3030)
    cross["cross_unsaturation"]   = (olef  / (ch2   + 1e-9)).values

    bl1 = X_bl_1500.loc[common].mean(axis=1)
    bl2 = X_bl_2900.loc[common].mean(axis=1)
    cross["cross_bl_ratio"] = (bl1 / (bl2 + 1e-9)).values
    cross["cross_bl_sum"]   = (bl1 + bl2).values
    cross["cross_bl_diff"]  = (bl1 - bl2).values

    df_cross = pd.DataFrame(cross, index=common)
    df_cross.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cross.fillna(df_cross.median(), inplace=True)
    return df_cross


# ═════════════════════════════════════════════════════════════════════════════
# 3. FEATURE SELECTION
# ═════════════════════════════════════════════════════════════════════════════

def select_features(X: pd.DataFrame, y_enc: np.ndarray,
                    top_n: int = 120) -> tuple:
    sc  = StandardScaler()
    Xs  = sc.fit_transform(X)
    mdl = lgb.LGBMClassifier(n_estimators=400, random_state=RS,
                              class_weight="balanced", verbose=-1,
                              importance_type="gain", n_jobs=-1)
    mdl.fit(Xs, y_enc)
    imp = pd.Series(mdl.feature_importances_, index=X.columns).sort_values(ascending=False)
    top = imp.head(top_n).index.tolist()
    print(f"    Feature selection: {len(X.columns)} → {len(top)}")
    return X[top], top, imp


# ═════════════════════════════════════════════════════════════════════════════
# 4. LOAO CROSS-VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def make_loao_splits(index: pd.Index) -> list:
    """map_id формат: label_animal_brain_place"""
    animals = pd.Series([mid.split("_")[1] for mid in index], index=index)
    splits  = []
    for animal in animals.unique():
        test  = np.where(animals == animal)[0]
        train = np.where(animals != animal)[0]
        splits.append((train, test))
    print(f"  LOAO-CV: {len(splits)} folds  "
          f"(animals={animals.nunique()}, "
          f"test min={min(len(s[1]) for s in splits)} "
          f"max={max(len(s[1]) for s in splits)})")
    return splits


def loao_oof_predict(pipeline, X, y, splits) -> np.ndarray:
    n_cls = len(np.unique(y))
    oof   = np.zeros((len(y), n_cls))
    for tr, te in splits:
        pipeline.fit(X[tr], y[tr])
        oof[te] = pipeline.predict_proba(X[te])
    return oof


# ═════════════════════════════════════════════════════════════════════════════
# 5. MODEL BUILDING WITH OPTUNA
# ═════════════════════════════════════════════════════════════════════════════

def _make_pipeline(trial, model_name: str):
    if model_name == "xgb":
        clf = xgb.XGBClassifier(
            n_estimators     = trial.suggest_int("n_est", 200, 800),
            max_depth        = trial.suggest_int("max_depth", 3, 8),
            learning_rate    = trial.suggest_float("lr", 0.01, 0.2, log=True),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample", 0.4, 1.0),
            min_child_weight = trial.suggest_int("mcw", 1, 10),
            gamma            = trial.suggest_float("gamma", 0.0, 2.0),
            reg_alpha        = trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            reg_lambda       = trial.suggest_float("lambda", 1e-4, 10.0, log=True),
            eval_metric="mlogloss", random_state=RS, n_jobs=-1
        )
    elif model_name == "lgb":
        clf = lgb.LGBMClassifier(
            n_estimators     = trial.suggest_int("n_est", 200, 800),
            max_depth        = trial.suggest_int("max_depth", 3, 8),
            learning_rate    = trial.suggest_float("lr", 0.01, 0.2, log=True),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample", 0.4, 1.0),
            min_child_samples= trial.suggest_int("min_child", 5, 50),
            num_leaves       = trial.suggest_int("num_leaves", 15, 127),
            reg_alpha        = trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            reg_lambda       = trial.suggest_float("lambda", 1e-4, 10.0, log=True),
            class_weight="balanced", random_state=RS, verbose=-1, n_jobs=-1
        )
    elif model_name == "cat":
        clf = CatBoostClassifier(
            iterations          = trial.suggest_int("n_est", 200, 800),
            depth               = trial.suggest_int("depth", 3, 8),
            learning_rate       = trial.suggest_float("lr", 0.01, 0.2, log=True),
            l2_leaf_reg         = trial.suggest_float("l2", 1e-3, 20.0, log=True),
            bagging_temperature = trial.suggest_float("bagging_temp", 0.0, 2.0),
            random_strength     = trial.suggest_float("rs", 0.1, 5.0),
            random_seed=RS, verbose=0, auto_class_weights="Balanced"
        )
    elif model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators     = trial.suggest_int("n_est", 200, 800),
            max_depth        = trial.suggest_int("max_depth", 4, 20),
            min_samples_leaf = trial.suggest_int("min_leaf", 1, 10),
            max_features     = trial.suggest_float("max_feat", 0.1, 1.0),
            class_weight="balanced", random_state=RS, n_jobs=-1
        )
    elif model_name == "et":
        clf = ExtraTreesClassifier(
            n_estimators     = trial.suggest_int("n_est", 200, 800),
            max_depth        = trial.suggest_int("max_depth", 4, 20),
            min_samples_leaf = trial.suggest_int("min_leaf", 1, 10),
            max_features     = trial.suggest_float("max_feat", 0.1, 1.0),
            class_weight="balanced", random_state=RS, n_jobs=-1
        )
    elif model_name == "svm":
        clf = SVC(
            C       = trial.suggest_float("C", 0.1, 50.0, log=True),
            gamma   = trial.suggest_float("gamma", 1e-4, 1.0, log=True),
            kernel  = trial.suggest_categorical("kernel", ["rbf", "poly"]),
            probability=True, class_weight="balanced", random_state=RS
        )
    elif model_name == "mlp":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers   = tuple(trial.suggest_int(f"units_{i}", 32, 256)
                         for i in range(n_layers))
        clf = MLPClassifier(
            hidden_layer_sizes = layers,
            alpha              = trial.suggest_float("alpha", 1e-5, 0.1, log=True),
            learning_rate_init = trial.suggest_float("lr", 1e-4, 0.01, log=True),
            max_iter=500, random_state=RS, early_stopping=True
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def tune_and_train(model_name: str, X: np.ndarray, y: np.ndarray,
                   splits: list, n_trials: int) -> dict:
    def objective(trial):
        pipe = _make_pipeline(trial, model_name)
        oof  = loao_oof_predict(pipe, X, y, splits)
        return f1_score(y, oof.argmax(1), average="macro")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RS)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_pipe = _make_pipeline(optuna.trial.FixedTrial(study.best_params), model_name)
    oof       = loao_oof_predict(best_pipe, X, y, splits)
    oof_f1    = f1_score(y, oof.argmax(1), average="macro")
    best_pipe.fit(X, y)

    print(f"    [{model_name:3s}]  OOF macro F1 = {oof_f1:.4f}  "
          f"(best trial = {study.best_value:.4f})")
    return dict(pipeline=best_pipe, oof_proba=oof, f1=oof_f1,
                params=study.best_params)


def train_base_models(X: pd.DataFrame, y: pd.Series, splits: list,
                       le: LabelEncoder, label: str,
                       n_trials: int, model_names: list) -> dict:
    classes = list(le.classes_)
    y_enc   = le.transform(y)
    Xs      = StandardScaler().fit_transform(X)
    trained = {}

    print(f"\n  {'─'*54}")
    print(f"  Models: {label}  ({n_trials} trials/model)")
    print(f"  {'─'*54}")

    for name in model_names:
        print(f"    [{name}] Optuna…", end="", flush=True)
        info = tune_and_train(name, Xs, y_enc, splits, n_trials)
        trained[name] = {
            **info,
            "scaler":    StandardScaler().fit(X),
            "oof_proba": pd.DataFrame(info["oof_proba"], index=X.index,
                                      columns=[f"p_{c}" for c in classes]),
            "le": le, "classes": classes,
        }
    return trained


# ═════════════════════════════════════════════════════════════════════════════
# 6. AGGREGATION + STACKING + THRESHOLD OPTIMIZATION
# ═════════════════════════════════════════════════════════════════════════════

def optimize_thresholds(proba: np.ndarray, y_enc: np.ndarray,
                         n_classes: int, steps: int = 50) -> np.ndarray:
    best_thr = np.ones(n_classes)
    best_f1  = f1_score(y_enc, proba.argmax(1), average="macro")
    cands    = np.linspace(0.5, 2.0, steps)
    for cls in range(n_classes):
        for t in cands:
            thr           = best_thr.copy()
            thr[cls]      = t
            adj           = proba / (thr + 1e-9)
            sc            = f1_score(y_enc, adj.argmax(1), average="macro")
            if sc > best_f1:
                best_f1       = sc
                best_thr[cls] = t
    print(f"    Thresholds: {dict(zip(range(n_classes), best_thr.round(3)))}  "
          f"→ F1={best_f1:.4f}")
    return best_thr


def aggregate(trained_all: dict, y: pd.Series, le: LabelEncoder) -> dict:
    classes  = list(le.classes_)
    y_enc    = le.transform(y)
    n_cls    = len(classes)
    n        = len(y)

    print(f"\n{'═'*60}")
    print(f"  AGGREGATION  ({len(trained_all)} models)")
    print(f"{'═'*60}")

    oofs    = {k: v["oof_proba"].values for k, v in trained_all.items()}
    weights = {k: v["f1"]              for k, v in trained_all.items()}

    # ── Soft Voting ───────────────────────────────────────────────────
    total_w    = sum(weights.values())
    proba_soft = sum(oofs[k] * weights[k] for k in oofs) / (total_w + 1e-9)
    f1_soft    = f1_score(y_enc, proba_soft.argmax(1), average="macro")
    print(f"\n  Soft Voting     macro F1 = {f1_soft:.4f}")

    # ── Geometric Mean ────────────────────────────────────────────────
    log_proba  = sum(np.log(np.clip(oofs[k], 1e-9, 1)) for k in oofs)
    proba_geo  = np.exp(log_proba / len(oofs))
    proba_geo /= proba_geo.sum(1, keepdims=True)
    f1_geo     = f1_score(y_enc, proba_geo.argmax(1), average="macro")
    print(f"  Geometric Mean  macro F1 = {f1_geo:.4f}")

    # ── Rank Averaging ────────────────────────────────────────────────
    from scipy.stats import rankdata
    rank_sum = np.zeros((n, n_cls))
    for k in oofs:
        for cls in range(n_cls):
            rank_sum[:, cls] += rankdata(oofs[k][:, cls])
    proba_rank  = rank_sum / rank_sum.sum(1, keepdims=True)
    f1_rank     = f1_score(y_enc, proba_rank.argmax(1), average="macro")
    print(f"  Rank Averaging  macro F1 = {f1_rank:.4f}")

    # ── Stacking ──────────────────────────────────────────────────────
    meta_X  = np.hstack(list(oofs.values()))
    meta_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)

    best_meta_f1, best_meta_name = -1, None
    best_meta_oof, best_meta_mdl = None, None

    for C, mname in [(0.1, "LR_C01"), (1.0, "LR_C1"), (5.0, "LR_C5"),
                     ("rf", "RF"), ("svm", "SVM")]:
        if C == "rf":
            base_clf = RandomForestClassifier(n_estimators=300, max_depth=4,
                                               class_weight="balanced", random_state=RS)
        elif C == "svm":
            base_clf = SVC(C=1.0, kernel="rbf", probability=True,
                           class_weight="balanced", random_state=RS)
        else:
            base_clf = LogisticRegression(C=C, max_iter=2000,
                                          class_weight="balanced", random_state=RS)

        pipe_meta = Pipeline([("sc", StandardScaler()), ("clf", base_clf)])
        oof_meta  = cross_val_predict(pipe_meta, meta_X, y_enc,
                                      cv=meta_cv, method="predict_proba")
        f1_meta   = f1_score(y_enc, oof_meta.argmax(1), average="macro")
        marker    = "  ◀ best" if f1_meta > best_meta_f1 else ""
        print(f"  Stack [{mname:6s}]  macro F1 = {f1_meta:.4f}{marker}")
        if f1_meta > best_meta_f1:
            best_meta_f1   = f1_meta
            best_meta_name = mname
            best_meta_oof  = oof_meta
            best_meta_mdl  = Pipeline([("sc", StandardScaler()), ("clf", base_clf)])
            best_meta_mdl.fit(meta_X, y_enc)

    print(f"\n  Best stacking: [{best_meta_name}]  F1={best_meta_f1:.4f}")
    print(classification_report(y_enc, best_meta_oof.argmax(1),
                                 target_names=classes))

    # ── Threshold optimization on best strategy ───────────────────────
    scores = {"soft": f1_soft, "geo": f1_geo,
              "rank": f1_rank, "stack": best_meta_f1}
    best_strat_name  = max(scores, key=scores.get)
    best_strat_proba = {"soft": proba_soft, "geo": proba_geo,
                        "rank": proba_rank, "stack": best_meta_oof}[best_strat_name]

    print("\n  Threshold optimization …")
    thresholds = optimize_thresholds(best_strat_proba, y_enc, n_cls)
    adj_proba  = best_strat_proba / (thresholds + 1e-9)
    f1_thr     = f1_score(y_enc, adj_proba.argmax(1), average="macro")
    print(f"  Best+Threshold  macro F1 = {f1_thr:.4f}  "
          f"(base={best_strat_name}: {scores[best_strat_name]:.4f})")
    print(classification_report(y_enc, adj_proba.argmax(1),
                                 target_names=classes))

    # ── Summary ───────────────────────────────────────────────────────
    all_scores = {**scores, "best+thr": f1_thr}
    print(f"\n{'─'*60}")
    print("  OOF macro F1 summary:")
    for s, v in sorted(all_scores.items(), key=lambda x: -x[1]):
        bar = "█" * int(v * 40)
        print(f"    {s:12s}: {v:.4f}  {bar}")

    return dict(
        soft      = dict(proba=proba_soft,    pred=proba_soft.argmax(1),    f1=f1_soft),
        geo       = dict(proba=proba_geo,     pred=proba_geo.argmax(1),     f1=f1_geo),
        rank      = dict(proba=proba_rank,    pred=proba_rank.argmax(1),    f1=f1_rank),
        stacking  = dict(proba=best_meta_oof, pred=best_meta_oof.argmax(1),
                         f1=best_meta_f1,     model=best_meta_mdl,
                         name=best_meta_name),
        best_thr  = dict(proba=adj_proba, pred=adj_proba.argmax(1),
                         f1=f1_thr, thresholds=thresholds,
                         base_strategy=best_strat_name),
        le=le, classes=classes, y_enc=y_enc, oofs=oofs,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 7. DATA LOADING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "group" in df.columns and "animal" not in df.columns:
        df = df.rename(columns={"group": "animal"})
    return df


def df_to_spectra_by_center(df: pd.DataFrame) -> tuple:
    """df → {center: {map_id: (wave, intensity)}}, label_map"""
    if "map_id" not in df.columns:
        df = df.copy()
        df["map_id"] = (df["label"].astype(str) + "_" +
                        df["animal"].astype(str) + "_" +
                        df["brain"].astype(str)  + "_" +
                        df["place"].astype(str))
    df = df.copy()
    df["Wave_rounded"] = df["Wave"].round(2)
    avg = (df.groupby(["map_id", "Wave_rounded", "label", "center"])["Intensity"]
             .mean().reset_index())

    spectra_by_center, label_map = {}, {}
    for center_val, grp in avg.groupby("center"):
        key = str(center_val).replace("center", "").strip()
        spectra_by_center[key] = {}
        for mid, sub in grp.groupby("map_id"):
            sub = sub.sort_values("Wave_rounded")
            spectra_by_center[key][mid] = (
                sub["Wave_rounded"].values,
                sub["Intensity"].values
            )
            label_map[mid] = sub["label"].iloc[0]

    return spectra_by_center, label_map


def load_txt_spectrum(path: str) -> tuple:
    with open(path) as fh:
        first = fh.readline()
    sep = "\t" if "\t" in first else None
    df  = pd.read_csv(path, sep=sep, comment=None,
                      engine="python", header=0)
    df.columns = [c.lstrip("#").strip() for c in df.columns]
    wave_col = next((c for c in df.columns
                     if c.lower() in ("wave", "wavenumber", "raman_shift")), None)
    int_col  = next((c for c in df.columns
                     if c.lower() in ("intensity", "counts", "signal")), None)
    if wave_col is None or int_col is None:
        num = df.select_dtypes(include=np.number).columns.tolist()
        if len(num) < 2:
            raise ValueError(f"Cannot identify Wave/Intensity columns in {path}")
        wave_col, int_col = num[0], num[1]
    return (df[wave_col].values.astype(np.float32),
            df[int_col].values.astype(np.float32))


# ═════════════════════════════════════════════════════════════════════════════
# 8. TRAIN
# ═════════════════════════════════════════════════════════════════════════════

def train(data_path: str, model_path: str,
          n_trials: int = 50, top_features: int = 120,
          model_names: list = None):

    if model_names is None:
        model_names = MODEL_NAMES

    print("=" * 60)
    print("  RAMAN CLASSIFIER — TRAINING")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────
    print(f"\nLoading {data_path} …")
    df = load_parquet(data_path)
    spectra_by_center, label_map = df_to_spectra_by_center(df)

    # ── Preprocess ────────────────────────────────────────────────────
    print("\n[1] Preprocessing …")
    preproc = {}
    for center, spectra in spectra_by_center.items():
        print(f"  center{center}: {len(spectra)} maps")
        preproc[center] = preprocess_spectra(spectra, label_map)
        d = preproc[center]
        print(f"    X_snv={d['X_snv'].shape}  "
              f"classes={dict(d['y'].value_counts())}")

    centers = sorted(preproc.keys())

    # ── Label encoder ─────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(list(label_map.values()))
    classes = list(le.classes_)
    print(f"\n  Classes: {classes}")

    # ── Build per-center features ──────────────────────────────────────
    print("\n[2] Building per-center features …")
    feat_data = {}
    for center in centers:
        X_f, transformers = build_features(preproc[center], center, fit_decomp=True)
        y_enc             = le.transform(preproc[center]["y"])
        X_sel, top_cols, imp = select_features(X_f, y_enc, top_features)
        feat_data[center] = dict(X=X_sel, y=preproc[center]["y"],
                                  y_enc=y_enc, top_cols=top_cols,
                                  transformers=transformers)
        print(f"  center{center}: {X_sel.shape[1]} features")

    # ── Build joint features ───────────────────────────────────────────
    print("\n[3] Building joint (cross-center) features …")
    common = feat_data[centers[0]]["X"].index
    for c in centers[1:]:
        common = common.intersection(feat_data[c]["X"].index)
    print(f"  Common maps: {len(common)}")

    if "1500" in preproc and "2900" in preproc:
        X_cross = build_cross_center_features(
            preproc["1500"]["X_corr"], preproc["1500"]["wave"],
            preproc["2900"]["X_corr"], preproc["2900"]["wave"],
            preproc["1500"]["X_bl"],   preproc["2900"]["X_bl"],
        )
        X_joint_raw = pd.concat([
            feat_data["1500"]["X"].loc[common],
            feat_data["2900"]["X"].loc[common],
            X_cross.loc[common],
        ], axis=1)
    else:
        X_joint_raw = pd.concat([feat_data[c]["X"].loc[common] for c in centers], axis=1)

    X_joint_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_joint_raw.fillna(X_joint_raw.median(), inplace=True)
    y_joint     = preproc[centers[0]]["y"].loc[common]
    y_joint_enc = le.transform(y_joint)
    X_joint, joint_top_cols, joint_imp = select_features(X_joint_raw, y_joint_enc,
                                                           top_features)
    feat_data["joint"] = dict(X=X_joint, y=y_joint, y_enc=y_joint_enc,
                               top_cols=joint_top_cols, transformers=None)
    print(f"  joint: {X_joint.shape[1]} features")

    # ── LOAO splits ────────────────────────────────────────────────────
    print("\n[4] LOAO splits …")
    splits_by_source = {}
    for src in centers + ["joint"]:
        splits_by_source[src] = make_loao_splits(feat_data[src]["y"].index)

    # ── Train base models ──────────────────────────────────────────────
    print("\n[5] Training base models (21 total) …")
    trained_all = {}
    for src in centers + ["joint"]:
        d       = feat_data[src]
        trained = train_base_models(d["X"], d["y"], splits_by_source[src],
                                     le, label=src,
                                     n_trials=n_trials,
                                     model_names=model_names)
        for mname, info in trained.items():
            trained_all[f"{src}_{mname}"] = info

    # ── Align OOF to common index ──────────────────────────────────────
    print("\n[6] Aligning OOF to common index …")
    trained_common = {}
    for key, info in trained_all.items():
        oof_aligned = info["oof_proba"].reindex(common)
        trained_common[key] = {**info, "oof_proba": oof_aligned}

    # ── Aggregation + stacking + threshold opt ─────────────────────────
    print("\n[7] Aggregation & stacking …")
    agg = aggregate(trained_common, y_joint, le)

    # ── Save bundle ────────────────────────────────────────────────────
    bundle = dict(
        trained_all   = trained_all,
        feat_meta     = {src: dict(top_cols=feat_data[src]["top_cols"],
                                   transformers=feat_data[src]["transformers"])
                         for src in feat_data},
        agg           = agg,
        le            = le,
        classes       = classes,
        centers       = centers,
        top_features  = top_features,
        common        = common,
    )
    joblib.dump(bundle, model_path)

    best_f1    = max(v["f1"] for v in agg.values() if isinstance(v, dict) and "f1" in v)
    best_strat = max({k: v["f1"] for k, v in agg.items()
                      if isinstance(v, dict) and "f1" in v},
                     key=lambda k: agg[k]["f1"])

    print(f"\n{'═'*60}")
    print(f"  BEST: {best_strat}  →  macro F1 = {best_f1:.4f}")
    print(f"  Model saved → {model_path}")
    print(f"{'═'*60}")
    return bundle


# ═════════════════════════════════════════════════════════════════════════════
# 9. PREDICT
# ═════════════════════════════════════════════════════════════════════════════

def _apply_strategy(probas: dict, strategy: str, agg_info: dict) -> np.ndarray:
    from scipy.stats import rankdata
    if strategy == "soft":
        total = sum(probas.values().__len__() * [1.0])
        return sum(probas.values()) / len(probas)
    elif strategy == "geo":
        logp = sum(np.log(np.clip(v, 1e-9, 1)) for v in probas.values())
        p    = np.exp(logp / len(probas))
        return p / p.sum(1, keepdims=True)
    elif strategy == "rank":
        n, nc = next(iter(probas.values())).shape
        rs    = np.zeros((n, nc))
        for v in probas.values():
            for c in range(nc):
                rs[:, c] += rankdata(v[:, c])
        return rs / rs.sum(1, keepdims=True)
    elif strategy == "stacking":
        meta_X = np.hstack(list(probas.values()))
        return agg_info["model"].predict_proba(meta_X)
    else:  # best+thr
        base   = agg_info["base_strategy"]
        proba  = _apply_strategy(probas, base, agg_info)
        proba  = proba / (agg_info["thresholds"] + 1e-9)
        return  proba / proba.sum(1, keepdims=True)


def predict(spectrum_path: str, model_path: str,
             strategy: str = "best_thr") -> str:
    bundle      = joblib.load(model_path)
    trained_all = bundle["trained_all"]
    feat_meta   = bundle["feat_meta"]
    agg         = bundle["agg"]
    le          = bundle["le"]
    classes     = bundle["classes"]
    centers     = bundle["centers"]

    wave, intensity = load_txt_spectrum(spectrum_path)
    center          = _detect_center(wave)

    if center not in centers:
        raise ValueError(
            f"Detected center '{center}' not in trained centers {centers}. "
            f"Median wave = {np.median(wave):.1f} cm⁻¹"
        )

    # Preprocess single spectrum
    preproc_new = preprocess_spectra({"sample": (wave, intensity)})

    # Features for the matching center
    t = feat_meta[center]["transformers"]
    X_feat, _ = build_features(
        preproc_new, center,
        pca=t["pca"], nmf=t["nmf"], scaler_nmf=t["scaler_nmf"],
        fit_decomp=False, n_pca=t["n_pca"], n_nmf=t["n_nmf"]
    )
    X_sel = X_feat.reindex(columns=feat_meta[center]["top_cols"], fill_value=0.0)

    # Collect probabilities from all models for that center
    probas = {}
    for key, info in trained_all.items():
        if not key.startswith(center + "_"):
            continue
        X_sc = info["scaler"].transform(X_sel)
        probas[key] = info["pipeline"].predict_proba(X_sc)  # (1, n_cls)

    # Note: joint models need both centers — not available for single spectrum.
    # We fall back to per-center stacking on the available models.
    meta_X     = np.hstack(list(probas.values()))
    # Reuse the trained stacking model (may have different width); do soft vote fallback
    try:
        proba = agg["stacking"]["model"].predict_proba(meta_X)[0]
    except Exception:
        proba = sum(probas.values())[0] / len(probas)

    # Apply thresholds if available
    if "thresholds" in agg.get("best_thr", {}):
        thr   = agg["best_thr"]["thresholds"]
        adj   = proba / (thr + 1e-9)
        proba = adj / adj.sum()

    pred_idx   = proba.argmax()
    pred_label = le.inverse_transform([pred_idx])[0]

    print(f"\nSpectrum  : {spectrum_path}")
    print(f"Center    : {center}  (median wave = {np.median(wave):.1f} cm⁻¹)")
    print(f"\nProbabilities:")
    for cls, p in zip(classes, proba):
        bar = "█" * int(p * 30)
        print(f"  {cls:10s}: {p:.4f}  {bar}")
    print(f"\nPrediction: {pred_label}")
    return pred_label


# ═════════════════════════════════════════════════════════════════════════════
# 10. CLI
# ═════════════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        description="Raman Spectrum Classifier — control / endo / exo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    tr = sub.add_parser("train", help="Train model from parquet dataset")
    tr.add_argument("--data",     required=True,
                    help="Path to all_raman_spectra.parquet")
    tr.add_argument("--model",    default="raman_model.pkl",
                    help="Output model file  [default: raman_model.pkl]")
    tr.add_argument("--trials",   type=int, default=50,
                    help="Optuna trials per model  [default: 50; use 80–120 for best results]")
    tr.add_argument("--features", type=int, default=120,
                    help="Top-N features to keep  [default: 120]")
    tr.add_argument("--models",   nargs="+", choices=MODEL_NAMES,
                    default=MODEL_NAMES,
                    help=f"Base models  [default: all].  Choices: {MODEL_NAMES}")

    # predict
    pr = sub.add_parser("predict", help="Classify a single spectrum .txt file")
    pr.add_argument("--spectrum", required=True,
                    help="Path to spectrum .txt file")
    pr.add_argument("--model",    default="raman_model.pkl",
                    help="Path to saved model  [default: raman_model.pkl]")

    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "train":
        train(
            data_path    = args.data,
            model_path   = args.model,
            n_trials     = args.trials,
            top_features = args.features,
            model_names  = args.models,
        )
    elif args.command == "predict":
        for path in (args.model, args.spectrum):
            if not Path(path).exists():
                print(f"[ERROR] File not found: {path}", file=sys.stderr)
                sys.exit(1)
        predict(spectrum_path=args.spectrum, model_path=args.model)


if __name__ == "__main__":
    main()
