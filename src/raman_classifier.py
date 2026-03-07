#!/usr/bin/env python3

"""
Raman Spectrum Classifier
=========================
Классификация рамановских спектров мозга крыс: control / endo / exo.

Использование
-------------
# Обучение (читает parquet, сохраняет модель):
    python raman_classifier.py train \
        --data    path/to/all_raman_spectra.parquet \
        --model   raman_model.pkl \
        --trials  50

# Предсказание (читает один .txt спектр):
    python raman_classifier.py predict \
        --spectrum  path/to/spectrum.txt \
        --model     raman_model.pkl

Формат входного .txt файла (для predict):
    #Wave       #Intensity
    2002.417969 12803.853516
    2001.458008 13013.024414
    ...

Примечание: при обучении автоматически определяется диапазон спектра
(center1500 ≈ 600–1800 cm⁻¹  или  center2900 ≈ 2800–3100 cm⁻¹).
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from itertools import combinations

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
from sklearn.metrics import f1_score, classification_report

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Константы
# ─────────────────────────────────────────────────────────────────────────────

GRID_POINTS   = 1500
ALS_LAM       = 1e5
ALS_P         = 0.01
ALS_ITER      = 15
SG_WINDOW     = 11
SG_POLYORDER  = 3
RS            = 42

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


# ─────────────────────────────────────────────────────────────────────────────
# 1. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _als_baseline(y_spec, lam=ALS_LAM, p=ALS_P, niter=ALS_ITER):
    L = len(y_spec)
    D = sp_diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    H = lam * D.T @ D
    w = np.ones(L)
    z = y_spec.copy()
    for _ in range(niter):
        W  = sp_diags(w, 0, shape=(L, L))
        z  = spsolve(W + H, w * y_spec)
        w  = p * (y_spec > z) + (1 - p) * (y_spec <= z)
    return z


def _detect_center(wave_values: np.ndarray) -> str:
    """Определяем диапазон спектра по медиане длин волн."""
    med = np.median(wave_values)
    return "2900" if med > 2500 else "1500"


def preprocess_spectra(spectra: dict, label_map: dict = None) -> dict:
    """
    spectra : { map_id: (wave_array, intensity_array) }
    label_map: { map_id: label_str }  (None при инференсе)

    Возвращает dict: wave, X_raw, X_bl, X_corr, X_snv, y (или None).
    """
    # Общая сетка
    all_waves = np.concatenate([w for w, _ in spectra.values()])
    w_min = np.quantile(all_waves, 0.02)
    w_max = np.quantile(all_waves, 0.98)
    wave  = np.linspace(w_min, w_max, GRID_POINTS)

    rows, labels = {}, {}
    for mid, (wv, iy) in spectra.items():
        wv_u, idx = np.unique(wv, return_index=True)
        iy_u      = iy[idx]
        f         = interp1d(wv_u, iy_u, kind="linear",
                             bounds_error=False, fill_value="extrapolate")
        rows[mid]   = f(wave)
        if label_map:
            labels[mid] = label_map[mid]

    X_raw  = pd.DataFrame(rows, index=wave).T
    X_raw.index.name = "map_id"

    # ALS baseline correction
    bl_arr  = np.array([_als_baseline(r) for r in X_raw.values])
    X_corr  = pd.DataFrame(X_raw.values - bl_arr,
                            index=X_raw.index, columns=X_raw.columns)
    X_bl    = pd.DataFrame(bl_arr,
                            index=X_raw.index, columns=X_raw.columns)

    # Savitzky-Golay smoothing
    sm      = savgol_filter(X_corr.values, SG_WINDOW, SG_POLYORDER, axis=1)
    X_smooth= pd.DataFrame(sm, index=X_raw.index, columns=X_raw.columns)

    # SNV normalisation
    mu      = X_smooth.mean(axis=1)
    sd      = X_smooth.std(axis=1).replace(0, 1)
    X_snv   = X_smooth.sub(mu, axis=0).div(sd, axis=0)

    y = pd.Series(labels, name="label") if labels else None

    return dict(wave=wave, X_raw=X_raw, X_bl=X_bl,
                X_corr=X_corr, X_snv=X_snv, y=y)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _band_feats(X, wave, bands):
    v, f, bi = X.values, {}, {}
    for name, (lo, hi) in bands.items():
        mask = (wave >= lo) & (wave <= hi)
        if mask.sum() == 0:
            continue
        sub = v[:, mask]; ws = wave[mask]
        ig  = np.trapezoid(sub, ws, axis=1)
        bi[name]             = ig
        f[f"{name}_area"]    = ig
        f[f"{name}_max"]     = sub.max(1)
        f[f"{name}_mean"]    = sub.mean(1)
        f[f"{name}_std"]     = sub.std(1)
        f[f"{name}_pkpos"]   = ws[sub.argmax(1)]
        f[f"{name}_skew"]    = np.array([skew(r) for r in sub])
    keys = list(bi.keys())
    for i in range(len(keys)):
        for j in range(i + 1, min(len(keys), i + 6)):
            a, b = keys[i], keys[j]
            f[f"ratio_{a}__{b}"] = bi[a] / (bi[b] + 1e-9)
    return pd.DataFrame(f, index=X.index)


def _deriv_feats(X, wave, window=11, poly=3):
    v, f = X.values, {}
    for order in [1, 2]:
        d   = savgol_filter(v, window, poly, deriv=order, axis=1)
        p   = f"d{order}"
        f[f"{p}_mean"]    = d.mean(1)
        f[f"{p}_std"]     = d.std(1)
        f[f"{p}_absmax"]  = np.abs(d).max(1)
        f[f"{p}_energy"]  = (d ** 2).sum(1)
        f[f"{p}_zc"]      = (np.diff(np.sign(d), axis=1) != 0).sum(1)
        f[f"{p}_skew"]    = np.array([skew(r)     for r in d])
        f[f"{p}_kurt"]    = np.array([kurtosis(r) for r in d])
    return pd.DataFrame(f, index=X.index)


def _wavelet_feats(X, wavelet="db8", levels=5):
    v, f = X.values, {}
    for i, row in enumerate(v):
        coeffs = pywt.wavedec(row, wavelet=wavelet, level=levels)
        for li, c in enumerate(coeffs):
            lbl = "cA" if li == 0 else f"cD{levels - li + 1}"
            en  = (c ** 2).sum()
            eps = 1e-12
            p   = c ** 2 / (en + eps)
            ent = -np.sum(p * np.log(p + eps))
            for k, val in [("en", en), ("mean", c.mean()),
                            ("std", c.std()), ("entropy", ent)]:
                key = f"{lbl}_{k}"
                f.setdefault(key, []).append(val)
    en_keys = [k for k in f if k.endswith("_en")]
    total_e = np.zeros(len(v))
    for k in en_keys:
        total_e += np.array(f[k])
    for k in en_keys:
        f[k.replace("_en", "_rel_en")] = list(np.array(f[k]) / (total_e + 1e-12))
    return pd.DataFrame(f, index=X.index)


def _zone_feats(X, wave, n_zones=8):
    v, f = X.values, {}
    zi   = np.array_split(np.arange(len(wave)), n_zones)
    zm   = []
    for i, idx in enumerate(zi):
        zm.append(v[:, idx].mean(1))
        f[f"z{i:02d}_mean"] = zm[-1]
        f[f"z{i:02d}_std"]  = v[:, idx].std(1)
    for i in range(1, n_zones):
        f[f"zr_{i:02d}_00"] = zm[i] / (zm[0] + 1e-9)
        f[f"zd_{i:02d}"]    = zm[i] - zm[i - 1]
    f["integral"] = np.trapezoid(v, wave, axis=1)
    f["skew"]     = np.array([skew(r)     for r in v])
    f["kurt"]     = np.array([kurtosis(r) for r in v])
    return pd.DataFrame(f, index=X.index)


def build_features(preproc: dict, center: str,
                   pca=None, nmf=None, scaler_nmf=None,
                   fit_decomp=True,
                   n_pca=20, n_nmf=15) -> tuple:
    """Строит матрицу признаков. Возвращает (X_feat, transformers_dict)."""
    d     = preproc
    wave  = d["wave"]
    bands = RAMAN_BANDS_1500 if center == "1500" else RAMAN_BANDS_2900
    p     = center + "__"

    fb = _band_feats(d["X_corr"], wave, bands).add_prefix(p + "bnd__")
    fd = _deriv_feats(d["X_snv"],  wave).add_prefix(p + "drv__")
    fw = _wavelet_feats(d["X_snv"]).add_prefix(p + "wt__")
    fz = _zone_feats(d["X_snv"],   wave).add_prefix(p + "z__")

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

    X_feat = pd.concat([fb, fd, fw, fz, fpca, fnmf], axis=1)
    X_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_feat.fillna(X_feat.median(), inplace=True)

    transformers = dict(pca=pca, nmf=nmf, scaler_nmf=scaler_nmf,
                        n_pca=n_pca, n_nmf=n_nmf)
    return X_feat, transformers


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature selection
# ─────────────────────────────────────────────────────────────────────────────

def select_features(X: pd.DataFrame, y_enc: np.ndarray,
                    top_n: int = 120) -> tuple:
    sc  = StandardScaler()
    Xs  = sc.fit_transform(X)
    mdl = lgb.LGBMClassifier(n_estimators=400, random_state=RS,
                              class_weight="balanced", verbose=-1,
                              importance_type="gain", n_jobs=-1)
    mdl.fit(Xs, y_enc)
    imp  = pd.Series(mdl.feature_importances_, index=X.columns).sort_values(ascending=False)
    top  = imp.head(top_n).index.tolist()
    print(f"    Feature selection: {len(X.columns)} → {len(top)}")
    return X[top], top, imp


# ─────────────────────────────────────────────────────────────────────────────
# 4. LOAO cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def make_loao_splits(index: pd.Index) -> list:
    """Leave-One-Animal-Out. map_id format: label_animal_brain_place"""
    animals = pd.Series([mid.split("_")[1] for mid in index], index=index)
    splits  = []
    for animal in animals.unique():
        test_mask  = animals == animal
        train_mask = ~test_mask
        splits.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    print(f"  LOAO-CV: {len(splits)} folds  "
          f"(animals: {animals.nunique()}, "
          f"test size min={min(len(s[1]) for s in splits)} "
          f"max={max(len(s[1]) for s in splits)})")
    return splits


def loao_oof_predict(pipeline, X, y, splits) -> np.ndarray:
    n_cls = len(np.unique(y))
    oof   = np.zeros((len(y), n_cls))
    for tr, te in splits:
        pipeline.fit(X[tr], y[tr])
        oof[te] = pipeline.predict_proba(X[te])
    return oof


# ─────────────────────────────────────────────────────────────────────────────
# 5. Model building with Optuna
# ─────────────────────────────────────────────────────────────────────────────

def _make_pipeline(trial, model_name: str):
    if model_name == "xgb":
        clf = xgb.XGBClassifier(
            n_estimators     = trial.suggest_int("n_est", 200, 600),
            max_depth        = trial.suggest_int("max_depth", 3, 8),
            learning_rate    = trial.suggest_float("lr", 0.01, 0.2, log=True),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample", 0.4, 1.0),
            min_child_weight = trial.suggest_int("mcw", 1, 10),
            reg_alpha        = trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            reg_lambda       = trial.suggest_float("lambda", 1e-4, 10.0, log=True),
            eval_metric="mlogloss", random_state=RS, n_jobs=-1
        )
    elif model_name == "lgb":
        clf = lgb.LGBMClassifier(
            n_estimators     = trial.suggest_int("n_est", 200, 600),
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
            iterations         = trial.suggest_int("n_est", 200, 600),
            depth              = trial.suggest_int("depth", 3, 8),
            learning_rate      = trial.suggest_float("lr", 0.01, 0.2, log=True),
            l2_leaf_reg        = trial.suggest_float("l2", 1e-3, 20.0, log=True),
            bagging_temperature= trial.suggest_float("bagging_temp", 0.0, 2.0),
            random_strength    = trial.suggest_float("rs", 0.1, 5.0),
            random_seed=RS, verbose=0, auto_class_weights="Balanced"
        )
    elif model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators     = trial.suggest_int("n_est", 200, 600),
            max_depth        = trial.suggest_int("max_depth", 4, 20),
            min_samples_leaf = trial.suggest_int("min_leaf", 1, 10),
            max_features     = trial.suggest_float("max_feat", 0.1, 1.0),
            class_weight="balanced", random_state=RS, n_jobs=-1
        )
    elif model_name == "et":
        clf = ExtraTreesClassifier(
            n_estimators     = trial.suggest_int("n_est", 200, 600),
            max_depth        = trial.suggest_int("max_depth", 4, 20),
            min_samples_leaf = trial.suggest_int("min_leaf", 1, 10),
            max_features     = trial.suggest_float("max_feat", 0.1, 1.0),
            class_weight="balanced", random_state=RS, n_jobs=-1
        )
    elif model_name == "svm":
        clf = SVC(
            C           = trial.suggest_float("C", 0.1, 50.0, log=True),
            gamma       = trial.suggest_float("gamma", 1e-4, 1.0, log=True),
            kernel      = trial.suggest_categorical("kernel", ["rbf", "poly"]),
            probability=True, class_weight="balanced", random_state=RS
        )
    elif model_name == "mlp":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers   = tuple(trial.suggest_int(f"units_{i}", 32, 256)
                         for i in range(n_layers))
        clf = MLPClassifier(
            hidden_layer_sizes=layers,
            alpha              = trial.suggest_float("alpha", 1e-5, 0.1, log=True),
            learning_rate_init = trial.suggest_float("lr", 1e-4, 0.01, log=True),
            max_iter=500, random_state=RS, early_stopping=True
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def tune_and_train(model_name: str, X: np.ndarray, y: np.ndarray,
                   splits: list, n_trials: int) -> dict:
    """Optuna tuning + final fit on all data. Returns info dict."""
    def objective(trial):
        pipe = _make_pipeline(trial, model_name)
        oof  = loao_oof_predict(pipe, X, y, splits)
        return f1_score(y, oof.argmax(1), average="macro")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RS)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_pipe = _make_pipeline(
        optuna.trial.FixedTrial(study.best_params), model_name
    )
    oof   = loao_oof_predict(best_pipe, X, y, splits)
    oof_f1= f1_score(y, oof.argmax(1), average="macro")
    best_pipe.fit(X, y)

    print(f"    [{model_name:3s}]  OOF macro F1 = {oof_f1:.4f}  "
          f"(best trial = {study.best_value:.4f})")
    return dict(pipeline=best_pipe, oof_proba=oof, f1=oof_f1,
                params=study.best_params)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Stacking meta-learner
# ─────────────────────────────────────────────────────────────────────────────

def build_stacking(trained: dict, y_enc: np.ndarray,
                   classes: list) -> dict:
    """Trains LR stacking on OOF probabilities."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    meta_X  = np.hstack([v["oof_proba"] for v in trained.values()])
    meta_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)

    best_f1, best_name, best_oof, best_mdl = -1, None, None, None
    for C, name in [(0.1, "LR_C01"), (1.0, "LR_C1"), (5.0, "LR_C5")]:
        mdl      = Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=2000,
                                       class_weight="balanced",
                                       random_state=RS))
        ])
        oof_meta = cross_val_predict(mdl, meta_X, y_enc,
                                     cv=meta_cv, method="predict_proba")
        f1_meta  = f1_score(y_enc, oof_meta.argmax(1), average="macro")
        print(f"    Stack [{name}]  macro F1 = {f1_meta:.4f}"
              + ("  ◀ best" if f1_meta > best_f1 else ""))
        if f1_meta > best_f1:
            best_f1, best_name, best_oof = f1_meta, name, oof_meta
            best_mdl = Pipeline([
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(C=C, max_iter=2000,
                                           class_weight="balanced",
                                           random_state=RS))
            ])
            best_mdl.fit(meta_X, y_enc)

    print(f"\n  Best stack: [{best_name}]  F1 = {best_f1:.4f}")
    print(classification_report(y_enc, best_oof.argmax(1),
                                 target_names=classes))
    return dict(model=best_mdl, f1=best_f1,
                oof_proba=best_oof, name=best_name)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "group" in df.columns and "animal" not in df.columns:
        df = df.rename(columns={"group": "animal"})
    return df


def df_to_spectra(df: pd.DataFrame) -> tuple:
    """
    Converts the full training dataframe to two dicts of spectra
    (one per center) and a label map.

    Returns: spectra_1500, spectra_2900, label_map
    """
    if "map_id" not in df.columns:
        df = df.copy()
        df["map_id"] = (df["label"].astype(str) + "_" +
                        df["animal"].astype(str) + "_" +
                        df["brain"].astype(str)  + "_" +
                        df["place"].astype(str))

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
    """
    Loads a single spectrum .txt file.
    Expected columns: Wave (or #Wave) and Intensity (or #Intensity).
    Returns (wave_array, intensity_array).
    """
    with open(path) as fh:
        first = fh.readline()

    sep = "\t" if "\t" in first else None  # auto-detect
    df  = pd.read_csv(path, sep=sep, comment=None, engine="python",
                      header=0 if first.startswith("#") else 0)
    df.columns = [c.lstrip("#").strip() for c in df.columns]

    wave_col = next((c for c in df.columns
                     if c.lower() in ("wave", "wavenumber", "raman_shift")), None)
    int_col  = next((c for c in df.columns
                     if c.lower() in ("intensity", "counts", "signal")), None)

    if wave_col is None or int_col is None:
        # Fall back: first two numeric columns
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols) < 2:
            raise ValueError(f"Cannot identify Wave/Intensity columns in {path}")
        wave_col, int_col = num_cols[0], num_cols[1]

    return df[wave_col].values.astype(np.float32), \
           df[int_col].values.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Train entry point
# ─────────────────────────────────────────────────────────────────────────────

def train(data_path: str, model_path: str,
          n_trials: int = 50, top_features: int = 120,
          models: list = None):

    if models is None:
        models = MODEL_NAMES

    print("=" * 60)
    print("  RAMAN CLASSIFIER — TRAINING")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────
    print(f"\nLoading {data_path} …")
    df = load_parquet(data_path)
    spectra_by_center, label_map = df_to_spectra(df)

    # ── Preprocess ────────────────────────────────────────────────────
    print("\n[1] Preprocessing …")
    preproc = {}
    for center, spectra in spectra_by_center.items():
        print(f"  center{center}: {len(spectra)} maps")
        preproc[center] = preprocess_spectra(spectra, label_map)
        d = preproc[center]
        print(f"    X_snv={d['X_snv'].shape}  "
              f"classes={dict(d['y'].value_counts())}")

    centers = list(preproc.keys())

    # ── Label encoder ─────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(list(label_map.values()))
    classes = list(le.classes_)
    print(f"\n  Classes: {classes}")

    # ── Build features ────────────────────────────────────────────────
    print("\n[2] Building features …")
    feat_data = {}
    for center in centers:
        y    = preproc[center]["y"]
        X_f, transformers = build_features(preproc[center], center,
                                            fit_decomp=True)
        y_enc             = le.transform(y)
        X_sel, top_cols, imp = select_features(X_f, y_enc, top_features)
        feat_data[center] = dict(X=X_sel, y=y, y_enc=y_enc,
                                  top_cols=top_cols,
                                  transformers=transformers)
        print(f"  center{center}: {X_sel.shape[1]} features selected")

    # ── LOAO splits ───────────────────────────────────────────────────
    print("\n[3] LOAO cross-validation splits …")
    ref_center = centers[0]
    splits     = make_loao_splits(feat_data[ref_center]["y"].index)

    # ── Train base models ─────────────────────────────────────────────
    print("\n[4] Training base models …")
    trained_all = {}
    for center in centers:
        d      = feat_data[center]
        Xs     = StandardScaler().fit_transform(d["X"])
        center_splits = make_loao_splits(d["y"].index)
        print(f"\n  Center {center}:")
        for mname in models:
            info = tune_and_train(mname, Xs, d["y_enc"],
                                   center_splits, n_trials)
            trained_all[f"{center}_{mname}"] = {
                **info,
                "scaler": StandardScaler().fit(d["X"]),
                "oof_proba": pd.DataFrame(
                    info["oof_proba"], index=d["X"].index,
                    columns=[f"p_{c}" for c in classes]
                ),
                "le": le, "classes": classes,
            }

    # ── Stacking ──────────────────────────────────────────────────────
    print("\n[5] Stacking …")
    # Use common index across all centers
    common  = feat_data[centers[0]]["y"].index
    for c in centers[1:]:
        common = common.intersection(feat_data[c]["y"].index)

    oofs_common = {}
    for key, info in trained_all.items():
        oof_df = info["oof_proba"].loc[common]
        oofs_common[key] = oof_df.values

    y_common = feat_data[centers[0]]["y"].loc[common]
    y_common_enc = le.transform(y_common)

    stacking = build_stacking(
        {k: {"oof_proba": v} for k, v in oofs_common.items()},
        y_common_enc, classes
    )

    # ── Bundle & save ─────────────────────────────────────────────────
    bundle = dict(
        trained_all  = trained_all,
        feat_data    = {c: {k: v for k, v in d.items()
                            if k in ("top_cols", "transformers")}
                        for c, d in feat_data.items()},
        stacking     = stacking,
        le           = le,
        classes      = classes,
        centers      = centers,
        top_features = top_features,
    )

    joblib.dump(bundle, model_path)
    print(f"\n  Model saved → {model_path}")
    print(f"  Final stacking F1 (OOF) = {stacking['f1']:.4f}")
    print("=" * 60)
    return bundle


# ─────────────────────────────────────────────────────────────────────────────
# 9. Predict entry point
# ─────────────────────────────────────────────────────────────────────────────

def predict(spectrum_path: str, model_path: str) -> str:
    """
    Loads a saved model bundle and classifies a single spectrum .txt file.
    Returns the predicted label string.
    """
    bundle       = joblib.load(model_path)
    trained_all  = bundle["trained_all"]
    feat_data    = bundle["feat_data"]
    stacking     = bundle["stacking"]
    le           = bundle["le"]
    classes      = bundle["classes"]
    centers      = bundle["centers"]
    top_features = bundle["top_features"]

    # ── Load spectrum ─────────────────────────────────────────────────
    wave, intensity = load_txt_spectrum(spectrum_path)
    center          = _detect_center(wave)

    if center not in centers:
        raise ValueError(
            f"Detected spectrum center '{center}' not in trained centers {centers}. "
            f"Median wave = {np.median(wave):.1f} cm⁻¹"
        )

    # ── Preprocess ────────────────────────────────────────────────────
    spectra = {"sample": (wave, intensity)}
    preproc = preprocess_spectra(spectra, label_map=None)

    # ── Features ──────────────────────────────────────────────────────
    t = feat_data[center]["transformers"]
    X_feat, _ = build_features(
        preproc, center,
        pca        = t["pca"],
        nmf        = t["nmf"],
        scaler_nmf = t["scaler_nmf"],
        fit_decomp = False,
        n_pca      = t["n_pca"],
        n_nmf      = t["n_nmf"],
    )
    top_cols = feat_data[center]["top_cols"]
    # Keep only trained columns; fill any missing with 0
    X_sel = X_feat.reindex(columns=top_cols, fill_value=0.0)

    # ── Per-model probabilities ───────────────────────────────────────
    probas = {}
    for key, info in trained_all.items():
        if not key.startswith(center + "_"):
            continue
        sc     = info["scaler"]
        X_sc   = sc.transform(X_sel)
        probas[key] = info["pipeline"].predict_proba(X_sc)  # (1, n_cls)

    if not probas:
        raise RuntimeError(f"No trained models found for center '{center}'")

    # ── Stacking prediction ───────────────────────────────────────────
    meta_X  = np.hstack(list(probas.values()))          # (1, n_models*n_cls)
    # Pad to expected width (in case fewer models match during predict)
    expected_width = stacking["model"].named_steps["clf"].coef_.shape[1] \
                     if hasattr(stacking["model"].named_steps["clf"], "coef_") \
                     else None
    if expected_width and meta_X.shape[1] < expected_width:
        pad    = np.zeros((1, expected_width - meta_X.shape[1]))
        meta_X = np.hstack([meta_X, pad])

    proba = stacking["model"].predict_proba(meta_X)[0]  # (n_cls,)
    pred_idx  = proba.argmax()
    pred_label= le.inverse_transform([pred_idx])[0]

    # ── Print result ──────────────────────────────────────────────────
    print(f"\nSpectrum : {spectrum_path}")
    print(f"Center   : {center}  (median wave = {np.median(wave):.1f} cm⁻¹)")
    print(f"\nProbabilities:")
    for cls, p in zip(classes, proba):
        bar = "█" * int(p * 30)
        print(f"  {cls:10s}: {p:.4f}  {bar}")
    print(f"\nPrediction: {pred_label}")
    return pred_label


# ─────────────────────────────────────────────────────────────────────────────
# 10. CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Raman Spectrum Classifier (control / endo / exo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──────────────────────────────────────────────────────────
    tr = sub.add_parser("train", help="Train model from parquet dataset")
    tr.add_argument("--data",    required=True,
                    help="Path to all_raman_spectra.parquet")
    tr.add_argument("--model",   default="raman_model.pkl",
                    help="Output model file (default: raman_model.pkl)")
    tr.add_argument("--trials",  type=int, default=50,
                    help="Optuna trials per model (default: 50; use 80-120 for best results)")
    tr.add_argument("--features",type=int, default=120,
                    help="Number of top features to keep (default: 120)")
    tr.add_argument("--models",  nargs="+",
                    choices=MODEL_NAMES, default=MODEL_NAMES,
                    help=f"Base models to train (default: all). Choices: {MODEL_NAMES}")

    # ── predict ────────────────────────────────────────────────────────
    pr = sub.add_parser("predict", help="Predict label for a spectrum .txt file")
    pr.add_argument("--spectrum", required=True,
                    help="Path to spectrum .txt file")
    pr.add_argument("--model",    default="raman_model.pkl",
                    help="Path to saved model file (default: raman_model.pkl)")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "train":
        train(
            data_path    = args.data,
            model_path   = args.model,
            n_trials     = args.trials,
            top_features = args.features,
            models       = args.models,
        )

    elif args.command == "predict":
        if not Path(args.model).exists():
            print(f"[ERROR] Model file not found: {args.model}", file=sys.stderr)
            sys.exit(1)
        if not Path(args.spectrum).exists():
            print(f"[ERROR] Spectrum file not found: {args.spectrum}", file=sys.stderr)
            sys.exit(1)
        predict(
            spectrum_path = args.spectrum,
            model_path    = args.model,
        )


if __name__ == "__main__":
    main()
