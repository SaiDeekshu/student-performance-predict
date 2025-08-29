from pathlib import Path
import json, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, classification_report
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "features_config.json").read_text())

csv_path      = ROOT / CFG["dataset_csv"]
target_reg    = CFG["target_reg"]
target_cls    = CFG.get("target_cls")
use_cls       = bool(CFG.get("use_classification", False))
id_cols       = set(CFG.get("id_cols", []))
feature_map   = CFG["feature_map"]

df = pd.read_csv(csv_path)

mapped_cols = {k: v for k, v in feature_map.items() if v and v in df.columns}
feature_cols = list(mapped_cols.values())
feature_cols = [c for c in feature_cols if c not in id_cols and c not in [target_reg, target_cls]]

num_cols, cat_cols = [], []
for c in feature_cols:
    if pd.api.types.is_float_dtype(df[c]):
        num_cols.append(c)
    elif pd.api.types.is_integer_dtype(df[c]):
        (cat_cols if df[c].nunique() <= 6 else num_cols).append(c)
    else:
        cat_cols.append(c)

models_dir = ROOT / "models"
models_dir.mkdir(parents=True, exist_ok=True)

meta = {
    "dataset_csv": str(csv_path),
    "used_feature_cols": feature_cols,
    "numeric_cols": num_cols,
    "categorical_cols": cat_cols,
    "target_reg": target_reg,
    "use_classification": use_cls,
    "target_cls": target_cls
}

if target_reg in df.columns:
    Xr = df[feature_cols].copy()
    yr = df[target_reg].copy()

    pre_r = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    reg = RandomForestRegressor(n_estimators=400, random_state=42)
    pipe_r = make_pipeline(pre_r, reg)

    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    pipe_r.fit(Xr_tr, yr_tr)
    pr = pipe_r.predict(Xr_te)

    reg_report = {
        "r2": float(r2_score(yr_te, pr)),
        "rmse": float(np.sqrt(mean_squared_error(yr_te, pr))),
        "mae": float(mean_absolute_error(yr_te, pr))
    }
    joblib.dump(pipe_r, models_dir / "reg_model.pkl")
    (models_dir / "reg_report.json").write_text(json.dumps(reg_report, indent=2))
    meta["reg_report"] = reg_report
else:
    print(f"[WARN] target_reg '{target_reg}' not found; skipping regression.")

if use_cls and target_cls and target_cls in df.columns:
    Xc = df[feature_cols].copy()
    yc = df[target_cls].copy()

    pre_c = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    clf = RandomForestClassifier(n_estimators=400, random_state=42)
    pipe_c = make_pipeline(pre_c, clf)

    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        Xc, yc, test_size=0.2, random_state=42,
        stratify=yc if yc.nunique() > 1 else None
    )
    pipe_c.fit(Xc_tr, yc_tr)
    pc = pipe_c.predict(Xc_te)

    cls_report = {
        "accuracy": float(accuracy_score(yc_te, pc)),
        "f1_macro": float(f1_score(yc_te, pc, average="macro")),
        "details": classification_report(yc_te, pc, output_dict=True)
    }
    joblib.dump(pipe_c, models_dir / "cls_model.pkl")
    (models_dir / "cls_report.json").write_text(json.dumps(cls_report, indent=2))
    meta["cls_report"] = {k: v for k, v in cls_report.items() if k != "details"}
else:
    print("[INFO] classification disabled or target not found.")

(ROOT / "models" / "meta.json").write_text(json.dumps(meta, indent=2))
print("Saved meta:", meta)
print("Done.")
