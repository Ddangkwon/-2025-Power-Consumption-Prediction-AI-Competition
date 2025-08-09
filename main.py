#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DACON 2025 전력사용량 예측 - 고급 베이스라인 (argparse 미사용, run() 한 번으로 실행)

개선 포인트
- 모델: LightGBM -> XGBoost -> HistGradientBoosting -> RandomForest 순 폴백
- 시간 기반 교차검증(TimeSeriesSplit) + early stopping(가능한 모델에 한함)
- 타깃 log1p 변환 및 역변환
- 시간 파생(year, month, dow, hour, 주기형 sin/cos), 그룹 래깅/롤링(1, 3, 24)
- building_info.csv 자동 병합(공통 키 자동 탐색)
- sample_submission(num_date_time, answer) 포맷 엄수 및 값 검증

필요 패키지(최소)
    pip install pandas numpy scikit-learn
선택(성능 향상):
    pip install lightgbm xgboost
"""

import math
import warnings
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

# 우선 순위 모델 임포트(없는 경우 폴백)
try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.max_columns", 200)

# ------------------------------ 파일 탐색/적재 ------------------------------
def _list_candidate_files(root: Path) -> List[Path]:
    exts = [".csv", ".parquet"]
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def _find_by_keyword(files: List[Path], keywords: Tuple[str, ...]) -> Optional[Path]:
    name_map = {p.name.lower(): p for p in files}
    for k in keywords:
        if k in name_map:
            return name_map[k]
    cand = [p for p in files if any(k.split(".")[0] in p.name.lower() for k in keywords)]
    if cand:
        cand.sort(key=lambda x: (len(x.parts), len(x.name)))
        return cand[0]
    return None

def discover_paths(root: Path) -> Dict[str, Optional[Path]]:
    files = _list_candidate_files(root)
    if not files:
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없다: {root}")
    train = _find_by_keyword(files, ("train.csv", "train.parquet", "train"))
    test  = _find_by_keyword(files, ("test.csv", "test.parquet", "test"))
    sub   = _find_by_keyword(files, ("sample_submission.csv", "submission.csv", "sample_submission.parquet", "submission", "sample"))
    binfo = _find_by_keyword(files, ("building_info.csv", "building", "binfo"))
    if not (train and test):
        raise FileNotFoundError("train/test 파일을 찾지 못했다. 파일명이 표준인지 확인 필요")
    return {"train": train, "test": test, "submission": sub, "building": binfo}

def load_table(path: Path) -> pd.DataFrame:
    if path is None:
        return None
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"지원하지 않는 확장자: {path.suffix}")

# ------------------------------ 시간/타깃/전처리 ------------------------------
TIME_COL_CANDIDATES = ["num_date_time","datetime","date","timestamp","time","일시","측정일시","기준일시","기준시간","집계일시"]
TARGET_KEYWORDS = ["target","power","usage","electric","electricity","consumption","load","label","y","전력","전력사용량","소비","부하"]

def _safe_to_dt(v) -> bool:
    try:
        pd.to_datetime(v); return True
    except Exception:
        return False

def guess_time_col(df: pd.DataFrame) -> Optional[str]:
    lowmap = {c.lower(): c for c in df.columns}
    for key in TIME_COL_CANDIDATES:
        if key.lower() in lowmap:
            return lowmap[key.lower()]
    for c in df.columns:
        if df[c].dtype == object:
            sample = df[c].astype(str).head(24)
            ok = sum(_safe_to_dt(v) for v in sample)
            if ok >= len(sample) * 0.7:
                return c
    return None

def parse_time(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce", utc=True)
    # 파생
    out["year"] = out[time_col].dt.year
    out["month"] = out[time_col].dt.month
    out["day"] = out[time_col].dt.day
    out["hour"] = out[time_col].dt.hour
    out["dow"] = out[time_col].dt.weekday
    out["is_weekend"] = (out["dow"] >= 5).astype(np.int8)
    # 주기형
    if "hour" in out.columns:
        out["hour_sin"] = np.sin(2*np.pi*out["hour"]/24.0)
        out["hour_cos"] = np.cos(2*np.pi*out["hour"]/24.0)
    if "month" in out.columns:
        out["month_sin"] = np.sin(2*np.pi*out["month"]/12.0)
        out["month_cos"] = np.cos(2*np.pi*out["month"]/12.0)
    return out

def guess_target_col(train: pd.DataFrame, test: pd.DataFrame) -> str:
    diff = [c for c in train.columns if c not in test.columns]
    tcol = guess_time_col(train)
    keep = []
    for c in diff:
        if tcol and c == tcol:
            continue
        low = c.lower()
        if any(k == low or k in low for k in TARGET_KEYWORDS):
            keep.append(c)
    if len(keep) == 1:
        return keep[0]
    num_diff = [c for c in diff if pd.api.types.is_numeric_dtype(train[c])]
    if len(num_diff) == 1:
        return num_diff[0]
    for c in diff:
        if any(k in c.lower() for k in TARGET_KEYWORDS):
            return c
    if num_diff:
        return num_diff[-1]
    raise RuntimeError("타깃 컬럼을 추정하지 못했다. target 인자를 지정해야 한다.")

def _category_encode_inplace(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_categorical_dtype(out[c]):
            out[c] = out[c].astype("category")
            # LightGBM이 있으면 category 그대로 둔다. 없으면 코드화
            if not _HAS_LGBM:
                out[c] = out[c].cat.codes.astype("int32")
        elif pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype("int8")
    return out

# ------------------------------ building_info 병합 ------------------------------
def smart_join_building(train: pd.DataFrame, test: pd.DataFrame, binfo: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if binfo is None:
        return train, test
    # 공통 키 후보
    cand_keys = ["building_id", "meter_id", "site_id", "facility_id", "zone_id"]
    keys = [k for k in cand_keys if (k in train.columns and k in binfo.columns)]
    if not keys:
        # 열 이름이 조금 다른 경우를 위해 교집합 열로 소규모 키 추정
        inter = [c for c in binfo.columns if c in train.columns]
        # 식별성이 높을 법한 컬럼 우선
        for k in ["building","bldg","meter","site","facility","zone","b_id","bldg_id"]:
            for col in inter:
                if k in col.lower():
                    keys = [col]; break
            if keys: break
    if not keys:
        return train, test  # 병합 포기
    # 중복 키 제거
    binfo_uniq = binfo.drop_duplicates(subset=keys)
    # 병합
    train = train.merge(binfo_uniq, on=keys, how="left")
    test  = test.merge(binfo_uniq,  on=keys, how="left")
    return train, test

# ------------------------------ 래깅/롤링 특징 ------------------------------
def add_group_lag_roll(all_df: pd.DataFrame, time_col: str, group_keys: List[str], target_col: Optional[str]=None) -> pd.DataFrame:
    """
    train+test 를 합쳐 시간 정렬 후 그룹별 래깅/롤링을 계산한다.
    target_col 이 주어지면 해당 컬럼에 대한 롤링 통계도 함께 생성한다.
    """
    if not group_keys:
        group_keys = []
    df = all_df.sort_values(time_col).copy()
    # 안전: 연속 인덱스
    df.reset_index(drop=True, inplace=True)

    # 래그 대상 수치 컬럼(시간/ID/타깃 제외)
    exclude_cols = set([time_col])
    if target_col:
        exclude_cols.add(target_col)

    num_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_cols]

    # 그룹 기준
    if group_keys:
        gobj = df.groupby(group_keys, sort=False)
    else:
        gobj = [(None, df)]

    # 윈도우
    lags = [1]
    rolls = [3, 24]  # 빈도와 무관하게 완전 일반화는 불가하지만 경험적 기본값

    # 계산
    out_parts = []
    for key, g in (gobj if isinstance(gobj, list) else gobj):
        part = g.copy()
        # 래그
        for L in lags:
            for c in num_cols:
                part[f"{c}_lag{L}"] = g[c].shift(L) if not isinstance(gobj, list) else part[c].shift(L)
        # 롤링
        for W in rolls:
            for c in num_cols:
                roll = (g[c].rolling(W) if not isinstance(gobj, list) else part[c].rolling(W))
                part[f"{c}_rmean{W}"] = roll.mean()
                part[f"{c}_rstd{W}"]  = roll.std()
        # 타깃 기반 롤링(리크 방지: shift 후 rolling)
        if target_col and target_col in g.columns:
            tgt_shift = (g[target_col].shift(1) if not isinstance(gobj, list) else part[target_col].shift(1))
            part[f"{target_col}_lag1"] = tgt_shift
            for W in rolls:
                roll_t = tgt_shift.rolling(W)
                part[f"{target_col}_rmean{W}"] = roll_t.mean()
                part[f"{target_col}_rstd{W}"]  = roll_t.std()
        out_parts.append(part)

    df2 = pd.concat(out_parts, axis=0).sort_index()
    return df2

# ------------------------------ 피처 매트릭스 ------------------------------
def build_feature_matrix(train: pd.DataFrame, test: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], str]:
    time_col = guess_time_col(train)
    if time_col:
        train = parse_time(train, time_col)
        test  = parse_time(test,  time_col)

    # building_info 병합 시도
    # (호출부에서 이미 병합 후 들어오도록 할 수도 있으나 안전하게 한 번 더 보장 X)

    # 그룹 키 후보(존재하는 것만)
    group_keys = [k for k in ["building_id","meter_id","site_id","facility_id","zone_id"] if k in train.columns]
    if time_col:
        # 전체 합쳐 래깅/롤링 특징 생성
        all_df = pd.concat([train.assign(_is_train=1), test.assign(_is_train=0)], axis=0, ignore_index=True)
        all_df = add_group_lag_roll(all_df, time_col, group_keys, target_col=target)
        train = all_df[all_df["_is_train"] == 1].drop(columns=["_is_train"])
        test  = all_df[all_df["_is_train"] == 0].drop(columns=["_is_train"])
    # 불필요 열 제거 후보
    drop_cols = set([target])
    if time_col:
        drop_cols.add(time_col)

    X_train = train.drop(columns=[c for c in drop_cols if c in train.columns], errors="ignore")
    X_test  = test.drop(columns=[c for c in drop_cols if c in test.columns], errors="ignore")

    # 공통 컬럼만 사용
    common = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common].copy()
    X_test  = X_test[common].copy()

    # 인코딩(카테고리/불리언)
    X_train = _category_encode_inplace(X_train)
    X_test  = _category_encode_inplace(X_test)

    # 수치 결측/무한
    for df in (X_train, X_test):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in X_train.columns:
        if X_train[c].isna().any():
            X_train[c] = X_train[c].fillna(X_train[c].median())
    for c in X_test.columns:
        if X_test[c].isna().any():
            med = X_train[c].median() if c in X_train.columns else 0
            X_test[c] = X_test[c].fillna(med)

    return X_train, X_test, common, time_col

# ------------------------------ 제출 스키마(엄수) ------------------------------
def _ensure_submission(sample: pd.DataFrame, test: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    sub = sample.copy()
    # ID 병합
    id_col = "num_date_time" if "num_date_time" in sub.columns else None
    if id_col and id_col in test.columns:
        pred_df = pd.DataFrame({id_col: test[id_col].values, "answer": preds})
        sub = sub[[id_col]].merge(pred_df, on=id_col, how="left", validate="one_to_one")
    else:
        # 최후 방어: 행 순서 매칭
        sub["answer"] = preds[: len(sub)]

    # 값 검증/정리
    if sub["answer"].isna().any():
        # NaN 이면 0으로 대체(스코어에는 불리하지만 제출 에러는 방지)
        sub["answer"] = sub["answer"].fillna(0.0)
    sub["answer"] = np.where(np.isfinite(sub["answer"]), sub["answer"], 0.0)
    sub["answer"] = np.clip(sub["answer"], 0, None)

    # 컬럼 순서 강제
    if "num_date_time" in sub.columns:
        sub = sub[["num_date_time", "answer"]]
    else:
        # 안전망
        cols = [c for c in sub.columns if c != "answer"]
        sub = sub[cols + ["answer"]]
    return sub

# ------------------------------ 모델 학습/예측 ------------------------------
def _choose_model():
    """
    사용 가능 모델을 선택한다.
    우선순위: LightGBM -> XGBoost -> HistGBR -> RandomForest
    """
    if _HAS_LGBM:
        return "lgbm"
    if _HAS_XGB:
        return "xgb"
    return "hgb"  # sklearn HistGradientBoostingRegressor
    # (RandomForest 는 HGB 불가시로 폴백에서 사용)

def _fit_predict_cv(X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, time_index: Optional[pd.Series], n_splits: int = 3) -> Tuple[np.ndarray, float]:
    """
    시간 기반 CV 로 학습/예측. fold 평균 예측을 반환한다.
    """
    # 타깃 변환
    y_tr = np.log1p(np.clip(y, 0, None))

    # TimeSeriesSplit 구성(시간열이 있으면 시간순으로 정렬)
    if time_index is not None:
        order = np.argsort(time_index.values)
    else:
        order = np.arange(len(y_tr))
    X_ord = X.iloc[order].reset_index(drop=True)
    y_ord = y_tr[order]

    # 모델 선택
    model_name = _choose_model()
    preds_test_folds = []
    oof = np.zeros(len(y_ord), dtype=float)

    tss = TimeSeriesSplit(n_splits=n_splits)
    best_iters = []

    for fold, (tr_idx, va_idx) in enumerate(tss.split(X_ord), 1):
        X_tr_f, X_va_f = X_ord.iloc[tr_idx], X_ord.iloc[va_idx]
        y_tr_f, y_va_f = y_ord[tr_idx], y_ord[va_idx]

        if model_name == "lgbm":
            model = LGBMRegressor(
                n_estimators=5000, learning_rate=0.03, num_leaves=64,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=42, n_jobs=-1
            )
            model.fit(
                X_tr_f, y_tr_f,
                eval_set=[(X_va_f, y_va_f)],
                eval_metric="l1",
                callbacks=[],
                verbose=False
            )
            # LightGBM sklearn API 에서 early_stopping_rounds 미지정 -> best_iteration_ 사용 불가 시 대비
            best_iter = getattr(model, "best_iteration_", None)
            pred_va = model.predict(X_va_f, num_iteration=best_iter)
            pred_te = model.predict(X_test,  num_iteration=best_iter)

        elif model_name == "xgb":
            model = XGBRegressor(
                n_estimators=5000, learning_rate=0.03, max_depth=8,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                tree_method="hist", random_state=42, n_jobs=-1
            )
            model.fit(
                X_tr_f, y_tr_f,
                eval_set=[(X_va_f, y_va_f)],
                eval_metric="mae",
                verbose=False
            )
            # XGBRegressor 에서 best_ntree_limit 사용
            best_ntree = getattr(model, "best_ntree_limit", None)
            pred_va = model.predict(X_va_f, iteration_range=(0, best_ntree)) if best_ntree else model.predict(X_va_f)
            pred_te = model.predict(X_test,  iteration_range=(0, best_ntree)) if best_ntree else model.predict(X_test)

        else:
            # HistGradientBoostingRegressor(빠름, 기본값 좋음) 폴백
            try:
                model = HistGradientBoostingRegressor(
                    learning_rate=0.06, max_depth=None, max_iter=1500,
                    l2_regularization=0.0, early_stopping=True, random_state=42
                )
                model.fit(X_tr_f, y_tr_f)
                pred_va = model.predict(X_va_f)
                pred_te = model.predict(X_test)
            except Exception:
                # 최후의 폴백: RandomForest
                model = RandomForestRegressor(
                    n_estimators=1200, max_depth=None, min_samples_split=4,
                    max_features="sqrt", n_jobs=-1, random_state=42
                )
                model.fit(X_tr_f, y_tr_f)
                pred_va = model.predict(X_va_f)
                pred_te = model.predict(X_test)

        oof[va_idx] = pred_va
        preds_test_folds.append(pred_te)
        best_iters.append(len(getattr(model, "evals_result_", {})) if hasattr(model, "evals_result_") else None)

        mae = mean_absolute_error(np.expm1(y_va_f), np.expm1(pred_va))
        print(f"[CV] fold={fold} MAE={mae:.6f}")

    # CV 스코어
    mae_all = mean_absolute_error(np.expm1(y_ord), np.expm1(oof))
    print(f"[CV] overall MAE={mae_all:.6f}")

    # 테스트 예측 평균
    preds_test = np.mean(np.column_stack(preds_test_folds), axis=1)
    # 역변환
    preds_test = np.expm1(preds_test)
    # 음수 방지
    preds_test = np.clip(preds_test, 0, None)
    return preds_test, mae_all

# ------------------------------ run() ------------------------------
def run(data_path: Optional[str] = None, target: Optional[str] = None, output: str = "submission.csv") -> Path:
    """
    간단 실행 진입점.
    - data_path 가 None 이면 ./open.zip -> ./open 순서로 자동 탐색한다.
    - target 이 None 이면 train/test 비교로 자동 추정한다.
    - output 파일로 제출 파일을 저장한다.
    """
    cwd = Path.cwd()
    if data_path is None:
        if (cwd / "open.zip").exists():
            data_path = str(cwd / "open.zip")
        elif (cwd / "open").exists():
            data_path = str(cwd / "open")
        else:
            raise FileNotFoundError("open.zip 또는 open 디렉토리를 같은 폴더에 두어야 한다.")
    data_path = Path(data_path)

    # ZIP 처리
    work_dir = None
    if data_path.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory(); work_dir = Path(tmp.name)
        with zipfile.ZipFile(data_path, "r") as zf:
            zf.extractall(work_dir)
        root = work_dir
    else:
        root = data_path
    root = Path(root)

    paths = discover_paths(root)
    print("[INFO] Data files ->", {k: str(v) if v else None for k, v in paths.items()})

    train_df = load_table(paths["train"])
    test_df  = load_table(paths["test"])
    sub_df   = load_table(paths["submission"])
    binfo_df = load_table(paths["building"]) if paths.get("building") else None
    if sub_df is None:
        raise FileNotFoundError("sample_submission.csv 이 필요하다.")

    # building_info 병합(가능하면)
    if binfo_df is not None:
        train_df, test_df = smart_join_building(train_df, test_df, binfo_df)

    # 타깃 결정
    target_col = target if target else guess_target_col(train_df, test_df)
    if target_col not in train_df.columns:
        raise RuntimeError(f"타깃 컬럼을 찾지 못함: {target_col}")
    print(f"[INFO] Target column = {target_col}")

    # 시간열 추정(검증 분할용)
    time_col = guess_time_col(train_df)

    # 피처 구축
    X_train, X_test, feats, _ = build_feature_matrix(train_df, test_df, target_col)
    y = train_df[target_col].astype(float).values

    # 시간 인덱스
    t_index = None
    if time_col and time_col in train_df.columns:
        try:
            t_index = pd.to_datetime(train_df[time_col], errors="coerce")
        except Exception:
            t_index = None

    # 학습/예측(CV)
    preds, cv_mae = _fit_predict_cv(X_train, y, X_test, t_index, n_splits=4)

    # 제출 파일 생성
    submit = _ensure_submission(sub_df, test_df, preds)

    # 최종 저장
    out_path = Path(output)
    submit.to_csv(out_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"[INFO] Saved submission -> {out_path}  shape={submit.shape}")
    return out_path

# 스크립트 직접 실행 시
if __name__ == "__main__":
    run()
