#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import pickle
import shutil
import logging
import tempfile
import subprocess
import hashlib
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from fastmcp import FastMCP

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("rf-tacc-mcp")

# =============================================================================
# Base Directory
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# Environment / Config
# =============================================================================
MODEL_PATH = os.getenv(
    "RF_MODEL_PATH",
    os.path.join(BASE_DIR, "A_model_TACC.dat")
)

JONG_INFO_PATH = os.getenv(
    "JONG_INFO_PATH",
    os.path.join(BASE_DIR, "TACC_A_jong_info.csv")
)

ILEARN_SCRIPT_PATH = os.getenv(
    "ILEARN_SCRIPT_PATH",
    os.path.join(BASE_DIR, "iLearn", "iLearn-nucleotide-acc.py")
)

SEQ_TYPE = os.getenv("SEQ_TYPE", "DNA")  # DNA or RNA
TACC_LAG = os.getenv("TACC_LAG", "")     # "" 이면 lag 미지정

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

PYTHON_BIN = os.getenv("PYTHON_BIN", sys.executable)

# =============================================================================
# MCP Server
# =============================================================================
mcp = FastMCP("rf-tacc-species-mcp")

_MODEL = None
_CLASS_MAPPING = None


# =============================================================================
# Utility functions
# =============================================================================
def load_model(model_path: str):
    """pickle 우선, 실패 시 joblib fallback"""
    pickle_error = None
    joblib_error = None

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded via pickle: %s", model_path)
        return model
    except Exception as e:
        pickle_error = e
        logger.exception("pickle load failed")

    try:
        import joblib
        model = joblib.load(model_path)
        logger.info("Model loaded via joblib: %s", model_path)
        return model
    except Exception as e:
        joblib_error = e
        logger.exception("joblib load failed")

    raise RuntimeError(
        f"모델 로딩 실패: {model_path}\n"
        f"- pickle error: {pickle_error}\n"
        f"- joblib error: {joblib_error}"
    )


def get_model_expected_features(model) -> int:
    """모델이 기대하는 feature 개수 확인"""
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)

    if hasattr(model, "steps"):
        for _, step in reversed(model.steps):
            if hasattr(step, "n_features_in_"):
                return int(step.n_features_in_)

    raise AttributeError("모델에서 n_features_in_를 찾을 수 없습니다.")


def summarize_text(text: str, max_len: int = 240) -> str:
    text = str(text).replace("\n", "\\n")
    if len(text) <= max_len:
        return text
    half = max_len // 2
    return f"{text[:half]} ... {text[-half:]}"


def summarize_seq(seq: str, head: int = 40, tail: int = 40) -> str:
    seq = str(seq)
    if len(seq) <= head + tail:
        return seq
    return f"{seq[:head]}...{seq[-tail:]}"


def strip_code_fences(text: str) -> str:
    """``` ... ``` 코드블록이면 내부만 추출"""
    text = str(text).strip()

    m = re.match(
        r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*?)\n```\s*$",
        text,
        flags=re.DOTALL
    )
    if m:
        return m.group(1).strip()

    return text


def sanitize_sequence_text(sequence_text: str) -> str:
    """
    FASTA 헤더 제거 + 공백 제거 후 핵산 문자만 허용.
    이상한 문자가 있으면 에러 반환.
    """
    if not sequence_text or not str(sequence_text).strip():
        raise ValueError("빈 서열입니다.")

    lines = str(sequence_text).splitlines()
    lines = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith(">")]
    seq = "".join(lines).upper()
    seq = re.sub(r"\s+", "", seq)

    invalid_chars = sorted(set(re.findall(r"[^ACGTUN]", seq)))
    if invalid_chars:
        raise ValueError(
            f"허용되지 않은 문자가 포함되어 있습니다: {''.join(invalid_chars)}. "
            "A/C/G/T/U/N 문자만 포함해야 합니다."
        )

    if not seq:
        raise ValueError("유효한 서열을 읽지 못했습니다.")

    return seq


def write_temp_fasta(seq: str, out_dir: str, name: str = "query") -> str:
    """임시 FASTA 파일 생성"""
    fasta_path = os.path.join(out_dir, "query.fasta")
    with open(fasta_path, "w", encoding="utf-8") as f:
        f.write(f">{name}\n")
        for i in range(0, len(seq), 60):
            f.write(seq[i:i + 60] + "\n")
    return fasta_path


def extract_sequence_from_fasta(fasta_text: str) -> str:
    """FASTA 형식 문자열 엄격 파싱"""
    if not fasta_text or not str(fasta_text).strip():
        raise ValueError("빈 FASTA 입니다.")

    lines = [ln.strip() for ln in str(fasta_text).splitlines() if ln.strip()]
    if not lines:
        raise ValueError("빈 FASTA 입니다.")

    seq_lines: List[str] = []
    header_seen = False

    for ln in lines:
        if ln.startswith(">"):
            header_seen = True
            continue
        seq_lines.append(ln)

    if not header_seen:
        raise ValueError("FASTA header('>')가 없습니다.")

    if not seq_lines:
        raise ValueError("FASTA에서 sequence line을 찾지 못했습니다.")

    return "".join(seq_lines)


def extract_best_sequence_candidate(text: str) -> str:
    """
    Agent가 설명문, JSON, 코드블록, FASTA wrapper 등을 섞어 보내도
    가능한 핵산 서열만 최대한 추출.
    """
    if not text or not str(text).strip():
        raise ValueError("빈 입력입니다.")

    raw = strip_code_fences(str(text).strip())

    # 1) FASTA 우선
    if ">" in raw:
        try:
            fasta_seq = extract_sequence_from_fasta(raw)
            fasta_seq = re.sub(r"\s+", "", fasta_seq).upper()
            fasta_seq = re.sub(r"[^ACGTUN]", "", fasta_seq)
            if fasta_seq:
                return fasta_seq
        except Exception:
            pass

    # 2) JSON / key-value 형태에서 서열 값 추출
    key_patterns = [
        r'"sequence_text"\s*:\s*"([^"]+)"',
        r"'sequence_text'\s*:\s*'([^']+)'",
        r'"sequence"\s*:\s*"([^"]+)"',
        r"'sequence'\s*:\s*'([^']+)'",
        r'"fasta_text"\s*:\s*"([^"]+)"',
        r"'fasta_text'\s*:\s*'([^']+)'",
        r'sequence_text\s*=\s*([A-Za-z\s]+)',
        r'sequence\s*=\s*([A-Za-z\s]+)',
    ]

    for pat in key_patterns:
        m = re.search(pat, raw, flags=re.DOTALL)
        if m:
            candidate = m.group(1)
            candidate = re.sub(r"\s+", "", candidate).upper()
            candidate = re.sub(r"[^ACGTUN]", "", candidate)
            if candidate:
                return candidate

    # 3) 줄 단위로 핵산 비율이 높은 부분 수집
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    seq_like_parts: List[str] = []

    for ln in lines:
        ln2 = re.sub(r"\s+", "", ln).upper()
        if not ln2:
            continue

        if len(ln2) < 10:
            continue

        nuc_only = re.sub(r"[^ACGTUN]", "", ln2)
        if not nuc_only:
            continue

        ratio = len(nuc_only) / max(len(ln2), 1)

        if ratio >= 0.85:
            seq_like_parts.append(nuc_only)

    if seq_like_parts:
        merged = "".join(seq_like_parts)
        if merged:
            return merged

    # 4) 마지막 fallback
    blocks = re.findall(r"[ACGTUNacgtun\s]{20,}", raw)
    cleaned_blocks = []

    for b in blocks:
        c = re.sub(r"\s+", "", b).upper()
        c = re.sub(r"[^ACGTUN]", "", c)
        if c:
            cleaned_blocks.append(c)

    if cleaned_blocks:
        cleaned_blocks.sort(key=len, reverse=True)
        return cleaned_blocks[0]

    raise ValueError(
        "입력에서 유효한 핵산 서열을 추출하지 못했습니다. "
        "서열만 전달하거나 FASTA 형식으로 입력하세요."
    )


def normalize_agent_sequence_input(input_text: str) -> str:
    """Agent 입력에서 최종적으로 순수 핵산 서열만 반환"""
    candidate = extract_best_sequence_candidate(input_text)
    return sanitize_sequence_text(candidate)


def run_ilearn_tacc(
    ilearn_acc_py: str,
    fasta_path: str,
    out_csv: str,
    seq_type: str = "DNA",
    lag: Optional[int] = None,
) -> None:
    """iLearn TACC 실행"""
    if not os.path.exists(ilearn_acc_py):
        raise FileNotFoundError(f"iLearn script not found: {ilearn_acc_py}")

    cmd = [
        PYTHON_BIN,
        ilearn_acc_py,
        "--method", "TACC",
        "--file", fasta_path,
        "--format", "csv",
        "--out", out_csv,
        "--type", seq_type,
    ]

    if lag is not None:
        cmd += ["--lag", str(lag)]

    logger.info("Running iLearn command: %s", " ".join(cmd))

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        cwd=BASE_DIR,
    )

    logger.info("iLearn stdout preview: %s", summarize_text(proc.stdout, max_len=300))
    logger.info("iLearn stderr preview: %s", summarize_text(proc.stderr, max_len=500))

    if proc.returncode != 0:
        raise RuntimeError(
            "iLearn 실행 실패 | "
            f"returncode={proc.returncode} | "
            f"stderr={proc.stderr[:3000]}"
        )

    if not os.path.exists(out_csv):
        raise FileNotFoundError(f"iLearn 결과 파일이 생성되지 않았습니다: {out_csv}")

    if os.path.getsize(out_csv) == 0:
        raise RuntimeError("iLearn 결과 파일이 비어 있습니다.")


def load_ilearn_csv(out_csv: str) -> pd.DataFrame:
    """iLearn 출력 CSV/TSV를 견고하게 로드"""
    if not os.path.exists(out_csv):
        raise FileNotFoundError(f"iLearn 결과 파일이 없습니다: {out_csv}")

    with open(out_csv, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()

    has_alpha = any(ch.isalpha() for ch in first)
    header = 0 if has_alpha else None

    df = pd.read_csv(out_csv, sep=None, engine="python", header=header)

    if df.shape[0] == 0:
        df = pd.read_csv(out_csv, sep=None, engine="python", header=None)

    return df


def to_feature_matrix(df: pd.DataFrame, n_expected: Optional[int] = None) -> np.ndarray:
    """
    iLearn 출력에서 feature만 안정적으로 추출
    - 숫자 변환 후 전부 NaN인 컬럼 제거
    - 모델 기대 차원보다 1개 많으면 마지막 n_expected개 컬럼 사용
    """
    work = df.copy()

    if work.shape[0] == 0:
        raise ValueError("iLearn output is empty.")

    work = work.iloc[[0], :]

    logger.info("Raw iLearn columns before parsing: %s", list(work.columns))
    logger.info("Raw iLearn first row before parsing: %s", work.head(1).to_dict(orient="records"))

    drop_cols = []
    for c in work.columns:
        lc = str(c).strip().lower()

        if lc in {
            "name", "nameseq", "seqname", "id", "label", "class",
            "sequence", "seq", "sample", "sample_name", "sampleid",
            "sample_id", "samplename", "index", "no", "#"
        }:
            drop_cols.append(c)

        if lc.startswith("unnamed:"):
            drop_cols.append(c)

    if drop_cols:
        logger.info("Dropping obvious non-feature columns: %s", drop_cols)
        work = work.drop(columns=drop_cols, errors="ignore")

    work = work.apply(pd.to_numeric, errors="coerce")

    all_nan_cols = work.columns[work.isna().all()].tolist()
    if all_nan_cols:
        logger.info("Dropping all-NaN columns after numeric coercion: %s", all_nan_cols)
        work = work.drop(columns=all_nan_cols, errors="ignore")

    if work.isna().any().any():
        bad_cols = work.columns[work.isna().any()].tolist()
        raise ValueError(
            f"NaN values remain in parsed feature columns: {bad_cols} | "
            f"columns={list(work.columns)}"
        )

    logger.info("Parsed numeric columns before final dimension check: %s", list(work.columns))
    logger.info("Parsed numeric shape before final dimension check: rows=%d, cols=%d", work.shape[0], work.shape[1])

    if n_expected is not None and work.shape[1] == n_expected + 1:
        logger.warning(
            "Feature count is one more than expected (%d vs %d). "
            "Using the last %d columns as feature vector.",
            work.shape[1], n_expected, n_expected
        )
        work = work.iloc[:, -n_expected:]

    X = work.to_numpy(dtype=float)

    if X.shape[0] != 1:
        raise ValueError(f"Expected exactly 1 sample, got {X.shape[0]}")

    if n_expected is not None and X.shape[1] != n_expected:
        raise ValueError(
            f"Feature dimension mismatch: expected {n_expected}, got {X.shape[1]} | "
            f"columns={list(work.columns)} | values={work.head(1).to_dict(orient='records')}"
        )

    return X


def extract_scientific_name(class_name: str) -> str:
    """class_name에서 학명만 최대한 추출"""
    s = str(class_name).strip()

    if re.fullmatch(r"[A-Z][a-zA-Z\.-]+(?:\s+[a-z][a-zA-Z\.-]+){1,3}", s):
        return s

    m = re.search(r"\(([A-Z][a-zA-Z\.-]+(?:\s+[a-z][a-zA-Z\.-]+){1,3})\)", s)
    if m:
        return m.group(1).strip()

    m = re.match(r"^([A-Z][a-zA-Z\.-]+(?:\s+[a-z][a-zA-Z\.-]+){1,3})", s)
    if m:
        return m.group(1).strip()

    return s


def _normalize_mapping_key(value: Any) -> List[Any]:
    """예측 label / CSV key를 유연하게 매칭하기 위한 후보 키 생성"""
    candidates: List[Any] = [value]

    try:
        s = str(value).strip()
        candidates.append(s)
    except Exception:
        pass

    try:
        i = int(value)
        candidates.append(i)
        candidates.append(str(i))
    except Exception:
        pass

    try:
        f = float(value)
        candidates.append(f)
        if float(f).is_integer():
            i2 = int(f)
            candidates.append(i2)
            candidates.append(str(i2))
    except Exception:
        pass

    unique = []
    seen = set()
    for x in candidates:
        key = repr(x)
        if key not in seen:
            seen.add(key)
            unique.append(x)
    return unique


def parse_jong_info(csv_path: str) -> Dict[Any, Dict[str, Any]]:
    """TACC_A_jong_info.csv 파싱"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"클래스 매핑 CSV가 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    mapping: Dict[Any, Dict[str, Any]] = {}

    if df.shape[1] >= 2:
        key_col = df.columns[0]
        val_col = df.columns[1]

        for _, row in df.iterrows():
            raw_key = row[key_col]
            raw_val = row[val_col]
            class_name = str(raw_val).strip()

            info = {
                "class_index": raw_key,
                "class_name": class_name,
                "scientific_name": extract_scientific_name(class_name),
                "species_name": class_name,
            }

            for candidate in _normalize_mapping_key(raw_key):
                mapping[candidate] = info

        return mapping

    for i, v in enumerate(df.iloc[:, 0].astype(str).tolist()):
        class_name = str(v).strip()
        info = {
            "class_index": i,
            "class_name": class_name,
            "scientific_name": extract_scientific_name(class_name),
            "species_name": class_name,
        }
        for candidate in _normalize_mapping_key(i):
            mapping[candidate] = info

    return mapping


def predict(model, X: np.ndarray) -> Tuple[Any, Optional[np.ndarray]]:
    """
    최종 label은 model.predict() 결과를 그대로 사용.
    predict_proba는 참고용으로만 반환.
    """
    pred_label = model.predict(X)[0]
    proba = None

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
    except Exception as e:
        logger.warning("predict_proba failed: %s", e)

    return pred_label, proba


def ensure_resources_loaded() -> None:
    """최초 1회만 모델/클래스 매핑 로드"""
    global _MODEL, _CLASS_MAPPING

    logger.info("Python executable: %s", PYTHON_BIN)
    logger.info("BASE_DIR: %s", BASE_DIR)
    logger.info("MODEL_PATH: %s", MODEL_PATH)
    logger.info("JONG_INFO_PATH: %s", JONG_INFO_PATH)
    logger.info("ILEARN_SCRIPT_PATH: %s", ILEARN_SCRIPT_PATH)
    logger.info("SEQ_TYPE: %s", SEQ_TYPE)
    logger.info("TACC_LAG: %s", TACC_LAG if str(TACC_LAG).strip() else None)

    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Random Forest model not found: {MODEL_PATH}. "
                f"A_model_TACC.dat 파일 위치를 확인하세요."
            )
        _MODEL = load_model(MODEL_PATH)
        logger.info("Model loaded: %s", MODEL_PATH)
        if hasattr(_MODEL, "classes_"):
            logger.info(
                "Model classes_ preview: %s",
                summarize_text(list(_MODEL.classes_), max_len=200)
            )

    if _CLASS_MAPPING is None:
        _CLASS_MAPPING = parse_jong_info(JONG_INFO_PATH)
        logger.info("Class mapping loaded: %s", JONG_INFO_PATH)


def _lookup_species_info(pred_label: Any) -> Dict[str, Any]:
    mapping = _CLASS_MAPPING or {}
    for key in _normalize_mapping_key(pred_label):
        if key in mapping:
            return mapping[key]

    return {
        "class_index": pred_label,
        "class_name": str(pred_label),
        "scientific_name": str(pred_label),
        "species_name": str(pred_label),
    }


def build_result_from_prediction(pred: Any, proba: Optional[np.ndarray]) -> Dict[str, Any]:
    """최종 반환 JSON 생성"""
    info = _lookup_species_info(pred)

    result = {
        "predicted_label": str(pred),
        "class_name": info.get("class_name"),
        "scientific_name": info.get("scientific_name"),
        "species_name": info.get("species_name"),
    }

    if proba is not None:
        try:
            result["max_probability"] = float(np.max(proba))
        except Exception:
            pass

    logger.info("Prediction mapping info: %s", info)
    logger.info("Final response JSON: %s", result)

    return result


def _generate_feature_debug(sequence_text: str) -> Dict[str, Any]:
    """공통 디버그 로직"""
    work_dir = None
    try:
        ensure_resources_loaded()

        raw_input = str(sequence_text)
        logger.info("Received raw input length: %d", len(raw_input))
        logger.info("Received raw input preview: %s", summarize_text(raw_input, max_len=300))

        work_dir = tempfile.mkdtemp(prefix="rf_tacc_debug_")

        seq = normalize_agent_sequence_input(raw_input)
        logger.info("Normalized sequence length: %d", len(seq))
        logger.info("Normalized sequence preview: %s", summarize_seq(seq))

        fasta_path = write_temp_fasta(seq, work_dir, name="query")
        out_csv = os.path.join(work_dir, "result_TACC.csv")

        lag_value = int(TACC_LAG) if str(TACC_LAG).strip() else None

        run_ilearn_tacc(
            ilearn_acc_py=ILEARN_SCRIPT_PATH,
            fasta_path=fasta_path,
            out_csv=out_csv,
            seq_type=SEQ_TYPE,
            lag=lag_value,
        )

        logger.info("iLearn output file: %s", out_csv)
        logger.info("iLearn output file size: %d", os.path.getsize(out_csv))

        df_feat = load_ilearn_csv(out_csv)
        n_expected = get_model_expected_features(_MODEL)
        X = to_feature_matrix(df_feat, n_expected=n_expected)

        flat = X.flatten().astype(float)
        feature_digest = hashlib.sha256(flat.tobytes()).hexdigest()

        pred, proba = predict(_MODEL, X)
        species_info = _lookup_species_info(pred)

        return {
            "ok": True,
            "raw_length": len(raw_input),
            "normalized_length": len(seq),
            "normalized_preview": summarize_seq(seq),
            "raw_feature_shape": list(df_feat.shape),
            "raw_feature_columns": [str(c) for c in df_feat.columns],
            "parsed_feature_shape": list(X.shape),
            "expected_feature_count": n_expected,
            "feature_preview_first_20": flat[:20].tolist(),
            "feature_sha256": feature_digest,
            "predicted_label_raw": str(pred),
            "predicted_class_name": species_info.get("class_name"),
            "predicted_scientific_name": species_info.get("scientific_name"),
            "predicted_species_name": species_info.get("species_name"),
            "max_probability": float(np.max(proba)) if proba is not None else None,
            "csv_head": df_feat.head(1).to_dict(orient="records"),
            "model_path": MODEL_PATH,
            "jong_info_path": JONG_INFO_PATH,
            "ilearn_script_path": ILEARN_SCRIPT_PATH,
            "seq_type": SEQ_TYPE,
            "tacc_lag": int(TACC_LAG) if str(TACC_LAG).strip() else None,
        }

    finally:
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


# =============================================================================
# MCP Tools
# =============================================================================
@mcp.tool(
    description=(
        "Read-only species classification tool. "
        "Input is a nucleotide sequence or FASTA-like text. "
        "The tool internally extracts only the nucleotide sequence, "
        "runs TACC feature extraction and Random Forest prediction, "
        "and returns predicted_label, scientific_name, species_name, and max_probability. "
        "This tool does not modify files, create records, or perform external side effects."
    )
)
def classify_species_from_sequence(sequence_text: str) -> Dict[str, Any]:
    """
    입력된 핵산 서열 텍스트를 FASTA로 변환하고,
    iLearn TACC -> Random Forest -> class mapping 순서로 종 판별
    """
    work_dir = None

    try:
        ensure_resources_loaded()

        raw_input = str(sequence_text)
        logger.info("Received raw sequence input length: %d", len(raw_input))
        logger.info("Received raw sequence input preview: %s", summarize_text(raw_input, max_len=300))

        work_dir = tempfile.mkdtemp(prefix="rf_tacc_")

        seq = normalize_agent_sequence_input(raw_input)
        logger.info("Normalized sequence length: %d", len(seq))
        logger.info("Normalized sequence preview: %s", summarize_seq(seq))

        fasta_path = write_temp_fasta(seq, work_dir, name="query")
        logger.info("Temporary FASTA written: %s", fasta_path)

        out_csv = os.path.join(work_dir, "result_TACC.csv")
        lag_value = int(TACC_LAG) if str(TACC_LAG).strip() else None

        run_ilearn_tacc(
            ilearn_acc_py=ILEARN_SCRIPT_PATH,
            fasta_path=fasta_path,
            out_csv=out_csv,
            seq_type=SEQ_TYPE,
            lag=lag_value,
        )

        logger.info("iLearn output file: %s", out_csv)
        logger.info("iLearn output file size: %d", os.path.getsize(out_csv))

        df_feat = load_ilearn_csv(out_csv)
        logger.info("Loaded iLearn feature shape: rows=%d, cols=%d", df_feat.shape[0], df_feat.shape[1])
        logger.info("Loaded iLearn columns: %s", list(df_feat.columns))
        logger.info("Loaded iLearn first row: %s", df_feat.head(1).to_dict(orient="records"))

        n_expected = get_model_expected_features(_MODEL)
        logger.info("Model expected feature count: %d", n_expected)

        X = to_feature_matrix(df_feat, n_expected=n_expected)
        logger.info("Parsed feature matrix shape: rows=%d, cols=%d", X.shape[0], X.shape[1])

        pred, proba = predict(_MODEL, X)
        logger.info("Final predicted label: %s", str(pred))

        result = build_result_from_prediction(pred, proba)
        logger.info("Returning classification result: %s", result)

        return result

    except Exception as e:
        logger.exception("Inference failed")
        return {
            "predicted_label": None,
            "class_name": None,
            "scientific_name": None,
            "species_name": None,
            "error": f"Inference failed: {e}",
        }

    finally:
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


@mcp.tool(
    description=(
        "Read-only species classification tool for FASTA input. "
        "Returns predicted_label, scientific_name, species_name, and max_probability. "
        "This tool has no side effects."
    )
)
def classify_species_from_fasta(fasta_text: str) -> Dict[str, Any]:
    """FASTA 형식 문자열 입력용 보조 tool"""
    try:
        raw_input = str(fasta_text)
        logger.info("Received raw FASTA input length: %d", len(raw_input))
        logger.info("Received raw FASTA input preview: %s", summarize_text(raw_input, max_len=300))

        seq_only = normalize_agent_sequence_input(raw_input)
        logger.info("Normalized FASTA sequence length: %d", len(seq_only))
        logger.info("Normalized FASTA sequence preview: %s", summarize_seq(seq_only))

        return classify_species_from_sequence(seq_only)

    except Exception as e:
        logger.exception("FASTA parsing failed")
        return {
            "predicted_label": None,
            "class_name": None,
            "scientific_name": None,
            "species_name": None,
            "error": f"Failed to parse FASTA input: {e}",
        }


@mcp.tool(
    description=(
        "Read-only debug tool. "
        "Checks whether the nucleotide sequence is passed intact through MCP."
    )
)
def debug_sequence_input(sequence_text: str) -> Dict[str, Any]:
    try:
        raw = str(sequence_text)
        seq = normalize_agent_sequence_input(raw)

        return {
            "ok": True,
            "raw_length": len(raw),
            "normalized_length": len(seq),
            "raw_preview": summarize_text(raw, max_len=300),
            "normalized_preview": summarize_seq(seq),
            "contains_fasta_header": ">" in raw,
            "seq_type": SEQ_TYPE,
        }

    except Exception as e:
        logger.exception("Debug sequence input failed")
        return {
            "ok": False,
            "error": f"Debug failed: {e}",
        }


@mcp.tool(
    description="Read-only debug tool to inspect the exact feature vector generated from input sequence."
)
def debug_feature_vector(sequence_text: str) -> Dict[str, Any]:
    try:
        return _generate_feature_debug(sequence_text)
    except Exception as e:
        logger.exception("Debug feature vector failed")
        return {
            "ok": False,
            "error": f"Debug feature vector failed: {e}",
        }


@mcp.tool(
    description=(
        "Read-only debug tool to compare prediction pipeline details including "
        "parsed feature hash, predicted label, and current file paths."
    )
)
def debug_compare_prediction(sequence_text: str) -> Dict[str, Any]:
    try:
        result = _generate_feature_debug(sequence_text)
        return result
    except Exception as e:
        logger.exception("Debug compare prediction failed")
        return {
            "ok": False,
            "error": f"Debug compare prediction failed: {e}",
        }


@mcp.tool(
    description=(
        "Read-only health check. "
        "Checks whether model, class mapping CSV, and iLearn script are properly loaded and ready."
    )
)
def health_check() -> Dict[str, Any]:
    try:
        ensure_resources_loaded()
        return {
            "ok": True,
            "server": "rf-tacc-species-mcp",
            "base_dir": BASE_DIR,
            "model_path": MODEL_PATH,
            "jong_info_path": JONG_INFO_PATH,
            "ilearn_script_path": ILEARN_SCRIPT_PATH,
            "seq_type": SEQ_TYPE,
            "tacc_lag": int(TACC_LAG) if str(TACC_LAG).strip() else None,
            "notes": "Server is healthy and read-only classification tools are ready.",
        }
    except Exception as e:
        logger.exception("Health check failed")
        return {
            "ok": False,
            "server": "rf-tacc-species-mcp",
            "error": f"Health check failed: {e}",
        }


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    logger.info("Starting MCP server on %s:%s", HOST, PORT)
    logger.info("BASE_DIR: %s", BASE_DIR)
    logger.info("MODEL_PATH: %s", MODEL_PATH)
    logger.info("JONG_INFO_PATH: %s", JONG_INFO_PATH)
    logger.info("ILEARN_SCRIPT_PATH: %s", ILEARN_SCRIPT_PATH)
    logger.info("SEQ_TYPE: %s", SEQ_TYPE)
    logger.info("TACC_LAG: %s", TACC_LAG if str(TACC_LAG).strip() else None)

    mcp.run(transport="http", host=HOST, port=PORT)