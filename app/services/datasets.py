from __future__ import annotations

from io import BytesIO
from typing import Tuple, Dict, Any

import pandas as pd


def profile_csv(data: bytes, name: str | None = None) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    df = pd.read_csv(BytesIO(data))
    dataset_name = name or "dataset"
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    stats: Dict[str, Any] = {}
    for col in df.columns:
        s = df[col]
        col_stats: Dict[str, Any] = {
            "non_null": int(s.notna().sum()),
            "nulls": int(s.isna().sum()),
            "unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            col_stats.update({
                "mean": float(s.dropna().mean()) if s.notna().any() else None,
                "std": float(s.dropna().std()) if s.notna().any() else None,
                "min": float(s.dropna().min()) if s.notna().any() else None,
                "max": float(s.dropna().max()) if s.notna().any() else None,
            })
        else:
            top_values = s.dropna().value_counts().head(5)
            col_stats["top_values"] = [{"value": str(idx), "count": int(val)} for idx, val in top_values.items()]
        stats[col] = col_stats

    # Store a small sample for reference
    sample = df.head(20).to_dict(orient="records")
    stats["sample_head"] = sample
    return dataset_name, schema, stats

