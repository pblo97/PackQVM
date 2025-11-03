# pipeline_factors.py
from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd

__all__ = [
    "build_factor_frame",
    "_build_guardrails_base_from_snapshot",
    "apply_guardrails_logic",
]

# -------------------------------------------------------------------
# 1) DATA FETCH (puedes reemplazar por tus fuentes reales/FMP)
# -------------------------------------------------------------------
def fetch_raw_fundamentals(tickers: Iterable[str]) -> pd.DataFrame:
    """
    Devuelve un DF por 'symbol' con columnas crudas.
    Reemplaza este mock por tu descarga/caché real (FMP, etc.).
    """
    rows = []
    for sym in tickers:
        rows.append({
            "symbol": sym,
            "sector": "Technology",                  # TODO: sector real
            "market_cap": np.random.uniform(5e9, 5e11),

            # --- márgenes (para profit_hits y coverage) ---
            "ebit_margin": np.random.uniform(-0.1, 0.25),
            "cfo_margin":  np.random.uniform(-0.1, 0.25),
            "fcf_margin":  np.random.uniform(-0.1, 0.25),

            # --- guardrails ---
            "netdebt_ebitda": np.random.uniform(-1, 3),
            "accruals_ta":    np.random.uniform(-0.1, 0.1),
            "asset_growth":   np.random.uniform(-0.05, 0.3),
            "share_issuance": np.random.uniform(-0.05, 0.05),

            # --- otros (Tab3, señales, etc.) ---
            "BreakoutScore": np.random.uniform(0, 100),
            "RVOL20":        np.random.uniform(0.5, 3),
            "hits":          np.random.randint(0, 5),
        })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# 2) FACTOR FRAME (único): normaliza columnas que Tab2/Tab3 usan
# -------------------------------------------------------------------
def build_factor_frame(symbols: list[str]) -> pd.DataFrame:
    """
    Regresa un DF por 'symbol' que incluya al menos:
      ebit_margin, cfo_margin, fcf_margin,
      netdebt_ebitda, accruals_ta, asset_growth, share_issuance,
      sector, market_cap.
    Si tu fetch real trae otros nombres, mapea aquí.
    """
    raw = fetch_raw_fundamentals(symbols).copy()

    # Garantiza presencia y tipo numérico
    def _ensure_num(df: pd.DataFrame, name: str, default=np.nan):
        if name not in df.columns:
            df[name] = default
        df[name] = pd.to_numeric(df[name], errors="coerce")
        return df

    for c in [
        "ebit_margin", "cfo_margin", "fcf_margin",
        "netdebt_ebitda", "accruals_ta", "asset_growth", "share_issuance",
    ]:
        _ensure_num(raw, c)

    if "sector" not in raw.columns:
        raw["sector"] = "Unknown"
    if "market_cap" not in raw.columns:
        raw["market_cap"] = np.nan

    # Solo columnas clave + cualquier otra que ya traes
    return raw


# -------------------------------------------------------------------
# 3) BUILDER base usado por Tab2 (profit_hits + coverage_count)
# -------------------------------------------------------------------
def _build_guardrails_base_from_snapshot(snapshot_vfq: pd.DataFrame,
                                         uni_df: pd.DataFrame) -> pd.DataFrame:
    """
    - Reinyecta sector/market_cap desde 'uni_df'
    - Calcula profit_hits desde márgenes (>0) independiente del coverage
    - Calcula coverage_count por BLOQUES:
        {márgenes disponibles}, netdebt_ebitda, accruals_ta, asset_growth, share_issuance
    """
    if snapshot_vfq is None or snapshot_vfq.empty:
        return pd.DataFrame(columns=[
            "symbol","sector","market_cap",
            "ebit_margin","cfo_margin","fcf_margin",
            "netdebt_ebitda","accruals_ta","asset_growth","share_issuance",
            "profit_hits","coverage_count",
        ])

    df = snapshot_vfq.copy()

    # Asegura nombres esperados (si tu pipeline upstream cambia etiquetas, mapea aquí)
    alias = {
        "ebit_margin":    ["ebit_margin","ebitMargin","EBIT_margin"],
        "cfo_margin":     ["cfo_margin","cfoMargin","CFO_margin","oper_cf_margin"],
        "fcf_margin":     ["fcf_margin","fcfMargin","FCF_margin"],
        "netdebt_ebitda": ["netdebt_ebitda","netDebtToEbitda","netDebt_EBITDA","NetDebtEBITDA"],
        "accruals_ta":    ["accruals_ta","accrualsTA","accruals_total_assets"],
        "asset_growth":   ["asset_growth","assetGrowth","assets_growth_yoy"],
        "share_issuance": ["share_issuance","net_issuance","shares_net_issuance"],
    }

    def _ensure_col(name, candidates):
        for c in candidates:
            if c in df.columns:
                df[name] = pd.to_numeric(df[c], errors="coerce")
                return
        df[name] = np.nan

    for k, cand in alias.items():
        _ensure_col(k, cand)

    # sector / mcap desde universo actual (sin pisar symbol)
    merge_cols = [c for c in ["symbol","sector","market_cap"] if c in uni_df.columns]
    df = (
        df.drop(columns=["sector","market_cap"], errors="ignore")
          .merge(uni_df[merge_cols], on="symbol", how="left")
    )

    # profit_hits = cuenta de márgenes > 0 (EBIT, CFO, FCF)
    ebit_hit = pd.to_numeric(df["ebit_margin"], errors="coerce") > 0
    cfo_hit  = pd.to_numeric(df["cfo_margin"],  errors="coerce") > 0
    fcf_hit  = pd.to_numeric(df["fcf_margin"],  errors="coerce") > 0

    profit_hits = (
        pd.concat([ebit_hit, cfo_hit, fcf_hit], axis=1)
          .sum(axis=1, min_count=1)
          .astype("Float64")
    )
    df["profit_hits"] = profit_hits.astype("Int64")

    # coverage_count por BLOQUES (márgenes disponibles cuentan como 1)
    has_profit_any = pd.concat([
        pd.to_numeric(df["ebit_margin"], errors="coerce").notna(),
        pd.to_numeric(df["cfo_margin"],  errors="coerce").notna(),
        pd.to_numeric(df["fcf_margin"],  errors="coerce").notna(),
    ], axis=1).any(axis=1)

    coverage_cols = [
        has_profit_any,
        pd.to_numeric(df["netdebt_ebitda"], errors="coerce").notna(),
        pd.to_numeric(df["accruals_ta"],    errors="coerce").notna(),
        pd.to_numeric(df["asset_growth"],   errors="coerce").notna(),
        pd.to_numeric(df["share_issuance"], errors="coerce").notna(),
    ]
    df["coverage_count"] = pd.concat(coverage_cols, axis=1).sum(axis=1).astype(int)

    order = [
        "symbol","sector","market_cap",
        "ebit_margin","cfo_margin","fcf_margin",
        "profit_hits","coverage_count",
        "netdebt_ebitda","accruals_ta","asset_growth","share_issuance",
    ]
    return df[[c for c in order if c in df.columns]].copy()


# -------------------------------------------------------------------
# 4) LÓGICA de guardrails (única, sin recalcular factores)
# -------------------------------------------------------------------
def apply_guardrails_logic(
    df: pd.DataFrame,
    *,
    PROFIT_MIN_HITS: int      = 2,
    MAX_ISSUANCE: float       = 0.03,
    MAX_ASSET_GROWTH: float   = 0.20,
    MAX_ACCRUALS_ABS: float   = 0.10,
    MAX_NETDEBT_EBITDA: float = 3.0,
    MIN_COVERAGE: int         = 2,
) -> pd.DataFrame:
    """
    Aplica umbrales y marca pass_* / pass_all usando columnas ya presentes.
    """
    out = df.copy()

    # Reconstruye coverage_count si faltara (mismo criterio de BLOQUES)
    if "coverage_count" not in out.columns:
        has_profit_any = pd.concat([
            pd.to_numeric(out.get("ebit_margin"), errors="coerce").notna(),
            pd.to_numeric(out.get("cfo_margin"),  errors="coerce").notna(),
            pd.to_numeric(out.get("fcf_margin"),  errors="coerce").notna(),
        ], axis=1).any(axis=1)

        blocks = pd.concat([
            has_profit_any,
            pd.to_numeric(out.get("netdebt_ebitda"), errors="coerce").notna(),
            pd.to_numeric(out.get("accruals_ta"),    errors="coerce").notna(),
            pd.to_numeric(out.get("asset_growth"),   errors="coerce").notna(),
            pd.to_numeric(out.get("share_issuance"), errors="coerce").notna(),
        ], axis=1)
        out["coverage_count"] = blocks.sum(axis=1).astype(int)

    profit_hits  = pd.to_numeric(out.get("profit_hits"),      errors="coerce").fillna(0).astype(int)
    issuance     = pd.to_numeric(out.get("share_issuance"),   errors="coerce")
    asset_growth = pd.to_numeric(out.get("asset_growth"),     errors="coerce")
    accruals_ta  = pd.to_numeric(out.get("accruals_ta"),      errors="coerce")
    ndebt        = pd.to_numeric(out.get("netdebt_ebitda"),   errors="coerce")

    out["pass_profit"]   = profit_hits >= int(PROFIT_MIN_HITS)
    out["pass_issuance"] = issuance.abs()     <= float(MAX_ISSUANCE)
    out["pass_assets"]   = asset_growth.abs() <= float(MAX_ASSET_GROWTH)
    out["pass_accruals"] = accruals_ta.abs()  <= float(MAX_ACCRUALS_ABS)
    out["pass_ndebt"]    = ndebt              <= float(MAX_NETDEBT_EBITDA)
    out["pass_coverage"] = out["coverage_count"] >= int(MIN_COVERAGE)

    for c in ["pass_profit","pass_issuance","pass_assets","pass_accruals","pass_ndebt","pass_coverage"]:
        out[c] = out[c].fillna(False)

    checks = ["pass_profit","pass_issuance","pass_assets","pass_accruals","pass_ndebt","pass_coverage"]
    out["pass_all"] = out[checks].all(axis=1)
    return out
