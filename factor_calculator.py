"""
Factor Calculator V2 - Con neutralización por industria (versión robusta)
========================================================================

MEJORAS:
1) Sector-neutral ranking con mezcla (blend) para sectores pequeños
2) Composite más estable (winsorize + zscore robusto por defecto)
3) Inversión segura de múltiplos y mejor manejo de missing/inf
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Literal

# ============================================================================
# HELPERS ESTADÍSTICOS
# ============================================================================

def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    """Winsoriza serie al percentil p (ambas colas)."""
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 3:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)

def zscore_meanstd(s: pd.Series) -> pd.Series:
    """Z-score clásico (mean/std)."""
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(ddof=0, skipna=True)
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return (s - mu) / sd

def zscore_mad(s: pd.Series) -> pd.Series:
    """Z-score robusto usando mediana y MAD."""
    s = pd.to_numeric(s, errors="coerce")
    med = s.median(skipna=True)
    mad = (s - med).abs().median(skipna=True)
    if not np.isfinite(mad) or mad == 0:
        mad = 1.0
    return 0.6745 * (s - med) / mad  # 0.6745 ≈ escala normal

def robust_zscore(s: pd.Series, method: Literal["mad","std"]="mad") -> pd.Series:
    return zscore_mad(s) if method == "mad" else zscore_meanstd(s)

def rank_pct(s: pd.Series) -> pd.Series:
    """Ranking percentil (0-1)."""
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0.5, index=s.index)
    return s.rank(pct=True, method="average").fillna(0.5)

def safe_inv(s: pd.Series) -> pd.Series:
    """
    Inversión segura de múltiplos: solo invierte valores >0; si <=0 o NaN => NaN.
    Evita ±inf cuando hay 0 o negativos.
    """
    x = pd.to_numeric(s, errors="coerce")
    x = x.where(x > 0, np.nan)
    inv = 1.0 / x
    inv.replace([np.inf, -np.inf], np.nan, inplace=True)
    return inv

def _blend(a: pd.Series, b: pd.Series, alpha: float = 0.5) -> pd.Series:
    """
    Mezcla convexa de dos series (alineadas por índice): alpha*a + (1-alpha)*b.
    Si faltan valores en a, usa b; y viceversa.
    """
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    out = alpha * a + (1 - alpha) * b
    out = out.where(out.notna(), a).where(out.notna(), b)
    return out

def rank_within_groups(
    df: pd.DataFrame,
    col: str,
    group_col: str = "sector",
    min_group: int = 4,
    blend_alpha: float = 0.5
) -> pd.Series:
    """
    Ranking percentil dentro de grupos (sector-neutral) con mezcla:
    - Si el tamaño de grupo >= min_group → rank intra-sector
    - Si el tamaño < min_group        → mezcla rank global con intra-sector (poco confiable)
    """
    # Si no hay columna de grupo, rank global
    if group_col not in df.columns:
        return rank_pct(df[col])

    # Rank global (para fallback/mezcla)
    global_rank = rank_pct(df[col])

    # Rank por grupo
    ranked = df.groupby(group_col, group_keys=False)[col].apply(rank_pct)

    # Mezcla para sectores pequeños
    group_sizes = df.groupby(group_col).size()
    small_sectors = group_sizes[group_sizes < min_group].index

    if len(small_sectors) > 0:
        mask = df[group_col].isin(small_sectors)
        # Mezclar intra-sector (pobre) con global
        ranked.loc[mask] = _blend(ranked.loc[mask], global_rank.loc[mask], alpha=blend_alpha)

    return ranked.fillna(0.5)

# ============================================================================
# VALUE FACTORS (con sector-neutral)
# ============================================================================

def calculate_value_score(
    df: pd.DataFrame,
    sector_neutral: bool = True,
    z_method: Literal["mad","std"] = "mad"
) -> pd.Series:
    """
    Value compuesto con opción de sector-neutral.
    Mapea múltiplos altos → puntaje bajo (vía inversión segura).
    """
    ev_ebitda = pd.to_numeric(df.get("ev_ebitda"), errors="coerce")
    pb        = pd.to_numeric(df.get("pb"), errors="coerce")
    pe        = pd.to_numeric(df.get("pe"), errors="coerce")

    v1 = winsorize(safe_inv(ev_ebitda), 0.01)  # EV/EBITDA ↓ → mejor
    v2 = winsorize(safe_inv(pb), 0.01)         # P/B ↓ → mejor
    v3 = winsorize(safe_inv(pe), 0.01)         # P/E ↓ → mejor

    value_raw = 0.40 * robust_zscore(v1, z_method) \
              + 0.30 * robust_zscore(v2, z_method) \
              + 0.30 * robust_zscore(v3, z_method)

    if sector_neutral:
        df_temp = df.copy()
        df_temp["_value_raw"] = value_raw
        out = rank_within_groups(df_temp, "_value_raw", "sector")
    else:
        out = rank_pct(value_raw)

    return out

# ============================================================================
# QUALITY FACTORS (con sector-neutral)
# ============================================================================

def calculate_quality_score(
    df: pd.DataFrame,
    sector_neutral: bool = True,
    z_method: Literal["mad","std"] = "mad"
) -> pd.Series:
    """
    Quality con sector-neutral.
    Evita penalizar sectores regulados frente a sectores de alto ROE estructural.
    """
    roe          = winsorize(pd.to_numeric(df.get("roe"), errors="coerce"), 0.01)
    roic         = winsorize(pd.to_numeric(df.get("roic"), errors="coerce"), 0.01)
    gross_margin = winsorize(pd.to_numeric(df.get("gross_margin"), errors="coerce"), 0.01)

    quality_raw = 0.35 * robust_zscore(roe,  z_method) \
                + 0.35 * robust_zscore(roic, z_method) \
                + 0.30 * robust_zscore(gross_margin, z_method)

    if sector_neutral:
        df_temp = df.copy()
        df_temp["_quality_raw"] = quality_raw
        out = rank_within_groups(df_temp, "_quality_raw", "sector")
    else:
        out = rank_pct(quality_raw)

    return out

# ============================================================================
# PROFITABILITY (con sector-neutral)
# ============================================================================

def calculate_profitability_score(
    df: pd.DataFrame,
    sector_neutral: bool = True,
    z_method: Literal["mad","std"] = "mad"
) -> pd.Series:
    """
    Profitability score (FCF + OCF yields si hay market_cap; si no, niveles).
    """
    fcf = pd.to_numeric(df.get("fcf"), errors="coerce")
    ocf = pd.to_numeric(df.get("operating_cf"), errors="coerce")

    if "market_cap" in df.columns:
        mcap = pd.to_numeric(df["market_cap"], errors="coerce")
        fcf_yield = fcf / mcap.replace(0, np.nan)
        ocf_yield = ocf / mcap.replace(0, np.nan)
    else:
        # fallback a niveles si no hay mcap (menos comparable cross-sectores)
        fcf_yield = fcf
        ocf_yield = ocf

    p1 = winsorize(fcf_yield, 0.01)
    p2 = winsorize(ocf_yield, 0.01)

    profit_raw = 0.60 * robust_zscore(p1, z_method) \
               + 0.40 * robust_zscore(p2, z_method)

    if sector_neutral:
        df_temp = df.copy()
        df_temp["_profit_raw"] = profit_raw
        out = rank_within_groups(df_temp, "_profit_raw", "sector")
    else:
        out = rank_pct(profit_raw)

    return out

# ============================================================================
# MOMENTUM (placeholder - se reemplaza con momentum_calculator.py)
# ============================================================================

def calculate_momentum_placeholder(df: pd.DataFrame) -> pd.Series:
    """
    Momentum placeholder:
    si ya existen quality/value, mezcla para tener un proxy antes de integrar el real.
    """
    if "quality_score" in df.columns and "value_score" in df.columns:
        q = pd.to_numeric(df["quality_score"], errors="coerce").fillna(0.5)
        v = pd.to_numeric(df["value_score"], errors="coerce").fillna(0.5)
        momentum = 0.70 * q + 0.30 * v
    else:
        momentum = pd.Series(0.5, index=df.index)

    return rank_pct(momentum)

# ============================================================================
# COMPOSITE QVM (V2 con sector-neutral)
# ============================================================================

def calculate_qvm_composite(
    df: pd.DataFrame,
    w_quality: float = 0.40,
    w_value: float = 0.30,
    w_momentum: float = 0.30,
    sector_neutral: bool = True,
    z_method: Literal["mad","std"] = "mad",
) -> pd.DataFrame:
    """
    QVM composite V2 con sector-neutral y z-score robusto por defecto.
    """
    df = df.copy()

    df["value_score"]        = calculate_value_score(df, sector_neutral, z_method)
    df["quality_score"]      = calculate_quality_score(df, sector_neutral, z_method)
    df["profitability_score"]= calculate_profitability_score(df, sector_neutral, z_method)
    df["momentum_score"]     = calculate_momentum_placeholder(df)

    # Quality extendido
    df["quality_extended"] = 0.60 * df["quality_score"] + 0.40 * df["profitability_score"]

    # QVM compuesto
    df["qvm_score"] = (
        w_quality * df["quality_extended"]
        + w_value * df["value_score"]
        + w_momentum * df["momentum_score"]
    )

    # Percentil final
    df["qvm_rank"] = rank_pct(df["qvm_score"])

    # Limpieza de columnas auxiliares si quedaron
    aux_cols = [c for c in df.columns if c.startswith("_value_") or c.startswith("_quality_") or c.startswith("_profit_")]
    if aux_cols:
        df.drop(columns=aux_cols, inplace=True, errors="ignore")

    return df

# ============================================================================
# MAIN FUNCTION (V2)
# ============================================================================

def compute_all_factors(
    df_universe: pd.DataFrame,
    df_fundamentals: pd.DataFrame,
    sector_neutral: bool = True,
    w_quality: float = 0.40,
    w_value: float = 0.30,
    w_momentum: float = 0.30,
    z_method: Literal["mad","std"] = "mad",
) -> pd.DataFrame:
    """
    Función principal V2 con sector-neutral.
    Requiere que df_universe tenga: ['symbol','sector','market_cap'] (si es posible).
    """
    df = df_universe.merge(df_fundamentals, on="symbol", how="left")

    df = calculate_qvm_composite(
        df,
        w_quality=w_quality,
        w_value=w_value,
        w_momentum=w_momentum,
        sector_neutral=sector_neutral,
        z_method=z_method,
    )

    df = df.sort_values("qvm_score", ascending=False).reset_index(drop=True)
    return df

# ============================================================================
# ESTADÍSTICAS DE NEUTRALIZACIÓN
# ============================================================================

def neutralization_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estadísticas por sector para validar neutralización (medias ~0.5, sd similares).
    """
    if "sector" not in df.columns:
        return pd.DataFrame()

    agg = {
        "value_score": ["mean", "std", "count"],
        "quality_score": ["mean", "std"],
        "profitability_score": ["mean", "std"],
        "momentum_score": ["mean", "std"],
    }
    out = df.groupby("sector").agg(agg).round(3)
    return out

# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing factor_calculator V2 (robusto)...")

    df = pd.DataFrame({
        "symbol": ["TECH1", "TECH2", "UTIL1", "UTIL2", "UTIL3"],
        "sector": ["Technology", "Technology", "Utilities", "Utilities", "Utilities"],
        "market_cap": [3e12, 2e12, 1e11, 8e10, 7e10],
        "ev_ebitda": [20, 18, 12, 11, 10],
        "pb": [40, 35, 2, 1.8, 1.6],
        "pe": [28, 26, 15, 14, 13],
        "roe": [0.50, 0.45, 0.10, 0.09, 0.08],
        "roic": [0.35, 0.32, 0.08, 0.07, 0.06],
        "gross_margin": [0.42, 0.40, 0.25, 0.23, 0.22],
        "fcf": [1e11, 8e10, 5e9, 4e9, 3.5e9],
        "operating_cf": [1.2e11, 9e10, 6e9, 5e9, 4.2e9],
    })

    df_fund = df[["symbol","ev_ebitda","pb","pe","roe","roic","gross_margin","fcf","operating_cf"]]
    df_uni  = df[["symbol","sector","market_cap"]]

    print("\n📊 Sin sector-neutral:")
    res_global = compute_all_factors(df_uni, df_fund, sector_neutral=False)
    print(res_global[["symbol","sector","quality_score","value_score"]])

    print("\n📊 Con sector-neutral (blend para grupos pequeños):")
    res_neutral = compute_all_factors(df_uni, df_fund, sector_neutral=True)
    print(res_neutral[["symbol","sector","quality_score","value_score"]])

    print("\n📊 Estadísticas neutralización:")
    print(neutralization_stats(res_neutral))
