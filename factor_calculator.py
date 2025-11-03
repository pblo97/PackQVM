"""
Factor Calculator V2 - Con neutralización por industria
=======================================================

MEJORAS:
1. Sector-neutral ranking (compara tech vs tech, no tech vs utilities)
2. Composite scores más robustos
3. Mejor manejo de missing data
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional


# ============================================================================
# HELPERS ESTADÍSTICOS
# ============================================================================

def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    """Winsoriza serie al percentil p (ambas colas)"""
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 3:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def zscore(s: pd.Series) -> pd.Series:
    """Z-score robusto"""
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return (s - mu) / sd


def rank_pct(s: pd.Series) -> pd.Series:
    """Ranking percentil (0-1)"""
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0.5, index=s.index)
    return s.rank(pct=True, method="average").fillna(0.5)


def rank_within_groups(df: pd.DataFrame, col: str, group_col: str = "sector") -> pd.Series:
    """
    MEJORA CRÍTICA: Ranking dentro de grupos (sector-neutral).
    
    Evita que sectores cíclicos (energy, materials) tengan
    scores artificialmente bajos en recesiones.
    """
    if group_col not in df.columns:
        return rank_pct(df[col])
    
    def _rank_group(group):
        return rank_pct(group[col])
    
    # Rank dentro de cada sector
    ranked = df.groupby(group_col, group_keys=False).apply(_rank_group)
    
    # Fallback global para sectores muy pequeños (< 3 símbolos)
    group_sizes = df.groupby(group_col).size()
    small_sectors = group_sizes[group_sizes < 3].index
    
    if len(small_sectors) > 0:
        mask = df[group_col].isin(small_sectors)
        ranked.loc[mask] = rank_pct(df.loc[mask, col])
    
    return ranked


# ============================================================================
# VALUE FACTORS (con sector-neutral)
# ============================================================================

def calculate_value_score(df: pd.DataFrame, sector_neutral: bool = True) -> pd.Series:
    """
    Value compuesto con opción de sector-neutral.
    
    Args:
        sector_neutral: Si True, compara dentro del mismo sector
    """
    # Invertir múltiplos
    ev_ebitda = pd.to_numeric(df.get("ev_ebitda"), errors="coerce")
    pb = pd.to_numeric(df.get("pb"), errors="coerce")
    pe = pd.to_numeric(df.get("pe"), errors="coerce")
    
    v1 = winsorize(1.0 / ev_ebitda.replace(0, np.nan), 0.01)
    v2 = winsorize(1.0 / pb.replace(0, np.nan), 0.01)
    v3 = winsorize(1.0 / pe.replace(0, np.nan), 0.01)
    
    # Combinar
    value_raw = 0.40 * zscore(v1) + 0.30 * zscore(v2) + 0.30 * zscore(v3)
    
    # Sector-neutral o global
    if sector_neutral:
        df_temp = df.copy()
        df_temp["_value_raw"] = value_raw
        return rank_within_groups(df_temp, "_value_raw", "sector")
    else:
        return rank_pct(value_raw)


# ============================================================================
# QUALITY FACTORS (con sector-neutral)
# ============================================================================

def calculate_quality_score(df: pd.DataFrame, sector_neutral: bool = True) -> pd.Series:
    """
    Quality con sector-neutral.
    
    CRÍTICO: Evita que utilities (bajo ROE por regulación)
    sean castigados vs tech (alto ROE por naturaleza).
    """
    roe = pd.to_numeric(df.get("roe"), errors="coerce")
    roic = pd.to_numeric(df.get("roic"), errors="coerce")
    gross_margin = pd.to_numeric(df.get("gross_margin"), errors="coerce")
    
    q1 = winsorize(roe, 0.01)
    q2 = winsorize(roic, 0.01)
    q3 = winsorize(gross_margin, 0.01)
    
    quality_raw = 0.35 * zscore(q1) + 0.35 * zscore(q2) + 0.30 * zscore(q3)
    
    if sector_neutral:
        df_temp = df.copy()
        df_temp["_quality_raw"] = quality_raw
        return rank_within_groups(df_temp, "_quality_raw", "sector")
    else:
        return rank_pct(quality_raw)


# ============================================================================
# PROFITABILITY (con sector-neutral)
# ============================================================================

def calculate_profitability_score(df: pd.DataFrame, sector_neutral: bool = True) -> pd.Series:
    """Profitability score (FCF + OCF yields)"""
    fcf = pd.to_numeric(df.get("fcf"), errors="coerce")
    ocf = pd.to_numeric(df.get("operating_cf"), errors="coerce")
    
    if "market_cap" in df.columns:
        mcap = pd.to_numeric(df["market_cap"], errors="coerce")
        fcf_yield = fcf / mcap.replace(0, np.nan)
        ocf_yield = ocf / mcap.replace(0, np.nan)
    else:
        fcf_yield = fcf
        ocf_yield = ocf
    
    p1 = winsorize(fcf_yield, 0.01)
    p2 = winsorize(ocf_yield, 0.01)
    
    profit_raw = 0.60 * zscore(p1) + 0.40 * zscore(p2)
    
    if sector_neutral:
        df_temp = df.copy()
        df_temp["_profit_raw"] = profit_raw
        return rank_within_groups(df_temp, "_profit_raw", "sector")
    else:
        return rank_pct(profit_raw)


# ============================================================================
# MOMENTUM (placeholder - se reemplaza con momentum_calculator.py)
# ============================================================================

def calculate_momentum_placeholder(df: pd.DataFrame) -> pd.Series:
    """
    Momentum placeholder.
    
    IMPORTANTE: Usa momentum_calculator.integrate_real_momentum() 
    después de llamar a esta función.
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
    sector_neutral: bool = True,  # ← NUEVO parámetro
) -> pd.DataFrame:
    """
    QVM composite V2 con sector-neutral.
    
    Args:
        sector_neutral: Si True, todos los factores son sector-neutral
    """
    df = df.copy()
    
    # Calcular factores individuales (sector-neutral si aplica)
    df["value_score"] = calculate_value_score(df, sector_neutral)
    df["quality_score"] = calculate_quality_score(df, sector_neutral)
    df["profitability_score"] = calculate_profitability_score(df, sector_neutral)
    df["momentum_score"] = calculate_momentum_placeholder(df)
    
    # Quality extendido
    df["quality_extended"] = (
        0.60 * df["quality_score"] + 
        0.40 * df["profitability_score"]
    )
    
    # QVM compuesto
    df["qvm_score"] = (
        w_quality * df["quality_extended"] +
        w_value * df["value_score"] +
        w_momentum * df["momentum_score"]
    )
    
    # Percentil final
    df["qvm_rank"] = rank_pct(df["qvm_score"])
    
    return df


# ============================================================================
# MAIN FUNCTION (V2)
# ============================================================================

def compute_all_factors(
    df_universe: pd.DataFrame,
    df_fundamentals: pd.DataFrame,
    sector_neutral: bool = True,  # ← NUEVO parámetro
    w_quality: float = 0.40,
    w_value: float = 0.30,
    w_momentum: float = 0.30,
) -> pd.DataFrame:
    """
    Función principal V2 con sector-neutral.
    
    Args:
        sector_neutral: Si True, factores son sector-neutral (RECOMENDADO)
    """
    # Merge
    df = df_universe.merge(df_fundamentals, on="symbol", how="left")
    
    # Calcular QVM
    df = calculate_qvm_composite(
        df,
        w_quality=w_quality,
        w_value=w_value,
        w_momentum=w_momentum,
        sector_neutral=sector_neutral,
    )
    
    # Orden por QVM
    df = df.sort_values("qvm_score", ascending=False)
    
    return df


# ============================================================================
# ESTADÍSTICAS DE NEUTRALIZACIÓN
# ============================================================================

def neutralization_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estadísticas de sector para validar neutralización.
    
    Debería ver:
    - Media de value_score ~0.5 en cada sector
    - StdDev similar entre sectores
    """
    if "sector" not in df.columns:
        return pd.DataFrame()
    
    stats = df.groupby("sector").agg({
        "value_score": ["mean", "std", "count"],
        "quality_score": ["mean", "std"],
        "momentum_score": ["mean", "std"],
    }).round(3)
    
    return stats


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing factor_calculator V2...")
    
    # Mock data con sectores diferentes
    df = pd.DataFrame({
        "symbol": ["TECH1", "TECH2", "UTIL1", "UTIL2"],
        "sector": ["Technology", "Technology", "Utilities", "Utilities"],
        "market_cap": [3e12, 2e12, 1e11, 8e10],
        "ev_ebitda": [20, 18, 12, 11],  # Tech tiene múltiplos más altos
        "pb": [40, 35, 2, 1.8],
        "pe": [28, 26, 15, 14],
        "roe": [0.50, 0.45, 0.10, 0.09],  # Tech tiene ROE más alto por naturaleza
        "roic": [0.35, 0.32, 0.08, 0.07],
        "gross_margin": [0.42, 0.40, 0.25, 0.23],
        "fcf": [1e11, 8e10, 5e9, 4e9],
        "operating_cf": [1.2e11, 9e10, 6e9, 5e9],
    })
    
    df_fund = df[["symbol", "ev_ebitda", "pb", "pe", "roe", "roic", "gross_margin", "fcf", "operating_cf"]]
    df_uni = df[["symbol", "sector", "market_cap"]]
    
    # Sin sector-neutral
    print("\n📊 Sin sector-neutral:")
    result_global = compute_all_factors(df_uni, df_fund, sector_neutral=False)
    print(result_global[["symbol", "sector", "quality_score", "value_score"]])
    
    # Con sector-neutral
    print("\n📊 Con sector-neutral:")
    result_neutral = compute_all_factors(df_uni, df_fund, sector_neutral=True)
    print(result_neutral[["symbol", "sector", "quality_score", "value_score"]])
    
    print("\n✅ Nota la diferencia:")
    print("- Sin neutral: UTIL tiene scores bajos (comparado con TECH)")
    print("- Con neutral: Cada símbolo se compara con su sector")
    
    # Stats
    print("\n📊 Estadísticas de neutralización:")
    print(neutralization_stats(result_neutral))
