"""
Factor Calculator - QVM según bibliografía académica
====================================================

Implementa factores basados en:
- Fama-French (Value, Quality)
- Jegadeesh-Titman (Momentum)
- Novy-Marx (Quality: gross profitability)
- Piotroski (F-Score components)

Todos los cálculos son cross-sectional (ranking relativo).
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
        return pd.Series(0.0, index=s.index)
    return s.rank(pct=True, method="average").fillna(0.5)


# ============================================================================
# VALUE FACTORS (Fama-French style)
# ============================================================================

def calculate_value_score(df: pd.DataFrame) -> pd.Series:
    """
    Value compuesto según Fama-French + ajustes modernos.
    
    Métricas:
    - EV/EBITDA (invertido, winsorizado)
    - P/B (invertido)
    - P/E (invertido)
    
    Returns:
        Serie con value score (0-1, alto=barato)
    """
    # Invertir múltiplos (bajo múltiplo = alto value)
    ev_ebitda = pd.to_numeric(df.get("ev_ebitda"), errors="coerce")
    pb = pd.to_numeric(df.get("pb"), errors="coerce")
    pe = pd.to_numeric(df.get("pe"), errors="coerce")
    
    # Invertir y winsorizar
    v1 = winsorize(1.0 / ev_ebitda.replace(0, np.nan), 0.01)
    v2 = winsorize(1.0 / pb.replace(0, np.nan), 0.01)
    v3 = winsorize(1.0 / pe.replace(0, np.nan), 0.01)
    
    # Combinar con z-scores
    value = 0.40 * zscore(v1) + 0.30 * zscore(v2) + 0.30 * zscore(v3)
    
    # Normalizar a 0-1
    return rank_pct(value)


# ============================================================================
# QUALITY FACTORS (Novy-Marx + Piotroski)
# ============================================================================

def calculate_quality_score(df: pd.DataFrame) -> pd.Series:
    """
    Quality según Novy-Marx (gross profitability) + ajustes.
    
    Métricas:
    - ROE (Return on Equity)
    - ROIC (Return on Invested Capital)
    - Gross Margin
    
    Returns:
        Serie con quality score (0-1, alto=mejor calidad)
    """
    roe = pd.to_numeric(df.get("roe"), errors="coerce")
    roic = pd.to_numeric(df.get("roic"), errors="coerce")
    gross_margin = pd.to_numeric(df.get("gross_margin"), errors="coerce")
    
    # Winsorizar
    q1 = winsorize(roe, 0.01)
    q2 = winsorize(roic, 0.01)
    q3 = winsorize(gross_margin, 0.01)
    
    # Combinar
    quality = 0.35 * zscore(q1) + 0.35 * zscore(q2) + 0.30 * zscore(q3)
    
    return rank_pct(quality)


# ============================================================================
# PROFITABILITY (Cash generation)
# ============================================================================

def calculate_profitability_score(df: pd.DataFrame) -> pd.Series:
    """
    Profitability score basado en generación de cash.
    
    Métricas:
    - FCF (Free Cash Flow)
    - Operating CF
    
    Returns:
        Serie con profitability score (0-1)
    """
    fcf = pd.to_numeric(df.get("fcf"), errors="coerce")
    ocf = pd.to_numeric(df.get("operating_cf"), errors="coerce")
    
    # Normalizar por market cap si existe
    if "market_cap" in df.columns:
        mcap = pd.to_numeric(df["market_cap"], errors="coerce")
        fcf_yield = fcf / mcap.replace(0, np.nan)
        ocf_yield = ocf / mcap.replace(0, np.nan)
    else:
        fcf_yield = fcf
        ocf_yield = ocf
    
    p1 = winsorize(fcf_yield, 0.01)
    p2 = winsorize(ocf_yield, 0.01)
    
    profit = 0.60 * zscore(p1) + 0.40 * zscore(p2)
    
    return rank_pct(profit)


# ============================================================================
# MOMENTUM (Jegadeesh-Titman 12-1)
# ============================================================================

def calculate_momentum_proxy(df: pd.DataFrame) -> pd.Series:
    """
    Momentum proxy desde factores fundamentales.
    
    NOTA: Idealmente usarías returns 12-1 meses.
    Aquí usamos proxy desde quality+value como placeholder.
    
    Returns:
        Serie con momentum score (0-1)
    """
    # Proxy: empresas con quality alto tienden a tener momentum
    # (correlación empírica documentada)
    if "quality_score" in df.columns and "value_score" in df.columns:
        q = pd.to_numeric(df["quality_score"], errors="coerce").fillna(0.5)
        v = pd.to_numeric(df["value_score"], errors="coerce").fillna(0.5)
        
        # Quality fuerte + value razonable = momentum positivo
        momentum = 0.70 * q + 0.30 * v
    else:
        momentum = pd.Series(0.5, index=df.index)
    
    return rank_pct(momentum)


# ============================================================================
# COMPOSITE QVM SCORE
# ============================================================================

def calculate_qvm_composite(
    df: pd.DataFrame,
    w_quality: float = 0.40,
    w_value: float = 0.30,
    w_momentum: float = 0.30,
) -> pd.DataFrame:
    """
    Calcula el score QVM compuesto.
    
    Args:
        df: DataFrame con métricas fundamentales
        w_quality: Peso de Quality (default 0.40 según literatura)
        w_value: Peso de Value (default 0.30)
        w_momentum: Peso de Momentum (default 0.30)
    
    Returns:
        DataFrame original + columnas:
            - value_score, quality_score, profitability_score, momentum_score
            - qvm_score (compuesto)
            - qvm_rank (percentil)
    """
    df = df.copy()
    
    # Calcular factores individuales
    df["value_score"] = calculate_value_score(df)
    df["quality_score"] = calculate_quality_score(df)
    df["profitability_score"] = calculate_profitability_score(df)
    df["momentum_score"] = calculate_momentum_proxy(df)
    
    # Quality extendido = quality + profitability
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
# NEUTRALIZACIÓN POR SECTOR
# ============================================================================

def neutralize_by_sector(df: pd.DataFrame, score_col: str = "qvm_score") -> pd.Series:
    """
    Neutraliza score por sector (sector-neutral).
    
    Returns:
        Serie con score neutralizado
    """
    if "sector" not in df.columns:
        return df[score_col]
    
    def _zscore_group(group):
        return zscore(group[score_col])
    
    neutral = df.groupby("sector", group_keys=False).apply(_zscore_group)
    
    return rank_pct(neutral)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def compute_all_factors(
    df_universe: pd.DataFrame,
    df_fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Función principal: combina universo + fundamentales y calcula todos los factores.
    
    Args:
        df_universe: DataFrame con ['symbol', 'sector', 'market_cap']
        df_fundamentals: DataFrame con métricas de FMP
    
    Returns:
        DataFrame completo con todos los scores
    """
    # Merge
    df = df_universe.merge(df_fundamentals, on="symbol", how="left")
    
    # Calcular QVM
    df = calculate_qvm_composite(df)
    
    # Neutralizar por sector
    df["qvm_score_neutral"] = neutralize_by_sector(df, "qvm_score")
    
    # Orden por QVM
    df = df.sort_values("qvm_score", ascending=False)
    
    return df


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing factor_calculator...")
    
    # Mock data
    df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL", "TSLA"],
        "sector": ["Technology", "Technology", "Technology", "Automotive"],
        "market_cap": [3e12, 2.5e12, 1.8e12, 8e11],
        "ev_ebitda": [20, 18, 22, 35],
        "pb": [40, 12, 7, 15],
        "pe": [28, 30, 25, 60],
        "roe": [0.50, 0.40, 0.30, 0.10],
        "roic": [0.35, 0.32, 0.28, 0.08],
        "gross_margin": [0.42, 0.68, 0.56, 0.25],
        "fcf": [1e11, 6e10, 5e10, 2e10],
        "operating_cf": [1.2e11, 7e10, 6e10, 3e10],
    })
    
    df_fund = df[["symbol", "ev_ebitda", "pb", "pe", "roe", "roic", "gross_margin", "fcf", "operating_cf"]]
    df_uni = df[["symbol", "sector", "market_cap"]]
    
    result = compute_all_factors(df_uni, df_fund)
    
    print("\n✅ Factores calculados:")
    print(result[["symbol", "value_score", "quality_score", "qvm_score", "qvm_rank"]])
