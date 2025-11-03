"""
Screener Filters - Guardrails simples
======================================

Filtros de calidad basados en reglas fundamentales simples.
Sin complejidad innecesaria, solo checks claros.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple


# ============================================================================
# CONFIGURACIÓN DE FILTROS
# ============================================================================

@dataclass
class FilterConfig:
    """Configuración de todos los filtros"""
    # Profitabilidad
    min_roe: float = 0.10          # ROE >= 10%
    min_gross_margin: float = 0.20  # Margen bruto >= 20%
    
    # Cash generation
    require_positive_fcf: bool = True
    require_positive_ocf: bool = True
    
    # Valuación (excluir extremos)
    max_pe: float = 100.0          # P/E <= 100 (excluir loss-making extremos)
    max_ev_ebitda: float = 50.0    # EV/EBITDA <= 50
    
    # Liquidez
    min_volume: int = 500_000      # Volumen diario >= 500k
    min_market_cap: float = 5e8    # Market cap >= $500M


# ============================================================================
# FILTROS INDIVIDUALES
# ============================================================================

def filter_profitability(df: pd.DataFrame, config: FilterConfig) -> pd.Series:
    """
    Filtro de rentabilidad básica.
    
    Checks:
    - ROE >= min_roe
    - Gross margin >= min_gross_margin
    
    Returns:
        Serie booleana (True = pasa)
    """
    roe = pd.to_numeric(df.get("roe", np.nan), errors="coerce")
    gm = pd.to_numeric(df.get("gross_margin", np.nan), errors="coerce")
    
    pass_roe = (roe >= config.min_roe) | roe.isna()  # NaN = no penalizar
    pass_gm = (gm >= config.min_gross_margin) | gm.isna()
    
    return pass_roe & pass_gm


def filter_cash_generation(df: pd.DataFrame, config: FilterConfig) -> pd.Series:
    """
    Filtro de generación de efectivo.
    
    Checks:
    - FCF > 0 (si require_positive_fcf)
    - Operating CF > 0 (si require_positive_ocf)
    
    Returns:
        Serie booleana (True = pasa)
    """
    fcf = pd.to_numeric(df.get("fcf", np.nan), errors="coerce")
    ocf = pd.to_numeric(df.get("operating_cf", np.nan), errors="coerce")
    
    if config.require_positive_fcf:
        pass_fcf = (fcf > 0) | fcf.isna()
    else:
        pass_fcf = pd.Series(True, index=df.index)
    
    if config.require_positive_ocf:
        pass_ocf = (ocf > 0) | ocf.isna()
    else:
        pass_ocf = pd.Series(True, index=df.index)
    
    return pass_fcf & pass_ocf


def filter_valuation(df: pd.DataFrame, config: FilterConfig) -> pd.Series:
    """
    Filtro de valuación (excluir extremos absurdos).
    
    Checks:
    - P/E <= max_pe
    - EV/EBITDA <= max_ev_ebitda
    
    Returns:
        Serie booleana (True = pasa)
    """
    pe = pd.to_numeric(df.get("pe", np.nan), errors="coerce")
    ev_ebitda = pd.to_numeric(df.get("ev_ebitda", np.nan), errors="coerce")
    
    pass_pe = (pe <= config.max_pe) | pe.isna()
    pass_ev = (ev_ebitda <= config.max_ev_ebitda) | ev_ebitda.isna()
    
    return pass_pe & pass_ev


def filter_liquidity(df: pd.DataFrame, config: FilterConfig) -> pd.Series:
    """
    Filtro de liquidez.
    
    Checks:
    - Volume >= min_volume
    - Market cap >= min_market_cap
    
    Returns:
        Serie booleana (True = pasa)
    """
    volume = pd.to_numeric(df.get("volume", np.nan), errors="coerce")
    mcap = pd.to_numeric(df.get("market_cap", np.nan), errors="coerce")
    
    pass_vol = (volume >= config.min_volume) | volume.isna()
    pass_mcap = (mcap >= config.min_market_cap) | mcap.isna()
    
    return pass_vol & pass_mcap


# ============================================================================
# APLICADOR PRINCIPAL
# ============================================================================

def apply_all_filters(
    df: pd.DataFrame,
    config: FilterConfig = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica todos los filtros y devuelve diagnóstico.
    
    Args:
        df: DataFrame con datos fundamentales
        config: Configuración de filtros (usa default si None)
    
    Returns:
        (passed, diagnostics)
        
        passed: DataFrame con símbolos que pasaron TODO
        diagnostics: DataFrame con flags individuales por símbolo
    """
    if config is None:
        config = FilterConfig()
    
    df = df.copy()
    
    # Aplicar cada filtro
    df["pass_profitability"] = filter_profitability(df, config)
    df["pass_cash"] = filter_cash_generation(df, config)
    df["pass_valuation"] = filter_valuation(df, config)
    df["pass_liquidity"] = filter_liquidity(df, config)
    
    # Agregado
    filter_cols = ["pass_profitability", "pass_cash", "pass_valuation", "pass_liquidity"]
    df["pass_all"] = df[filter_cols].all(axis=1)
    
    # Razón de rechazo
    def _reason(row):
        failed = [col.replace("pass_", "") for col in filter_cols if not row[col]]
        return ",".join(failed) if failed else ""
    
    df["reason"] = df.apply(_reason, axis=1)
    
    # Split
    passed = df[df["pass_all"] == True][["symbol"]].copy()
    
    diag_cols = ["symbol"] + filter_cols + ["pass_all", "reason"]
    diagnostics = df[diag_cols].copy()
    
    return passed, diagnostics


# ============================================================================
# FILTRO POR QVM SCORE
# ============================================================================

def filter_by_qvm(
    df: pd.DataFrame,
    min_qvm_rank: float = 0.50,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Filtra por QVM score.
    
    Args:
        df: DataFrame con 'qvm_rank'
        min_qvm_rank: Percentil mínimo (0-1)
        top_n: Si se especifica, devuelve solo top N
    
    Returns:
        DataFrame filtrado
    """
    df = df.copy()
    
    if "qvm_rank" not in df.columns:
        return df
    
    # Filtro por percentil
    mask = df["qvm_rank"] >= min_qvm_rank
    result = df[mask].copy()
    
    # Si se pide top N
    if top_n is not None and len(result) > top_n:
        result = result.nlargest(top_n, "qvm_rank")
    
    return result


# ============================================================================
# TESTS
# ============================================================================

from typing import Optional

if __name__ == "__main__":
    print("🧪 Testing screener_filters...")
    
    # Mock data
    df = pd.DataFrame({
        "symbol": ["PASS", "FAIL_PROF", "FAIL_CASH", "FAIL_VAL"],
        "roe": [0.25, 0.05, 0.30, 0.25],
        "gross_margin": [0.40, 0.40, 0.40, 0.40],
        "fcf": [1e9, 1e9, -1e8, 1e9],
        "operating_cf": [1.2e9, 1.2e9, 1.2e9, 1.2e9],
        "pe": [25, 25, 25, 150],
        "ev_ebitda": [15, 15, 15, 15],
        "volume": [1e6, 1e6, 1e6, 1e6],
        "market_cap": [10e9, 10e9, 10e9, 10e9],
    })
    
    passed, diag = apply_all_filters(df)
    
    print(f"\n✅ Pasaron: {len(passed)}")
    print(passed)
    
    print("\n📊 Diagnóstico:")
    print(diag)
    
    assert len(passed) == 1, "Expected 1 pass"
    assert passed["symbol"].iloc[0] == "PASS", "Wrong symbol passed"
    
    print("\n✅ All tests passed!")
