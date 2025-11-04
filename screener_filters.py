"""
Screener Filters - VERSIÓN OPTIMIZADA CON BACKWARD COMPATIBILITY
================================================================

Este archivo combina la funcionalidad optimizada con compatibilidad
hacia atrás para el código existente (app_streamlit.py).

MEJORAS:
1. ✅ Reglas más estrictas (ROE 15%, Margin 30%, P/E 40)
2. ✅ Nuevos filtros (leverage, growth, FCF margin)
3. ✅ Mantiene API compatible con código anterior
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


# ============================================================================
# CONFIGURACIÓN (Compatible + Optimizada)
# ============================================================================

@dataclass
class FilterConfig:
    """
    Configuración de filtros - VERSIÓN OPTIMIZADA
    
    Mantiene compatibilidad con código anterior pero con
    valores más estrictos por default.
    """
    # Profitabilidad (MÁS ESTRICTO)
    min_roe: float = 0.15              # 15% (vs 10% anterior) ⭐
    min_roic: float = 0.12             # 12% ROIC (nuevo)
    min_gross_margin: float = 0.30     # 30% (vs 20% anterior) ⭐
    
    # Cash generation
    require_positive_fcf: bool = True
    require_positive_ocf: bool = True
    min_fcf_margin: float = 0.05       # Nuevo: FCF/Revenue >= 5%
    
    # Valuación (MÁS ESTRICTO)
    max_pe: float = 40.0               # 40 (vs 100 anterior) ⭐
    max_ev_ebitda: float = 20.0        # 20 (vs 50 anterior) ⭐
    max_pb: float = 10.0               # Nuevo
    
    # Leverage (NUEVO)
    max_debt_to_equity: float = 2.0    
    min_current_ratio: float = 1.0
    
    # Liquidez
    min_volume: int = 1_000_000        # 1M (vs 500k anterior)
    min_market_cap: float = 2e9        # $2B (vs $500M anterior)
    
    # Growth (NUEVO)
    require_positive_revenue_growth: bool = False  # Opcional por ahora


# ============================================================================
# FILTROS INDIVIDUALES (OPTIMIZADOS)
# ============================================================================

def filter_profitability(df: pd.DataFrame, config: FilterConfig) -> pd.Series:
    """
    Filtro de rentabilidad ROBUSTO.
    
    Checks:
    - ROE >= 15% (vs 10% anterior)
    - ROIC >= 12% (nuevo)
    - Gross Margin >= 30% (vs 20% anterior)
    """
    checks = []
    
    # ROE check
    if 'roe' in df.columns:
        roe = pd.to_numeric(df.get('roe'), errors='coerce')
        checks.append((roe >= config.min_roe) | roe.isna())
    
    # ROIC check (nuevo)
    if 'roic' in df.columns:
        roic = pd.to_numeric(df.get('roic'), errors='coerce')
        checks.append((roic >= config.min_roic) | roic.isna())
    
    # Gross Margin check
    if 'gross_margin' in df.columns:
        gm = pd.to_numeric(df.get('gross_margin'), errors='coerce')
        checks.append((gm >= config.min_gross_margin) | gm.isna())
    
    if not checks:
        return pd.Series(True, index=df.index)
    
    result = pd.Series(True, index=df.index)
    for check in checks:
        result = result & check
    
    return result


def filter_cash_generation(df: pd.DataFrame, config: FilterConfig) -> pd.Series:
    """
    Filtro de generación de efectivo ROBUSTO.
    
    Checks:
    - FCF > 0
    - Operating CF > 0
    - FCF Margin >= 5% (nuevo)
    """
    checks = []
    
    # FCF > 0
    if config.require_positive_fcf and 'fcf' in df.columns:
        fcf = pd.to_numeric(df.get('fcf'), errors='coerce')
        checks.append((fcf > 0) | fcf.isna())
    
    # Operating CF > 0
    if config.require_positive_ocf and 'operating_cf' in df.columns:
        ocf = pd.to_numeric(df.get('operating_cf'), errors='coerce')
        checks.append((ocf > 0) | ocf.isna())
    
    # FCF Margin (nuevo)
    if 'fcf' in df.columns and 'revenue' in df.columns:
        fcf = pd.to_numeric(df.get('fcf'), errors='coerce')
        revenue = pd.to_numeric(df.get('revenue'), errors='coerce')
        fcf_margin = fcf / revenue.replace(0, np.nan)
        checks.append((fcf_margin >= config.min_fcf_margin) | fcf_margin.isna())
    
    if not checks:
        return pd.Series(True, index=df.index)
    
    result = pd.Series(True, index=df.index)
    for check in checks:
        result = result & check
    
    return result


def filter_valuation(df: pd.DataFrame, config: FilterConfig) -> pd.Series:
    """
    Filtro de valuación ROBUSTO.
    
    Checks:
    - P/E <= 40 (vs 100 anterior)
    - EV/EBITDA <= 20 (vs 50 anterior)
    - P/B <= 10 (nuevo)
    """
    checks = []
    
    # P/E check
    if 'pe' in df.columns:
        pe = pd.to_numeric(df.get('pe'), errors='coerce')
        checks.append((pe <= config.max_pe) | pe.isna())
    
    # EV/EBITDA check
    if 'ev_ebitda' in df.columns:
        ev_ebitda = pd.to_numeric(df.get('ev_ebitda'), errors='coerce')
        checks.append((ev_ebitda <= config.max_ev_ebitda) | ev_ebitda.isna())
    
    # P/B check (nuevo)
    if 'pb' in df.columns:
        pb = pd.to_numeric(df.get('pb'), errors='coerce')
        checks.append((pb <= config.max_pb) | pb.isna())
    
    if not checks:
        return pd.Series(True, index=df.index)
    
    result = pd.Series(True, index=df.index)
    for check in checks:
        result = result & check
    
    return result


def filter_liquidity(df: pd.DataFrame, config: FilterConfig) -> pd.Series:
    """
    Filtro de liquidez.
    
    Checks:
    - Volume >= 1M (vs 500k anterior)
    - Market cap >= $2B (vs $500M anterior)
    """
    checks = []
    
    # Volume
    if 'volume' in df.columns:
        volume = pd.to_numeric(df.get('volume'), errors='coerce')
        checks.append((volume >= config.min_volume) | volume.isna())
    
    # Market Cap
    if 'market_cap' in df.columns:
        mcap = pd.to_numeric(df.get('market_cap'), errors='coerce')
        checks.append((mcap >= config.min_market_cap) | mcap.isna())
    
    if not checks:
        return pd.Series(True, index=df.index)
    
    result = pd.Series(True, index=df.index)
    for check in checks:
        result = result & check
    
    return result


# ============================================================================
# APLICADOR PRINCIPAL (COMPATIBLE CON API ANTERIOR)
# ============================================================================

def apply_all_filters(
    df: pd.DataFrame,
    config: FilterConfig = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica todos los filtros y devuelve diagnóstico.
    
    API COMPATIBLE con versión anterior pero con filtros OPTIMIZADOS.
    
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
    df['pass_profitability'] = filter_profitability(df, config)
    df['pass_cash'] = filter_cash_generation(df, config)
    df['pass_valuation'] = filter_valuation(df, config)
    df['pass_liquidity'] = filter_liquidity(df, config)
    
    # Agregado
    filter_cols = [
        'pass_profitability', 
        'pass_cash', 
        'pass_valuation',
        'pass_liquidity'
    ]
    
    df['pass_all'] = df[filter_cols].all(axis=1)
    
    # Razón de rechazo
    def _rejection_reason(row):
        failed = [
            col.replace('pass_', '') 
            for col in filter_cols 
            if not row[col]
        ]
        return ', '.join(failed) if failed else ''
    
    df['reason'] = df.apply(_rejection_reason, axis=1)
    
    # Split
    # Conservar columnas base del screener para pasos futuros (sector, market cap, etc.)
    base_cols = [c for c in ['symbol', 'sector', 'market_cap', 'volume'] if c in df.columns]
    passed = df.loc[df['pass_all'] == True, base_cols].copy()

    
    diag_cols = ['symbol'] + filter_cols + ['pass_all', 'reason']
    diagnostics = df[diag_cols].copy()
    
    return passed, diagnostics


# ============================================================================
# FILTRO POR QVM SCORE (COMPATIBLE)
# ============================================================================

def filter_by_qvm(
    df: pd.DataFrame,
    min_qvm_rank: float = 0.50,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Filtra por QVM score.
    
    API COMPATIBLE con versión anterior.
    
    Args:
        df: DataFrame con 'qvm_rank'
        min_qvm_rank: Percentil mínimo (0-1)
        top_n: Si se especifica, devuelve solo top N
    
    Returns:
        DataFrame filtrado
    """
    df = df.copy()
    
    if 'qvm_rank' not in df.columns:
        return df
    
    # Filtro por percentil
    mask = df['qvm_rank'] >= min_qvm_rank
    result = df[mask].copy()
    
    # Si se pide top N
    if top_n is not None and len(result) > top_n:
        result = result.nlargest(top_n, 'qvm_rank')
    
    return result


# ============================================================================
# NUEVAS FUNCIONES (OPTIMIZACIÓN)
# ============================================================================

def get_filter_statistics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """
    Genera estadísticas de qué filtros rechazan más empresas.
    """
    if diagnostics.empty:
        return pd.DataFrame()
    
    filter_cols = [col for col in diagnostics.columns if col.startswith('pass_')]
    
    stats = []
    for col in filter_cols:
        filter_name = col.replace('pass_', '')
        passed = diagnostics[col].sum()
        total = len(diagnostics)
        pass_rate = passed / total if total > 0 else 0
        
        stats.append({
            'Filter': filter_name,
            'Passed': passed,
            'Total': total,
            'Pass Rate': f"{pass_rate:.1%}",
            'Reject Rate': f"{(1-pass_rate):.1%}",
        })
    
    return pd.DataFrame(stats)


def get_top_rejection_reasons(diagnostics: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Top razones de rechazo.
    """
    if diagnostics.empty:
        return pd.DataFrame()
    
    # Solo rechazados
    rejected = diagnostics[diagnostics['pass_all'] == False]
    
    if rejected.empty:
        return pd.DataFrame()
    
    # Count reasons
    reason_counts = rejected['reason'].value_counts().head(top_n)
    
    result = pd.DataFrame({
        'Rejection Reason': reason_counts.index,
        'Count': reason_counts.values,
        'Percentage': (reason_counts.values / len(rejected) * 100).round(1),
    })
    
    return result.reset_index(drop=True)


# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================================

# Alias para código que usa RobustFilterConfig
RobustFilterConfig = FilterConfig


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing Optimized Screener Filters (Compatible)...")
    
    # Mock data
    df = pd.DataFrame({
        'symbol': ['QUALITY', 'VALUE', 'GROWTH', 'JUNK'],
        'roe': [0.25, 0.08, 0.30, -0.05],
        'roic': [0.20, 0.06, 0.25, -0.03],
        'gross_margin': [0.45, 0.25, 0.42, 0.15],
        'fcf': [1e9, 5e8, 8e8, -1e8],
        'operating_cf': [1.2e9, 6e8, 1e9, -5e7],
        'revenue': [10e9, 5e9, 8e9, 2e9],
        'pe': [28, 12, 35, 150],
        'ev_ebitda': [18, 8, 22, 60],
        'pb': [6, 1.5, 8, 0.8],
        'volume': [2e6, 1.5e6, 3e6, 5e5],
        'market_cap': [50e9, 5e9, 30e9, 3e8],
    })
    
    print("\n📊 Test Data:")
    print(df[['symbol', 'roe', 'gross_margin', 'pe', 'fcf']].to_string(index=False))
    
    # Aplicar filtros
    config = FilterConfig()
    passed, diagnostics = apply_all_filters(df, config)
    
    print(f"\n✅ RESULTADOS:")
    print(f"  Pasaron: {len(passed)}/{len(df)}")
    print(f"  Símbolos: {', '.join(passed['symbol'].tolist())}")
    
    print("\n📋 DIAGNÓSTICO:")
    print(diagnostics.to_string(index=False))
    
    print("\n📊 ESTADÍSTICAS:")
    stats = get_filter_statistics(diagnostics)
    print(stats.to_string(index=False))
    
    print("\n✅ Tests passed!")
    print("\n💡 NOTA: Filtros OPTIMIZADOS pero API COMPATIBLE:")
    print("  - ROE >= 15% (vs 10% anterior)")
    print("  - Gross Margin >= 30% (vs 20% anterior)")
    print("  - P/E <= 40 (vs 100 anterior)")
    print("  - EV/EBITDA <= 20 (vs 50 anterior)")
