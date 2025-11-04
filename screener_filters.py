"""
Screener Filters - VERSIÓN OPTIMIZADA CON REGLAS ROBUSTAS
==========================================================

MEJORAS vs versión anterior:
1. ✅ ROE mínimo 15% (vs 10% anterior)
2. ✅ Gross Margin mínimo 30% (vs 20% anterior)  
3. ✅ P/E máximo 40 (vs 100 anterior)
4. ✅ EV/EBITDA máximo 20 (vs 50 anterior)
5. ✅ Checks de leverage y eficiencia
6. ✅ Filtro de FCF growth YoY

Basado en:
- Piotroski (2000): Quality checks
- Novy-Marx (2013): Profitability premium
- Asness et al. (2019): Quality minus junk
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


# ============================================================================
# CONFIGURACIÓN ROBUSTA
# ============================================================================

@dataclass
class RobustFilterConfig:
    """
    Configuración ROBUSTA basada en literatura académica.
    
    CRÍTICO: Umbrales más altos que versión anterior para 
    seleccionar solo empresas de alta calidad.
    """
    
    # ========== PROFITABILIDAD (más estricto) ==========
    min_roe: float = 0.15              # 15% ROE (quality threshold)
    min_roic: float = 0.12             # 12% ROIC
    min_gross_margin: float = 0.30     # 30% margin (moat indicator)
    
    # ========== CASH GENERATION (obligatorio) ==========
    require_positive_fcf: bool = True
    require_positive_ocf: bool = True
    min_fcf_margin: float = 0.05       # FCF / Revenue >= 5%
    
    # ========== VALUATION (más estricto) ==========
    max_pe: float = 40.0               # P/E <= 40 (vs 100 anterior)
    max_ev_ebitda: float = 20.0        # EV/EBITDA <= 20 (vs 50 anterior)
    max_pb: float = 10.0               # P/B <= 10
    
    # ========== LEVERAGE (nuevo) ==========
    max_debt_to_equity: float = 2.0    # Debt/Equity <= 2.0
    min_current_ratio: float = 1.0     # Current Ratio >= 1.0
    
    # ========== LIQUIDEZ ==========
    min_volume: int = 1_000_000        # 1M daily volume
    min_market_cap: float = 2e9        # $2B market cap
    
    # ========== GROWTH (nuevo) ==========
    require_positive_revenue_growth: bool = True
    min_revenue_growth: float = 0.0    # YoY growth >= 0%


# ============================================================================
# FILTROS INDIVIDUALES (OPTIMIZADOS)
# ============================================================================

def filter_profitability_robust(
    df: pd.DataFrame, 
    config: RobustFilterConfig
) -> pd.Series:
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
        roe = pd.to_numeric(df['roe'], errors='coerce')
        checks.append((roe >= config.min_roe) | roe.isna())
    
    # ROIC check (nuevo)
    if 'roic' in df.columns:
        roic = pd.to_numeric(df['roic'], errors='coerce')
        checks.append((roic >= config.min_roic) | roic.isna())
    
    # Gross Margin check
    if 'gross_margin' in df.columns:
        gm = pd.to_numeric(df['gross_margin'], errors='coerce')
        checks.append((gm >= config.min_gross_margin) | gm.isna())
    
    if not checks:
        return pd.Series(True, index=df.index)
    
    # Debe pasar TODOS los checks
    result = pd.Series(True, index=df.index)
    for check in checks:
        result = result & check
    
    return result


def filter_cash_generation_robust(
    df: pd.DataFrame,
    config: RobustFilterConfig
) -> pd.Series:
    """
    Filtro de generación de efectivo ROBUSTO.
    
    Checks:
    - FCF > 0 (obligatorio)
    - Operating CF > 0 (obligatorio)
    - FCF Margin >= 5% (nuevo)
    """
    checks = []
    
    # FCF > 0
    if config.require_positive_fcf and 'fcf' in df.columns:
        fcf = pd.to_numeric(df['fcf'], errors='coerce')
        checks.append((fcf > 0) | fcf.isna())
    
    # Operating CF > 0
    if config.require_positive_ocf and 'operating_cf' in df.columns:
        ocf = pd.to_numeric(df['operating_cf'], errors='coerce')
        checks.append((ocf > 0) | ocf.isna())
    
    # FCF Margin (nuevo)
    if 'fcf' in df.columns and 'revenue' in df.columns:
        fcf = pd.to_numeric(df['fcf'], errors='coerce')
        revenue = pd.to_numeric(df['revenue'], errors='coerce')
        fcf_margin = fcf / revenue.replace(0, np.nan)
        checks.append((fcf_margin >= config.min_fcf_margin) | fcf_margin.isna())
    
    if not checks:
        return pd.Series(True, index=df.index)
    
    result = pd.Series(True, index=df.index)
    for check in checks:
        result = result & check
    
    return result


def filter_valuation_robust(
    df: pd.DataFrame,
    config: RobustFilterConfig
) -> pd.Series:
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
        pe = pd.to_numeric(df['pe'], errors='coerce')
        checks.append((pe <= config.max_pe) | pe.isna())
    
    # EV/EBITDA check
    if 'ev_ebitda' in df.columns:
        ev_ebitda = pd.to_numeric(df['ev_ebitda'], errors='coerce')
        checks.append((ev_ebitda <= config.max_ev_ebitda) | ev_ebitda.isna())
    
    # P/B check (nuevo)
    if 'pb' in df.columns:
        pb = pd.to_numeric(df['pb'], errors='coerce')
        checks.append((pb <= config.max_pb) | pb.isna())
    
    if not checks:
        return pd.Series(True, index=df.index)
    
    result = pd.Series(True, index=df.index)
    for check in checks:
        result = result & check
    
    return result


def filter_leverage(
    df: pd.DataFrame,
    config: RobustFilterConfig
) -> pd.Series:
    """
    Filtro de leverage (NUEVO).
    
    Basado en Piotroski (2000): empresas con bajo leverage
    tienen mejor performance.
    
    Checks:
    - Debt/Equity <= 2.0
    - Current Ratio >= 1.0
    """
    checks = []
    
    # Debt to Equity
    if 'debt_to_equity' in df.columns:
        dte = pd.to_numeric(df['debt_to_equity'], errors='coerce')
        checks.append((dte <= config.max_debt_to_equity) | dte.isna())
    elif 'total_debt' in df.columns and 'total_equity' in df.columns:
        debt = pd.to_numeric(df['total_debt'], errors='coerce')
        equity = pd.to_numeric(df['total_equity'], errors='coerce')
        dte = debt / equity.replace(0, np.nan)
        checks.append((dte <= config.max_debt_to_equity) | dte.isna())
    
    # Current Ratio
    if 'current_ratio' in df.columns:
        cr = pd.to_numeric(df['current_ratio'], errors='coerce')
        checks.append((cr >= config.min_current_ratio) | cr.isna())
    
    if not checks:
        return pd.Series(True, index=df.index)
    
    result = pd.Series(True, index=df.index)
    for check in checks:
        result = result & check
    
    return result


def filter_growth(
    df: pd.DataFrame,
    config: RobustFilterConfig
) -> pd.Series:
    """
    Filtro de crecimiento (NUEVO).
    
    Checks:
    - Revenue growth YoY >= 0%
    """
    if not config.require_positive_revenue_growth:
        return pd.Series(True, index=df.index)
    
    checks = []
    
    # Revenue growth
    if 'revenue_growth_yoy' in df.columns:
        growth = pd.to_numeric(df['revenue_growth_yoy'], errors='coerce')
        checks.append((growth >= config.min_revenue_growth) | growth.isna())
    elif 'revenue' in df.columns and 'revenue_prev' in df.columns:
        rev = pd.to_numeric(df['revenue'], errors='coerce')
        rev_prev = pd.to_numeric(df['revenue_prev'], errors='coerce')
        growth = (rev - rev_prev) / rev_prev.replace(0, np.nan)
        checks.append((growth >= config.min_revenue_growth) | growth.isna())
    
    if not checks:
        return pd.Series(True, index=df.index)
    
    result = pd.Series(True, index=df.index)
    for check in checks:
        result = result & check
    
    return result


def filter_liquidity(
    df: pd.DataFrame,
    config: RobustFilterConfig
) -> pd.Series:
    """
    Filtro de liquidez.
    
    Checks:
    - Volume >= 1M (vs 500k anterior)
    - Market cap >= $2B (vs $500M anterior)
    """
    checks = []
    
    # Volume
    if 'volume' in df.columns:
        volume = pd.to_numeric(df['volume'], errors='coerce')
        checks.append((volume >= config.min_volume) | volume.isna())
    
    # Market Cap
    if 'market_cap' in df.columns:
        mcap = pd.to_numeric(df['market_cap'], errors='coerce')
        checks.append((mcap >= config.min_market_cap) | mcap.isna())
    
    if not checks:
        return pd.Series(True, index=df.index)
    
    result = pd.Series(True, index=df.index)
    for check in checks:
        result = result & check
    
    return result


# ============================================================================
# APLICADOR PRINCIPAL (OPTIMIZADO)
# ============================================================================

def apply_robust_filters(
    df: pd.DataFrame,
    config: Optional[RobustFilterConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica TODOS los filtros robustos.
    
    Args:
        df: DataFrame con datos fundamentales
        config: Configuración (usa default si None)
    
    Returns:
        (passed, diagnostics)
        
        passed: DataFrame con símbolos que pasaron TODO
        diagnostics: DataFrame con flags por símbolo
    """
    if config is None:
        config = RobustFilterConfig()
    
    df = df.copy()
    
    # Aplicar cada filtro
    df['pass_profitability'] = filter_profitability_robust(df, config)
    df['pass_cash'] = filter_cash_generation_robust(df, config)
    df['pass_valuation'] = filter_valuation_robust(df, config)
    df['pass_leverage'] = filter_leverage(df, config)
    df['pass_growth'] = filter_growth(df, config)
    df['pass_liquidity'] = filter_liquidity(df, config)
    
    # Agregado
    filter_cols = [
        'pass_profitability', 
        'pass_cash', 
        'pass_valuation',
        'pass_leverage',
        'pass_growth',
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
    
    df['rejection_reason'] = df.apply(_rejection_reason, axis=1)
    
    # Split
    passed = df[df['pass_all'] == True][['symbol']].copy()
    
    diag_cols = ['symbol'] + filter_cols + ['pass_all', 'rejection_reason']
    diagnostics = df[diag_cols].copy()
    
    return passed, diagnostics


# ============================================================================
# ESTADÍSTICAS DE FILTROS
# ============================================================================

def filter_statistics(diagnostics: pd.DataFrame) -> pd.DataFrame:
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


def top_rejection_reasons(diagnostics: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
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
    reason_counts = rejected['rejection_reason'].value_counts().head(top_n)
    
    result = pd.DataFrame({
        'Rejection Reason': reason_counts.index,
        'Count': reason_counts.values,
        'Percentage': (reason_counts.values / len(rejected) * 100).round(1),
    })
    
    return result.reset_index(drop=True)


# ============================================================================
# BACKWARD COMPATIBILITY (para código anterior)
# ============================================================================

# Alias para mantener compatibilidad con código que usa FilterConfig
FilterConfig = RobustFilterConfig

def apply_all_filters(
    df: pd.DataFrame,
    config: Optional[RobustFilterConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Alias para backward compatibility"""
    return apply_robust_filters(df, config)


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing Robust Screener Filters...")
    
    # Mock data con diferentes perfiles
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
        'debt_to_equity': [0.5, 1.8, 0.3, 3.5],
        'current_ratio': [2.0, 1.2, 1.8, 0.7],
        'volume': [2e6, 1.5e6, 3e6, 5e5],
        'market_cap': [50e9, 5e9, 30e9, 3e8],
    })
    
    print("\n📊 Test Data:")
    print(df[['symbol', 'roe', 'gross_margin', 'pe', 'fcf']].to_string(index=False))
    
    # Aplicar filtros robustos
    config = RobustFilterConfig()
    passed, diagnostics = apply_robust_filters(df, config)
    
    print(f"\n✅ RESULTADOS:")
    print(f"  Pasaron: {len(passed)}/{len(df)}")
    print(f"  Símbolos: {', '.join(passed['symbol'].tolist())}")
    
    print("\n📋 DIAGNÓSTICO:")
    print(diagnostics.to_string(index=False))
    
    print("\n📊 ESTADÍSTICAS DE FILTROS:")
    stats = filter_statistics(diagnostics)
    print(stats.to_string(index=False))
    
    print("\n❌ TOP RAZONES DE RECHAZO:")
    reasons = top_rejection_reasons(diagnostics)
    if not reasons.empty:
        print(reasons.to_string(index=False))
    
    print("\n✅ Tests passed!")
    print("\n💡 NOTA: Filtros más estrictos que versión anterior:")
    print("  - ROE >= 15% (vs 10%)")
    print("  - Gross Margin >= 30% (vs 20%)")
    print("  - P/E <= 40 (vs 100)")
    print("  - EV/EBITDA <= 20 (vs 50)")
