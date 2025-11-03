"""
Piotroski F-Score - Implementación Académica
=============================================

Basado en:
Piotroski, J. D. (2000). "Value investing: The use of historical 
financial statement information to separate winners from losers."
Journal of Accounting Research, 38, 1-41.

Sistema de 9 checks binarios que identifica empresas de alta calidad.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List


# ============================================================================
# PIOTROSKI F-SCORE (9 Puntos)
# ============================================================================

def calculate_piotroski_fscore(df: pd.DataFrame) -> pd.Series:
    """
    Calcula F-Score de Piotroski (0-9 puntos).
    
    Paper: Piotroski (2000)
    
    9 Checks Binarios:
    
    PROFITABILIDAD (4 puntos):
    1. ROA > 0
    2. Operating Cash Flow > 0
    3. Δ ROA > 0 (mejorando año-a-año)
    4. Accruals < 0 (OCF > Net Income = calidad de earnings)
    
    LEVERAGE/LIQUIDEZ (3 puntos):
    5. Δ Long-term Debt / Assets < 0 (reduciendo leverage)
    6. Δ Current Ratio > 0 (mejorando liquidez)
    7. No nueva emisión de acciones
    
    EFICIENCIA OPERATIVA (2 puntos):
    8. Δ Gross Margin > 0 (mejorando)
    9. Δ Asset Turnover > 0 (usando activos más eficientemente)
    
    Interpretación:
    - F-Score 8-9: High quality (comprar)
    - F-Score 6-7: Medium quality (considerar)
    - F-Score 0-5: Low quality (evitar)
    
    Returns:
        Serie con F-Score (0-9) por símbolo
    """
    score = pd.Series(0, index=df.index)
    
    # -------------------------
    # PROFITABILIDAD (4 puntos)
    # -------------------------
    
    # 1. ROA > 0
    if 'roa' in df.columns:
        roa = pd.to_numeric(df['roa'], errors='coerce')
        score += (roa > 0).fillna(False).astype(int)
    
    # 2. Operating Cash Flow > 0
    if 'operating_cf' in df.columns:
        ocf = pd.to_numeric(df['operating_cf'], errors='coerce')
        score += (ocf > 0).fillna(False).astype(int)
    
    # 3. Δ ROA > 0
    if 'roa' in df.columns and 'roa_prev' in df.columns:
        roa = pd.to_numeric(df['roa'], errors='coerce')
        roa_prev = pd.to_numeric(df['roa_prev'], errors='coerce')
        delta_roa = roa - roa_prev
        score += (delta_roa > 0).fillna(False).astype(int)
    
    # 4. Accruals < 0 (Quality of Earnings)
    # Accruals = Net Income - Operating CF
    if 'net_income' in df.columns and 'operating_cf' in df.columns:
        ni = pd.to_numeric(df['net_income'], errors='coerce')
        ocf = pd.to_numeric(df['operating_cf'], errors='coerce')
        accruals = ni - ocf
        score += (accruals < 0).fillna(False).astype(int)
    
    # -------------------------
    # LEVERAGE/LIQUIDEZ (3 puntos)
    # -------------------------
    
    # 5. Δ Leverage < 0 (reduciendo deuda)
    if 'long_term_debt' in df.columns and 'total_assets' in df.columns:
        if 'long_term_debt_prev' in df.columns and 'total_assets_prev' in df.columns:
            debt = pd.to_numeric(df['long_term_debt'], errors='coerce')
            assets = pd.to_numeric(df['total_assets'], errors='coerce')
            debt_prev = pd.to_numeric(df['long_term_debt_prev'], errors='coerce')
            assets_prev = pd.to_numeric(df['total_assets_prev'], errors='coerce')
            
            leverage = debt / assets.replace(0, np.nan)
            leverage_prev = debt_prev / assets_prev.replace(0, np.nan)
            
            delta_leverage = leverage - leverage_prev
            score += (delta_leverage < 0).fillna(False).astype(int)
    
    # 6. Δ Current Ratio > 0 (mejorando liquidez)
    if 'current_ratio' in df.columns and 'current_ratio_prev' in df.columns:
        cr = pd.to_numeric(df['current_ratio'], errors='coerce')
        cr_prev = pd.to_numeric(df['current_ratio_prev'], errors='coerce')
        delta_cr = cr - cr_prev
        score += (delta_cr > 0).fillna(False).astype(int)
    
    # 7. No dilución (shares outstanding no aumentó)
    if 'shares_outstanding' in df.columns and 'shares_outstanding_prev' in df.columns:
        shares = pd.to_numeric(df['shares_outstanding'], errors='coerce')
        shares_prev = pd.to_numeric(df['shares_outstanding_prev'], errors='coerce')
        delta_shares = shares - shares_prev
        score += (delta_shares <= 0).fillna(False).astype(int)
    
    # -------------------------
    # EFICIENCIA OPERATIVA (2 puntos)
    # -------------------------
    
    # 8. Δ Gross Margin > 0
    if 'gross_margin' in df.columns and 'gross_margin_prev' in df.columns:
        gm = pd.to_numeric(df['gross_margin'], errors='coerce')
        gm_prev = pd.to_numeric(df['gross_margin_prev'], errors='coerce')
        delta_gm = gm - gm_prev
        score += (delta_gm > 0).fillna(False).astype(int)
    
    # 9. Δ Asset Turnover > 0
    if 'revenue' in df.columns and 'total_assets' in df.columns:
        if 'revenue_prev' in df.columns and 'total_assets_prev' in df.columns:
            rev = pd.to_numeric(df['revenue'], errors='coerce')
            assets = pd.to_numeric(df['total_assets'], errors='coerce')
            rev_prev = pd.to_numeric(df['revenue_prev'], errors='coerce')
            assets_prev = pd.to_numeric(df['total_assets_prev'], errors='coerce')
            
            turnover = rev / assets.replace(0, np.nan)
            turnover_prev = rev_prev / assets_prev.replace(0, np.nan)
            
            delta_turnover = turnover - turnover_prev
            score += (delta_turnover > 0).fillna(False).astype(int)
    
    return score


# ============================================================================
# F-SCORE SIMPLIFICADO (cuando no hay datos históricos)
# ============================================================================

def calculate_simplified_fscore(df: pd.DataFrame) -> pd.Series:
    """
    F-Score simplificado para cuando solo tenemos datos TTM.
    
    Usa solo checks que no requieren comparación temporal (4 puntos máximo).
    """
    score = pd.Series(0, index=df.index)
    
    # 1. ROA > 0
    if 'roa' in df.columns:
        roa = pd.to_numeric(df['roa'], errors='coerce')
        score += (roa > 0).fillna(False).astype(int)
    elif 'roe' in df.columns:  # Fallback a ROE
        roe = pd.to_numeric(df['roe'], errors='coerce')
        score += (roe > 0).fillna(False).astype(int)
    
    # 2. Operating Cash Flow > 0
    if 'operating_cf' in df.columns:
        ocf = pd.to_numeric(df['operating_cf'], errors='coerce')
        score += (ocf > 0).fillna(False).astype(int)
    
    # 3. ROE > 15% (proxy de quality)
    if 'roe' in df.columns:
        roe = pd.to_numeric(df['roe'], errors='coerce')
        score += (roe > 0.15).fillna(False).astype(int)
    
    # 4. Free Cash Flow > 0
    if 'fcf' in df.columns:
        fcf = pd.to_numeric(df['fcf'], errors='coerce')
        score += (fcf > 0).fillna(False).astype(int)
    
    # Normalizar a 0-9 scale
    score = (score / 4.0) * 9.0
    
    return score.round(1)


# ============================================================================
# QUALITY FILTERS BASADOS EN F-SCORE
# ============================================================================

def filter_by_fscore(
    df: pd.DataFrame,
    min_score: int = 6,
    use_simplified: bool = True,
) -> pd.DataFrame:
    """
    Filtra empresas por F-Score.
    
    Args:
        df: DataFrame con datos fundamentales
        min_score: Score mínimo para pasar (6-9 recomendado)
        use_simplified: Si True, usa simplified F-Score (cuando no hay históricos)
    
    Returns:
        DataFrame filtrado
    """
    df = df.copy()
    
    if use_simplified:
        df['fscore'] = calculate_simplified_fscore(df)
    else:
        df['fscore'] = calculate_piotroski_fscore(df)
    
    # Filtrar
    passed = df[df['fscore'] >= min_score].copy()
    
    print(f"✅ F-Score Filter ({min_score}+): {len(passed)}/{len(df)} stocks pass")
    
    return passed


# ============================================================================
# ESTADÍSTICAS DE F-SCORE
# ============================================================================

def fscore_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera estadísticas de F-Score por rango.
    """
    if 'fscore' not in df.columns:
        return pd.DataFrame()
    
    # Categorizar
    df['fscore_category'] = pd.cut(
        df['fscore'],
        bins=[0, 3, 5, 7, 9],
        labels=['Low (0-3)', 'Medium-Low (4-5)', 'Medium-High (6-7)', 'High (8-9)']
    )
    
    # Estadísticas por categoría
    stats = df.groupby('fscore_category', observed=True).agg({
        'symbol': 'count',
        'roe': 'mean',
        'roic': 'mean',
        'fcf': 'mean',
    }).round(3)
    
    stats.columns = ['Count', 'Avg ROE', 'Avg ROIC', 'Avg FCF']
    
    return stats


# ============================================================================
# COMPONENTES INDIVIDUALES (para debugging)
# ============================================================================

def get_fscore_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna DataFrame con cada componente del F-Score por separado.
    
    Útil para debugging y entender qué checks pasa/falla cada empresa.
    """
    components = pd.DataFrame(index=df.index)
    components['symbol'] = df['symbol']
    
    # Profitability
    if 'roa' in df.columns:
        components['check_roa_positive'] = (pd.to_numeric(df['roa'], errors='coerce') > 0).astype(int)
    
    if 'operating_cf' in df.columns:
        components['check_ocf_positive'] = (pd.to_numeric(df['operating_cf'], errors='coerce') > 0).astype(int)
    
    if 'fcf' in df.columns:
        components['check_fcf_positive'] = (pd.to_numeric(df['fcf'], errors='coerce') > 0).astype(int)
    
    # Quality proxies
    if 'roe' in df.columns:
        components['check_roe_high'] = (pd.to_numeric(df['roe'], errors='coerce') > 0.15).astype(int)
    
    if 'gross_margin' in df.columns:
        components['check_margin_high'] = (pd.to_numeric(df['gross_margin'], errors='coerce') > 0.30).astype(int)
    
    # Total
    component_cols = [c for c in components.columns if c.startswith('check_')]
    components['total_score'] = components[component_cols].sum(axis=1)
    
    return components


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing piotroski_fscore...")
    
    # Mock data
    df = pd.DataFrame({
        'symbol': ['HIGH_QUALITY', 'MEDIUM', 'LOW_QUALITY'],
        'roa': [0.15, 0.08, -0.05],
        'roe': [0.25, 0.12, -0.10],
        'operating_cf': [1e9, 5e8, -1e8],
        'fcf': [8e8, 3e8, -2e8],
        'gross_margin': [0.45, 0.30, 0.15],
        'roic': [0.20, 0.10, -0.05],
    })
    
    # Simplified F-Score
    df['fscore'] = calculate_simplified_fscore(df)
    
    print("\n📊 F-Score Results:")
    print(df[['symbol', 'fscore', 'roe', 'fcf']])
    
    # Filter
    print("\n🔍 Filtering by F-Score >= 6:")
    passed = filter_by_fscore(df, min_score=6)
    print(passed[['symbol', 'fscore']])
    
    # Components
    print("\n🔧 F-Score Components:")
    components = get_fscore_components(df)
    print(components)
    
    print("\n✅ Tests complete!")
    print("\n💡 Interpretation:")
    print("  - F-Score 8-9: BUY (high quality)")
    print("  - F-Score 6-7: CONSIDER (medium quality)")
    print("  - F-Score 0-5: AVOID (low quality)")
