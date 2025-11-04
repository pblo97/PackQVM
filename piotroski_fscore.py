"""
Piotroski F-Score - Implementación Académica
=============================================

Basado en:
Piotroski, J. D. (2000). "Value investing: The use of historical
financial statement information to separate winners from losers."
Journal of Accounting Research, 38, 1-41.

Sistema de 9 checks binarios que identifica empresas de alta calidad.

Incluye:
- calculate_piotroski_fscore(df): F-Score académico (0-9) con deltas YoY
- calculate_simplified_fscore_no_roe(df): F-Score simplificado (3 checks TTM: ROA>0, OCF>0, FCF>0)
- get_fscore_components_no_roe(df): componentes binarios del simplificado (sin ROE)
- get_fscore_components(df): alias al no-ROE (compatibilidad)
- filter_by_fscore(..., use_simplified=True): filtra usando el simplificado por defecto (sin ROE)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List

# ----------------------------- Utils -----------------------------
def _num(s: pd.Series | None) -> pd.Series:
    if s is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(s, errors="coerce")

# =================================================================
#  PIOTROSKI F-SCORE (9 Puntos, versión académica)
# =================================================================
def calculate_piotroski_fscore(df: pd.DataFrame) -> pd.Series:
    """
    Calcula F-Score de Piotroski (0-9 puntos).

    PROFITABILIDAD (4):
      1) ROA > 0
      2) OCF > 0
      3) ΔROA > 0
      4) Accruals < 0  (OCF > Net Income)

    LEVERAGE/LIQUIDEZ (3):
      5) Δ(Long-term Debt / Assets) < 0
      6) Δ(Current Ratio) > 0
      7) No emisión de acciones (ΔShares <= 0)

    EFICIENCIA (2):
      8) ΔGross Margin > 0
      9) ΔAsset Turnover > 0
    """
    score = pd.Series(0, index=df.index, dtype="int64")

    # -------- PROFITABILIDAD --------
    if 'roa' in df.columns:
        roa = _num(df['roa'])
        score += (roa > 0).fillna(False).astype(int)

    if 'operating_cf' in df.columns:
        ocf = _num(df['operating_cf'])
        score += (ocf > 0).fillna(False).astype(int)

    if {'roa', 'roa_prev'} <= set(df.columns):
        delta_roa = _num(df['roa']) - _num(df['roa_prev'])
        score += (delta_roa > 0).fillna(False).astype(int)

    if {'net_income', 'operating_cf'} <= set(df.columns):
        accruals = _num(df['net_income']) - _num(df['operating_cf'])
        score += (accruals < 0).fillna(False).astype(int)

    # -------- LEVERAGE / LIQUIDEZ --------
    if {'long_term_debt', 'total_assets', 'long_term_debt_prev', 'total_assets_prev'} <= set(df.columns):
        leverage     = _num(df['long_term_debt']) / _num(df['total_assets']).replace(0, np.nan)
        leverage_prv = _num(df['long_term_debt_prev']) / _num(df['total_assets_prev']).replace(0, np.nan)
        score += ((leverage - leverage_prv) < 0).fillna(False).astype(int)

    if {'current_ratio', 'current_ratio_prev'} <= set(df.columns):
        score += ((_num(df['current_ratio']) - _num(df['current_ratio_prev'])) > 0).fillna(False).astype(int)

    if {'shares_outstanding', 'shares_outstanding_prev'} <= set(df.columns):
        score += ((_num(df['shares_outstanding']) - _num(df['shares_outstanding_prev'])) <= 0).fillna(False).astype(int)

    # -------- EFICIENCIA --------
    if {'gross_margin', 'gross_margin_prev'} <= set(df.columns):
        score += ((_num(df['gross_margin']) - _num(df['gross_margin_prev'])) > 0).fillna(False).astype(int)

    if {'revenue', 'revenue_prev', 'total_assets', 'total_assets_prev'} <= set(df.columns):
        at      = _num(df['revenue'])      / _num(df['total_assets']).replace(0, np.nan)
        at_prev = _num(df['revenue_prev']) / _num(df['total_assets_prev']).replace(0, np.nan)
        score += ((at - at_prev) > 0).fillna(False).astype(int)

    return score.astype("int64")

# =================================================================
#  F-SCORE SIMPLIFICADO (sin ROE, 3 checks TTM → 0..9)
# =================================================================
def calculate_simplified_fscore_no_roe(df: pd.DataFrame) -> pd.Series:
    """
    F-Score 'simplificado' SIN ROE (3 checks TTM):
      1) ROA > 0
      2) Operating CF > 0
      3) FCF > 0
    Normalizado a 0–9 (score/3 * 9).
    """
    if 'roa' in df.columns:
        roa_pos = (_num(df['roa']) > 0)
    else:
        roa_pos = (_num(df.get('net_income')) / _num(df.get('total_assets')) > 0)

    ocf_pos = (_num(df.get('operating_cf')) > 0)

    if 'fcf' in df.columns:
        fcf_pos = (_num(df['fcf']) > 0)
    else:
        fcf_pos = ((_num(df.get('operating_cf')) - _num(df.get('capex'))) > 0)

    checks = pd.concat([
        roa_pos.rename('check_roa_positive'),
        ocf_pos.rename('check_ocf_positive'),
        fcf_pos.rename('check_fcf_positive'),
    ], axis=1).fillna(False)

    raw = checks.sum(axis=1).astype(int)
    out = (raw * 9.0 / 3.0).round(2)
    return out

# =================================================================
#  COMPONENTES (debug) — SIN ROE
# =================================================================
def get_fscore_components_no_roe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve checks individuales (sin ROE) + fscore normalizado (0..9).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    idx = df['symbol'] if 'symbol' in df.columns else pd.RangeIndex(len(df))

    if 'roa' in df.columns:
        roa_pos = (_num(df['roa']) > 0)
    else:
        roa_pos = (_num(df.get('net_income')) / _num(df.get('total_assets')) > 0)

    ocf_pos = (_num(df.get('operating_cf')) > 0)
    if 'fcf' in df.columns:
        fcf_pos = (_num(df['fcf']) > 0)
    else:
        fcf_pos = ((_num(df.get('operating_cf')) - _num(df.get('capex'))) > 0)

    components = pd.DataFrame({
        'symbol': idx,
        'check_roa_positive': roa_pos.fillna(False),
        'check_ocf_positive': ocf_pos.fillna(False),
        'check_fcf_positive': fcf_pos.fillna(False),
    })

    raw = components[['check_roa_positive','check_ocf_positive','check_fcf_positive']].sum(axis=1).astype(int)
    components['fscore'] = (raw * 9.0 / 3.0).round(2)
    return components

# Alias de compatibilidad (Forma B)
def get_fscore_components(df: pd.DataFrame) -> pd.DataFrame:
    return get_fscore_components_no_roe(df)

# =================================================================
#  FILTRO POR F-SCORE
# =================================================================
def filter_by_fscore(
    df: pd.DataFrame,
    min_score: int = 6,
    use_simplified: bool = True,
) -> pd.DataFrame:
    """
    Filtra empresas por F-Score.
      - use_simplified=True: usa el simplificado SIN ROE (recomendado si no hay históricos).
      - use_simplified=False: usa el F-Score académico (requiere columnas *_prev).
    """
    df = df.copy()

    if use_simplified:
        df['fscore'] = calculate_simplified_fscore_no_roe(df)
        # Con 3 checks → posibles 0, 3, 6, 9. Umbral 6 = 2/3 checks.
        min_cut = max(min_score, 6)
    else:
        df['fscore'] = calculate_piotroski_fscore(df)
        min_cut = min_score

    passed = df[df['fscore'] >= min_cut].copy()
    print(f"✅ F-Score Filter ({min_cut}+): {len(passed)}/{len(df)} stocks pass")
    return passed

# =================================================================
#  ESTADÍSTICAS DE F-SCORE
# =================================================================
def fscore_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera estadísticas de F-Score por rango (no muta el DF original).
    """
    if df is None or df.empty or 'fscore' not in df.columns:
        return pd.DataFrame()

    tmp = df[['symbol','fscore']].copy()
    tmp['fscore_category'] = pd.cut(
        tmp['fscore'],
        bins=[-0.1, 3, 5, 7, 9.1],
        labels=['Low (0-3)', 'Medium-Low (4-5)', 'Medium-High (6-7)', 'High (8-9)']
    )

    # Adjuntar métricas si existen
    for col in ['roe','roic','fcf']:
        if col in df.columns:
            tmp[col] = df[col]

    stats = tmp.groupby('fscore_category', observed=True).agg({
        'symbol': 'count',
        **({ 'roe':'mean' }  if 'roe'  in tmp.columns else {}),
        **({ 'roic':'mean' } if 'roic' in tmp.columns else {}),
        **({ 'fcf':'mean' }  if 'fcf'  in tmp.columns else {}),
    }).round(3)

    # Renombrar si existen
    new_cols = []
    for c in stats.columns:
        if c == 'symbol':
            new_cols.append('Count')
        elif c == 'roe':
            new_cols.append('Avg ROE')
        elif c == 'roic':
            new_cols.append('Avg ROIC')
        elif c == 'fcf':
            new_cols.append('Avg FCF')
        else:
            new_cols.append(c)
    stats.columns = new_cols
    return stats

# =================================================================
#  TESTS BÁSICOS
# =================================================================
if __name__ == "__main__":
    print("🧪 Testing piotroski_fscore (Forma B: simplificado sin ROE)...")

    df = pd.DataFrame({
        'symbol': ['HIGH_QUALITY', 'MEDIUM', 'LOW_QUALITY'],
        'roa': [0.15, 0.08, -0.05],
        'roe': [0.25, 0.12, -0.10],        # no se usa en la forma B
        'operating_cf': [1e9, 5e8, -1e8],
        'fcf': [8e8, 3e8, -2e8],
        'gross_margin': [0.45, 0.30, 0.15],
        'roic': [0.20, 0.10, -0.05],
    })

    # Simplificado sin ROE
    df['fscore'] = calculate_simplified_fscore_no_roe(df)

    print("\n📊 F-Score (simplificado sin ROE):")
    print(df[['symbol', 'fscore', 'operating_cf', 'fcf']])

    # Filtro (umbral 6 → 2/3 checks)
    print("\n🔍 Filtering by F-Score >= 6 (simplificado sin ROE):")
    passed = filter_by_fscore(df, min_score=6, use_simplified=True)
    print(passed[['symbol', 'fscore']])

    # Componentes (alias funciona)
    print("\n🔧 F-Score Components (no ROE):")
    components = get_fscore_components(df)
    print(components)

    print("\n✅ Tests complete!")
    print("\n💡 Interpretación:")
    print("  - F-Score 8-9: BUY (high quality)")
    print("  - F-Score 6-7: CONSIDER (medium quality)")
    print("  - F-Score 0-5: AVOID (low quality)")
