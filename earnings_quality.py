"""
Earnings Quality Analysis - Evitar Value Traps
===============================================

Basado en:
- Sloan (1996): "Do Stock Prices Fully Reflect Information in Accruals?"
- Beneish (1999): "M-Score" para detectar earnings manipulation
- Abarbanell & Bushee (1998): "Fundamental Analysis"

Objetivo: Filtrar empresas con earnings de baja calidad que parecen baratas
pero son value traps (contabilidad agresiva, manipulación, etc.)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


# ============================================================================
# ACCRUALS QUALITY (Sloan 1996)
# ============================================================================

def calculate_accruals_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Accruals Ratio = (Net Income - Operating Cash Flow) / Total Assets

    Paper: Sloan (1996)
    Hallazgo: High accruals → poor future returns (accruals mean-revert)

    Interpretación:
    - Ratio < 5%: Earnings de alta calidad (cash-backed)
    - Ratio 5-10%: Aceptable
    - Ratio > 10%: Red flag (earnings inflados, posible manipulación)

    Args:
        df: DataFrame con columnas netIncome, operatingCashFlow, totalAssets

    Returns:
        Series con accruals_ratio
    """
    net_income = pd.to_numeric(df.get('netIncome', 0), errors='coerce')
    operating_cf = pd.to_numeric(df.get('operatingCashFlow', 0), errors='coerce')
    total_assets = pd.to_numeric(df.get('totalAssets', 1), errors='coerce')

    # Evitar división por cero
    total_assets = total_assets.replace(0, np.nan)

    accruals = net_income - operating_cf
    accruals_ratio = abs(accruals) / total_assets

    return accruals_ratio


def filter_low_accruals(df: pd.DataFrame, max_accruals: float = 0.10) -> pd.DataFrame:
    """
    Filtrar solo empresas con accruals bajos (alta calidad earnings).

    Default: accruals < 10% (conservador)

    Paper: Sloan (1996) muestra que high accruals portfolio underperforms
    by ~5-10% annually.
    """
    df = df.copy()
    df['accruals_ratio'] = calculate_accruals_ratio(df)

    return df[df['accruals_ratio'] <= max_accruals]


# ============================================================================
# WORKING CAPITAL QUALITY
# ============================================================================

def calculate_days_sales_outstanding(df: pd.DataFrame) -> pd.Series:
    """
    DSO = (Accounts Receivable / Revenue) * 365

    Mide cuántos días tarda la empresa en cobrar sus ventas.

    Interpretación:
    - DSO bajo y estable: Buena calidad de ventas, cobra rápido
    - DSO alto o creciente: Red flag (pueden estar "inflando" ventas
      con crédito laxo, o tienen clientes que no pagan)

    Requiere datos históricos para detectar tendencia.
    """
    accounts_receivable = pd.to_numeric(df.get('accountsReceivable', 0), errors='coerce')
    revenue = pd.to_numeric(df.get('revenue', 1), errors='coerce')

    # Evitar división por cero
    revenue = revenue.replace(0, np.nan)

    dso = (accounts_receivable / revenue) * 365

    return dso


def calculate_inventory_days(df: pd.DataFrame) -> pd.Series:
    """
    Inventory Days = (Inventory / COGS) * 365

    Mide cuántos días de inventario mantiene la empresa.

    Interpretación:
    - Inventory days bajo y estable: Buena rotación
    - Inventory days creciente: Red flag (inventario obsoleto,
      problemas de demanda, posible writedown futuro)
    """
    inventory = pd.to_numeric(df.get('inventory', 0), errors='coerce')
    cogs = pd.to_numeric(df.get('costOfRevenue', 1), errors='coerce')

    # Evitar división por cero
    cogs = cogs.replace(0, np.nan)

    inventory_days = (inventory / cogs) * 365

    return inventory_days


def calculate_working_capital_quality(df: pd.DataFrame) -> pd.Series:
    """
    Composite score de calidad de working capital.

    Combina:
    - Accruals ratio (lower mejor)
    - DSO (lower mejor, >90 días es red flag)
    - Inventory days (depende de industria, pero crecimiento es red flag)

    Returns:
        Score 0-1 (1 = máxima calidad)
    """
    # Normalizar accruals (0 = mejor, 0.20+ = pésimo)
    accruals = calculate_accruals_ratio(df)
    accruals_score = 1 - np.clip(accruals / 0.20, 0, 1)

    # Normalizar DSO (30 días = excelente, 90+ días = pésimo)
    dso = calculate_days_sales_outstanding(df)
    dso_score = 1 - np.clip((dso - 30) / 60, 0, 1)

    # Inventory days (depende de industria, más complejo)
    # Por simplicidad, penalizamos inventory > 120 días
    inv_days = calculate_inventory_days(df)
    inv_score = 1 - np.clip((inv_days - 60) / 120, 0, 1)

    # Composite (equal weight)
    composite = (accruals_score + dso_score + inv_score) / 3

    return composite


# ============================================================================
# BENEISH M-SCORE (Detectar Manipulación)
# ============================================================================

def calculate_beneish_m_score(df: pd.DataFrame, df_prev: pd.DataFrame) -> pd.Series:
    """
    Beneish M-Score para detectar earnings manipulation.

    Paper: Beneish (1999) - "Detection of Earnings Manipulation"

    M-Score > -2.22: Probable manipulator (red flag)
    M-Score < -2.22: Unlikely manipulator

    Requiere datos del año anterior (df_prev).

    Nota: Implementación simplificada. Full M-Score requiere 8 variables.
    Aquí usamos las 3 más importantes:
    1. DSRI (Days Sales Receivable Index)
    2. GMI (Gross Margin Index)
    3. TATA (Total Accruals to Total Assets)
    """
    # DSRI = (AR_t / Sales_t) / (AR_t-1 / Sales_t-1)
    # Si DSRI > 1.2 → Receivables creciendo más rápido que sales (red flag)
    ar_t = pd.to_numeric(df.get('accountsReceivable', 1), errors='coerce')
    sales_t = pd.to_numeric(df.get('revenue', 1), errors='coerce')
    ar_prev = pd.to_numeric(df_prev.get('accountsReceivable', 1), errors='coerce')
    sales_prev = pd.to_numeric(df_prev.get('revenue', 1), errors='coerce')

    dsri = (ar_t / sales_t) / (ar_prev / sales_prev + 1e-9)

    # GMI = (Gross Margin t-1) / (Gross Margin t)
    # Si GMI > 1.2 → Gross margin deteriorating (red flag)
    gm_t = (sales_t - pd.to_numeric(df.get('costOfRevenue', 0), errors='coerce')) / sales_t
    gm_prev = (sales_prev - pd.to_numeric(df_prev.get('costOfRevenue', 0), errors='coerce')) / sales_prev
    gmi = gm_prev / (gm_t + 1e-9)

    # TATA = Total Accruals / Total Assets (similar a Sloan)
    tata = calculate_accruals_ratio(df)

    # M-Score simplificado (solo 3 variables)
    # Coeficientes aproximados del paper original
    m_score = -4.84 + 0.92*dsri + 0.58*gmi + 4.68*tata

    return m_score


def filter_beneish_safe(df: pd.DataFrame, df_prev: pd.DataFrame, threshold: float = -2.22) -> pd.DataFrame:
    """
    Filtrar empresas que NO parecen manipular earnings según Beneish M-Score.

    Threshold = -2.22 (default del paper)
    """
    df = df.copy()
    df['m_score'] = calculate_beneish_m_score(df, df_prev)

    return df[df['m_score'] < threshold]


# ============================================================================
# COMPOSITE EARNINGS QUALITY SCORE
# ============================================================================

def calculate_earnings_quality_score(
    df: pd.DataFrame,
    df_prev: Optional[pd.DataFrame] = None,
    use_beneish: bool = True,
) -> pd.Series:
    """
    Score compuesto de calidad de earnings (0-100).

    Combina:
    1. Accruals quality (Sloan)
    2. Working capital quality (DSO, Inventory)
    3. Beneish M-Score (si df_prev disponible)

    Returns:
        Score 0-100 (100 = máxima calidad, 0 = pésima calidad)
    """
    # 1. Accruals
    accruals = calculate_accruals_ratio(df)
    accruals_score = 100 * (1 - np.clip(accruals / 0.20, 0, 1))

    # 2. Working capital
    wc_quality = calculate_working_capital_quality(df)
    wc_score = 100 * wc_quality

    # 3. Beneish (si disponible)
    if use_beneish and df_prev is not None:
        m_score = calculate_beneish_m_score(df, df_prev)
        # Normalizar: m_score < -3 = 100, m_score > -1 = 0
        beneish_score = 100 * (1 - np.clip((m_score + 3) / 2, 0, 1))

        # Composite con Beneish
        composite = (0.40 * accruals_score + 0.30 * wc_score + 0.30 * beneish_score)
    else:
        # Sin Beneish
        composite = (0.60 * accruals_score + 0.40 * wc_score)

    return composite


def add_earnings_quality_metrics(df: pd.DataFrame, df_prev: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Agrega todas las métricas de earnings quality al DataFrame.

    Métricas agregadas:
    - accruals_ratio
    - dso (Days Sales Outstanding)
    - inventory_days
    - working_capital_quality
    - m_score (si df_prev disponible)
    - earnings_quality_score (0-100)

    Args:
        df: DataFrame actual con fundamentals
        df_prev: DataFrame del año anterior (opcional, para Beneish)

    Returns:
        DataFrame con nuevas columnas
    """
    df = df.copy()

    # Métricas individuales
    df['accruals_ratio'] = calculate_accruals_ratio(df)
    df['dso'] = calculate_days_sales_outstanding(df)
    df['inventory_days'] = calculate_inventory_days(df)
    df['working_capital_quality'] = calculate_working_capital_quality(df)

    # M-Score si tenemos data histórica
    if df_prev is not None and not df_prev.empty:
        df['m_score'] = calculate_beneish_m_score(df, df_prev)
    else:
        df['m_score'] = np.nan

    # Score compuesto
    df['earnings_quality_score'] = calculate_earnings_quality_score(df, df_prev)

    return df


# ============================================================================
# FILTROS RECOMENDADOS
# ============================================================================

def apply_earnings_quality_filters(
    df: pd.DataFrame,
    min_eq_score: float = 50.0,
    max_accruals: float = 0.10,
    max_dso: float = 90.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Aplica filtros conservadores de earnings quality.

    Defaults (conservadores):
    - Earnings Quality Score >= 50
    - Accruals <= 10%
    - DSO <= 90 días

    Paper: Sloan (1996) muestra que filtrar high accruals mejora returns 5-10%/año
    """
    df_filtered = df.copy()
    initial_count = len(df_filtered)

    # Filtro 1: Earnings Quality Score
    if 'earnings_quality_score' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['earnings_quality_score'] >= min_eq_score]
        if verbose:
            removed = initial_count - len(df_filtered)
            print(f"   Filtro EQ Score >= {min_eq_score}: -{removed} stocks")

    # Filtro 2: Accruals
    if 'accruals_ratio' in df_filtered.columns:
        count_before = len(df_filtered)
        df_filtered = df_filtered[df_filtered['accruals_ratio'] <= max_accruals]
        if verbose:
            removed = count_before - len(df_filtered)
            print(f"   Filtro Accruals <= {max_accruals:.1%}: -{removed} stocks")

    # Filtro 3: DSO
    if 'dso' in df_filtered.columns:
        count_before = len(df_filtered)
        df_filtered = df_filtered[df_filtered['dso'] <= max_dso]
        if verbose:
            removed = count_before - len(df_filtered)
            print(f"   Filtro DSO <= {max_dso} días: -{removed} stocks")

    if verbose:
        final_count = len(df_filtered)
        total_removed = initial_count - final_count
        pct_removed = 100 * total_removed / initial_count if initial_count > 0 else 0
        print(f"   Total filtrado: -{total_removed} stocks ({pct_removed:.1f}%)")

    return df_filtered
