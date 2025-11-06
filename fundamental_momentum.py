"""
Fundamental Momentum - Detectar Tendencias en Fundamentales
============================================================

Basado en:
- Novy-Marx (2013): "Gross Profitability Premium"
- Piotroski & So (2012): "Identifying Expectation Errors in Value/Glamour Strategies"
- Mohanram (2005): "Separating Winners from Losers"

Objetivo: Detectar empresas con TENDENCIAS positivas sostenidas en
fundamentales (no solo mejora year-over-year).

Diferencia vs Piotroski:
- Piotroski: ¿Mejoró vs año anterior? (1 periodo)
- Fundamental Momentum: ¿Mejora consistente multi-year? (tendencia)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


# ============================================================================
# TENDENCIAS EN REVENUE & PROFITABILITY
# ============================================================================

def calculate_revenue_growth_trend(financials_history: List[pd.Series]) -> float:
    """
    Calcula tendencia de revenue growth (acelerando o desacelerando).

    Args:
        financials_history: Lista de Series con financials (ordenados cronológicamente)
                           Debe incluir 'revenue' como key

    Returns:
        Pendiente de revenue growth (positivo = acelerando)
    """
    if len(financials_history) < 3:
        return np.nan

    try:
        revenues = [fs.get('revenue', np.nan) for fs in financials_history]
        revenues = [pd.to_numeric(r, errors='coerce') for r in revenues]

        # Filtrar NaN
        revenues = [r for r in revenues if not pd.isna(r) and r > 0]

        if len(revenues) < 3:
            return np.nan

        # Ajustar regresión lineal: y = mx + b
        # m > 0 → revenue creciendo
        # m < 0 → revenue decreciendo
        x = np.arange(len(revenues))
        y = np.array(revenues)

        slope = np.polyfit(x, y, deg=1)[0]

        # Normalizar por último revenue (slope relativo)
        slope_pct = slope / revenues[-1] if revenues[-1] > 0 else 0

        return float(slope_pct)

    except Exception:
        return np.nan


def calculate_margin_trend(
    financials_history: List[pd.Series],
    margin_type: str = 'gross',
) -> float:
    """
    Calcula tendencia de márgenes (gross, operating, net).

    Args:
        financials_history: Historia de financials
        margin_type: 'gross', 'operating', o 'net'

    Returns:
        Pendiente de margin (positivo = mejorando)
    """
    if len(financials_history) < 3:
        return np.nan

    try:
        margins = []

        for fs in financials_history:
            revenue = pd.to_numeric(fs.get('revenue', 0), errors='coerce')

            if margin_type == 'gross':
                cost = pd.to_numeric(fs.get('costOfRevenue', 0), errors='coerce')
                margin = (revenue - cost) / revenue if revenue > 0 else np.nan

            elif margin_type == 'operating':
                operating_income = pd.to_numeric(fs.get('operatingIncome', 0), errors='coerce')
                margin = operating_income / revenue if revenue > 0 else np.nan

            elif margin_type == 'net':
                net_income = pd.to_numeric(fs.get('netIncome', 0), errors='coerce')
                margin = net_income / revenue if revenue > 0 else np.nan

            else:
                margin = np.nan

            margins.append(margin)

        # Filtrar NaN
        margins = [m for m in margins if not pd.isna(m)]

        if len(margins) < 3:
            return np.nan

        # Ajustar tendencia
        x = np.arange(len(margins))
        y = np.array(margins)

        slope = np.polyfit(x, y, deg=1)[0]

        return float(slope)

    except Exception:
        return np.nan


def calculate_roe_trend(financials_history: List[pd.Series]) -> float:
    """
    Calcula tendencia de ROE (Return on Equity).

    Paper: Piotroski & So (2012) - ROE mejorando consistentemente
    es señal de calidad creciente.
    """
    if len(financials_history) < 3:
        return np.nan

    try:
        roes = []

        for fs in financials_history:
            net_income = pd.to_numeric(fs.get('netIncome', 0), errors='coerce')
            equity = pd.to_numeric(fs.get('totalStockholdersEquity', 1), errors='coerce')

            if equity > 0:
                roe = net_income / equity
                roes.append(roe)
            else:
                roes.append(np.nan)

        # Filtrar NaN
        roes = [r for r in roes if not pd.isna(r)]

        if len(roes) < 3:
            return np.nan

        # Ajustar tendencia
        x = np.arange(len(roes))
        y = np.array(roes)

        slope = np.polyfit(x, y, deg=1)[0]

        return float(slope)

    except Exception:
        return np.nan


# ============================================================================
# TENDENCIAS EN LEVERAGE & EFFICIENCY
# ============================================================================

def calculate_leverage_trend(financials_history: List[pd.Series]) -> float:
    """
    Calcula tendencia de leverage (Debt/Equity).

    Pendiente NEGATIVA = mejorando (reduciendo deuda)
    Pendiente POSITIVA = empeorando (aumentando deuda)
    """
    if len(financials_history) < 3:
        return np.nan

    try:
        leverages = []

        for fs in financials_history:
            total_debt = pd.to_numeric(fs.get('totalDebt', 0), errors='coerce')
            equity = pd.to_numeric(fs.get('totalStockholdersEquity', 1), errors='coerce')

            if equity > 0:
                leverage = total_debt / equity
                leverages.append(leverage)
            else:
                leverages.append(np.nan)

        # Filtrar NaN
        leverages = [lev for lev in leverages if not pd.isna(lev)]

        if len(leverages) < 3:
            return np.nan

        # Ajustar tendencia
        x = np.arange(len(leverages))
        y = np.array(leverages)

        slope = np.polyfit(x, y, deg=1)[0]

        # Invertir signo (negativo = bueno)
        return float(-slope)

    except Exception:
        return np.nan


def calculate_asset_turnover_trend(financials_history: List[pd.Series]) -> float:
    """
    Calcula tendencia de asset turnover (Revenue / Total Assets).

    Mide eficiencia en uso de activos.
    Pendiente positiva = mejorando eficiencia
    """
    if len(financials_history) < 3:
        return np.nan

    try:
        turnovers = []

        for fs in financials_history:
            revenue = pd.to_numeric(fs.get('revenue', 0), errors='coerce')
            total_assets = pd.to_numeric(fs.get('totalAssets', 1), errors='coerce')

            if total_assets > 0:
                turnover = revenue / total_assets
                turnovers.append(turnover)
            else:
                turnovers.append(np.nan)

        # Filtrar NaN
        turnovers = [t for t in turnovers if not pd.isna(t)]

        if len(turnovers) < 3:
            return np.nan

        # Ajustar tendencia
        x = np.arange(len(turnovers))
        y = np.array(turnovers)

        slope = np.polyfit(x, y, deg=1)[0]

        return float(slope)

    except Exception:
        return np.nan


# ============================================================================
# COMPOSITE FUNDAMENTAL MOMENTUM SCORE
# ============================================================================

def calculate_fundamental_momentum_score(
    financials_history: List[pd.Series],
    verbose: bool = False,
) -> Tuple[float, Dict]:
    """
    Score compuesto de fundamental momentum (0-100).

    Combina tendencias de:
    1. Revenue growth (25%)
    2. Gross margin (20%)
    3. Operating margin (20%)
    4. ROE (20%)
    5. Leverage reduction (10%)
    6. Asset turnover (5%)

    Args:
        financials_history: Lista de financials (3+ años recomendado)
        verbose: Si True, imprime breakdown

    Returns:
        (score, components_dict)
    """
    components = {}

    # 1. Revenue growth trend
    rev_trend = calculate_revenue_growth_trend(financials_history)
    components['revenue_trend'] = rev_trend
    rev_score = 50 + 50 * np.clip(rev_trend * 10, -1, 1) if not pd.isna(rev_trend) else 50

    # 2. Gross margin trend
    gm_trend = calculate_margin_trend(financials_history, 'gross')
    components['gross_margin_trend'] = gm_trend
    gm_score = 50 + 50 * np.clip(gm_trend * 100, -1, 1) if not pd.isna(gm_trend) else 50

    # 3. Operating margin trend
    om_trend = calculate_margin_trend(financials_history, 'operating')
    components['operating_margin_trend'] = om_trend
    om_score = 50 + 50 * np.clip(om_trend * 100, -1, 1) if not pd.isna(om_trend) else 50

    # 4. ROE trend
    roe_trend = calculate_roe_trend(financials_history)
    components['roe_trend'] = roe_trend
    roe_score = 50 + 50 * np.clip(roe_trend * 20, -1, 1) if not pd.isna(roe_trend) else 50

    # 5. Leverage trend (deleveraging)
    lev_trend = calculate_leverage_trend(financials_history)
    components['leverage_trend'] = lev_trend
    lev_score = 50 + 50 * np.clip(lev_trend * 5, -1, 1) if not pd.isna(lev_trend) else 50

    # 6. Asset turnover trend
    at_trend = calculate_asset_turnover_trend(financials_history)
    components['asset_turnover_trend'] = at_trend
    at_score = 50 + 50 * np.clip(at_trend * 10, -1, 1) if not pd.isna(at_trend) else 50

    # Composite score (weighted average)
    composite = (
        0.25 * rev_score +
        0.20 * gm_score +
        0.20 * om_score +
        0.20 * roe_score +
        0.10 * lev_score +
        0.05 * at_score
    )

    components['composite_score'] = composite

    if verbose:
        print(f"\n📊 Fundamental Momentum Breakdown:")
        print(f"   Revenue Growth Trend: {rev_trend:.4f} → Score: {rev_score:.1f}")
        print(f"   Gross Margin Trend:   {gm_trend:.4f} → Score: {gm_score:.1f}")
        print(f"   Operating Margin:     {om_trend:.4f} → Score: {om_score:.1f}")
        print(f"   ROE Trend:            {roe_trend:.4f} → Score: {roe_score:.1f}")
        print(f"   Leverage Trend:       {lev_trend:.4f} → Score: {lev_score:.1f}")
        print(f"   Asset Turnover:       {at_trend:.4f} → Score: {at_score:.1f}")
        print(f"   COMPOSITE:            {composite:.1f}/100")

    return float(composite), components


def filter_positive_fundamental_momentum(
    financials_history: List[pd.Series],
    min_score: float = 55.0,
) -> bool:
    """
    Filtrar empresas con fundamental momentum positivo.

    Args:
        financials_history: Historia de financials
        min_score: Score mínimo para pasar filtro (default 55 = ligeramente positivo)

    Returns:
        True si pasa filtro, False si no
    """
    score, _ = calculate_fundamental_momentum_score(financials_history)

    return score >= min_score


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def calculate_fundamental_momentum_batch(
    financials_dict: Dict[str, List[pd.Series]],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Calcula fundamental momentum para múltiples símbolos.

    Args:
        financials_dict: {symbol: [financials_history]}
        verbose: Print progress

    Returns:
        DataFrame con scores y componentes
    """
    results = []

    for symbol, history in financials_dict.items():
        if len(history) < 3:
            if verbose:
                print(f"⚠️  {symbol}: insuficiente historia (<3 años)")
            continue

        try:
            score, components = calculate_fundamental_momentum_score(history)

            results.append({
                'symbol': symbol,
                'fm_score': score,
                **components
            })

        except Exception as e:
            if verbose:
                print(f"⚠️  {symbol}: error calculando FM - {e}")
            continue

    df = pd.DataFrame(results)

    if verbose and not df.empty:
        print(f"\n✅ Fundamental Momentum calculado para {len(df)} símbolos")
        print(f"   Score promedio: {df['fm_score'].mean():.1f}")
        print(f"   Score mediano:  {df['fm_score'].median():.1f}")

    return df


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def add_fundamental_momentum_to_df(
    df: pd.DataFrame,
    financials_dict: Dict[str, List[pd.Series]],
) -> pd.DataFrame:
    """
    Agrega fundamental momentum score al DataFrame principal.

    Args:
        df: DataFrame con símbolos
        financials_dict: {symbol: [financials_history]}

    Returns:
        DataFrame con columna 'fm_score' agregada
    """
    df = df.copy()

    # Calcular FM batch
    fm_df = calculate_fundamental_momentum_batch(financials_dict)

    if fm_df.empty:
        df['fm_score'] = 50.0  # Neutral
        return df

    # Merge
    df = df.merge(fm_df[['symbol', 'fm_score']], on='symbol', how='left')
    df['fm_score'] = df['fm_score'].fillna(50.0)

    return df


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing fundamental_momentum...")

    # Mock data: empresa mejorando
    improving_company = [
        pd.Series({
            'revenue': 1000,
            'costOfRevenue': 600,
            'operatingIncome': 200,
            'netIncome': 150,
            'totalStockholdersEquity': 1000,
            'totalDebt': 500,
            'totalAssets': 2000,
        }),
        pd.Series({
            'revenue': 1200,
            'costOfRevenue': 680,  # Gross margin mejorando
            'operatingIncome': 260,
            'netIncome': 200,
            'totalStockholdersEquity': 1100,
            'totalDebt': 450,  # Deleveraging
            'totalAssets': 2100,
        }),
        pd.Series({
            'revenue': 1500,
            'costOfRevenue': 825,  # Gross margin sigue mejorando
            'operatingIncome': 350,
            'netIncome': 280,
            'totalStockholdersEquity': 1250,
            'totalDebt': 400,  # Continúa deleveraging
            'totalAssets': 2200,
        }),
    ]

    # Mock data: empresa deteriorándose
    declining_company = [
        pd.Series({
            'revenue': 1500,
            'costOfRevenue': 750,
            'operatingIncome': 400,
            'netIncome': 300,
            'totalStockholdersEquity': 1200,
            'totalDebt': 300,
            'totalAssets': 2200,
        }),
        pd.Series({
            'revenue': 1300,
            'costOfRevenue': 700,  # Margin comprimido
            'operatingIncome': 280,
            'netIncome': 200,
            'totalStockholdersEquity': 1100,
            'totalDebt': 400,  # Aumentando deuda
            'totalAssets': 2100,
        }),
        pd.Series({
            'revenue': 1100,
            'costOfRevenue': 650,  # Margin sigue comprimido
            'operatingIncome': 180,
            'netIncome': 120,
            'totalStockholdersEquity': 1000,
            'totalDebt': 500,  # Más deuda
            'totalAssets': 2000,
        }),
    ]

    print("\n📈 Empresa Mejorando:")
    score_imp, _ = calculate_fundamental_momentum_score(improving_company, verbose=True)

    print("\n📉 Empresa Deteriorándose:")
    score_dec, _ = calculate_fundamental_momentum_score(declining_company, verbose=True)

    print(f"\n✅ Tests complete!")
    print(f"   Empresa mejorando: {score_imp:.1f}/100 (esperado >55)")
    print(f"   Empresa deteriorándose: {score_dec:.1f}/100 (esperado <45)")
