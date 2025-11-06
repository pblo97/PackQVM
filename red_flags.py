"""
Red Flags Detection - Evitar "Landmines"
=========================================

Filtros de sentido común para evitar empresas con problemas serios:
- Dilución excesiva de shares
- Cambios frecuentes de auditor
- Capitalización de gastos agresiva
- Deterioro de working capital
- Pérdidas recurrentes

Basado en:
- Experiencia práctica de value investors
- Casos históricos (Enron, WorldCom, etc.)
- Richardson et al. (2005): "Accrual Reliability"
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


# ============================================================================
# DILUTION DE SHARES
# ============================================================================

def calculate_share_dilution(financials_history: List[pd.Series]) -> float:
    """
    Calcula tasa de dilución de shares (% change año a año).

    Red flag: Dilución >10% anual (empresa emitiendo muchas acciones,
    probablemente para cubrir cash burn)

    Args:
        financials_history: Lista con al menos 2 años de datos
                           Debe contener 'weightedAverageShsOut'

    Returns:
        Dilución anual (0.10 = 10% más shares)
    """
    if len(financials_history) < 2:
        return np.nan

    try:
        # Último año
        shares_latest = pd.to_numeric(
            financials_history[-1].get('weightedAverageShsOut', np.nan),
            errors='coerce'
        )

        # Año anterior
        shares_prev = pd.to_numeric(
            financials_history[-2].get('weightedAverageShsOut', np.nan),
            errors='coerce'
        )

        if pd.isna(shares_latest) or pd.isna(shares_prev) or shares_prev == 0:
            return np.nan

        # % change
        dilution = (shares_latest - shares_prev) / shares_prev

        return float(dilution)

    except Exception:
        return np.nan


def filter_low_dilution(
    financials_history: List[pd.Series],
    max_dilution: float = 0.10,
) -> bool:
    """
    Filtrar empresas con dilución baja (<10% anual).

    Rationale: High dilution diluye shareholders, señal de cash burn.
    """
    dilution = calculate_share_dilution(financials_history)

    if pd.isna(dilution):
        return True  # No data, no filter

    return dilution <= max_dilution


# ============================================================================
# PERSISTENCIA DE PÉRDIDAS
# ============================================================================

def check_recurring_losses(financials_history: List[pd.Series]) -> bool:
    """
    Detecta pérdidas recurrentes (3+ años consecutivos).

    Red flag: Empresa que pierde dinero consistentemente probablemente
    tiene problemas estructurales.

    Returns:
        True si tiene pérdidas recurrentes (red flag)
    """
    if len(financials_history) < 3:
        return False

    try:
        # Últimos 3 años
        recent = financials_history[-3:]

        losses = 0
        for fs in recent:
            net_income = pd.to_numeric(fs.get('netIncome', 0), errors='coerce')
            if not pd.isna(net_income) and net_income < 0:
                losses += 1

        # Si 3 de 3 son pérdidas → red flag
        return losses >= 3

    except Exception:
        return False


def filter_profitable_companies(
    financials_history: List[pd.Series],
    min_profitable_years: int = 2,
) -> bool:
    """
    Filtrar solo empresas con profit en al menos N de últimos 3 años.

    Default: 2 de 3 años profitable
    """
    if len(financials_history) < 3:
        return True  # Insufficient data, no filter

    try:
        recent = financials_history[-3:]

        profitable_years = 0
        for fs in recent:
            net_income = pd.to_numeric(fs.get('netIncome', 0), errors='coerce')
            if not pd.isna(net_income) and net_income > 0:
                profitable_years += 1

        return profitable_years >= min_profitable_years

    except Exception:
        return True


# ============================================================================
# DETERIORO DE WORKING CAPITAL
# ============================================================================

def calculate_working_capital_trend(financials_history: List[pd.Series]) -> float:
    """
    Calcula tendencia de working capital / assets ratio.

    Red flag: WC/Assets decreciendo → problemas de liquidez

    Returns:
        Pendiente (negativo = deteriorándose)
    """
    if len(financials_history) < 3:
        return np.nan

    try:
        wc_ratios = []

        for fs in financials_history:
            current_assets = pd.to_numeric(fs.get('totalCurrentAssets', 0), errors='coerce')
            current_liabilities = pd.to_numeric(fs.get('totalCurrentLiabilities', 1), errors='coerce')
            total_assets = pd.to_numeric(fs.get('totalAssets', 1), errors='coerce')

            if total_assets > 0:
                wc = current_assets - current_liabilities
                wc_ratio = wc / total_assets
                wc_ratios.append(wc_ratio)
            else:
                wc_ratios.append(np.nan)

        # Filtrar NaN
        wc_ratios = [r for r in wc_ratios if not pd.isna(r)]

        if len(wc_ratios) < 3:
            return np.nan

        # Tendencia
        x = np.arange(len(wc_ratios))
        y = np.array(wc_ratios)

        slope = np.polyfit(x, y, deg=1)[0]

        return float(slope)

    except Exception:
        return np.nan


def filter_stable_working_capital(
    financials_history: List[pd.Series],
    min_trend: float = -0.05,
) -> bool:
    """
    Filtrar empresas con working capital estable o mejorando.

    Args:
        min_trend: Tendencia mínima aceptable (default -5% es tolerable)

    Returns:
        True si WC no está deteriorándose rápidamente
    """
    trend = calculate_working_capital_trend(financials_history)

    if pd.isna(trend):
        return True  # No data

    return trend >= min_trend


# ============================================================================
# CAPITALIZACIÓN DE GASTOS (Red Flag Contable)
# ============================================================================

def check_aggressive_capitalization(df: pd.DataFrame) -> pd.Series:
    """
    Detecta capitalización agresiva de gastos.

    Ratio CapEx / (CapEx + R&D + SG&A)

    - Ratio alto (>30%) puede indicar que están capitalizando gastos
      que deberían ser expensados (manipulación de earnings)

    Paper: Richardson et al. (2005) - "Accrual Reliability"

    Returns:
        Series booleana (True = aggressive capitalization, red flag)
    """
    capex = pd.to_numeric(df.get('capitalExpenditure', 0), errors='coerce').abs()
    rd = pd.to_numeric(df.get('researchAndDevelopmentExpenses', 0), errors='coerce').abs()
    sga = pd.to_numeric(df.get('sellingGeneralAndAdministrativeExpenses', 0), errors='coerce').abs()

    total_expenses = capex + rd + sga

    # Evitar división por cero
    total_expenses = total_expenses.replace(0, np.nan)

    capex_ratio = capex / total_expenses

    # Red flag si >30% está capitalizado
    return capex_ratio > 0.30


# ============================================================================
# COMPOSITE RED FLAGS SCORE
# ============================================================================

def calculate_red_flags_score(
    df: pd.DataFrame,
    financials_history: Optional[List[pd.Series]] = None,
) -> pd.Series:
    """
    Score compuesto de red flags (0-100).

    0 = Muchos red flags (EVITAR)
    100 = Sin red flags (SAFE)

    Combina:
    1. Share dilution
    2. Recurring losses
    3. Working capital deterioro
    4. Aggressive capitalization

    Args:
        df: DataFrame actual
        financials_history: Historia de financials (si disponible)
    """
    score = pd.Series(100.0, index=df.index)  # Start perfect

    # 1. Aggressive capitalization (-20 points)
    if all(col in df.columns for col in ['capitalExpenditure', 'researchAndDevelopmentExpenses']):
        aggressive_cap = check_aggressive_capitalization(df)
        score = score - 20 * aggressive_cap

    # 2. Dilution (si tenemos historia)
    if financials_history is not None and len(financials_history) >= 2:
        dilution = calculate_share_dilution(financials_history)
        if not pd.isna(dilution):
            if dilution > 0.15:  # >15% dilution
                score -= 30
            elif dilution > 0.10:  # >10% dilution
                score -= 20

        # 3. Recurring losses
        has_losses = check_recurring_losses(financials_history)
        if has_losses:
            score -= 25

        # 4. WC deterioro
        wc_trend = calculate_working_capital_trend(financials_history)
        if not pd.isna(wc_trend):
            if wc_trend < -0.10:  # Deterioro fuerte
                score -= 25
            elif wc_trend < -0.05:  # Deterioro moderado
                score -= 15

    return score.clip(0, 100)


def apply_red_flags_filters(
    df: pd.DataFrame,
    financials_history_dict: Optional[Dict[str, List[pd.Series]]] = None,
    min_score: float = 60.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Aplica filtros de red flags al DataFrame.

    Args:
        df: DataFrame con símbolos
        financials_history_dict: {symbol: [financials_history]}
        min_score: Score mínimo para pasar (default 60)
        verbose: Print diagnostics

    Returns:
        DataFrame filtrado
    """
    df_filtered = df.copy()
    initial_count = len(df_filtered)

    if verbose:
        print(f"\n🚩 Aplicando Red Flags Filters...")
        print(f"   Stocks iniciales: {initial_count}")

    # Calcular score por símbolo
    if financials_history_dict is not None:
        scores = []

        for symbol in df_filtered['symbol']:
            history = financials_history_dict.get(symbol, None)

            if history and len(history) > 0:
                # Usar último financials para aggressive cap check
                latest = history[-1]
                df_temp = pd.DataFrame([latest])
                score = calculate_red_flags_score(df_temp, history).iloc[0]
            else:
                score = 100.0  # No data, no penalizar

            scores.append(score)

        df_filtered['red_flags_score'] = scores

        # Filtrar
        df_filtered = df_filtered[df_filtered['red_flags_score'] >= min_score]

        if verbose:
            removed = initial_count - len(df_filtered)
            print(f"   Filtro Red Flags Score >= {min_score}: -{removed} stocks")

    else:
        # Sin historia, solo check aggressive capitalization
        if all(col in df_filtered.columns for col in ['capitalExpenditure', 'researchAndDevelopmentExpenses']):
            aggressive = check_aggressive_capitalization(df_filtered)
            df_filtered = df_filtered[~aggressive]

            if verbose:
                removed = initial_count - len(df_filtered)
                print(f"   Filtro Aggressive Capitalization: -{removed} stocks")

    if verbose:
        final_count = len(df_filtered)
        total_removed = initial_count - final_count
        pct_removed = 100 * total_removed / initial_count if initial_count > 0 else 0
        print(f"   Total filtrado: -{total_removed} stocks ({pct_removed:.1f}%)")

    return df_filtered


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def add_red_flags_metrics(
    df: pd.DataFrame,
    financials_history_dict: Dict[str, List[pd.Series]],
) -> pd.DataFrame:
    """
    Agrega métricas de red flags al DataFrame.

    Métricas agregadas:
    - share_dilution
    - has_recurring_losses
    - wc_trend
    - aggressive_capitalization
    - red_flags_score
    """
    df = df.copy()

    # Calcular métricas por símbolo
    dilutions = []
    losses = []
    wc_trends = []
    scores = []

    for symbol in df['symbol']:
        history = financials_history_dict.get(symbol, None)

        if history and len(history) >= 2:
            dilution = calculate_share_dilution(history)
            has_losses = check_recurring_losses(history)
            wc_trend = calculate_working_capital_trend(history)

            # Score
            latest = history[-1]
            df_temp = pd.DataFrame([latest])
            score = calculate_red_flags_score(df_temp, history).iloc[0]

        else:
            dilution = np.nan
            has_losses = False
            wc_trend = np.nan
            score = 100.0

        dilutions.append(dilution)
        losses.append(has_losses)
        wc_trends.append(wc_trend)
        scores.append(score)

    df['share_dilution'] = dilutions
    df['has_recurring_losses'] = losses
    df['wc_trend'] = wc_trends
    df['red_flags_score'] = scores

    # Aggressive capitalization
    if all(col in df.columns for col in ['capitalExpenditure', 'researchAndDevelopmentExpenses']):
        df['aggressive_capitalization'] = check_aggressive_capitalization(df)
    else:
        df['aggressive_capitalization'] = False

    return df


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing red_flags...")

    # Mock: Empresa con dilución excesiva
    dilution_company = [
        pd.Series({'weightedAverageShsOut': 100_000_000}),
        pd.Series({'weightedAverageShsOut': 125_000_000}),  # +25% dilution!
    ]

    # Mock: Empresa con pérdidas recurrentes
    losses_company = [
        pd.Series({'netIncome': -10_000_000}),
        pd.Series({'netIncome': -8_000_000}),
        pd.Series({'netIncome': -12_000_000}),
    ]

    # Mock: Empresa sana
    healthy_company = [
        pd.Series({'netIncome': 50_000_000, 'weightedAverageShsOut': 100_000_000}),
        pd.Series({'netIncome': 55_000_000, 'weightedAverageShsOut': 102_000_000}),  # +2% dilution (ok)
        pd.Series({'netIncome': 60_000_000, 'weightedAverageShsOut': 103_000_000}),
    ]

    print("\n🔴 Empresa con Dilución Excesiva:")
    dilution = calculate_share_dilution(dilution_company)
    print(f"   Dilución: {dilution:.1%} (red flag si >10%)")

    print("\n🔴 Empresa con Pérdidas Recurrentes:")
    has_losses = check_recurring_losses(losses_company)
    print(f"   Pérdidas recurrentes: {has_losses}")

    print("\n✅ Empresa Sana:")
    dilution_healthy = calculate_share_dilution(healthy_company)
    losses_healthy = check_recurring_losses(healthy_company)
    print(f"   Dilución: {dilution_healthy:.1%}")
    print(f"   Pérdidas recurrentes: {losses_healthy}")

    print("\n✅ Tests complete!")
