"""
Quality-Value Score - Sin Multicolinealidad
============================================

Score compuesto que combina:
1. Piotroski Score (0-9) - Captura calidad operacional completa
2. Value Score - Basado en múltiplos de valoración (EV/EBITDA, P/B, P/E)
3. Momentum Score (opcional) - Basado en retornos de precio

IMPORTANTE: NO mezclamos Piotroski con métricas crudas (ROA, ROIC, etc.)
porque Piotroski ya las incorpora → evitamos multicolinealidad.

Bibliografía:
- Piotroski (2000): "Value Investing: The Use of Historical Financial Statement Information"
- Asness, Frazzini & Pedersen (2019): "Quality Minus Junk"
- Fama & French (1992): "The Cross-Section of Expected Stock Returns"
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Literal


# ============================================================================
# HELPERS
# ============================================================================

def _safe_float(x):
    """Convierte a float de manera segura."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _normalize_score(series: pd.Series, lower_is_better: bool = False) -> pd.Series:
    """
    Normaliza una serie a rango [0, 1].
    - lower_is_better=False: valores altos → 1, valores bajos → 0
    - lower_is_better=True: valores bajos → 1, valores altos → 0
    """
    s = pd.to_numeric(series, errors='coerce')

    # Eliminar infinitos y valores extremos
    s = s.replace([np.inf, -np.inf], np.nan)

    if s.notna().sum() < 2:
        return pd.Series(0.5, index=s.index)

    # Winsorizar al 1% y 99% para remover outliers
    p1, p99 = s.quantile(0.01), s.quantile(0.99)
    s = s.clip(p1, p99)

    # Normalizar a [0, 1]
    min_val, max_val = s.min(), s.max()
    if max_val == min_val:
        return pd.Series(0.5, index=s.index)

    normalized = (s - min_val) / (max_val - min_val)

    if lower_is_better:
        normalized = 1 - normalized

    return normalized.fillna(0.5)


def _rank_score(series: pd.Series) -> pd.Series:
    """Convierte a rank percentil [0, 1]."""
    s = pd.to_numeric(series, errors='coerce')
    if s.notna().sum() == 0:
        return pd.Series(0.5, index=s.index)
    return s.rank(pct=True, method='average').fillna(0.5)


# ============================================================================
# QUALITY SCORE (basado en Piotroski)
# ============================================================================

def calculate_quality_score(df: pd.DataFrame) -> pd.Series:
    """
    Quality Score basado ÚNICAMENTE en Piotroski Score (0-9).

    Normaliza a [0, 1] donde:
    - 9 puntos → 1.0 (máxima calidad)
    - 0 puntos → 0.0 (mínima calidad)

    NO incluye ROA, ROIC, ROE, márgenes, etc. porque Piotroski ya los captura.
    """
    if 'piotroski_score' not in df.columns:
        print("⚠️  Warning: piotroski_score not found, using 0.5 as default")
        return pd.Series(0.5, index=df.index)

    piotroski = pd.to_numeric(df['piotroski_score'], errors='coerce')

    # Normalizar de 0-9 a 0-1
    quality_score = piotroski / 9.0

    return quality_score.fillna(0.5)


# ============================================================================
# VALUE SCORE (múltiplos de valoración)
# ============================================================================

def calculate_value_score(df: pd.DataFrame) -> pd.Series:
    """
    Value Score basado en múltiplos de valoración.

    Usa métricas INDEPENDIENTES de Piotroski:
    - EV/EBITDA (lower is better)
    - P/B (lower is better)
    - P/E (lower is better)

    Ponderación: 40% EV/EBITDA, 30% P/B, 30% P/E
    """
    scores = []
    weights = []

    # EV/EBITDA (más peso porque es más completo)
    if 'ev_ebitda' in df.columns:
        ev_ebitda = pd.to_numeric(df['ev_ebitda'], errors='coerce')
        # Filtrar valores negativos o extremos
        ev_ebitda = ev_ebitda.where((ev_ebitda > 0) & (ev_ebitda < 100), np.nan)
        if ev_ebitda.notna().sum() > 1:
            scores.append(_normalize_score(ev_ebitda, lower_is_better=True))
            weights.append(0.40)

    # P/B
    if 'pb' in df.columns:
        pb = pd.to_numeric(df['pb'], errors='coerce')
        pb = pb.where((pb > 0) & (pb < 50), np.nan)
        if pb.notna().sum() > 1:
            scores.append(_normalize_score(pb, lower_is_better=True))
            weights.append(0.30)

    # P/E
    if 'pe' in df.columns:
        pe = pd.to_numeric(df['pe'], errors='coerce')
        pe = pe.where((pe > 0) & (pe < 100), np.nan)
        if pe.notna().sum() > 1:
            scores.append(_normalize_score(pe, lower_is_better=True))
            weights.append(0.30)

    if not scores:
        print("⚠️  Warning: no valuation metrics found, using 0.5 as default")
        return pd.Series(0.5, index=df.index)

    # Normalizar pesos
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Combinar scores
    value_score = sum(s * w for s, w in zip(scores, weights))

    return value_score.fillna(0.5)


# ============================================================================
# MOMENTUM SCORE (opcional)
# ============================================================================

def calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
    """
    Momentum Score basado en retornos de precio.

    Si hay columna 'momentum_12m' o 'return_12m', la usa.
    Si no, devuelve 0.5 (neutral).
    """
    for col in ['momentum_12m', 'return_12m', 'momentum_score']:
        if col in df.columns:
            momentum = pd.to_numeric(df[col], errors='coerce')
            return _rank_score(momentum)

    # Si no hay momentum, usar neutral
    return pd.Series(0.5, index=df.index)


# ============================================================================
# FCF YIELD SCORE (adicional, independiente)
# ============================================================================

def calculate_fcf_yield_score(df: pd.DataFrame) -> pd.Series:
    """
    FCF Yield Score - Independiente de Piotroski.

    FCF Yield = FCF / Market Cap

    Aunque Piotroski usa FCF en checks binarios, el FCF Yield como
    métrica de valoración es independiente y complementaria.
    """
    if 'fcf_yield' in df.columns:
        fcf_yield = pd.to_numeric(df['fcf_yield'], errors='coerce')
        # Filtrar valores extremos negativos
        fcf_yield = fcf_yield.where(fcf_yield > -0.5, np.nan)
        if fcf_yield.notna().sum() > 1:
            return _normalize_score(fcf_yield, lower_is_better=False)

    # Si no hay fcf_yield pero hay fcf y market_cap, calcularlo
    if 'fcf' in df.columns and 'market_cap' in df.columns:
        fcf = pd.to_numeric(df['fcf'], errors='coerce')
        mcap = pd.to_numeric(df['market_cap'], errors='coerce')

        fcf_yield = fcf / mcap.replace(0, np.nan)
        fcf_yield = fcf_yield.where(fcf_yield > -0.5, np.nan)

        if fcf_yield.notna().sum() > 1:
            return _normalize_score(fcf_yield, lower_is_better=False)

    return pd.Series(0.5, index=df.index)


# ============================================================================
# QUALITY-VALUE SCORE COMPUESTO
# ============================================================================

def calculate_quality_value_score(
    df: pd.DataFrame,
    w_quality: float = 0.40,
    w_value: float = 0.35,
    w_fcf_yield: float = 0.15,
    w_momentum: float = 0.10,
    normalize_output: bool = True,
) -> pd.DataFrame:
    """
    Calcula Quality-Value Score compuesto SIN multicolinealidad.

    Componentes:
    1. Quality (40%): Piotroski Score normalizado
    2. Value (35%): Múltiplos de valoración (EV/EBITDA, P/B, P/E)
    3. FCF Yield (15%): Free Cash Flow Yield
    4. Momentum (10%): Retornos de precio

    Returns:
        DataFrame con columnas adicionales:
        - quality_score_component
        - value_score_component
        - fcf_yield_component
        - momentum_component
        - qv_score (score compuesto final)
        - qv_rank (rank percentil del score)
    """
    df = df.copy()

    # Calcular componentes
    df['quality_score_component'] = calculate_quality_score(df)
    df['value_score_component'] = calculate_value_score(df)
    df['fcf_yield_component'] = calculate_fcf_yield_score(df)
    df['momentum_component'] = calculate_momentum_score(df)

    # Normalizar pesos
    total_weight = w_quality + w_value + w_fcf_yield + w_momentum
    w_quality /= total_weight
    w_value /= total_weight
    w_fcf_yield /= total_weight
    w_momentum /= total_weight

    # Score compuesto
    df['qv_score'] = (
        w_quality * df['quality_score_component'] +
        w_value * df['value_score_component'] +
        w_fcf_yield * df['fcf_yield_component'] +
        w_momentum * df['momentum_component']
    )

    # Rank percentil del score final
    if normalize_output:
        df['qv_rank'] = _rank_score(df['qv_score'])
    else:
        df['qv_rank'] = df['qv_score']

    return df


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def compute_quality_value_factors(
    df_universe: pd.DataFrame,
    df_fundamentals: pd.DataFrame,
    w_quality: float = 0.40,
    w_value: float = 0.35,
    w_fcf_yield: float = 0.15,
    w_momentum: float = 0.10,
) -> pd.DataFrame:
    """
    Función principal que combina datos del universo y fundamentales,
    y calcula el Quality-Value Score.

    Args:
        df_universe: DataFrame con symbol, sector, market_cap, etc.
        df_fundamentals: DataFrame con piotroski_score, ev_ebitda, pb, pe, fcf, etc.
        w_quality: Peso del componente Quality (default 40%)
        w_value: Peso del componente Value (default 35%)
        w_fcf_yield: Peso del FCF Yield (default 15%)
        w_momentum: Peso del Momentum (default 10%)

    Returns:
        DataFrame ordenado por qv_score descendente
    """
    # Merge
    df = df_universe.merge(df_fundamentals, on='symbol', how='left')

    # Calcular score
    df = calculate_quality_value_score(
        df,
        w_quality=w_quality,
        w_value=w_value,
        w_fcf_yield=w_fcf_yield,
        w_momentum=w_momentum,
    )

    # Ordenar por score descendente
    df = df.sort_values('qv_score', ascending=False).reset_index(drop=True)

    return df


# ============================================================================
# ANÁLISIS Y ESTADÍSTICAS
# ============================================================================

def analyze_score_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analiza la distribución de cada componente del score.
    """
    if not all(col in df.columns for col in ['quality_score_component', 'value_score_component']):
        print("⚠️  Score components not found in dataframe")
        return pd.DataFrame()

    stats = df[['quality_score_component', 'value_score_component',
                'fcf_yield_component', 'momentum_component', 'qv_score']].describe()

    return stats


def top_quality_value_stocks(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    Retorna los top N stocks por Quality-Value Score.
    """
    if 'qv_score' not in df.columns:
        print("⚠️  qv_score not found in dataframe")
        return pd.DataFrame()

    cols_to_show = ['symbol', 'qv_score', 'qv_rank', 'piotroski_score',
                    'quality_score_component', 'value_score_component']

    # Incluir sector y market_cap si existen
    if 'sector' in df.columns:
        cols_to_show.insert(1, 'sector')
    if 'market_cap' in df.columns:
        cols_to_show.insert(2, 'market_cap')

    available_cols = [c for c in cols_to_show if c in df.columns]

    return df[available_cols].head(n)


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing quality_value_score...")

    # Datos de prueba
    test_df = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'XOM', 'T', 'F'],
        'sector': ['Technology', 'Technology', 'Energy', 'Telecom', 'Auto'],
        'market_cap': [3e12, 2.5e12, 4e11, 1.5e11, 5e10],

        # Piotroski (0-9)
        'piotroski_score': [8, 7, 5, 4, 3],

        # Valoración
        'ev_ebitda': [25, 22, 8, 6, 12],
        'pb': [45, 40, 1.5, 1.2, 0.8],
        'pe': [30, 28, 10, 8, 15],

        # FCF
        'fcf': [1e11, 8e10, 2e10, 5e9, -2e9],

        # Momentum (opcional)
        'momentum_12m': [0.25, 0.30, 0.10, -0.05, -0.15],
    })

    # Calcular FCF Yield manualmente
    test_df['fcf_yield'] = test_df['fcf'] / test_df['market_cap']

    print("\n📊 Input Data:")
    print(test_df[['symbol', 'piotroski_score', 'ev_ebitda', 'pb', 'pe', 'fcf_yield']])

    # Calcular Quality-Value Score
    result = calculate_quality_value_score(test_df)

    print("\n✅ Quality-Value Score Results:")
    print(result[['symbol', 'quality_score_component', 'value_score_component',
                  'fcf_yield_component', 'momentum_component', 'qv_score', 'qv_rank']])

    print("\n📈 Score Statistics:")
    print(analyze_score_components(result))

    print("\n🏆 Top Stocks by QV Score:")
    print(top_quality_value_stocks(result, n=5))

    print("\n✅ Test complete!")
    print("\n💡 Interpretación:")
    print("  - qv_score > 0.7: STRONG BUY (alta calidad + barato)")
    print("  - qv_score 0.5-0.7: BUY (buena combinación)")
    print("  - qv_score 0.3-0.5: HOLD (neutro)")
    print("  - qv_score < 0.3: AVOID (baja calidad o caro)")
