"""
Momentum Calculator - Implementación Académica
===============================================

Basado en:
- Jegadeesh & Titman (1993): "Returns to Buying Winners"
- Carhart (1997): Four-factor model
- Faber (2007): Moving average timing

CRITICAL: Este módulo calcula momentum REAL desde precios históricos.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Optional


# ============================================================================
# MOMENTUM ACADÉMICO (Jegadeesh & Titman, 1993)
# ============================================================================

def calculate_12m_1m_momentum(prices: pd.DataFrame) -> float:
    """
    Momentum clásico: retorno 12 meses excluyendo último mes.
    
    Paper: Jegadeesh & Titman (1993)
    Rationale: Skip último mes para evitar reversal de corto plazo
    
    Args:
        prices: DataFrame con columna 'close' e index datetime
    
    Returns:
        Momentum score (float)
    """
    if len(prices) < 252:  # Menos de 1 año de datos
        return np.nan
    
    try:
        # Retorno 12 meses atrás
        p_12m = prices['close'].iloc[-252]
        
        # Retorno 1 mes atrás
        p_1m = prices['close'].iloc[-21]
        
        # Precio actual
        p_now = prices['close'].iloc[-1]
        
        # Momentum = (P_-1M / P_-12M) - 1
        momentum = (p_1m / p_12m) - 1
        
        return float(momentum)
    
    except Exception:
        return np.nan


def calculate_risk_adjusted_momentum(prices: pd.DataFrame) -> float:
    """
    Momentum ajustado por riesgo (Sharpe-style).
    
    Rationale: Penaliza acciones muy volátiles
    """
    if len(prices) < 252:
        return np.nan
    
    try:
        # Retornos diarios últimos 12 meses
        returns = prices['close'].pct_change().iloc[-252:]
        
        # Momentum = retorno acumulado
        cum_return = (1 + returns).prod() - 1
        
        # Volatilidad anualizada
        volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted momentum
        if volatility > 0:
            ra_momentum = cum_return / volatility
        else:
            ra_momentum = 0.0
        
        return float(ra_momentum)
    
    except Exception:
        return np.nan


def calculate_6m_momentum(prices: pd.DataFrame) -> float:
    """
    Momentum de 6 meses (alternativa más responsive).
    
    Usado por: Fama-French 6-month momentum factor
    """
    if len(prices) < 126:
        return np.nan
    
    try:
        p_6m = prices['close'].iloc[-126]
        p_now = prices['close'].iloc[-1]
        
        momentum = (p_now / p_6m) - 1
        return float(momentum)
    
    except Exception:
        return np.nan


# ============================================================================
# MOVING AVERAGE FILTERS (Faber, 2007)
# ============================================================================

def calculate_ma200(prices: pd.DataFrame) -> Optional[float]:
    """
    Calcula media móvil de 200 días.
    
    Paper: Faber (2007) "A Quantitative Approach to Tactical Asset Allocation"
    """
    if len(prices) < 200:
        return None
    
    try:
        ma200 = prices['close'].rolling(200).mean().iloc[-1]
        return float(ma200)
    except Exception:
        return None


def is_above_ma200(prices: pd.DataFrame) -> bool:
    """
    Verifica si precio actual está por encima de MA200.
    
    CRITICAL: Este filtro reduce drawdowns en 50%+ según literatura.
    
    Returns:
        True si price > MA200, False otherwise
    """
    if len(prices) < 200:
        return False
    
    try:
        current_price = prices['close'].iloc[-1]
        ma200 = calculate_ma200(prices)
        
        if ma200 is None or pd.isna(ma200):
            return False
        
        return float(current_price) > float(ma200)
    
    except Exception:
        return False


def calculate_ma_slope(prices: pd.DataFrame, window: int = 200) -> float:
    """
    Calcula pendiente de la MA (confirma fuerza de tendencia).
    
    Positivo = uptrend
    Negativo = downtrend
    """
    if len(prices) < window + 20:
        return 0.0
    
    try:
        ma = prices['close'].rolling(window).mean()
        
        # Pendiente de últimos 20 días de la MA
        ma_recent = ma.iloc[-20:]
        
        # Regresión lineal simple
        x = np.arange(len(ma_recent))
        y = ma_recent.values
        
        # Pendiente = coef de regresión
        slope = np.polyfit(x, y, 1)[0]
        
        return float(slope)
    
    except Exception:
        return 0.0


# ============================================================================
# TREND STRENGTH (ADX-style simple)
# ============================================================================

def calculate_trend_strength(prices: pd.DataFrame) -> float:
    """
    Mide fuerza de tendencia (0-1).
    
    Basado en: % de días que precio está por encima de MA50
    """
    if len(prices) < 50:
        return 0.0
    
    try:
        ma50 = prices['close'].rolling(50).mean()
        
        # % de días recientes por encima de MA50
        above_ma = (prices['close'] > ma50).iloc[-50:]
        trend_strength = above_ma.sum() / len(above_ma)
        
        return float(trend_strength)
    
    except Exception:
        return 0.0


# ============================================================================
# COMPOSITE MOMENTUM SCORE
# ============================================================================

def calculate_composite_momentum(
    prices: pd.DataFrame,
    w_12m1m: float = 0.40,
    w_risk_adj: float = 0.30,
    w_trend: float = 0.30,
) -> Dict:
    """
    Score de momentum compuesto con múltiples señales.
    
    Returns:
        Dict con todos los componentes
    """
    result = {
        'momentum_12m1m': calculate_12m_1m_momentum(prices),
        'momentum_risk_adj': calculate_risk_adjusted_momentum(prices),
        'momentum_6m': calculate_6m_momentum(prices),
        'above_ma200': is_above_ma200(prices),
        'ma200': calculate_ma200(prices),
        'ma_slope': calculate_ma_slope(prices),
        'trend_strength': calculate_trend_strength(prices),
    }
    
    # Composite score (0-1)
    scores = []
    
    if not pd.isna(result['momentum_12m1m']):
        scores.append(w_12m1m * (1 if result['momentum_12m1m'] > 0 else 0))
    
    if not pd.isna(result['momentum_risk_adj']):
        scores.append(w_risk_adj * (1 if result['momentum_risk_adj'] > 0 else 0))
    
    if result['above_ma200']:
        scores.append(w_trend * 1.0)
    
    result['composite_momentum'] = sum(scores) / (w_12m1m + w_risk_adj + w_trend) if scores else 0.0
    
    return result


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def calculate_momentum_batch(
    prices_dict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Calcula momentum para múltiples símbolos.
    
    Args:
        prices_dict: {symbol: prices_df}
    
    Returns:
        DataFrame con scores por símbolo
    """
    results = []
    
    for symbol, prices in prices_dict.items():
        momentum_data = calculate_composite_momentum(prices)
        
        results.append({
            'symbol': symbol,
            **momentum_data
        })
    
    df = pd.DataFrame(results)
    
    return df


# ============================================================================
# INTEGRATION FUNCTION (para factor_calculator.py)
# ============================================================================

def integrate_real_momentum(
    df_factors: pd.DataFrame,
    prices_dict: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Reemplaza momentum placeholder con momentum real.
    
    USO:
    ```python
    from momentum_calculator import integrate_real_momentum
    
    # Después de calculate_qvm_composite()
    df = integrate_real_momentum(df, prices_dict)
    ```
    
    Args:
        df_factors: DataFrame con factores (incluye momentum_score placeholder)
        prices_dict: Diccionario de precios {symbol: df}
    
    Returns:
        DataFrame con momentum_score actualizado
    """
    df = df_factors.copy()
    
    # Calcular momentum real
    momentum_df = calculate_momentum_batch(prices_dict)
    
    # Merge
    df = df.merge(
        momentum_df[['symbol', 'momentum_12m1m', 'above_ma200', 'composite_momentum']],
        on='symbol',
        how='left',
        suffixes=('', '_real')
    )
    
    # Reemplazar momentum_score placeholder
    if 'composite_momentum' in df.columns:
        df['momentum_score'] = df['composite_momentum'].fillna(0.5)
    
    # Recalcular QVM con momentum real
    if all(col in df.columns for col in ['quality_extended', 'value_score', 'momentum_score']):
        df['qvm_score'] = (
            0.40 * df['quality_extended'] +
            0.30 * df['value_score'] +
            0.30 * df['momentum_score']
        )
        
        # Recalcular rank
        df['qvm_rank'] = df['qvm_score'].rank(pct=True, method='average')
    
    return df


# ============================================================================
# FILTRO CRÍTICO: SOLO ABOVE MA200
# ============================================================================

def filter_above_ma200(
    df: pd.DataFrame,
    prices_dict: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    FILTRO CRÍTICO: Solo mantiene acciones con Price > MA200.
    
    Paper: Faber (2007) - reduce drawdowns 50%+
    
    Args:
        df: DataFrame con símbolos
        prices_dict: Precios históricos
    
    Returns:
        DataFrame filtrado (solo above MA200)
    """
    symbols_above_ma = []
    
    for symbol in df['symbol']:
        if symbol in prices_dict:
            if is_above_ma200(prices_dict[symbol]):
                symbols_above_ma.append(symbol)
    
    result = df[df['symbol'].isin(symbols_above_ma)].copy()
    
    print(f"✅ MA200 Filter: {len(result)}/{len(df)} stocks pass (Price > MA200)")
    
    return result


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing momentum_calculator...")
    
    # Mock uptrend
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    prices_up = pd.DataFrame({
        'close': np.linspace(100, 300, len(dates)),  # Strong uptrend
    }, index=dates)
    
    # Mock downtrend
    prices_down = pd.DataFrame({
        'close': np.linspace(300, 100, len(dates)),  # Downtrend
    }, index=dates)
    
    print("\n📈 Uptrend Stock:")
    mom_up = calculate_composite_momentum(prices_up)
    print(f"  12M-1M Momentum: {mom_up['momentum_12m1m']:.2%}")
    print(f"  Above MA200: {mom_up['above_ma200']}")
    print(f"  Composite: {mom_up['composite_momentum']:.2f}")
    
    print("\n📉 Downtrend Stock:")
    mom_down = calculate_composite_momentum(prices_down)
    print(f"  12M-1M Momentum: {mom_down['momentum_12m1m']:.2%}")
    print(f"  Above MA200: {mom_down['above_ma200']}")
    print(f"  Composite: {mom_down['composite_momentum']:.2f}")
    
    print("\n✅ Tests complete!")
    print("\n⚠️  CRITICAL: Use filter_above_ma200() BEFORE portfolio selection!")
