"""
Backtest Engine - Simple y limpio
==================================

Backtest básico de estrategia long-only con rebalanceo mensual.
Sin complejidad innecesaria.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


# ============================================================================
# HELPERS
# ============================================================================

def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y valida DataFrame de precios"""
    df = df.copy()
    
    # Asegurar index datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index(pd.to_datetime(df["date"]))
        else:
            df.index = pd.to_datetime(df.index)
    
    # Asegurar columna 'close'
    if "close" not in df.columns:
        for col in ["Close", "adjClose", "adj_close"]:
            if col in df.columns:
                df["close"] = df[col]
                break
    
    if "close" not in df.columns:
        raise ValueError("No 'close' column found")
    
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    
    # Limpiar
    df = df[df["close"] > 0].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    
    return df[["close"]]


# ============================================================================
# MÉTRICAS
# ============================================================================

def calculate_cagr(returns: pd.Series) -> float:
    """CAGR anualizado"""
    if returns.empty:
        return 0.0
    
    equity = (1 + returns).cumprod()
    years = len(returns) / 12  # Asumiendo monthly returns
    
    if years <= 0 or equity.iloc[-1] <= 0:
        return 0.0
    
    return float((equity.iloc[-1] ** (1/years)) - 1)


def calculate_sharpe(returns: pd.Series) -> float:
    """Sharpe ratio anualizado"""
    if returns.empty:
        return 0.0
    
    mu = returns.mean() * 12  # Anualizar
    sd = returns.std() * np.sqrt(12)
    
    return float(mu / sd) if sd > 0 else 0.0


def calculate_max_dd(equity: pd.Series) -> float:
    """Maximum drawdown"""
    if equity.empty:
        return 0.0
    
    peak = equity.cummax()
    dd = (equity - peak) / peak
    
    return float(dd.min())


# ============================================================================
# BACKTEST CORE
# ============================================================================

@dataclass
class BacktestResult:
    """Resultado de backtest"""
    symbol: str
    cagr: float
    sharpe: float
    max_dd: float
    n_periods: int
    equity_curve: pd.Series


def backtest_single_symbol(
    prices: pd.DataFrame,
    symbol: str,
    rebalance_freq: str = "M",  # Monthly
) -> BacktestResult:
    """
    Backtest simple: compra y mantiene con rebalanceo.
    
    Args:
        prices: DataFrame con columna 'close' e index datetime
        symbol: Nombre del símbolo
        rebalance_freq: Frecuencia de rebalanceo ('M' = mensual)
    
    Returns:
        BacktestResult con métricas
    """
    try:
        prices = clean_prices(prices)
    except Exception:
        return BacktestResult(symbol, 0, 0, 0, 0, pd.Series(dtype=float))
    
    if len(prices) < 20:  # Mínimo de datos
        return BacktestResult(symbol, 0, 0, 0, 0, pd.Series(dtype=float))
    
    # Resample a frecuencia de rebalanceo (último precio de cada período)
    resampled = prices.resample(rebalance_freq).last().dropna()
    
    if len(resampled) < 2:
        return BacktestResult(symbol, 0, 0, 0, 0, pd.Series(dtype=float))
    
    # Calcular returns
    returns = resampled["close"].pct_change().dropna()
    
    # Equity curve
    equity = (1 + returns).cumprod()
    equity = pd.concat([pd.Series([1.0], index=[returns.index[0]]), equity])
    
    # Métricas
    cagr = calculate_cagr(returns)
    sharpe = calculate_sharpe(returns)
    max_dd = calculate_max_dd(equity)
    
    return BacktestResult(
        symbol=symbol,
        cagr=cagr,
        sharpe=sharpe,
        max_dd=max_dd,
        n_periods=len(returns),
        equity_curve=equity,
    )


def backtest_portfolio(
    prices_dict: Dict[str, pd.DataFrame],
    symbols: list[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Backtest de múltiples símbolos.
    
    Args:
        prices_dict: Dict {symbol: DataFrame de precios}
        symbols: Lista de símbolos a testear (None = todos)
    
    Returns:
        (metrics_df, equity_curves)
        
        metrics_df: DataFrame con métricas por símbolo
        equity_curves: Dict {symbol: equity_curve}
    """
    if symbols is None:
        symbols = list(prices_dict.keys())
    
    results = []
    equity_curves = {}
    
    for symbol in symbols:
        if symbol not in prices_dict:
            continue
        
        result = backtest_single_symbol(prices_dict[symbol], symbol)
        
        if result.n_periods > 0:
            results.append({
                "symbol": symbol,
                "CAGR": result.cagr,
                "Sharpe": result.sharpe,
                "MaxDD": result.max_dd,
                "Periods": result.n_periods,
            })
            equity_curves[symbol] = result.equity_curve
    
    metrics = pd.DataFrame(results)
    
    if not metrics.empty:
        metrics = metrics.sort_values("Sharpe", ascending=False)
    
    return metrics, equity_curves


# ============================================================================
# PORTFOLIO METRICS
# ============================================================================

def calculate_portfolio_metrics(equity_curves: Dict[str, pd.Series]) -> Dict:
    """
    Calcula métricas de portfolio (equal-weight).
    
    Args:
        equity_curves: Dict {symbol: equity_series}
    
    Returns:
        Dict con métricas agregadas
    """
    if not equity_curves:
        return {"CAGR": 0, "Sharpe": 0, "MaxDD": 0}
    
    # Combine equity curves (equal weight)
    df = pd.DataFrame(equity_curves)
    
    # Portfolio equity (promedio)
    portfolio_equity = df.mean(axis=1)
    
    # Returns
    returns = portfolio_equity.pct_change().dropna()
    
    return {
        "CAGR": calculate_cagr(returns),
        "Sharpe": calculate_sharpe(returns),
        "MaxDD": calculate_max_dd(portfolio_equity),
        "N_Symbols": len(equity_curves),
    }


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing backtest_engine...")
    
    # Mock price data
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    
    # Símbolo que sube
    df_up = pd.DataFrame({
        "date": dates,
        "close": np.linspace(100, 200, len(dates)),
    })
    
    # Símbolo que baja
    df_down = pd.DataFrame({
        "date": dates,
        "close": np.linspace(100, 50, len(dates)),
    })
    
    prices_dict = {
        "UP": df_up,
        "DOWN": df_down,
    }
    
    # Backtest individual
    result = backtest_single_symbol(df_up, "UP")
    print(f"\n✅ UP: CAGR={result.cagr:.2%}, Sharpe={result.sharpe:.2f}")
    
    # Backtest portfolio
    metrics, curves = backtest_portfolio(prices_dict)
    print(f"\n📊 Portfolio metrics:")
    print(metrics)
    
    # Portfolio agregado
    port_metrics = calculate_portfolio_metrics(curves)
    print(f"\n💼 Portfolio (equal-weight):")
    print(port_metrics)
