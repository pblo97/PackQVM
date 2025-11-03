"""
Backtest Engine V2 - Con transaction costs realistas
====================================================

MEJORAS:
1. Transaction costs (bps por trade)
2. Execution lag (1-2 días)
3. Turnover tracking
4. Sharpe más realista
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


# ============================================================================
# CONFIGURACIÓN DE COSTOS
# ============================================================================

@dataclass
class TradingCosts:
    """Costos de trading realistas"""
    commission_bps: int = 5        # Comisión (5 bps = $5 por $10k)
    slippage_bps: int = 5          # Slippage (5 bps típico)
    market_impact_bps: int = 2     # Impacto de mercado (small cap)
    
    @property
    def total_bps(self) -> int:
        """Total one-way cost"""
        return self.commission_bps + self.slippage_bps + self.market_impact_bps
    
    @property
    def round_trip_bps(self) -> int:
        """Round-trip cost (compra + venta)"""
        return self.total_bps * 2


# ============================================================================
# HELPERS
# ============================================================================

def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y valida DataFrame de precios"""
    df = df.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index(pd.to_datetime(df["date"]))
        else:
            df.index = pd.to_datetime(df.index)
    
    if "close" not in df.columns:
        for col in ["Close", "adjClose", "adj_close"]:
            if col in df.columns:
                df["close"] = df[col]
                break
    
    if "close" not in df.columns:
        raise ValueError("No 'close' column found")
    
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df[df["close"] > 0].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    
    return df[["close"]]


# ============================================================================
# MÉTRICAS
# ============================================================================

def calculate_cagr(returns: pd.Series, periods_per_year: int = 12) -> float:
    """CAGR anualizado"""
    if returns.empty:
        return 0.0
    
    equity = (1 + returns).cumprod()
    years = len(returns) / periods_per_year
    
    if years <= 0 or equity.iloc[-1] <= 0:
        return 0.0
    
    return float((equity.iloc[-1] ** (1/years)) - 1)


def calculate_sharpe(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Sharpe ratio anualizado"""
    if returns.empty:
        return 0.0
    
    mu = returns.mean() * periods_per_year
    sd = returns.std(ddof=1) * np.sqrt(periods_per_year)
    
    return float(mu / sd) if sd > 0 else 0.0


def calculate_sortino(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Sortino ratio (penaliza solo downside)"""
    if returns.empty:
        return 0.0
    
    mu = returns.mean() * periods_per_year
    downside = returns[returns < 0]
    
    if len(downside) < 2:
        return 0.0
    
    dd_std = downside.std(ddof=1) * np.sqrt(periods_per_year)
    
    return float(mu / dd_std) if dd_std > 0 else 0.0


def calculate_max_dd(equity: pd.Series) -> float:
    """Maximum drawdown"""
    if equity.empty:
        return 0.0
    
    peak = equity.cummax()
    dd = (equity - peak) / peak
    
    return float(dd.min())


def calculate_calmar(cagr: float, max_dd: float) -> float:
    """Calmar ratio = CAGR / |MaxDD|"""
    if max_dd >= 0:  # No drawdown o mal calculado
        return 0.0
    return float(cagr / abs(max_dd))


# ============================================================================
# BACKTEST CORE (V2 con costos)
# ============================================================================

@dataclass
class BacktestResult:
    """Resultado de backtest V2"""
    symbol: str
    cagr: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float
    turnover: float          # ← NUEVO
    avg_holding_days: float  # ← NUEVO
    n_trades: int            # ← NUEVO
    n_periods: int
    equity_curve: pd.Series
    returns: pd.Series       # ← NUEVO (para análisis)


def backtest_single_symbol(
    prices: pd.DataFrame,
    symbol: str,
    rebalance_freq: str = "M",
    costs: TradingCosts = None,
    execution_lag_days: int = 1,  # ← NUEVO
) -> BacktestResult:
    """
    Backtest V2 con transaction costs y execution lag.
    
    Args:
        prices: DataFrame con 'close' e index datetime
        symbol: Nombre del símbolo
        rebalance_freq: Frecuencia ('M' = mensual, 'W' = semanal)
        costs: Costos de trading (usa default si None)
        execution_lag_days: Días de lag en ejecución (1-2 realista)
    """
    if costs is None:
        costs = TradingCosts()
    
    try:
        prices = clean_prices(prices)
    except Exception:
        return BacktestResult(
            symbol, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            pd.Series(dtype=float), pd.Series(dtype=float)
        )
    
    if len(prices) < 20:
        return BacktestResult(
            symbol, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            pd.Series(dtype=float), pd.Series(dtype=float)
        )
    
    # Resample a frecuencia de rebalanceo
    resampled = prices.resample(rebalance_freq).last().dropna()
    
    if len(resampled) < 2:
        return BacktestResult(
            symbol, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            pd.Series(dtype=float), pd.Series(dtype=float)
        )
    
    # Returns GROSS
    returns_gross = resampled["close"].pct_change().dropna()
    
    # -------------------------
    # APLICAR COSTOS
    # -------------------------
    # Asumimos rebalanceo completo cada período (round-trip cost)
    cost_per_period = costs.round_trip_bps / 10000  # bps a decimal
    returns_net = returns_gross - cost_per_period
    
    # -------------------------
    # APLICAR LAG
    # -------------------------
    # Simulamos que ejecutamos 'execution_lag_days' después de la señal
    if execution_lag_days > 0:
        returns_net = returns_net.shift(execution_lag_days).dropna()
    
    if returns_net.empty:
        return BacktestResult(
            symbol, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            pd.Series(dtype=float), pd.Series(dtype=float)
        )
    
    # Equity curve
    equity = (1 + returns_net).cumprod()
    equity = pd.concat([pd.Series([1.0], index=[returns_net.index[0]]), equity])
    
    # -------------------------
    # MÉTRICAS
    # -------------------------
    freq_map = {"M": 12, "W": 52, "Q": 4}
    periods_per_year = freq_map.get(rebalance_freq, 12)
    
    cagr = calculate_cagr(returns_net, periods_per_year)
    sharpe = calculate_sharpe(returns_net, periods_per_year)
    sortino = calculate_sortino(returns_net, periods_per_year)
    max_dd = calculate_max_dd(equity)
    calmar = calculate_calmar(cagr, max_dd)
    
    # Turnover (100% cada rebalanceo asumido)
    turnover = 1.0  # 100% anual típico para monthly rebalance
    
    # Holding period estimado
    avg_holding_days = 365 / periods_per_year
    
    # Number of trades (2 por período: compra + venta)
    n_trades = len(returns_net) * 2
    
    return BacktestResult(
        symbol=symbol,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_dd=max_dd,
        calmar=calmar,
        turnover=turnover,
        avg_holding_days=avg_holding_days,
        n_trades=n_trades,
        n_periods=len(returns_net),
        equity_curve=equity,
        returns=returns_net,
    )


def backtest_portfolio(
    prices_dict: Dict[str, pd.DataFrame],
    symbols: list[str] = None,
    costs: TradingCosts = None,
    execution_lag_days: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Backtest V2 de múltiples símbolos con costos.
    """
    if symbols is None:
        symbols = list(prices_dict.keys())
    
    if costs is None:
        costs = TradingCosts()
    
    results = []
    equity_curves = {}
    
    for symbol in symbols:
        if symbol not in prices_dict:
            continue
        
        result = backtest_single_symbol(
            prices_dict[symbol],
            symbol,
            costs=costs,
            execution_lag_days=execution_lag_days,
        )
        
        if result.n_periods > 0:
            results.append({
                "symbol": symbol,
                "CAGR": result.cagr,
                "Sharpe": result.sharpe,
                "Sortino": result.sortino,
                "MaxDD": result.max_dd,
                "Calmar": result.calmar,
                "Turnover": result.turnover,
                "N_Trades": result.n_trades,
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

def calculate_portfolio_metrics(
    equity_curves: Dict[str, pd.Series],
    costs: TradingCosts = None,
) -> Dict:
    """Métricas de portfolio con costos incluidos"""
    if not equity_curves:
        return {
            "CAGR": 0, "Sharpe": 0, "Sortino": 0,
            "MaxDD": 0, "Calmar": 0, "N_Symbols": 0
        }
    
    if costs is None:
        costs = TradingCosts()
    
    # Combine equity curves (equal weight)
    df = pd.DataFrame(equity_curves)
    portfolio_equity = df.mean(axis=1)
    
    # Returns
    returns = portfolio_equity.pct_change().dropna()
    
    cagr = calculate_cagr(returns, 12)
    sharpe = calculate_sharpe(returns, 12)
    sortino = calculate_sortino(returns, 12)
    max_dd = calculate_max_dd(portfolio_equity)
    calmar = calculate_calmar(cagr, max_dd)
    
    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": max_dd,
        "Calmar": calmar,
        "N_Symbols": len(equity_curves),
        "Total_Cost_Impact": f"-{costs.round_trip_bps}bps per rebalance",
    }


# ============================================================================
# COMPARACIÓN DE COSTOS
# ============================================================================

def compare_cost_scenarios(
    prices_dict: Dict[str, pd.DataFrame],
    symbols: list[str],
) -> pd.DataFrame:
    """
    Compara backtest con diferentes escenarios de costos.
    
    Útil para entender el impacto de costos en Sharpe.
    """
    scenarios = {
        "No Costs": TradingCosts(0, 0, 0),
        "Low Costs (Retail)": TradingCosts(3, 3, 1),
        "Medium Costs (Realistic)": TradingCosts(5, 5, 2),
        "High Costs (Small Cap)": TradingCosts(10, 10, 5),
    }
    
    results = []
    
    for name, costs in scenarios.items():
        metrics, curves = backtest_portfolio(prices_dict, symbols, costs)
        
        if not metrics.empty:
            port_metrics = calculate_portfolio_metrics(curves, costs)
            
            results.append({
                "Scenario": name,
                "Total_Costs_bps": costs.round_trip_bps,
                "CAGR": port_metrics["CAGR"],
                "Sharpe": port_metrics["Sharpe"],
                "Sortino": port_metrics["Sortino"],
            })
    
    return pd.DataFrame(results)


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing backtest_engine V2...")
    
    # Mock data
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    
    df_up = pd.DataFrame({
        "date": dates,
        "close": np.linspace(100, 200, len(dates)),
    })
    
    prices_dict = {"UP": df_up}
    
    # Sin costos
    print("\n📊 Sin costos:")
    result_no_cost = backtest_single_symbol(
        df_up, "UP",
        costs=TradingCosts(0, 0, 0)
    )
    print(f"CAGR: {result_no_cost.cagr:.2%}")
    print(f"Sharpe: {result_no_cost.sharpe:.2f}")
    
    # Con costos realistas
    print("\n📊 Con costos realistas (12 bps round-trip):")
    result_with_cost = backtest_single_symbol(
        df_up, "UP",
        costs=TradingCosts(5, 5, 2)
    )
    print(f"CAGR: {result_with_cost.cagr:.2%}")
    print(f"Sharpe: {result_with_cost.sharpe:.2f}")
    
    impact = (result_no_cost.sharpe - result_with_cost.sharpe) / result_no_cost.sharpe
    print(f"\n⚠️ Impacto de costos: {impact:.1%} reducción en Sharpe")
