# backtest_engine.py
"""
Backtest Engine - Simple y limpio (compat V2)
=============================================

- Long-only, equal-weight
- Usa retornos diarios y agrega a mensual para métricas
- Soporta costos de entrada (bps) y lag de ejecución (días)
- API compatible con app_streamlit.py:
    - TradingCosts
    - backtest_portfolio(prices_dict, costs, execution_lag_days)
    - calculate_portfolio_metrics(equity_curves, costs)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# LIMPIEZA
# ============================================================================
def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza DataFrame de precios a index datetime + columna 'close'."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["close"])

    x = df.copy()
    # index datetime
    if not isinstance(x.index, pd.DatetimeIndex):
        if "date" in x.columns:
            x["date"] = pd.to_datetime(x["date"], errors="coerce")
            x = x.dropna(subset=["date"]).set_index("date")
        else:
            x.index = pd.to_datetime(x.index, errors="coerce")
            x = x.dropna(subset=[x.index.name])

    # columna close
    if "close" not in x.columns:
        for alt in ("Close", "adjClose", "Adj Close", "adj_close"):
            if alt in x.columns:
                x["close"] = x[alt]
                break

    if "close" not in x.columns:
        return pd.DataFrame(columns=["close"])

    x["close"] = pd.to_numeric(x["close"], errors="coerce")
    x = x.sort_index()
    x = x[~x.index.duplicated(keep="last")]
    x = x[x["close"] > 0]
    return x[["close"]]


# ============================================================================
# COSTOS
# ============================================================================
@dataclass
class TradingCosts:
    commission_bps: float = 5.0      # 0.05%
    slippage_bps: float = 5.0        # 0.05%
    market_impact_bps: float = 2.0   # 0.02%

    @property
    def total_bps(self) -> float:
        return float(self.commission_bps + self.slippage_bps + self.market_impact_bps)

    @property
    def total_frac(self) -> float:
        return self.total_bps / 10_000.0


# ============================================================================
# MÉTRICAS
# ============================================================================
def _cagr_from_monthly(returns_m: pd.Series) -> float:
    if returns_m is None or returns_m.empty:
        return np.nan
    eq = (1 + returns_m).cumprod()
    years = len(returns_m) / 12.0
    if years <= 0 or eq.iloc[-1] <= 0:
        return np.nan
    return float(eq.iloc[-1] ** (1 / years) - 1)


def _sharpe_from_monthly(returns_m: pd.Series) -> float:
    if returns_m is None or returns_m.empty:
        return np.nan
    mu_a = returns_m.mean() * 12.0
    sd_a = returns_m.std(ddof=0) * np.sqrt(12.0)
    return float(mu_a / sd_a) if (sd_a and np.isfinite(sd_a) and sd_a != 0) else np.nan


def _sortino_from_monthly(returns_m: pd.Series) -> float:
    if returns_m is None or returns_m.empty:
        return np.nan
    mu_a = returns_m.mean() * 12.0
    downside = returns_m[returns_m < 0]
    d_sd_a = downside.std(ddof=0) * np.sqrt(12.0) if len(downside) > 0 else np.nan
    return float(mu_a / d_sd_a) if (d_sd_a and np.isfinite(d_sd_a) and d_sd_a != 0) else np.nan


def _maxdd_from_equity(equity: pd.Series) -> float:
    if equity is None or equity.empty:
        return np.nan
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if not dd.empty else np.nan


# ============================================================================
# BACKTEST
# ============================================================================
def _apply_entry_cost_once(returns_d: pd.Series, cost_frac: float) -> pd.Series:
    """Resta el costo una sola vez en el primer retorno disponible."""
    r = returns_d.copy()
    if not r.empty:
        first_idx = r.first_valid_index()
        if first_idx is not None:
            r.loc[first_idx] = r.loc[first_idx] - cost_frac
    return r


def _daily_to_monthly_returns(returns_d: pd.Series) -> pd.Series:
    """Agrega retornos diarios a mensuales: (1+r_d).prod()-1 por mes."""
    if returns_d is None or returns_d.empty:
        return pd.Series(dtype=float)
    return (1.0 + returns_d).resample("M").prod().dropna() - 1.0


def backtest_portfolio(
    prices_dict: Dict[str, pd.DataFrame],
    costs: TradingCosts | None = None,
    execution_lag_days: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Backtest buy&hold por símbolo (sin rebalanceo periódico) y métricas por activo.
    - Aplica costo de entrada una vez (total de costs) por símbolo.
    - Aplica lag en días en los retornos diarios (shift positivo desplaza la ejecución).

    Parameters
    ----------
    prices_dict : dict[str, DataFrame]
        Cada DF debe tener columnas ['date','close'] o index datetime + 'close'.
    costs : TradingCosts | None
    execution_lag_days : int
        Días de lag para comenzar (>=0).

    Returns
    -------
    metrics_df : DataFrame con columnas:
        ['Symbol','CAGR','Sharpe','Sortino','MaxDD','Calmar']
    equity_curves : dict[str, Series]
        Curva de equity diaria normalizada a 1.0 (desde primer día válido post-lag).
    """
    if not prices_dict:
        return pd.DataFrame(), {}

    metrics_rows = []
    equity_curves: Dict[str, pd.Series] = {}
    cost_frac = costs.total_frac if isinstance(costs, TradingCosts) else 0.0
    lag = max(int(execution_lag_days), 0)

    for sym, df in prices_dict.items():
        px = clean_prices(df)
        if px.empty or "close" not in px.columns:
            continue

        # retornos diarios
        r_d = px["close"].pct_change()

        # lag de ejecución
        if lag > 0:
            r_d = r_d.shift(lag)

        # costo de entrada una sola vez
        r_d = _apply_entry_cost_once(r_d, cost_frac)

        # quitar NaNs iniciales tras lag
        r_d = r_d.dropna()
        if r_d.empty:
            continue

        # equity diaria
        eq = (1.0 + r_d.fillna(0.0)).cumprod()
        equity_curves[sym] = eq

        # mensuales para métricas
        r_m = _daily_to_monthly_returns(r_d)

        cagr = _cagr_from_monthly(r_m)
        sharpe = _sharpe_from_monthly(r_m)
        sortino = _sortino_from_monthly(r_m)
        maxdd = _maxdd_from_equity(eq)
        calmar = (cagr / abs(maxdd)) if (cagr is not None and maxdd is not None and maxdd not in (0, np.nan)) else np.nan

        metrics_rows.append({
            "Symbol": sym,
            "CAGR": cagr,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "MaxDD": maxdd,
            "Calmar": calmar,
        })

    metrics_df = pd.DataFrame(metrics_rows)
    # ordenar columnas si no está vacío
    if not metrics_df.empty:
        metrics_df = metrics_df[["Symbol", "CAGR", "Sharpe", "Sortino", "MaxDD", "Calmar"]]

    return metrics_df, equity_curves


def calculate_portfolio_metrics(
    equity_curves: Dict[str, pd.Series],
    costs: TradingCosts | None = None,
) -> dict:
    """
    Combina curvas individuales (equal-weight) y calcula métricas agregadas.
    Devuelve: {'CAGR','Sharpe','Sortino','MaxDD','Calmar'}
    """
    if not equity_curves:
        return {"CAGR": np.nan, "Sharpe": np.nan, "Sortino": np.nan, "MaxDD": np.nan, "Calmar": np.nan}

    # Alinear por fecha
    ec_df = pd.concat(equity_curves.values(), axis=1)
    ec_df.columns = list(equity_curves.keys())
    ec_df = ec_df.dropna(how="all").ffill().dropna(how="all")

    # Equity promedio (equal-weight)
    port_eq = ec_df.mean(axis=1)

    # Retornos diarios y mensuales
    port_r_d = port_eq.pct_change().dropna()
    port_r_m = _daily_to_monthly_returns(port_r_d)

    cagr = _cagr_from_monthly(port_r_m)
    sharpe = _sharpe_from_monthly(port_r_m)
    sortino = _sortino_from_monthly(port_r_m)
    maxdd = _maxdd_from_equity(port_eq)
    calmar = (cagr / abs(maxdd)) if (cagr is not None and maxdd is not None and maxdd not in (0, np.nan)) else np.nan

    return {"CAGR": cagr, "Sharpe": sharpe, "Sortino": sortino, "MaxDD": maxdd, "Calmar": calmar}


# ============================================================================
# TEST RÁPIDO
# ============================================================================
if __name__ == "__main__":
    print("🧪 Smoke test backtest_engine.py")

    dates = pd.date_range("2023-01-01", "2025-11-01", freq="B")
    up = pd.DataFrame({"date": dates, "close": np.linspace(100, 180, len(dates))})
    dn = pd.DataFrame({"date": dates, "close": np.linspace(100, 60, len(dates))})
    prices = {"UP": up, "DN": dn}

    costs = TradingCosts(5, 5, 2)
    m, ec = backtest_portfolio(prices, costs=costs, execution_lag_days=1)
    print(m.head())

    port = calculate_portfolio_metrics(ec, costs)
    print(port)
