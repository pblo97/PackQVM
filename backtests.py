# === backtests.py (robusto p/ app) ===
from __future__ import annotations
import math
import pandas as pd
import numpy as np
from dataclasses import dataclass

# ----------------- Helpers de limpieza -----------------
_CLOSE_CANDIDATES = ["close", "Close", "adjclose", "Adj Close", "AdjClose", "adjusted_close", "price", "Price"]

def _to_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        # intenta columna 'date' o index->datetime
        if "date" in df.columns:
            df = df.set_index(pd.to_datetime(df["date"], errors="coerce"))
            df = df.drop(columns=["date"], errors="ignore")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    # ordenar y deduplicar conservando el último
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

def _ensure_close(df: pd.DataFrame) -> pd.DataFrame:
    if "close" not in df.columns:
        for c in _CLOSE_CANDIDATES:
            if c in df.columns:
                df = df.rename(columns={c: "close"})
                break
    # si aún no hay 'close', construir con la 1ª numérica disponible
    if "close" not in df.columns:
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                df = df.rename(columns={c: "close"})
                break
    # forzar numérico y limpiar
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df[df["close"].notna()]
    return df

def _coerce_price_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out = _to_dt_index(out)
    if out.empty:
        return out
    out = _ensure_close(out)
    # quitar ceros/spikes imposibles
    out = out[out["close"] > 0]
    return out

# -------------- Métricas básicas (mensual/semanal) --------------
def _cagr(returns: pd.Series, freq_per_year=12) -> float:
    r = returns.dropna()
    if r.empty:
        return 0.0
    equity = (1 + r).cumprod()
    n_years = len(r) / float(freq_per_year)
    if n_years <= 0 or equity.iloc[-1] <= 0:
        return 0.0
    return float(equity.iloc[-1] ** (1.0 / n_years) - 1.0)

def _sharpe(returns: pd.Series, freq_per_year=12) -> float:
    r = returns.dropna()
    if r.empty:
        return 0.0
    mu = r.mean() * freq_per_year
    sd = r.std(ddof=0) * math.sqrt(freq_per_year)
    return 0.0 if sd == 0 or np.isnan(sd) else float(mu / sd)

def _sortino(returns: pd.Series, freq_per_year=12) -> float:
    r = returns.dropna()
    if r.empty:
        return 0.0
    downside = r[r < 0]
    dd = downside.std(ddof=0) * math.sqrt(freq_per_year)
    mu = r.mean() * freq_per_year
    return 0.0 if dd == 0 or np.isnan(dd) else float(mu / dd)

def _maxdd(equity: pd.Series) -> float:
    if equity is None or equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min()) if len(dd) else 0.0

# -------- Señales: MA200 (diaria) y Momentum 12-1 (≈252-21 días) --------
def _ensure_sig_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ma200" not in out.columns:
        out["ma200"] = out["close"].rolling(200, min_periods=200).mean()
    if "mom_12_1" not in out.columns:
        lag_12 = out["close"].shift(252)
        lag_1  = out["close"].shift(21)
        out["mom_12_1"] = (lag_1 / lag_12) - 1.0
    return out

def _passes_signal(row, use_and_condition: bool) -> bool:
    cond_ma  = pd.notna(row.get("ma200")) and (row["close"] > row["ma200"])
    cond_mom = pd.notna(row.get("mom_12_1")) and (row["mom_12_1"] > 0)
    return (cond_ma and cond_mom) if use_and_condition else (cond_ma or cond_mom)

@dataclass
class BTResult:
    symbol: str
    cagr: float
    sharpe: float
    sortino: float
    maxdd: float
    turnover: float
    n_trades: int
    equity: pd.Series
    rets: pd.Series

# ------------------- Core backtest unitario -------------------
def backtest_single_symbol(
    df_price: pd.DataFrame,
    symbol: str,
    cost_bps: int = 10,
    lag_days: int = 0,
    use_and_condition: bool = False,
    rebalance_freq: str = "M",   # "M" o "W"
) -> BTResult:
    """
    Backtest long-only binario con regla (MA200 OR/AND Mom12-1>0).
    Rebalance mensual o semanal; 'lag_days' desplaza la ejecución.
    """
    df = _coerce_price_df(df_price)
    if df.empty or "close" not in df.columns:
        return BTResult(symbol, 0,0,0,0,0,0, pd.Series(dtype=float), pd.Series(dtype=float))

    df = _ensure_sig_cols(df)

    # Fechas de rebalance
    if rebalance_freq not in ("M", "W"):
        rebalance_freq = "M"
    freq_per_year = 12 if rebalance_freq == "M" else 52

    # Si la serie es muy corta, aborta
    if len(df) < (252 + 22):  # suficiente para mom_12_1 y ma200
        return BTResult(symbol, 0,0,0,0,0,0, pd.Series(dtype=float), pd.Series(dtype=float))

    # Puntos de rebalance = último punto de cada período
    rb_ends = df.resample(rebalance_freq).last().index
    if len(rb_ends) < 2:
        return BTResult(symbol, 0,0,0,0,0,0, pd.Series(dtype=float), pd.Series(dtype=float))

    lag = pd.Timedelta(days=int(lag_days)) if lag_days else pd.Timedelta(0)

    equity_vals = [1.0]
    rets = []
    prev_w = 0.0
    turnover_hist = []
    n_trades = 0

    for t0, t1 in zip(rb_ends[:-1], rb_ends[1:]):
        # Señal evaluada al final del período t0
        row_t0 = df.loc[:t0].iloc[-1]
        in_signal = _passes_signal(row_t0, use_and_condition)
        w = 1.0 if in_signal else 0.0

        # "turnover" binario
        tw = abs(w - prev_w) * 0.5
        turnover_hist.append(tw)
        if w != prev_w:
            n_trades += 1

        # precios con lag
        p0s = df.loc[:t0 + lag, "close"]
        p1s = df.loc[:t1 + lag, "close"]
        if p0s.empty or p1s.empty:
            r = 0.0
        else:
            p0 = p0s.iloc[-1]
            p1 = p1s.iloc[-1]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                gross = (p1 / p0 - 1.0) * w
                cost = tw * (cost_bps / 1e4)
                r = gross - cost
            else:
                r = 0.0

        rets.append(r)
        equity_vals.append(equity_vals[-1] * (1.0 + r))
        prev_w = w

    rets = pd.Series(rets, index=rb_ends[1:], name=symbol)
    equity = pd.Series(equity_vals, index=[rb_ends[0]] + list(rb_ends[1:]), name=symbol)

    cagr   = _cagr(rets, freq_per_year=freq_per_year)
    sharpe = _sharpe(rets, freq_per_year=freq_per_year)
    sortino= _sortino(rets, freq_per_year=freq_per_year)
    maxdd  = _maxdd(equity)
    turn   = float(np.nanmean(turnover_hist)) if turnover_hist else 0.0

    return BTResult(symbol, cagr, sharpe, sortino, maxdd, turn, n_trades, equity, rets)

# ------------------- Backtest por lote -------------------
def backtest_many(
    *,
    panel: dict[str, pd.DataFrame],
    symbols: list[str],
    cost_bps: int = 10,
    lag_days: int = 0,
    use_and_condition: bool = False,
    rebalance_freq: str = "M",
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """
    Ejecuta el unitario para cada símbolo y devuelve:
      - metrics_df: ['symbol','CAGR','Sharpe','Sortino','MaxDD','Turnover','Trades']
      - curves: {symbol: equity_series}
    """
    rows = []
    curves: dict[str, pd.Series] = {}

    for s in symbols:
        df = panel.get(s)
        if df is None or df.empty:
            continue
        res = backtest_single_symbol(
            df_price=df,
            symbol=s,
            cost_bps=int(cost_bps),
            lag_days=int(lag_days),
            use_and_condition=bool(use_and_condition),
            rebalance_freq=str(rebalance_freq),
        )
        if res.equity is None or res.equity.empty:
            continue
        rows.append({
            "symbol":   s,
            "CAGR":     res.cagr,
            "Sharpe":   res.sharpe,
            "Sortino":  res.sortino,
            "MaxDD":    res.maxdd,
            "Turnover": res.turnover,
            "Trades":   res.n_trades,
        })
        curves[s] = res.equity

    metrics = pd.DataFrame(rows)
    if not metrics.empty and "CAGR" in metrics.columns:
        metrics = metrics.sort_values(["CAGR","Sharpe"], ascending=False).reset_index(drop=True)
    return metrics, curves

# (Opcional) Si necesitas métricas desde una serie de retornos externa:
def perf_metrics_from_returns(rets: pd.Series, periods_per_year: int) -> dict:
    rets = rets.dropna()
    if rets.empty:
        return {"CAGR":0.0,"Sharpe":0.0,"Sortino":0.0,"MaxDD":0.0}
    eq = (1 + rets).cumprod()
    return {
        "CAGR":   _cagr(rets, periods_per_year),
        "Sharpe": _sharpe(rets, periods_per_year),
        "Sortino":_sortino(rets, periods_per_year),
        "MaxDD":  _maxdd(eq),
    }
