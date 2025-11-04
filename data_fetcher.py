"""
Data Fetcher - Módulo limpio
============================

Responsabilidad: Descargar datos desde FMP con rate limiting y caché.

- fetch_screener(...): universo base con columnas normalizadas
- fetch_fundamentals_batch(symbols): fundamentales TTM para F-Score simplificado (Forma B)
- fetch_prices_daily(symbol, lookback_days): precios diarios (close/volume) para MA200 y 12-1
"""

from __future__ import annotations
import os
import time
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import requests

# --------------------------- Config ---------------------------

FMP_API_KEY = os.getenv("FMP_API_KEY", "")
BASE = "https://financialmodelingprep.com/api/v3"
CACHE_DIR = Path(".cache/fmp")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Rate limiting suave: 25 req/seg es el máximo del plan premium; usamos margen.
SLEEP_BETWEEN_CALLS = float(os.getenv("FMP_SLEEP", "0.08"))

# TTLs
TTL_SCREENER_SEC = 3 * 60         # 3 min
TTL_FUND_TTM_SEC = 15 * 60        # 15 min
TTL_PRICES_SEC   = 60 * 60        # 1 hora

# ------------------------ Utilidades --------------------------

def _cache_path(key: str) -> Path:
    safe = key.replace("/", "_").replace("?", "_").replace("&", "_").replace("=", "_")
    return CACHE_DIR / f"{safe}.json"

def _read_cache(key: str, ttl_s: int) -> Optional[dict]:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        st = p.stat().st_mtime
        if (time.time() - st) > ttl_s:
            return None
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _write_cache(key: str, data: dict) -> None:
    p = _cache_path(key)
    try:
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass

def _http_get(endpoint: str, params: Dict, ttl_s: int, cache_key: Optional[str] = None) -> dict:
    """GET con caché. endpoint puede ser absoluto o relativo a BASE."""
    if not endpoint.startswith("http"):
        url = f"{BASE.rstrip('/')}/{endpoint.lstrip('/')}"
    else:
        url = endpoint

    params = dict(params or {})
    if FMP_API_KEY:
        params["apikey"] = FMP_API_KEY

    key = cache_key or (url + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params)))
    cached = _read_cache(key, ttl_s)
    if cached is not None:
        return cached

    # Llamada real
    resp = requests.get(url, params=params, timeout=30)
    time.sleep(SLEEP_BETWEEN_CALLS)
    resp.raise_for_status()
    data = resp.json()
    _write_cache(key, data)
    return data

def _to_df(obj, cols: Optional[List[str]] = None) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame(columns=cols or [])
    if isinstance(obj, list):
        df = pd.DataFrame(obj)
    elif isinstance(obj, dict):
        df = pd.DataFrame([obj])
    else:
        return pd.DataFrame(columns=cols or [])
    if cols:
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df[cols]
    return df

# ---------------------- Screener / Universo --------------------

def fetch_screener(
    limit: int = 1500,
    mcap_min: int = 200_000_000,
    volume_min: int = 100_000,
    exchanges: Optional[List[str]] = None,  # p.ej. ["NYSE","NASDAQ"]
) -> pd.DataFrame:
    """
    Devuelve universo base con columnas normalizadas:
    ['symbol','sector','market_cap','volume','price','exchange']

    Hace un único hit a /stock-screener con filtros básicos.
    """
    params = {
        "marketCapMoreThan": mcap_min,
        "volumeMoreThan": volume_min,
        "limit": limit,
        "isEtf": "false",
        "isFund": "false",
        "isActivelyTrading": "true",
    }
    if exchanges:
        params["exchange"] = ",".join(exchanges)

    data = _http_get("stock-screener", params, TTL_SCREENER_SEC)
    df = _to_df(data)

    # Normalizaciones
    rename_map = {
        "symbol": "symbol",
        "sector": "sector",
        "marketCap": "market_cap",
        "volume": "volume",
        "price": "price",
        "exchangeShortName": "exchange",
    }
    for src, dst in rename_map.items():
        if src not in df.columns:
            df[src] = np.nan
    df = df[list(rename_map.keys())].rename(columns=rename_map)

    # Tipos
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["sector"] = df["sector"].fillna("Unknown")
    df["exchange"] = df["exchange"].fillna("")

    # Limpieza
    df = df.dropna(subset=["symbol"]).drop_duplicates("symbol")
    return df.reset_index(drop=True)

# ---------------------- Fundamentales (TTM) --------------------

def _fetch_ratios_ttm(symbols: List[str]) -> pd.DataFrame:
    """
    /ratios-ttm: trae ROA, ROE, grossMargin, assetTurnover, etc.
    """
    rows = []
    for sym in symbols:
        try:
            data = _http_get(f"ratios-ttm/{sym}", {}, TTL_FUND_TTM_SEC)
            df = _to_df(data)
            if df.empty:
                continue
            # ratios-ttm viene ordenado más reciente al inicio normalmente
            last = df.iloc[0]
            rows.append({
                "symbol": sym,
                "roa": pd.to_numeric(last.get("returnOnAssetsTTM"), errors="coerce"),
                "roe": pd.to_numeric(last.get("returnOnEquityTTM"), errors="coerce"),
                "roic": pd.to_numeric(last.get("returnOnCapitalEmployedTTM"), errors="coerce"),
                "gross_margin": pd.to_numeric(last.get("grossProfitMarginTTM"), errors="coerce"),
                "asset_turnover": pd.to_numeric(last.get("assetTurnoverTTM"), errors="coerce"),
                "current_ratio": pd.to_numeric(last.get("currentRatioTTM"), errors="coerce"),
                "long_term_debt_ratio": pd.to_numeric(last.get("longTermDebtToTotalAssetsTTM"), errors="coerce"),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

def _fetch_cashflow_ttm(symbols: List[str]) -> pd.DataFrame:
    """
    /cash-flow-statement-ttm: trae operatingCashFlowTTM y freeCashFlowTTM.
    """
    rows = []
    for sym in symbols:
        try:
            data = _http_get(f"cash-flow-statement-ttm/{sym}", {}, TTL_FUND_TTM_SEC)
            df = _to_df(data)
            if df.empty:
                continue
            last = df.iloc[0]
            rows.append({
                "symbol": sym,
                "operating_cf": pd.to_numeric(last.get("operatingCashFlowTTM"), errors="coerce"),
                "fcf": pd.to_numeric(last.get("freeCashFlowTTM"), errors="coerce"),
                "capex": pd.to_numeric(last.get("capitalExpenditureTTM"), errors="coerce"),
                "net_income": pd.to_numeric(last.get("netIncomeTTM"), errors="coerce"),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

def _fetch_balance_ttm(symbols: List[str]) -> pd.DataFrame:
    """
    /balance-sheet-statement-ttm: totalAssetsTTM, longTermDebtTTM.
    (No siempre presente; usar ratios si no).
    """
    rows = []
    for sym in symbols:
        try:
            data = _http_get(f"balance-sheet-statement-ttm/{sym}", {}, TTL_FUND_TTM_SEC)
            df = _to_df(data)
            if df.empty:
                continue
            last = df.iloc[0]
            rows.append({
                "symbol": sym,
                "total_assets": pd.to_numeric(last.get("totalAssetsTTM"), errors="coerce"),
                "long_term_debt": pd.to_numeric(last.get("longTermDebtTTM"), errors="coerce"),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

def fetch_fundamentals_batch(symbols: List[str], chunk_size: int = 50) -> pd.DataFrame:
    """
    Devuelve DF con columnas clave para el F-Score simplificado (Forma B):
      ['symbol','roa','operating_cf','fcf','capex','roe','roic','gross_margin',
       'asset_turnover','current_ratio','long_term_debt','total_assets','net_income']

    Además incluye columnas *_prev como NaN para compatibilidad con la versión académica.
    """
    symbols = [s for s in pd.unique(pd.Series(symbols).dropna().astype(str)) if s]
    if not symbols:
        cols = ["symbol","roa","operating_cf","fcf","capex","roe","roic","gross_margin",
                "asset_turnover","current_ratio","long_term_debt","total_assets","net_income",
                "roa_prev","current_ratio_prev","long_term_debt_prev","total_assets_prev",
                "revenue","revenue_prev","gross_margin_prev","shares_outstanding","shares_outstanding_prev"]
        return pd.DataFrame(columns=cols)

    # Batches para controlar caché/latencia
    dfs = []
    for i in range(0, len(symbols), chunk_size):
        batch = symbols[i:i+chunk_size]

        ratios = _fetch_ratios_ttm(batch)
        cf     = _fetch_cashflow_ttm(batch)
        bs     = _fetch_balance_ttm(batch)

        df = pd.merge(ratios, cf, on="symbol", how="outer")
        df = pd.merge(df, bs, on="symbol", how="outer")

        # Añadir columnas esperadas por otros módulos si no existen
        ensure_cols = [
            "roa","operating_cf","fcf","capex","roe","roic","gross_margin",
            "asset_turnover","current_ratio","long_term_debt","total_assets","net_income",
            # Académico *_prev y otros
            "roa_prev","current_ratio_prev","long_term_debt_prev","total_assets_prev",
            "revenue","revenue_prev","gross_margin_prev",
            "shares_outstanding","shares_outstanding_prev"
        ]
        for c in ensure_cols:
            if c not in df.columns:
                df[c] = np.nan

        # Tipos numéricos seguros
        num_cols = [c for c in df.columns if c != "symbol"]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    out = out.dropna(subset=["symbol"]).drop_duplicates("symbol").reset_index(drop=True)
    return out

# -------------------------- Precios ----------------------------

def fetch_prices_daily(symbol: str, lookback_days: int = 800) -> pd.DataFrame:
    """
    Devuelve OHLCV diario (al menos 'date','close','volume') para el símbolo.
    Usado para MA200 y momentum 12-1. 800 días ≈ 3 años borsátiles.
    """
    if not symbol:
        return pd.DataFrame(columns=["date","close","volume"])

    # historical-price-full limita por 'from'/'to' o 'serietype=line'
    # Usamos serietype 'line' + timeseries para close.
    params = {
        "timeseries": lookback_days,
        "serietype": "line",
    }
    data = _http_get(f"historical-price-full/{symbol}", params, TTL_PRICES_SEC)
    hist = data.get("historical", [])
    df = _to_df(hist)
    if df.empty:
        # Fallback: chart/1day? Preferimos no, retornamos vacío
        return pd.DataFrame(columns=["date","close","volume"])

    # Normalizar
    for c in ["date","close","volume"]:
        if c not in df.columns:
            df[c] = np.nan
    df = df[["date","close","volume"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = df.dropna(subset=["date","close"]).sort_values("date").reset_index(drop=True)
    return df

# --------------------------- Tests -----------------------------

if __name__ == "__main__":
    print("🧪 Testing data_fetcher...")

    uni = fetch_screener(limit=10)
    print(f"✅ Screener: {len(uni)} símbolos")
    print(uni.head())

    if not uni.empty:
        syms = uni["symbol"].head(5).tolist()
        fund = fetch_fundamentals_batch(syms)
        print(f"\n✅ Fundamentales ({len(syms)}): {fund.shape}")
        print(fund.head())

        px = fetch_prices_daily(syms[0])
        print(f"\n✅ Prices {syms[0]}: {px.shape}")
        print(px.tail())
