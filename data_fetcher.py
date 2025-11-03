"""
Data Fetcher - Módulo limpio desde cero
========================================

Responsabilidad única: Descargar datos desde FMP con rate limiting y caché.
"""

from __future__ import annotations
import os
import time
import json
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
CACHE_DIR = Path(".cache_qvm")
CACHE_DIR.mkdir(exist_ok=True)

# Rate limiting: 4 req/s = 240/min (conservador para free tier)
MIN_REQUEST_INTERVAL = 0.25  # segundos

# Session HTTP con retries
_session = requests.Session()
_retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_adapter = HTTPAdapter(max_retries=_retry)
_session.mount("https://", _adapter)

_last_request_time = 0.0


# ============================================================================
# RATE LIMITER
# ============================================================================

def _rate_limit():
    """Rate limiter simple y thread-safe"""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


def _http_get(url: str, params: Optional[Dict] = None) -> Dict | List:
    """GET con rate limiting y manejo de errores"""
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY no configurada")
    
    params = dict(params or {})
    params["apikey"] = FMP_API_KEY
    
    _rate_limit()
    
    try:
        r = _session.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ HTTP error: {url} - {e}")
        return {}


# ============================================================================
# CACHÉ
# ============================================================================

def _cache_path(key: str) -> Path:
    """Path del archivo de caché"""
    return CACHE_DIR / f"{key}.parquet"


def load_from_cache(key: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
    """Carga desde caché si existe y no está expirado"""
    path = _cache_path(key)
    
    if not path.exists():
        return None
    
    # Check age
    age = time.time() - path.stat().st_mtime
    if age > max_age_hours * 3600:
        return None
    
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def save_to_cache(df: pd.DataFrame, key: str):
    """Guarda en caché"""
    try:
        df.to_parquet(_cache_path(key))
    except Exception as e:
        print(f"⚠️ Error guardando caché: {e}")


# ============================================================================
# SCREENER
# ============================================================================

def fetch_screener(
    limit: int = 300,
    mcap_min: float = 5e8,
    volume_min: int = 500_000,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Descarga universo base desde FMP screener.
    
    Returns:
        DataFrame con ['symbol', 'sector', 'market_cap', 'volume']
    """
    cache_key = f"screener_{limit}_{int(mcap_min)}_{volume_min}"
    
    if use_cache:
        cached = load_from_cache(cache_key)
        if cached is not None:
            return cached
    
    url = "https://financialmodelingprep.com/api/v3/stock-screener"
    params = {
        "marketCapMoreThan": mcap_min,
        "volumeMoreThan": volume_min,
        "limit": limit,
        "isEtf": "false",
        "isFund": "false",
        "isActivelyTrading": "true",
    }
    
    data = _http_get(url, params)
    
    if not data or not isinstance(data, list):
        return pd.DataFrame(columns=["symbol", "sector", "market_cap", "volume"])
    
    df = pd.DataFrame(data)
    
    # Normalización
    if "symbol" not in df.columns:
        return pd.DataFrame(columns=["symbol", "sector", "market_cap", "volume"])
    
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["sector"] = df.get("sector", "Unknown").fillna("Unknown").astype(str)
    df["market_cap"] = pd.to_numeric(df.get("marketCap", np.nan), errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", np.nan), errors="coerce")
    
    # Filtrar
    df = df[df["market_cap"] >= mcap_min]
    df = df[df["volume"] >= volume_min]
    
    result = df[["symbol", "sector", "market_cap", "volume"]].copy()
    
    if use_cache:
        save_to_cache(result, cache_key)
    
    return result


# ============================================================================
# FUNDAMENTALES
# ============================================================================

def fetch_fundamentals_batch(symbols: List[str], use_cache: bool = True) -> pd.DataFrame:
    """
    Descarga métricas fundamentales para lista de símbolos.
    
    Returns:
        DataFrame con métricas de value/quality/profitability
    """
    cache_key = f"fund_{'_'.join(sorted(symbols[:5]))}"
    
    if use_cache:
        cached = load_from_cache(cache_key)
        if cached is not None:
            return cached
    
    rows = []
    
    for symbol in symbols:
        # Key metrics TTM
        try:
            km = _http_get(f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{symbol}")
            km = km[0] if isinstance(km, list) and km else {}
        except Exception:
            km = {}
        
        # Ratios TTM
        try:
            rt = _http_get(f"https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol}")
            rt = rt[0] if isinstance(rt, list) and rt else {}
        except Exception:
            rt = {}
        
        # Cash flow TTM
        try:
            cf = _http_get(f"https://financialmodelingprep.com/api/v3/cash-flow-statement-ttm/{symbol}")
            cf = cf if isinstance(cf, dict) else {}
        except Exception:
            cf = {}
        
        rows.append({
            "symbol": symbol,
            # Value
            "ev_ebitda": km.get("enterpriseValueOverEBITDATTM"),
            "pb": km.get("pbRatioTTM"),
            "pe": km.get("peRatioTTM"),
            # Quality
            "roe": rt.get("returnOnEquityTTM"),
            "roic": rt.get("returnOnCapitalEmployedTTM"),
            "gross_margin": rt.get("grossProfitMarginTTM"),
            # Profitability
            "fcf": cf.get("freeCashFlowTTM"),
            "operating_cf": cf.get("netCashProvidedByOperatingActivitiesTTM"),
        })
    
    df = pd.DataFrame(rows)
    
    # Conversión numérica
    for col in df.columns:
        if col != "symbol":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    if use_cache:
        save_to_cache(df, cache_key)
    
    return df


# ============================================================================
# PRECIOS
# ============================================================================

def fetch_prices(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    Descarga precios históricos para un símbolo.
    
    Returns:
        DataFrame con index=date, columnas=['close', 'volume']
    """
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
    params = {"from": start, "to": end}
    
    data = _http_get(url, params)
    
    if not data or "historical" not in data:
        return None
    
    df = pd.DataFrame(data["historical"])
    
    if df.empty or "date" not in df.columns:
        return None
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    
    # Normalizar close
    for col in ["adjClose", "close"]:
        if col in df.columns:
            df["close"] = pd.to_numeric(df[col], errors="coerce")
            break
    
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    
    return df[["close", "volume"]].dropna(subset=["close"])


# ============================================================================
# UTILS
# ============================================================================

import numpy as np

def clear_cache():
    """Limpia todo el caché"""
    for f in CACHE_DIR.glob("*.parquet"):
        f.unlink()
    print(f"✅ Caché limpiado: {CACHE_DIR}")


if __name__ == "__main__":
    # Test básico
    print("🧪 Testing data_fetcher...")
    
    df = fetch_screener(limit=10)
    print(f"✅ Screener: {len(df)} símbolos")
    print(df.head())
    
    if not df.empty:
        symbol = df["symbol"].iloc[0]
        fund = fetch_fundamentals_batch([symbol])
        print(f"\n✅ Fundamentales de {symbol}:")
        print(fund)
