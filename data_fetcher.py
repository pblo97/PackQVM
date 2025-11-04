"""
Data Fetcher - FMP client con rate limiting y caché
===================================================

Funciones públicas:
- fetch_screener(limit, mcap_min, volume_min)
- fetch_fundamentals_batch(symbols)
- fetch_prices_daily(symbol, lookback_days=800)

Requiere: export FMP_API_KEY=<tu_api_key>
"""

from __future__ import annotations
import os
import time
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
FMP_API_KEY = os.getenv("FMP_API_KEY", "").strip()
BASE = "https://financialmodelingprep.com/api/v3"
CACHE_DIR = Path(".cache/fmp")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Rate limiting simple (respetuoso)
_MIN_INTERVAL = 0.15  # ~6-7 req/s
_last_call_ts = 0.0
_session = requests.Session()


def _rate_limit():
    global _last_call_ts
    now = time.time()
    dt = now - _last_call_ts
    if dt < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - dt)
    _last_call_ts = time.time()


# -----------------------------------------------------------------------------
# Utilidades de caché
# -----------------------------------------------------------------------------
def _cache_path(namespace: str, key: str) -> Path:
    safe = key.replace("/", "_").replace("?", "_").replace("&", "_").replace("=", "_").replace(",", "_")
    return CACHE_DIR / f"{namespace}__{safe}.json"


def _cache_load(namespace: str, key: str, max_age_sec: int) -> Optional[dict]:
    p = _cache_path(namespace, key)
    if not p.exists():
        return None
    try:
        if time.time() - p.stat().st_mtime > max_age_sec:
            return None
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _cache_save(namespace: str, key: str, obj: dict) -> None:
    p = _cache_path(namespace, key)
    try:
        with p.open("w", encoding="utf-8") as f:
            json.dump(obj, f)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# HTTP helper
# -----------------------------------------------------------------------------
def _http_get(path: str, params: Optional[Dict] = None, cache_ns: str = "", cache_key: str = "", max_age_sec: int = 3600):
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY no configurada en el entorno")

    params = dict(params or {})
    params["apikey"] = FMP_API_KEY

    if cache_ns and cache_key:
        hit = _cache_load(cache_ns, cache_key, max_age_sec)
        if hit is not None:
            return hit

    _rate_limit()
    url = f"{BASE}/{path.lstrip('/')}"
    r = _session.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if cache_ns and cache_key:
        _cache_save(cache_ns, cache_key, data)
    return data


# -----------------------------------------------------------------------------
# Normalizadores
# -----------------------------------------------------------------------------
def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _clean_sector(x: Optional[str]) -> Optional[str]:
    if not x:
        return None
    return str(x).strip()


# -----------------------------------------------------------------------------
# Screener
# -----------------------------------------------------------------------------
def fetch_screener(limit: int = 300, mcap_min: float = 2e9, volume_min: int = 1_000_000) -> pd.DataFrame:
    """
    Devuelve un universo con columnas mínimas: symbol, sector, market_cap
    Usamos /stock-screener para filtrar por marketCap y volumen.
    """
    # FMP stock-screener admite filtros; usamos un rango amplio y filtramos localmente
    params = {
        "marketCapMoreThan": int(max(mcap_min, 0)),
        "volumeMoreThan": int(max(volume_min, 0)),
        "limit": int(limit * 2),  # pedimos más y luego recortamos
        "exchange": "",           # all exchanges
        "isEtf": "false",
        "isActivelyTrading": "true",
    }
    raw = _http_get(
        "stock-screener",
        params=params,
        cache_ns="screener",
        cache_key=f"mcap_{int(mcap_min)}_vol_{int(volume_min)}_limit_{int(limit)}",
        max_age_sec=3600,
    )

    rows = []
    for it in raw or []:
        sym = it.get("symbol")
        if not sym:
            continue
        rows.append({
            "symbol": sym,
            "sector": _clean_sector(it.get("sector")),
            "market_cap": _to_float(it.get("marketCap")),
            "price": _to_float(it.get("price")),
            "volume": int(it.get("volume") or 0),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Filtrado local por si el endpoint devolvió más
    df = df.sort_values("market_cap", ascending=False)
    df = df.head(limit).reset_index(drop=True)

    # Fallback de sector si está vacío → intentamos /profile (light cache)
    if df["sector"].isna().any():
        missing = df[df["sector"].isna()]["symbol"].tolist()[:50]  # no saturar
        for s in missing:
            try:
                prof = _http_get(
                    f"profile/{s}",
                    cache_ns="profile",
                    cache_key=s,
                    max_age_sec=24 * 3600,
                )
                if isinstance(prof, list) and prof:
                    sec = _clean_sector(prof[0].get("sector"))
                    if sec:
                        df.loc[df["symbol"] == s, "sector"] = sec
            except Exception:
                pass

    return df


# -----------------------------------------------------------------------------
# Fundamentales (batch)
# -----------------------------------------------------------------------------
def _fetch_ratios(symbol: str) -> dict:
    # ratios TTM: PE, PB, EV/EBITDA (evEbitda)
    data = _http_get(
        f"ratios-ttm/{symbol}",
        cache_ns="ratios_ttm",
        cache_key=symbol,
        max_age_sec=24 * 3600,
    )
    if isinstance(data, list) and data:
        d = data[0]
        return {
            "pe": _to_float(d.get("priceEarningsRatioTTM")),
            "pb": _to_float(d.get("priceToBookRatioTTM")),
            "ev_ebitda": _to_float(d.get("enterpriseValueOverEBITDATTM")),
        }
    return {}


def _fetch_key_metrics(symbol: str) -> dict:
    # key-metrics TTM: ROE, ROIC, grossMarginTTM, etc.
    data = _http_get(
        f"key-metrics-ttm/{symbol}",
        cache_ns="keymetrics_ttm",
        cache_key=symbol,
        max_age_sec=24 * 3600,
    )
    if isinstance(data, list) and data:
        d = data[0]
        return {
            "roe": _to_float(d.get("roeTTM")),
            "roic": _to_float(d.get("roicTTM")),
            "gross_margin": _to_float(d.get("grossProfitMarginTTM")),
        }
    return {}


def _fetch_cashflow_ttm(symbol: str) -> dict:
    # operatingCF TTM y CAPEX TTM para aproximar FCF
    data = _http_get(
        f"cash-flow-statement-ttm/{symbol}",
        cache_ns="cf_ttm",
        cache_key=symbol,
        max_age_sec=24 * 3600,
    )
    if isinstance(data, list) and data:
        d = data[0]
        ocf = _to_float(d.get("operatingCashFlowTTM"))
        capex = _to_float(d.get("capitalExpenditureTTM"))
        fcf = None
        if ocf is not None and capex is not None:
            fcf = ocf - capex
        return {
            "operating_cf": ocf,
            "capex": capex,
            "fcf": fcf,
        }
    return {}


def fetch_fundamentals_batch(symbols: List[str]) -> pd.DataFrame:
    """
    Devuelve DataFrame con columnas:
    symbol, ev_ebitda, pb, pe, roe, roic, gross_margin, fcf, operating_cf
    """
    symbols = [s for s in (symbols or []) if isinstance(s, str)]
    out_rows: List[Dict] = []

    for s in symbols:
        row = {"symbol": s}
        try:
            row.update(_fetch_ratios(s))
        except Exception:
            pass
        try:
            row.update(_fetch_key_metrics(s))
        except Exception:
            pass
        try:
            row.update(_fetch_cashflow_ttm(s))
        except Exception:
            pass
        out_rows.append(row)

    df = pd.DataFrame(out_rows).drop_duplicates(subset=["symbol"])
    return df


# -----------------------------------------------------------------------------
# Precios diarios
# -----------------------------------------------------------------------------
def fetch_prices_daily(symbol: str, lookback_days: int = 800) -> pd.DataFrame:
    """
    Devuelve OHLCV diario (DataFrame) con al menos 'date' y 'close'.
    Intenta traer ~lookback_days hacia atrás.
    """
    # FMP historical-price-full admite "serietype=line" para más compacto
    params = {
        "serietype": "line"
    }
    data = _http_get(
        f"historical-price-full/{symbol}",
        params=params,
        cache_ns="prices",
        cache_key=f"{symbol}_line",
        max_age_sec=6 * 3600,
    )

    hist = (data or {}).get("historical") or []
    if not hist:
        # fallback OHLC completo (más pesado)
        data2 = _http_get(
            f"historical-price-full/{symbol}",
            params={},
            cache_ns="prices",
            cache_key=f"{symbol}_full",
            max_age_sec=6 * 3600,
        )
        hist = (data2 or {}).get("historical") or []

    if not hist:
        return pd.DataFrame()

    df = pd.DataFrame(hist)
    # Asegurar columnas
    if "date" not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").dropna(subset=["date"]).reset_index(drop=True)

    # Nos quedamos con la ventana requerida (últimos lookback_days)
    if len(df) > lookback_days:
        df = df.iloc[-lookback_days:].copy()

    # Asegurar columnas mínimas
    if "close" not in df.columns:
        # si usamos serietype=line, tendremos "close"; si no, la traemos
        close_col = "close" if "close" in df.columns else None
        if close_col is None:
            return pd.DataFrame()

    return df[["date", "close"]].copy()


# -----------------------------------------------------------------------------
# Script de prueba
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("🧪 Testing data_fetcher...")
    print("API key set?" , bool(FMP_API_KEY))

    try:
        uni = fetch_screener(limit=20, mcap_min=2e9, volume_min=1_000_000)
        print(f"Screener size: {len(uni)}")
        print(uni.head())
    except Exception as e:
        print("Screener error:", e)

    try:
        if not uni.empty:
            sample = uni["symbol"].head(5).tolist()
            fund = fetch_fundamentals_batch(sample)
            print("Fundamentals cols:", fund.columns.tolist())
            print(fund.head())
    except Exception as e:
        print("Fundamentals error:", e)

    try:
        if not uni.empty:
            sym = uni["symbol"].iloc[0]
            px = fetch_prices_daily(sym, 800)
            print(sym, px.shape, px.head())
    except Exception as e:
        print("Prices error:", e)
