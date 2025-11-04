# data_fetcher.py
"""
Data Fetcher - Módulo limpio desde cero
========================================
Responsabilidad única: Descargar datos desde FMP con rate limiting y caché.
Variables de entorno:
- FMP_API_KEY (obligatoria)
"""

from __future__ import annotations
import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
FMP_API_KEY = os.getenv("FMP_API_KEY", "").strip()
BASE = "https://financialmodelingprep.com/api/v3"
CACHE_DIR = Path(".cache_fmp")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Ajusta a tu plan FMP (en seg por request)
_MIN_INTERVAL = 0.15  # ~6–7 req/seg (360–420 rpm)
_session = requests.Session()
_last_call = 0.0


# -----------------------------------------------------------------------------
# Utilidades de caché
# -----------------------------------------------------------------------------
def _cache_key(endpoint: str, params: Dict) -> Path:
    payload = json.dumps({"endpoint": endpoint, "params": params}, sort_keys=True)
    h = hashlib.sha256(payload.encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"

def _cache_get(endpoint: str, params: Dict, ttl: int | None) -> Optional[dict | list]:
    if ttl is None:
        return None
    fp = _cache_key(endpoint, params)
    if not fp.exists():
        return None
    age = time.time() - fp.stat().st_mtime
    if age > ttl:
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return None

def _cache_put(endpoint: str, params: Dict, data: dict | list) -> None:
    fp = _cache_key(endpoint, params)
    try:
        fp.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass

def clear_cache() -> None:
    for p in CACHE_DIR.glob("*.json"):
        p.unlink(missing_ok=True)


# -----------------------------------------------------------------------------
# Llamadas HTTP con rate limiting
# -----------------------------------------------------------------------------
def _rate_limit():
    global _last_call
    now = time.time()
    wait = _MIN_INTERVAL - (now - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()

def _http_get(endpoint: str, params: Optional[Dict] = None, ttl: int | None = 600) -> dict | list:
    """
    GET con rate limiting y caché por (endpoint, params, apikey)
    """
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY no configurada")

    params = dict(params or {})
    params["apikey"] = FMP_API_KEY

    # caché
    cached = _cache_get(endpoint, params, ttl)
    if cached is not None:
        return cached

    _rate_limit()
    url = f"{BASE}/{endpoint.lstrip('/')}"
    try:
        r = _session.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        _cache_put(endpoint, params, data)
        return data
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"FMP error: {e}") from e


# -----------------------------------------------------------------------------
# Screener
# -----------------------------------------------------------------------------
def fetch_screener(limit: int = 300, mcap_min: float = 2e9, volume_min: int = 1_000_000, use_cache: bool = True) -> pd.DataFrame:
    """
    Usa /stock-screener (FMP) y devuelve columnas:
      symbol, companyName, sector, market_cap, price, volume
    """
    params = {
        "limit": int(limit),
        "marketCapMoreThan": float(mcap_min),
        "volumeMoreThan": int(volume_min),
        # puedes agregar: "betaMoreThan", "country", etc.
    }
    ttl = 900 if use_cache else None
    data = _http_get("stock-screener", params=params, ttl=ttl)
    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame(columns=["symbol","sector","market_cap","price","volume"])

    df = pd.DataFrame(data)
    # normaliza nombres
    rename = {
        "marketCap": "market_cap",
        "companyName": "companyName",
        "sector": "sector",
        "price": "price",
        "volume": "volume",
        "symbol": "symbol",
    }
    for k, v in rename.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)

    keep = [c for c in ["symbol","companyName","sector","market_cap","price","volume"] if c in df.columns]
    df = df[keep].dropna(subset=["symbol"]).drop_duplicates("symbol").reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Fundamentales (TTM)
# -----------------------------------------------------------------------------
def _get_key_metrics_ttm(symbol: str, use_cache: bool = True) -> Dict:
    # ROIC, ROE, ratios TTM
    ttl = 3600 if use_cache else None
    data = _http_get(f"key-metrics-ttm/{symbol}", params={}, ttl=ttl)
    if isinstance(data, list) and data:
        return data[0]
    return {}

def _get_ratios_ttm(symbol: str, use_cache: bool = True) -> Dict:
    # Margen bruto, PE, PB
    ttl = 3600 if use_cache else None
    data = _http_get(f"ratios-ttm/{symbol}", params={}, ttl=ttl)
    if isinstance(data, list) and data:
        return data[0]
    return {}

def _get_enterprise_values(symbol: str, use_cache: bool = True) -> Dict:
    # EV/EBITDA TTM (a veces viene mejor en key-metrics-ttm; esto es backup)
    ttl = 3600 if use_cache else None
    data = _http_get(f"enterprise-values/{symbol}", params={"limit": 1}, ttl=ttl)
    if isinstance(data, list) and data:
        return data[0]
    return {}

def _get_cashflow_ttm(symbol: str, use_cache: bool = True) -> Dict:
    # OCF y FCF TTM
    ttl = 3600 if use_cache else None
    data = _http_get(f"cash-flow-statement-ttm/{symbol}", params={}, ttl=ttl)
    if isinstance(data, list) and data:
        return data[0]
    return {}

def fetch_fundamentals_batch(symbols: List[str], use_cache: bool = True) -> pd.DataFrame:
    """
    Devuelve DataFrame con columnas estandarizadas:
      symbol, ev_ebitda, pb, pe, roe, roic, gross_margin, fcf, operating_cf
    """
    rows: List[Dict] = []
    syms = [s for s in (symbols or []) if isinstance(s, str)]
    for sym in syms:
        sym = sym.strip().upper()
        try:
            km = _get_key_metrics_ttm(sym, use_cache)
            rt = _get_ratios_ttm(sym, use_cache)
            cf = _get_cashflow_ttm(sym, use_cache)

            # Fuentes:
            # - key-metrics-ttm: returnOnEquityTTM, returnOnInvestedCapitalTTM, enterpriseValueOverEBITDATTM
            # - ratios-ttm: priceEarningsRatioTTM, priceToBookRatioTTM, grossProfitMarginTTM
            # - cash-flow-statement-ttm: operatingCashFlowTTM, freeCashFlowTTM

            row = {
                "symbol": sym,
                "ev_ebitda": float(km.get("enterpriseValueOverEBITDATTM") or rt.get("enterpriseValueOverEBITDATTM") or "nan"),
                "pb": float(rt.get("priceToBookRatioTTM") or "nan"),
                "pe": float(rt.get("priceEarningsRatioTTM") or "nan"),
                "roe": float(km.get("returnOnEquityTTM") or rt.get("returnOnEquityTTM") or "nan"),
                "roic": float(km.get("returnOnInvestedCapitalTTM") or "nan"),
                "gross_margin": float(rt.get("grossProfitMarginTTM") or "nan"),
                "operating_cf": float(cf.get("operatingCashFlowTTM") or "nan"),
                "fcf": float(cf.get("freeCashFlowTTM") or "nan"),
            }
            rows.append(row)
        except Exception:
            # Si algo falla para un símbolo, empuja NaNs pero no detiene batch
            rows.append({
                "symbol": sym, "ev_ebitda": float("nan"), "pb": float("nan"), "pe": float("nan"),
                "roe": float("nan"), "roic": float("nan"), "gross_margin": float("nan"),
                "operating_cf": float("nan"), "fcf": float("nan"),
            })

    df = pd.DataFrame(rows).drop_duplicates("symbol")
    return df


# -----------------------------------------------------------------------------
# Precios diarios
# -----------------------------------------------------------------------------
def fetch_prices(symbol: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Devuelve DataFrame con columnas ['date','close'] en orden ascendente.
    Usa /historical-price-full?from=&to=&serietype=line
    """
    params = {"from": start, "to": end, "serietype": "line"}
    ttl = 3600 if use_cache else None
    data = _http_get(f"historical-price-full/{symbol}", params=params, ttl=ttl)
    hist = (data or {}).get("historical", [])
    if not hist:
        return pd.DataFrame(columns=["date","close"])

    df = pd.DataFrame(hist)
    # normalizar
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "close" not in df.columns:
        # FMP devuelve 'close' en este endpoint; por si acaso:
        for alt in ("adjClose", "Adj Close"):
            if alt in df.columns:
                df["close"] = df[alt]
                break
    df = df.dropna(subset=["date","close"]).sort_values("date")
    return df[["date","close"]].reset_index(drop=True)


# -----------------------------------------------------------------------------
# Prueba rápida
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("🧪 Testing data_fetcher...")

    try:
        scr = fetch_screener(limit=10, mcap_min=2e9, volume_min=1_000_000)
        print(f"Screener: {len(scr)}")
        print(scr.head())
    except Exception as e:
        print("Screener error:", e)

    try:
        if not scr.empty:
            syms = scr["symbol"].head(3).tolist()
            fund = fetch_fundamentals_batch(syms)
            print("\nFundamentals:")
            print(fund)
    except Exception as e:
        print("Fundamentals error:", e)

    try:
        if not scr.empty:
            sym = scr["symbol"].iloc[0]
            px = fetch_prices(sym, "2022-01-01", "2025-12-31")
            print("\nPrices:", sym, len(px))
            print(px.head())
    except Exception as e:
        print("Prices error:", e)
