# data_fetcher.py  (drop-in)

from __future__ import annotations
import os, json, time, hashlib, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
FMP_API_KEY = os.getenv("FMP_API_KEY", "").strip()
BASE = "https://financialmodelingprep.com/api/v3"

# Unifica caché (puedes setear FMP_CACHE_DIR si quieres otro path)
CACHE_DIR = Path(os.getenv("FMP_CACHE_DIR", ".cache_fmp"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Ajusta a tu plan FMP
_MIN_INTERVAL = 0.15          # espaciado base entre reqs
_MAX_RETRIES  = 3             # reintentos en 429/5xx
_BACKOFF0     = 0.6           # backoff exponencial inicial

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
    try:
        _cache_key(endpoint, params).write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass

def clear_cache() -> None:
    for p in CACHE_DIR.glob("*.json"):
        p.unlink(missing_ok=True)


# -----------------------------------------------------------------------------
# Rate limit + backoff
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
    GET con rate limiting, backoff (429/5xx) y caché.
    """
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY no configurada")

    params = dict(params or {})
    params["apikey"] = FMP_API_KEY

    # caché
    cached = _cache_get(endpoint, params, ttl)
    if cached is not None:
        return cached

    url = f"{BASE}/{endpoint.lstrip('/')}"
    backoff = _BACKOFF0
    for att in range(_MAX_RETRIES):
        _rate_limit()
        try:
            r = _session.get(url, params=params, timeout=30)
            # Si hay 429/5xx, backoff
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff *= 2.0
                continue
            r.raise_for_status()
            data = r.json()
            _cache_put(endpoint, params, data)
            return data
        except requests.exceptions.RequestException as e:
            if att < _MAX_RETRIES - 1:
                time.sleep(backoff)
                backoff *= 2.0
                continue
            raise RuntimeError(f"FMP error: {e}") from e

    return []  # fallback


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def _sanitize_symbol(sym: str) -> List[str]:
    """
    Devuelve posibles variantes que FMP acepta mejor para TTM.
    Ej: 'BRK.B' -> ['BRK.B', 'BRK-B']; 'RDS.A' -> ['RDS.A','RDS-A'].
    """
    s = (sym or "").strip().upper()
    cands = [s]
    if "." in s:
        cands.append(s.replace(".", "-"))
    if "-" in s:
        cands.append(s.replace("-", "."))
    # quitar sufijos de clase extraños si todo falla:
    if "." in s:
        base = s.split(".")[0]
        cands.append(base)
    return list(dict.fromkeys(cands))  # unique, keep order


# -----------------------------------------------------------------------------
# Screener
# -----------------------------------------------------------------------------
def fetch_screener(
    limit: int = 300,
    mcap_min: float = 2e9,
    volume_min: int = 1_000_000,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Devuelve SIEMPRE columnas normalizadas:
      ['symbol','companyName','sector','market_cap','price','volume']
    """
    params = {
        "limit": int(limit),
        "marketCapMoreThan": float(mcap_min),
        "volumeMoreThan": int(volume_min),
    }
    ttl = 900 if use_cache else None
    data = _http_get("stock-screener", params=params, ttl=ttl)

    cols_final = ["symbol", "companyName", "sector", "market_cap", "price", "volume"]
    if not isinstance(data, list) or not data:
        return pd.DataFrame(columns=cols_final)

    df = pd.DataFrame(data)

    # Normalización básica
    rename = {
        "marketCap": "market_cap",
        "companyName": "companyName",
        "company": "companyName",
        "price": "price",
        "volume": "volume",
        "symbol": "symbol",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "symbol" not in df.columns:
        return pd.DataFrame(columns=cols_final)

    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df = df[df["symbol"].ne("")].drop_duplicates("symbol")

    # Sector robusto: sector → sectorName → industry → industryTitle → subSector
    def _first_nonnull(*cols):
        for c in cols:
            if c in df.columns:
                return (
                    df[c].astype(str)
                        .str.strip()
                        .replace({"": "Unknown", "None": "Unknown", "nan": "Unknown"})
                )
        return None

    sec_series = _first_nonnull("sector", "sectorName", "industry", "industryTitle", "subSector")
    if sec_series is None:
        df["sector"] = "Unknown"
    else:
        df["sector"] = sec_series.fillna("Unknown")

    # numéricos
    for c in ("market_cap", "price", "volume"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    if "companyName" not in df.columns:
        df["companyName"] = pd.NA

    out = (
        df[["symbol", "companyName", "sector", "market_cap", "price", "volume"]]
        .dropna(subset=["symbol"])
        .drop_duplicates("symbol")
        .reset_index(drop=True)
    )
    out["sector"] = out["sector"].astype(str).str.strip().replace({"": "Unknown"}).fillna("Unknown")
    return out


# -----------------------------------------------------------------------------
# Fundamentales (TTM)
# -----------------------------------------------------------------------------
def _get_key_metrics_ttm(symbol: str, use_cache: bool = True) -> Dict:
    ttl = 3600 if use_cache else None
    # probar variantes del símbolo
    for candidate in _sanitize_symbol(symbol):
        data = _http_get(f"key-metrics-ttm/{candidate}", params={}, ttl=ttl)
        if isinstance(data, list) and data:
            return data[0]
    return {}

def _get_ratios_ttm(symbol: str, use_cache: bool = True) -> Dict:
    ttl = 3600 if use_cache else None
    for candidate in _sanitize_symbol(symbol):
        data = _http_get(f"ratios-ttm/{candidate}", params={}, ttl=ttl)
        if isinstance(data, list) and data:
            return data[0]
    return {}

def _get_cashflow_ttm(symbol: str, use_cache: bool = True) -> Dict:
    ttl = 3600 if use_cache else None
    for candidate in _sanitize_symbol(symbol):
        data = _http_get(f"cash-flow-statement-ttm/{candidate}", params={}, ttl=ttl)
        if isinstance(data, list) and data:
            return data[0]
    return {}

def fetch_fundamentals_batch(symbols: List[str], use_cache: bool = True, debug: bool = False) -> pd.DataFrame:
    """
    Devuelve columnas:
      symbol, ev_ebitda, pb, pe, roe, roic, gross_margin, fcf, operating_cf
    """
    rows: List[Dict] = []
    syms = [s for s in (symbols or []) if isinstance(s, str)]
    for sym in syms:
        s = sym.strip().upper()
        try:
            km = _get_key_metrics_ttm(s, use_cache)
            rt = _get_ratios_ttm(s, use_cache)
            cf = _get_cashflow_ttm(s, use_cache)

            row = {
                "symbol": s,
                "ev_ebitda": _safe_float(km.get("enterpriseValueOverEBITDATTM") or rt.get("enterpriseValueOverEBITDATTM")),
                "pb": _safe_float(rt.get("priceToBookRatioTTM")),
                "pe": _safe_float(rt.get("priceEarningsRatioTTM")),
                "roe": _safe_float(km.get("returnOnEquityTTM") or rt.get("returnOnEquityTTM")),
                "roic": _safe_float(km.get("returnOnInvestedCapitalTTM")),
                "gross_margin": _safe_float(rt.get("grossProfitMarginTTM")),
                "operating_cf": _safe_float(cf.get("operatingCashFlowTTM")),
                "fcf": _safe_float(cf.get("freeCashFlowTTM")),
            }
            rows.append(row)
        except Exception:
            rows.append({
                "symbol": s, "ev_ebitda": float("nan"), "pb": float("nan"), "pe": float("nan"),
                "roe": float("nan"), "roic": float("nan"), "gross_margin": float("nan"),
                "operating_cf": float("nan"), "fcf": float("nan"),
            })

    df = pd.DataFrame(rows).drop_duplicates("symbol")

    if debug and not df.empty:
        numeric_cols = ["ev_ebitda","pb","pe","roe","roic","gross_margin","operating_cf","fcf"]
        coverage = df[numeric_cols].notna().sum().to_dict()
        total = len(df)
        print("Fundamentals coverage:", {k: f"{v}/{total}" for k, v in coverage.items()})
    return df


# -----------------------------------------------------------------------------
# Precios diarios
# -----------------------------------------------------------------------------
def fetch_prices(symbol: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Devuelve ['date','close'] ascendente usando historical-price-full (serietype=line)
    """
    params = {"from": start, "to": end, "serietype": "line"}
    ttl = 3600 if use_cache else None

    # probar variantes del símbolo aquí también por si acaso
    hist = []
    for candidate in _sanitize_symbol(symbol):
        data = _http_get(f"historical-price-full/{candidate}", params=params, ttl=ttl)
        hist = (data or {}).get("historical", [])
        if hist:
            break

    if not hist:
        return pd.DataFrame(columns=["date","close"])

    df = pd.DataFrame(hist)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "close" not in df.columns:
        for alt in ("adjClose", "Adj Close"):
            if alt in df.columns:
                df["close"] = df[alt]
                break
    df = df.dropna(subset=["date","close"]).sort_values("date")
    return df[["date","close"]].reset_index(drop=True)

def fetch_financial_scores(symbols: list[str], use_cache: bool = True) -> pd.DataFrame:
    """
    Llama a FMP /stable/financial-scores y retorna SIEMPRE columnas normalizadas:
      ['symbol','altmanZScore','piotroskiScore','date']
    - Si hay múltiples filas por símbolo, conserva la MÁS RECIENTE por 'date'
    - 'symbol' upper/strip
    """
    if not symbols:
        return pd.DataFrame(columns=["symbol","altmanZScore","piotroskiScore","date"])

    # FMP acepta múltiples símbolos en una sola query (separa por coma).
    # Para listas largas: chunkea (p.ej., 100 por request).
    out = []
    chunk_size = 100
    ttl = 900 if use_cache else None
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i+chunk_size]
        params = {"symbol": ",".join(chunk)}
        data = _http_get("stable/financial-scores", params=params, ttl=ttl)
        if isinstance(data, list) and data:
            out.extend(data)

    if not out:
        return pd.DataFrame(columns=["symbol","altmanZScore","piotroskiScore","date"])

    df = pd.DataFrame(out)

    # Normalizar columnas esperadas
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    else:
        df["symbol"] = pd.NA

    # Coerce numéricas
    for col in ["altmanZScore", "piotroskiScore"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.NA

    # Fecha (si existe)
    if "date" not in df.columns:
        df["date"] = pd.NaT
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Mantener SOLO la fila más reciente por símbolo
    df = df.sort_values(["symbol","date"], ascending=[True, False])
    df = df.drop_duplicates("symbol", keep="first").reset_index(drop=True)

    # Solo columnas útiles (con default si faltan)
    keep = ["symbol","altmanZScore","piotroskiScore","date"]
    for k in keep:
        if k not in df.columns:
            df[k] = pd.NA

    return df[keep]

# -----------------------------------------------------------------------------
# Prueba rápida
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("🧪 Testing data_fetcher...")

    try:
        scr = fetch_screener(limit=20, mcap_min=2e9, volume_min=1_000_000, exchange="NYSE,NASDAQ")
        print(f"Screener: {len(scr)}"); print(scr.head())
    except Exception as e:
        print("Screener error:", e)

    try:
        if not scr.empty:
            syms = scr["symbol"].head(5).tolist()
            fund = fetch_fundamentals_batch(syms, debug=True)
            print("\nFundamentals:"); print(fund.head())
    except Exception as e:
        print("Fundamentals error:", e)

    try:
        if not scr.empty:
            sym = scr["symbol"].iloc[0]
            px = fetch_prices(sym, "2022-01-01", "2025-12-31")
            print("\nPrices:", sym, len(px)); print(px.head())
    except Exception as e:
        print("Prices error:", e)
