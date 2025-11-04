# data_fetcher.py  (drop-in)

from __future__ import annotations
import os, json, time, hashlib, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd

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
    exchange: Optional[str] = "NYSE,NASDAQ",   # <- ayuda a que TTM exista
    country: Optional[str] = None,
) -> pd.DataFrame:
    """
    Usa /stock-screener y devuelve:
      symbol, companyName, sector, market_cap, price, volume
    """
    params = {
        "limit": int(limit),
        "marketCapMoreThan": float(mcap_min),
        "volumeMoreThan": int(volume_min),
    }
    if exchange:
        params["exchange"] = exchange
    if country:
        params["country"] = country

    ttl = 900 if use_cache else None
    data = _http_get("stock-screener", params=params, ttl=ttl)
    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame(columns=["symbol","companyName","sector","market_cap","price","volume"])

    df = pd.DataFrame(data)
    rename = {
        "symbol": "symbol",
        "companyName": "companyName",
        "sector": "sector",
        "marketCap": "market_cap",
        "price": "price",
        "volume": "volume",
    }
    for src, dst in rename.items():
        if src in df.columns:
            df.rename(columns={src: dst}, inplace=True)

    keep = [c for c in ["symbol","companyName","sector","market_cap","price","volume"] if c in df.columns]
    df = (
        df[keep]
        .dropna(subset=["symbol"])
        .drop_duplicates("symbol")
        .reset_index(drop=True)
    )
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    return df


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
def fetch_prices(
    limit: int = 300,
    mcap_min: float = 2e9,
    volume_min: int = 1_000_000,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Usa /stock-screener (FMP) y devuelve SIEMPRE columnas normalizadas:
      ['symbol','companyName','sector','market_cap','price','volume']

    - sector: mapeado desde varias variantes posibles; fallback "Unknown"
    - symbol: uppercase y sin espacios
    """
    params = {
        "limit": int(limit),
        "marketCapMoreThan": float(mcap_min),
        "volumeMoreThan": int(volume_min),
        # puedes agregar filtros opcionales: betaMoreThan, country, exchange, etc.
    }
    ttl = 900 if use_cache else None
    data = _http_get("stock-screener", params=params, ttl=ttl)

    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame(columns=["symbol","companyName","sector","market_cap","price","volume"])

    df = pd.DataFrame(data)

    # --- Normalización de nombres base (cuando existen) ---
    # FMP suele traer marketCap / price / volume / companyName / sector
    rename = {
        "marketCap": "market_cap",
        "companyName": "companyName",
        "price": "price",
        "volume": "volume",
        "symbol": "symbol",
        # "sector": "sector"  # se maneja aparte con fallback inteligente
    }
    for src, dst in rename.items():
        if src in df.columns and src != dst:
            df.rename(columns={src: dst}, inplace=True)

    # --- Symbol limpio ---
    if "symbol" in df.columns:
        df["symbol"] = (
            df["symbol"]
            .astype(str)
            .str.strip()
            .str.upper()
        )
    else:
        # Sin symbol no hay nada que hacer…
        return pd.DataFrame(columns=["symbol","companyName","sector","market_cap","price","volume"])

    # --- Sector robusto: tomar primera columna disponible ---
    # Orden de preferencia: 'sector' -> 'sectorName' -> 'industry' -> 'industryTitle' -> 'subSector'
    sector_col = None
    for cand in ["sector", "sectorName", "industry", "industryTitle", "subSector"]:
        if cand in df.columns:
            sector_col = cand
            break

    if sector_col is None:
        # crea columna sector si no existe ninguna fuente
        df["sector"] = "Unknown"
    else:
        # copia y normaliza a string no nulo
        df["sector"] = (
            df[sector_col]
            .astype(str)
            .str.strip()
            .replace({"": "Unknown", "None": "Unknown", "nan": "Unknown"})
            .fillna("Unknown")
        )

    # --- market_cap / price / volume numéricos ---
    for col, target in [
        ("market_cap", "market_cap"),
        ("price", "price"),
        ("volume", "volume"),
    ]:
        if col in df.columns:
            df[target] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[target] = pd.NA

    # --- companyName opcional pero útil para UI ---
    if "companyName" not in df.columns:
        # algunos endpoints traen 'companyName', otros 'company'
        if "company" in df.columns:
            df["companyName"] = df["company"]
        else:
            df["companyName"] = pd.NA

    # --- Selección y limpieza final ---
    keep = ["symbol", "companyName", "sector", "market_cap", "price", "volume"]
    out = (
        df[keep]
        .dropna(subset=["symbol"])
        .drop_duplicates("symbol")
        .reset_index(drop=True)
    )

    return out



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
