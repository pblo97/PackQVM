# data_fetcher.py  (drop-in)

from __future__ import annotations
import os, json, time, hashlib, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import requests
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
# Try to get API key from Streamlit secrets first, then fall back to env variable
try:
    import streamlit as st
    FMP_API_KEY = st.secrets.get("FMP_API_KEY", "")
except:
    FMP_API_KEY = os.getenv("FMP_API_KEY", "")

FMP_API_KEY = FMP_API_KEY.strip()
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

# -----------------------------------------------------------------------------
# Financial Statements (Full Statements for detailed analysis)
# -----------------------------------------------------------------------------
def _fetch_income_statement(symbol: str, limit: int = 2) -> List[dict]:
    """
    Obtiene income statements históricos (últimos 'limit' períodos).
    Retorna lista de dicts con fecha y métricas de ingresos.
    """
    data = _http_get(
        f"income-statement/{symbol}",
        params={"limit": limit},
        cache_ns="income_stmt",
        cache_key=f"{symbol}_{limit}",
        max_age_sec=24 * 3600,
    )
    if isinstance(data, list):
        return data
    return []


def _fetch_balance_sheet(symbol: str, limit: int = 2) -> List[dict]:
    """
    Obtiene balance sheets históricos (últimos 'limit' períodos).
    Retorna lista de dicts con fecha y métricas de balance.
    """
    data = _http_get(
        f"balance-sheet-statement/{symbol}",
        params={"limit": limit},
        cache_ns="balance_sheet",
        cache_key=f"{symbol}_{limit}",
        max_age_sec=24 * 3600,
    )
    if isinstance(data, list):
        return data
    return []


def _fetch_cash_flow_statement(symbol: str, limit: int = 2) -> List[dict]:
    """
    Obtiene cash flow statements históricos (últimos 'limit' períodos).
    Retorna lista de dicts con fecha y métricas de flujo de efectivo.
    """
    data = _http_get(
        f"cash-flow-statement/{symbol}",
        params={"limit": limit},
        cache_ns="cashflow_stmt",
        cache_key=f"{symbol}_{limit}",
        max_age_sec=24 * 3600,
    )
    if isinstance(data, list):
        return data
    return []


def _calculate_piotroski_components(income: List[dict], balance: List[dict], cashflow: List[dict]) -> dict:
    """
    Calcula los componentes del Piotroski F-Score usando estados financieros completos.
    Requiere al menos 2 períodos (actual y anterior) para calcular deltas.

    Retorna dict con:
    - Cada componente del F-Score (9 checks)
    - F-Score total (0-9)
    - Métricas individuales usadas
    """
    result = {
        "piotroski_score": 0,
        "roa": None,
        "roa_positive": 0,
        "cfo_positive": 0,
        "delta_roa_positive": 0,
        "accruals_quality": 0,
        "delta_leverage": 0,
        "delta_liquidity": 0,
        "delta_shares": 0,
        "delta_gross_margin": 0,
        "delta_asset_turnover": 0,
    }

    if not income or not balance or not cashflow:
        return result

    # Necesitamos al menos 2 períodos para calcular deltas
    if len(income) < 2 or len(balance) < 2:
        # Solo calculamos checks que no requieren histórico
        curr_income = income[0]
        curr_balance = balance[0]
        curr_cf = cashflow[0] if cashflow else {}

        # 1. ROA > 0 (profitability)
        net_income = _to_float(curr_income.get("netIncome"))
        total_assets = _to_float(curr_balance.get("totalAssets"))
        if net_income and total_assets and total_assets > 0:
            roa = net_income / total_assets
            result["roa"] = roa
            result["roa_positive"] = 1 if roa > 0 else 0
            result["piotroski_score"] += result["roa_positive"]

        # 2. CFO > 0 (cash flow profitability)
        cfo = _to_float(curr_cf.get("operatingCashFlow"))
        if cfo and cfo > 0:
            result["cfo_positive"] = 1
            result["piotroski_score"] += 1

        return result

    # Tenemos histórico completo - calculamos todos los checks
    curr_income = income[0]
    prev_income = income[1]
    curr_balance = balance[0]
    prev_balance = balance[1]
    curr_cf = cashflow[0] if cashflow else {}
    prev_cf = cashflow[1] if len(cashflow) > 1 else {}

    # === PROFITABILITY (4 checks) ===

    # 1. ROA > 0
    net_income = _to_float(curr_income.get("netIncome"))
    total_assets = _to_float(curr_balance.get("totalAssets"))
    if net_income and total_assets and total_assets > 0:
        roa = net_income / total_assets
        result["roa"] = roa
        result["roa_positive"] = 1 if roa > 0 else 0
        result["piotroski_score"] += result["roa_positive"]

    # 2. CFO > 0
    cfo = _to_float(curr_cf.get("operatingCashFlow"))
    if cfo and cfo > 0:
        result["cfo_positive"] = 1
        result["piotroski_score"] += 1

    # 3. ΔROA > 0 (ROA improvement)
    prev_net_income = _to_float(prev_income.get("netIncome"))
    prev_total_assets = _to_float(prev_balance.get("totalAssets"))
    if all([net_income, total_assets, prev_net_income, prev_total_assets,
            total_assets > 0, prev_total_assets > 0]):
        prev_roa = prev_net_income / prev_total_assets
        if roa > prev_roa:
            result["delta_roa_positive"] = 1
            result["piotroski_score"] += 1

    # 4. Accruals < 0 (CFO > Net Income → quality of earnings)
    if net_income and cfo and total_assets and total_assets > 0:
        accruals = (net_income - cfo) / total_assets
        if accruals < 0:
            result["accruals_quality"] = 1
            result["piotroski_score"] += 1

    # === LEVERAGE/LIQUIDITY/SOURCE OF FUNDS (3 checks) ===

    # 5. Δ Long-term Debt / Assets < 0 (decreasing leverage)
    curr_ltd = _to_float(curr_balance.get("longTermDebt")) or 0
    prev_ltd = _to_float(prev_balance.get("longTermDebt")) or 0
    prev_assets = _to_float(prev_balance.get("totalAssets"))
    if total_assets and prev_assets and total_assets > 0 and prev_assets > 0:
        curr_leverage = curr_ltd / total_assets
        prev_leverage = prev_ltd / prev_assets
        if curr_leverage < prev_leverage:
            result["delta_leverage"] = 1
            result["piotroski_score"] += 1

    # 6. Δ Current Ratio > 0 (improving liquidity)
    curr_assets_current = _to_float(curr_balance.get("totalCurrentAssets"))
    curr_liab_current = _to_float(curr_balance.get("totalCurrentLiabilities"))
    prev_assets_current = _to_float(prev_balance.get("totalCurrentAssets"))
    prev_liab_current = _to_float(prev_balance.get("totalCurrentLiabilities"))

    if all([curr_assets_current, curr_liab_current, prev_assets_current, prev_liab_current,
            curr_liab_current > 0, prev_liab_current > 0]):
        curr_ratio = curr_assets_current / curr_liab_current
        prev_ratio = prev_assets_current / prev_liab_current
        if curr_ratio > prev_ratio:
            result["delta_liquidity"] = 1
            result["piotroski_score"] += 1

    # 7. No new equity issued (ΔShares <= 0)
    curr_shares = _to_float(curr_income.get("weightedAverageShsOut"))
    prev_shares = _to_float(prev_income.get("weightedAverageShsOut"))
    if curr_shares and prev_shares:
        if curr_shares <= prev_shares:
            result["delta_shares"] = 1
            result["piotroski_score"] += 1

    # === OPERATING EFFICIENCY (2 checks) ===

    # 8. Δ Gross Margin > 0
    curr_revenue = _to_float(curr_income.get("revenue"))
    curr_gross_profit = _to_float(curr_income.get("grossProfit"))
    prev_revenue = _to_float(prev_income.get("revenue"))
    prev_gross_profit = _to_float(prev_income.get("grossProfit"))

    if all([curr_revenue, curr_gross_profit, prev_revenue, prev_gross_profit,
            curr_revenue > 0, prev_revenue > 0]):
        curr_gm = curr_gross_profit / curr_revenue
        prev_gm = prev_gross_profit / prev_revenue
        if curr_gm > prev_gm:
            result["delta_gross_margin"] = 1
            result["piotroski_score"] += 1

    # 9. Δ Asset Turnover > 0
    if all([curr_revenue, total_assets, prev_revenue, prev_total_assets,
            total_assets > 0, prev_total_assets > 0]):
        curr_turnover = curr_revenue / total_assets
        prev_turnover = prev_revenue / prev_total_assets
        if curr_turnover > prev_turnover:
            result["delta_asset_turnover"] = 1
            result["piotroski_score"] += 1

    return result


def _calculate_advanced_metrics(income: List[dict], balance: List[dict], cashflow: List[dict], market_cap: Optional[float] = None) -> dict:
    """
    Calcula métricas financieras avanzadas:
    - ROIC (Return on Invested Capital)
    - FCF Yield (Free Cash Flow Yield)
    - ROA (Return on Assets)
    - ROE (Return on Equity)
    - Gross Margin
    - Operating Margin
    - Net Margin
    """
    result = {
        "roic": None,
        "fcf_yield": None,
        "roa": None,
        "roe": None,
        "gross_margin": None,
        "operating_margin": None,
        "net_margin": None,
        "fcf": None,
        "operating_cf": None,
    }

    if not income or not balance or not cashflow:
        return result

    curr_income = income[0]
    curr_balance = balance[0]
    curr_cf = cashflow[0] if cashflow else {}

    # Valores básicos
    net_income = _to_float(curr_income.get("netIncome"))
    revenue = _to_float(curr_income.get("revenue"))
    operating_income = _to_float(curr_income.get("operatingIncome"))
    gross_profit = _to_float(curr_income.get("grossProfit"))

    total_assets = _to_float(curr_balance.get("totalAssets"))
    total_equity = _to_float(curr_balance.get("totalStockholdersEquity"))
    total_debt = _to_float(curr_balance.get("totalDebt")) or 0

    operating_cf = _to_float(curr_cf.get("operatingCashFlow"))
    capex = abs(_to_float(curr_cf.get("capitalExpenditure")) or 0)

    # Free Cash Flow
    if operating_cf is not None:
        result["operating_cf"] = operating_cf
        fcf = operating_cf - capex
        result["fcf"] = fcf

        # FCF Yield (requiere market cap)
        if market_cap and market_cap > 0:
            result["fcf_yield"] = fcf / market_cap

    # ROIC = NOPAT / Invested Capital
    # NOPAT ≈ Operating Income * (1 - Tax Rate)
    # Invested Capital ≈ Total Equity + Total Debt
    if operating_income and total_equity:
        tax_expense = _to_float(curr_income.get("incomeTaxExpense")) or 0
        income_before_tax = _to_float(curr_income.get("incomeBeforeTax"))

        if income_before_tax and income_before_tax > 0:
            tax_rate = tax_expense / income_before_tax
        else:
            tax_rate = 0.21  # Tasa impositiva por defecto

        nopat = operating_income * (1 - tax_rate)
        invested_capital = total_equity + total_debt

        if invested_capital > 0:
            result["roic"] = nopat / invested_capital

    # ROA = Net Income / Total Assets
    if net_income and total_assets and total_assets > 0:
        result["roa"] = net_income / total_assets

    # ROE = Net Income / Total Equity
    if net_income and total_equity and total_equity > 0:
        result["roe"] = net_income / total_equity

    # Margins
    if revenue and revenue > 0:
        if gross_profit is not None:
            result["gross_margin"] = gross_profit / revenue
        if operating_income is not None:
            result["operating_margin"] = operating_income / revenue
        if net_income is not None:
            result["net_margin"] = net_income / revenue

    return result


def fetch_financial_statements_batch(symbols: List[str]) -> pd.DataFrame:
    """
    Descarga estados financieros completos y calcula métricas avanzadas
    incluyendo Piotroski Score, ROIC, FCF Yield, etc.

    Retorna DataFrame con:
    - symbol
    - piotroski_score (0-9)
    - Componentes del Piotroski (9 checks binarios)
    - roic, fcf_yield, roa, roe
    - gross_margin, operating_margin, net_margin
    - fcf, operating_cf
    """
    symbols = [s for s in (symbols or []) if isinstance(s, str)]
    out_rows: List[Dict] = []

    for s in symbols:
        row = {"symbol": s}

        try:
            # Descargar estados financieros (últimos 2 años para calcular deltas)
            income = _fetch_income_statement(s, limit=2)
            balance = _fetch_balance_sheet(s, limit=2)
            cashflow = _fetch_cash_flow_statement(s, limit=2)

            # Calcular Piotroski Score y componentes
            piotroski = _calculate_piotroski_components(income, balance, cashflow)
            row.update(piotroski)

            # Calcular métricas avanzadas
            advanced = _calculate_advanced_metrics(income, balance, cashflow)
            row.update(advanced)

        except Exception as e:
            print(f"⚠️  Error fetching {s}: {e}")
            pass

        out_rows.append(row)

    df = pd.DataFrame(out_rows).drop_duplicates(subset=["symbol"])
    return df


def fetch_fundamentals_batch(symbols: List[str], use_full_statements: bool = False) -> pd.DataFrame:
    """
    Devuelve DataFrame con columnas:
    symbol, ev_ebitda, pb, pe, roe, roic, gross_margin, fcf, operating_cf

    Si use_full_statements=True, descarga estados financieros completos y calcula
    Piotroski Score y métricas avanzadas. Esto es más lento pero más completo.
    """
    symbols = [s for s in (symbols or []) if isinstance(s, str)]

    if use_full_statements:
        # Usar estados financieros completos (más completo pero más lento)
        return fetch_financial_statements_batch(symbols)

    # Usar TTM endpoints (más rápido pero menos completo)
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
                # 👇 ROA directo desde ratios-ttm (esto habilita el F-score local aunque falle basics)
                "roa": _safe_float(rt.get("returnOnAssetsTTM")),
                # 👇 TTM cash flow (si falla, quedará NaN y lo veremos en el debug)
                "operating_cf": _safe_float(cf.get("operatingCashFlowTTM") or cf.get("netCashProvidedByOperatingActivities")),
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

def fetch_financial_scores(symbols: Iterable[str], use_cache: bool = True) -> pd.DataFrame:
    """
    Intenta primero el endpoint batched:
      /stable/financial-scores?symbol=AAPL,MSFT,...
    Si no trae nada, cae al endpoint por-símbolo:
      /financial-scores?symbol=SYMB

    Devuelve SIEMPRE: ['symbol','piotroskiScore','altmanZScore','date? (si hubo)']
    """
    syms = [str(s).strip().upper() for s in symbols if isinstance(s, str)]
    if not syms:
        return pd.DataFrame(columns=["symbol","piotroskiScore","altmanZScore","date"])

    ttl = 900 if use_cache else None

    # -------- intento batched --------
    try:
        out = []
        chunk = 100
        for i in range(0, len(syms), chunk):
            seg = syms[i:i+chunk]
            data = _http_get("stable/financial-scores", {"symbol": ",".join(seg)}, ttl=ttl)
            if isinstance(data, list) and data:
                out.extend(data)
        if out:
            df = pd.DataFrame(out)
            df["symbol"] = df.get("symbol", pd.NA).astype(str).str.strip().str.upper()
            for c in ["piotroskiScore","altmanZScore"]:
                df[c] = pd.to_numeric(df.get(c), errors="coerce")
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            else:
                df["date"] = pd.NaT
            df = df.sort_values(["symbol","date"], ascending=[True, False]).drop_duplicates("symbol", keep="first")
            keep = ["symbol","piotroskiScore","altmanZScore","date"]
            for k in keep:
                if k not in df.columns:
                    df[k] = pd.NA
            return df[keep]
    except Exception:
        pass

    # -------- fallback por símbolo --------
    rows = []
    for s in syms:
        try:
            d = _http_get("financial-scores", {"symbol": s}, ttl=ttl)
            if isinstance(d, list) and d and isinstance(d[0], dict):
                d = d[0]
            elif not isinstance(d, dict):
                d = {}
            rows.append({
                "symbol": s,
                "piotroskiScore": pd.to_numeric(d.get("piotroskiScore"), errors="coerce"),
                "altmanZScore": pd.to_numeric(d.get("altmanZScore"), errors="coerce"),
                "date": pd.NaT,
            })
        except Exception:
            rows.append({"symbol": s, "piotroskiScore": pd.NA, "altmanZScore": pd.NA, "date": pd.NaT})

    return pd.DataFrame(rows).drop_duplicates("symbol")


# data_fetcher.py
from typing import Iterable, List, Dict
import pandas as pd

def fetch_financial_basics(
    symbols: Iterable[str],
    period: str = "annual",   # "annual" | "quarter"
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Devuelve SIEMPRE columnas:
      symbol, revenue, gross_margin, net_income, total_assets,
      operating_cf, capex, fcf, current_ratio, long_term_debt,
      shares_outstanding, asset_turnover, roa
    """
    syms = [str(s).strip().upper() for s in symbols if isinstance(s, str)]
    cols = [
        "symbol","revenue","gross_margin","net_income","total_assets",
        "operating_cf","capex","fcf","current_ratio","long_term_debt",
        "shares_outstanding","asset_turnover","roa"
    ]
    if not syms:
        return pd.DataFrame(columns=cols)

    ttl = 900 if use_cache else None
    rows: List[Dict] = []

    def _num(d: dict, *keys):
        for k in keys:
            if k in d and d[k] is not None:
                return pd.to_numeric(d.get(k), errors="coerce")
        return pd.NA

    for s in syms:
        try:
            # -------- 1) Trae ANUAL; si viene vacío, intenta QUARTER --------
            is_ = _http_get("income-statement", {"symbol": s, "period": period, "limit": 1}, ttl=ttl)
            cf_ = _http_get("cash-flow-statement", {"symbol": s, "period": period, "limit": 1}, ttl=ttl)
            bs_ = _http_get("balance-sheet-statement", {"symbol": s, "period": period, "limit": 1}, ttl=ttl)

            if (not is_ or not isinstance(is_, list)) and period == "annual":
                is_ = _http_get("income-statement", {"symbol": s, "period": "quarter", "limit": 1}, ttl=ttl)
            if (not cf_ or not isinstance(cf_, list)) and period == "annual":
                cf_ = _http_get("cash-flow-statement", {"symbol": s, "period": "quarter", "limit": 1}, ttl=ttl)
            if (not bs_ or not isinstance(bs_, list)) and period == "annual":
                bs_ = _http_get("balance-sheet-statement", {"symbol": s, "period": "quarter", "limit": 1}, ttl=ttl)

            is_d = is_[0] if isinstance(is_, list) and is_ else {}
            cf_d = cf_[0] if isinstance(cf_, list) and cf_ else {}
            bs_d = bs_[0] if isinstance(bs_, list) and bs_ else {}

            # -------- Income Statement --------
            revenue      = _num(is_d, "revenue", "totalRevenue")
            gross_profit = _num(is_d, "grossProfit")
            net_income   = _num(is_d, "netIncome", "netIncomeApplicableToCommonShares")
            shs_avg      = _num(is_d, "weightedAverageShsOut", "weightedAverageShsOutDil", "weightedAverageShares")

            # -------- Cash Flow --------
            ocf = _num(cf_d, "netCashProvidedByOperatingActivities", "operatingCashFlow")
            capex = _num(cf_d, "capitalExpenditure", "purchaseOfPropertyPlantAndEquipment")
            # Nota: en FMP capex suele venir NEGATIVO; fcf = ocf - capex ya lo maneja
            fcf = (ocf - capex) if pd.notna(ocf) and pd.notna(capex) else pd.NA

            # -------- Balance Sheet --------
            total_assets   = _num(bs_d, "totalAssets")
            current_assets = _num(bs_d, "totalCurrentAssets")
            current_liabs  = _num(bs_d, "totalCurrentLiabilities")
            long_debt      = _num(bs_d, "longTermDebt", "longTermDebtNoncurrent")
            shs_out        = _num(bs_d, "commonStockSharesOutstanding", "commonSharesOutstanding")

            # -------- Derivados --------
            gross_margin   = (gross_profit / revenue) if pd.notna(gross_profit) and pd.notna(revenue) and revenue != 0 else pd.NA
            current_ratio  = (current_assets / current_liabs) if pd.notna(current_assets) and pd.notna(current_liabs) and current_liabs != 0 else pd.NA
            asset_turnover = (revenue / total_assets) if pd.notna(revenue) and pd.notna(total_assets) and total_assets != 0 else pd.NA
            roa            = (net_income / total_assets) if pd.notna(net_income) and pd.notna(total_assets) and total_assets != 0 else pd.NA

            rows.append({
                "symbol": s,
                "revenue": revenue,
                "gross_margin": gross_margin,
                "net_income": net_income,
                "total_assets": total_assets,
                "operating_cf": ocf,
                "capex": capex,
                "fcf": fcf,
                "current_ratio": current_ratio,
                "long_term_debt": long_debt,
                "shares_outstanding": shs_avg if pd.notna(shs_avg) else shs_out,
                "asset_turnover": asset_turnover,
                "roa": roa,
            })
        except Exception:
            rows.append({"symbol": s})

    out = pd.DataFrame(rows).drop_duplicates("symbol")

    # ---------- Debug de cobertura (visible en Tab 4) ----------
    try:
        import streamlit as st
        if isinstance(out, pd.DataFrame) and not out.empty:
            cov = out[["net_income","total_assets","operating_cf","fcf","current_ratio","long_term_debt","roa"]].notna().sum().to_dict()
            st.session_state["__basics_coverage__"] = {"count": int(len(out)), "nonnull": cov}
    except Exception:
        pass

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

    # Test 1: TTM fundamentals (rápido)
    try:
        if not uni.empty:
            sample = uni["symbol"].head(3).tolist()
            print(f"\n📊 Testing TTM fundamentals for {sample}...")
            fund = fetch_fundamentals_batch(sample, use_full_statements=False)
            print("Fundamentals cols:", fund.columns.tolist())
            print(fund.head())
    except Exception as e:
        print("Fundamentals error:", e)

    # Test 2: Full statements + Piotroski Score (más lento)
    try:
        if not uni.empty:
            sample = uni["symbol"].head(2).tolist()
            print(f"\n📊 Testing FULL financial statements + Piotroski for {sample}...")
            full_fund = fetch_financial_statements_batch(sample)
            print("\n✅ Full fundamentals cols:", full_fund.columns.tolist())

            if not full_fund.empty:
                print("\n📈 Financial Metrics:")
                for col in ["piotroski_score", "roic", "fcf_yield", "roa", "roe", "gross_margin"]:
                    if col in full_fund.columns:
                        print(f"  {col}: {full_fund[col].tolist()}")

                print("\n🎯 Piotroski Components:")
                piotroski_cols = [c for c in full_fund.columns if 'positive' in c or 'delta_' in c or 'accruals' in c]
                for col in piotroski_cols:
                    print(f"  {col}: {full_fund[col].tolist()}")
    except Exception as e:
        print("Full statements error:", e)

    # Test 3: Prices
    try:
        if not uni.empty:
            sym = uni["symbol"].iloc[0]
            px = fetch_prices_daily(sym, 800)
            print(f"\n📊 Prices for {sym}: {px.shape}")
            print(px.head())
    except Exception as e:
        print("Prices error:", e)


# data_fetcher.py



