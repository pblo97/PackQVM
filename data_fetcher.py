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
