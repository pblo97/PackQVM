# qvm_trend/fundamentals.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
import concurrent.futures as cf
import time
import math

import numpy as np
import pandas as pd

# HTTP común (robusto, con rate limit/backoff) provisto en data_io.py
from data_io import _http_get

# Cache opcional (si no está, definimos no-ops)
try:
    from cache_io import save_df, load_df
except Exception:
    def save_df(df: pd.DataFrame, key: str):
        return
    def load_df(key: str) -> Optional[pd.DataFrame]:
        return None

# ======================================================================
# Helpers genéricos / numéricos robustos
# ======================================================================

CAP_Z = 3.0  # límite de seguridad para z-scores

def _first_obj(x):
    """Devuelve el primer objeto si es lista; si es dict lo devuelve; si no, {}."""
    if isinstance(x, list):
        return x[0] if x else {}
    return x if isinstance(x, dict) else {}

def _safe_float(x):
    try:
        if x in ("", None):
            return None
        return float(x)
    except Exception:
        return None

def _yr_series(items, key):
    """Convierte list[dict] anual/quarter en lista de (fecha, valor) con coerción numérica."""
    out = []
    for it in (items or []):
        d = it.get("date")
        v = _safe_float(it.get(key))
        if d and v is not None:
            out.append((pd.to_datetime(d), v))
    out.sort(key=lambda z: z[0])
    return out

def _to_float(s: pd.Series | np.ndarray | None) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    s = pd.to_numeric(s, errors="coerce")
    return s.astype(float)

def _winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    s = _to_float(s)
    if s.notna().sum() < 3 or p <= 0:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)

def _zscore(s: pd.Series) -> pd.Series:
    s = _to_float(s)
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return (s - mu) / sd

def _safe_div(a, b) -> pd.Series:
    a = _to_float(a)
    b = _to_float(b)
    out = a.div(b)
    return out.replace([np.inf, -np.inf], np.nan)

def _rank_pct(s: pd.Series) -> pd.Series:
    s = _to_float(s)
    return s.rank(pct=True, method="average")

def _winsor(s: pd.Series, p: float = 0.01) -> pd.Series:
    # Alias interno para módulos que usaban _winsor en vez de _winsorize
    return _winsorize(s, p)

def _first_num(d: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Devuelve la primera columna numérica disponible entre 'candidates' (merge de valores)."""
    out = pd.Series(np.nan, index=d.index, dtype=float)
    for c in candidates:
        if c in d.columns:
            out = out.fillna(pd.to_numeric(d[c], errors="coerce"))
    return out

def _num_or_nan(d: pd.DataFrame, col: str) -> pd.Series:
    """Versión simple: toma una sola columna si existe, NaN si no."""
    return pd.to_numeric(d[col], errors="coerce") if col in d.columns else pd.Series(np.nan, index=d.index, dtype=float)

# ======================================================================
# Intangibles / I+D
# ======================================================================

def capitalize_rd(df: pd.DataFrame, rd_col="rd_expense_ttm", amort_years: int = 3) -> pd.DataFrame:
    """
    Capitaliza I+D (80%) y genera:
      - rd_asset (activo intangible por I+D)
      - op_income_xrd (EBIT operativo ajustado + amort. I+D)
      - assets_xrd (activos + rd_asset)
    Requiere columnas:
      rd_expense_ttm, operating_income_ttm, total_assets_ttm
    """
    out = df.copy()
    needed = {rd_col, "operating_income_ttm", "total_assets_ttm"}
    if not needed.issubset(out.columns):
        return out.assign(
            rd_asset=np.nan,
            op_income_xrd=np.nan,
            assets_xrd=out.get("total_assets_ttm", np.nan),
        )

    for col in [rd_col, "operating_income_ttm", "total_assets_ttm"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    rd = out[rd_col].fillna(0.0)
    cap_ratio = 0.80
    rd_asset = cap_ratio * rd * amort_years
    amort = rd_asset / amort_years

    out["rd_asset"] = rd_asset
    out["op_income_xrd"] = out["operating_income_ttm"].fillna(0) + amort
    out["assets_xrd"] = out["total_assets_ttm"].fillna(0) + rd_asset
    return out

# ======================================================================
# Value y Quality (growth/intangible-aware)
# ======================================================================

def value_growth_aware(df: pd.DataFrame) -> pd.Series:
    """
    Value “growth-aware”:
      40% EV/EBITDA NTM (invertido)
      30% EV/Gross Profit TTM (invertido)
      30% EV/Sales NTM penalizado por Capex/Sales (invertido)
    Overrides:
      +boost si FCF_yield_5y (ajustada por SBC) está en top quintil sectorial
    Requiere: ev, ebitda_ntm, gross_profit_ttm, sales_ntm, capex_ttm, sbc_ttm
             y opcionalmente fcf_5y_median (si no, se aproxima con fcf_ttm)
    """
    out = df.copy()

    # Forzar numérico en columnas usadas
    for col in ["ev","ebitda_ntm","gross_profit_ttm","sales_ntm",
                "capex_ttm","sbc_ttm","fcf_ttm","fcf_5y_median"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    ev = out.get("ev")
    gp = out.get("gross_profit_ttm")
    ebitda_ntm = out.get("ebitda_ntm")
    sales_ntm = out.get("sales_ntm")
    capex = out.get("capex_ttm", pd.Series(index=out.index, data=np.nan))
    sbc = out.get("sbc_ttm", pd.Series(index=out.index, data=0.0)).fillna(0.0)
    fcf_ttm = out.get("fcf_ttm", pd.Series(index=out.index, data=np.nan))
    fcf_5y_median = out.get("fcf_5y_median", fcf_ttm)

    # Flags de calidad mínima (evita disparos por divisiones con ~0)
    pre_rev   = (pd.to_numeric(sales_ntm, errors="coerce") <= 0) | (pd.to_numeric(gp, errors="coerce") <= 0)
    bad_ebitda= (pd.to_numeric(ebitda_ntm, errors="coerce") <= 0)

    ev_over_ebitda = _safe_div(ev, ebitda_ntm)
    ev_over_gp     = _safe_div(ev, gp)
    ev_over_sales  = _safe_div(ev, sales_ntm)

    capex_sales = _safe_div(capex, sales_ntm).fillna(0.0).clip(lower=0.0, upper=1.0)  # tope razonable
    ev_over_sales_pen = ev_over_sales * (1 + capex_sales)

    # Invertidos + winsor + CAP de z (evita outliers absurdos)
    def _inv_w(s):
        inv = 1.0 / s.replace(0, np.nan)
        return _winsorize(inv, 0.01).fillna(0.0)

    v1 = _inv_w(ev_over_ebitda)
    v2 = _inv_w(ev_over_gp)
    v3 = _inv_w(ev_over_sales_pen)

    raw = 0.40 * _zscore(v1) + 0.30 * _zscore(v2) + 0.30 * _zscore(v3)
    raw = raw.clip(-CAP_Z, CAP_Z)

    # Penalizaciones explícitas
    penalty = pd.Series(0.0, index=raw.index)
    penalty = penalty.mask(pre_rev,   -1.5)  # sin ventas o sin GP ⇒ fuerte castigo
    penalty = penalty.mask(bad_ebitda, -0.8) # EBITDA ≤ 0 ⇒ castigo moderado

    # Boost por FCF 5y ajustado por SBC (cap suave)
    fcf_yield5 = _safe_div((fcf_5y_median - sbc), ev)
    f5_pct = _rank_pct(fcf_yield5)
    boost = (f5_pct >= 0.80).astype(float) * 0.25

    return (raw + penalty + boost).fillna(-1.0)

def quality_intangible_aware(df: pd.DataFrame) -> pd.Series:
    """
    Quality ajustado por intangibles:
      - GP/Assets_xRD
      - ROIC_xRD (NOPAT_xRD / InvestedCapital_xRD)
      - Estabilidad de márgenes (inv. de la desviación 5y)
      - Accruals (NOA) bajos
      - NetCash/EBITDA
    Requiere: gross_profit_ttm, operating_income_ttm, total_assets_ttm,
              (opcional) rd_expense_ttm para capitalización,
              ebitda_ttm/ntm, net_debt_ttm, noa_ttm, invested_capital_ttm,
              current_liabilities_ttm, tax_rate
    """
    out = capitalize_rd(df).copy()

    # Coerción numérica segura
    for col in ["gross_profit_ttm","assets_xrd","total_assets_ttm","ebitda_ttm",
                "ebitda_ntm","net_debt_ttm","noa_ttm","invested_capital_ttm",
                "current_liabilities_ttm","operating_income_ttm","op_income_xrd",
                "tax_rate"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    gp = out.get("gross_profit_ttm")
    assets_xrd = out.get("assets_xrd", out.get("total_assets_ttm"))
    ebitda = out.get("ebitda_ttm", out.get("ebitda_ntm"))
    net_debt = out.get("net_debt_ttm")
    noa = out.get("noa_ttm")
    ic = out.get("invested_capital_ttm", out.get("total_assets_ttm", 0) - out.get("current_liabilities_ttm", 0))

    tax_rate = out.get("tax_rate", pd.Series(index=out.index, data=0.20)).fillna(0.20)
    op_xrd = out.get("op_income_xrd", out.get("operating_income_ttm", 0))
    nopat_xrd = _to_float(op_xrd) * (1 - _to_float(tax_rate))

    gp_assets = _winsorize(_safe_div(gp, assets_xrd), 0.01)
    roic_xrd = _winsorize(_safe_div(nopat_xrd, ic), 0.01)

    # Estabilidad de márgenes: si hay historial de margen operativo por fila (lista/array)
    if "op_margin_hist" in out.columns:
        std_margin = out["op_margin_hist"].apply(
            lambda xs: np.nanstd(np.asarray(xs), ddof=0) if isinstance(xs, (list, tuple, np.ndarray)) else np.nan
        )
    else:
        std_margin = pd.Series(index=out.index, data=np.nan)
    stab = -_zscore(_winsorize(std_margin.fillna(std_margin.median()), 0.01))

    accruals = _winsorize(noa.fillna(noa.median()) if noa is not None else pd.Series(index=out.index, data=0), 0.01)
    accruals_score = -_zscore(accruals)

    netcash_ebitda = _winsorize(-_safe_div(net_debt.fillna(0), _to_float(ebitda).abs() + 1e-9), 0.01)

    score = (
        0.35 * _zscore(gp_assets) +
        0.35 * _zscore(roic_xrd) +
        0.10 * stab +
        0.10 * _zscore(netcash_ebitda) +
        0.10 * accruals_score
    ).clip(-CAP_Z, CAP_Z).fillna(-1.0)

    return score

# ======================================================================
# Neutralización por sector/capitalización y QVM
# ======================================================================

def neutralize_by_sector_cap(df: pd.DataFrame, score_col: str, sector_col: str = "sector",
                             mcap_col: str = "market_cap",
                             buckets=(("Mega", 150e9, np.inf),
                                      ("Large", 10e9, 150e9),
                                      ("Mid", 2e9, 10e9),
                                      ("Small", 0, 2e9))) -> pd.Series:
    """
    Devuelve score neutralizado por sector y bucket de market cap:
      final = 0.5*z_sector + 0.5*z_capbucket
    - Se ordenan los buckets por su borde inferior para garantizar bins crecientes.
    """
    out = df.copy()
    out[mcap_col] = pd.to_numeric(out.get(mcap_col, np.nan), errors="coerce")

    # Ordenar buckets por límite inferior y construir bins crecientes
    b_sorted = sorted(list(buckets), key=lambda b: float(b[1]))
    # edges: [low0, low1, low2, ..., high_last]
    edges = [b_sorted[0][1]] + [b[1] for b in b_sorted[1:]] + [b_sorted[-1][2]]
    # Asegurar estrictamente creciente
    edges = [float(x) for x in edges]
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = np.nextafter(edges[i-1], np.inf)

    labels = [b[0] for b in b_sorted]
    try:
        out["_cap_bucket"] = pd.cut(out[mcap_col], bins=edges, labels=labels, include_lowest=True, right=False)
    except Exception:
        out["_cap_bucket"] = pd.Series(np.nan, index=out.index, dtype="object")

    def z_by(group):
        return _zscore(group[score_col])

    # z por sector
    z_sector = out.groupby(sector_col, group_keys=False, dropna=False).apply(z_by).rename("z_sector")

    # z por bucket (si todo NaN, devuelve NaN y el promedio final lo trata)
    z_cap = out.groupby("_cap_bucket", group_keys=False, dropna=False).apply(z_by).rename("z_cap")

    final = 0.5 * z_sector + 0.5 * z_cap
    return final

def compute_qvm_scores(df: pd.DataFrame,
                       w_quality: float = 0.40,
                       w_value: float = 0.25,
                       w_momentum: float = 0.35,
                       momentum_col: str = "momentum_score",
                       sector_col: str = "sector",
                       mcap_col: str = "market_cap") -> pd.DataFrame:
    """
    Calcula Value y Quality “growth-aware”,
    neutraliza por sector+cap y devuelve:
      value_adj, quality_adj, value_adj_neut, quality_adj_neut, qvm_score
    """
    df = df.copy()
    df["value_adj"] = value_growth_aware(df)
    df["quality_adj"] = quality_intangible_aware(df)
    df["value_adj_neut"] = neutralize_by_sector_cap(df, "value_adj", sector_col, mcap_col)
    df["quality_adj_neut"] = neutralize_by_sector_cap(df, "quality_adj", sector_col, mcap_col)
    m = _zscore(_to_float(df.get(momentum_col, np.nan)))
    df["qvm_score"] = (
        w_quality * _zscore(df["quality_adj_neut"]) +
        w_value   * _zscore(df["value_adj_neut"])   +
        w_momentum* m
    )
    return df

def apply_megacap_rules(df: pd.DataFrame,
                        momentum_col="momentum_score",
                        quality_col="quality_adj_neut",
                        value_col="value_adj_neut") -> pd.DataFrame:
    """
    Reglas:
     - Permite peso aunque Value quede 35–45p si Momentum>=70p y Quality>=55p (sectorial).
     - Si Quality<45p o profit warning, marca 'quality_too_low'.
    """
    out = df.copy()
    out["q_pct_sector"] = out.groupby("sector")[quality_col].transform(lambda s: s.rank(pct=True))
    out["v_pct_sector"] = out.groupby("sector")[value_col].transform(lambda s: s.rank(pct=True))
    out["m_pct_global"] = out[momentum_col].rank(pct=True)
    out["mega_exception_ok"] = (
        (out["m_pct_global"] >= 0.70) &
        (out["q_pct_sector"] >= 0.55) &
        (out["v_pct_sector"] >= 0.35)
    )
    out["quality_too_low"] = out["q_pct_sector"] < 0.45
    return out

# ======================================================================
# FUNDAMENTALES (mínimo de batalla) → para VFQ
# ======================================================================

def _num(x):
    try:
        return float(x)
    except Exception:
        return None

def _fetch_min_battle_fmp(symbol: str, market_cap_hint: float | None = None) -> Dict[str, Any]:
    """
    Descarga el set mínimo y normaliza nombres:
      evToEbitda, fcf_ttm, cfo_ttm, ebit_ttm, grossProfitTTM, totalAssetsTTM,
      roic, roa, netMargin, marketCap (si hay/ hint)
    Usa TTM y cae en annual si falta.
    """
    s = symbol.strip().upper()
    out: Dict[str, Any] = {"symbol": s}

    # --- KEY METRICS TTM (ev/ebitda, grossProfitTTM, totalAssetsTTM) ---
    try:
        j = _http_get(f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{s}")
        kmttm = j[0] if isinstance(j, list) and j else (j if isinstance(j, dict) else {})
    except Exception:
        kmttm = {}

    # --- RATIOS TTM (roic/roa/netMargin) ---
    try:
        j = _http_get(f"https://financialmodelingprep.com/api/v3/ratios-ttm/{s}")
        rttm = j[0] if isinstance(j, list) and j else (j if isinstance(j, dict) else {})
    except Exception:
        rttm = {}

    # --- CASH-FLOW TTM (CFO/FCF) ---
    try:
        cfttm = _http_get(f"https://financialmodelingprep.com/api/v3/cash-flow-statement-ttm/{s}")
        cfttm = cfttm if isinstance(cfttm, dict) else {}
    except Exception:
        cfttm = {}

    # --- INCOME TTM (EBIT aprox) ---
    try:
        incttm = _http_get(f"https://financialmodelingprep.com/api/v3/income-statement-ttm/{s}")
        incttm = incttm if isinstance(incttm, dict) else {}
    except Exception:
        incttm = {}

    # Map TTM → normalizados
    evttm  = _num(kmttm.get("enterpriseValueOverEBITDATTM"))
    gpttm  = _num(kmttm.get("grossProfitTTM"))
    tattm  = _num(kmttm.get("totalAssetsTTM"))
    fcf_t  = _num(cfttm.get("freeCashFlowTTM"))
    cfo_t  = _num(cfttm.get("netCashProvidedByOperatingActivitiesTTM"))
    ebit_t = _num(incttm.get("ebitTTM") or incttm.get("operatingIncomeTTM"))

    roic_t = _num(rttm.get("returnOnCapitalEmployedTTM") or rttm.get("returnOnInvestedCapitalTTM"))
    roa_t  = _num(rttm.get("returnOnAssetsTTM"))
    nmar_t = _num(rttm.get("netProfitMarginTTM"))

    out["evToEbitda"]        = evttm
    out["grossProfitTTM"]    = gpttm
    out["totalAssetsTTM"]    = tattm
    out["fcf_ttm"]           = fcf_t
    out["cfo_ttm"]           = cfo_t
    out["ebit_ttm"]          = ebit_t
    out["roic"]              = roic_t
    out["roa"]               = roa_t
    out["netMargin"]         = nmar_t
    out["marketCap"]         = _num(kmttm.get("marketCap")) or (market_cap_hint if market_cap_hint else None)

    # Flags de fuente
    out["__src_ev"]   = "ttm" if evttm is not None else None
    out["__src_gp"]   = "ttm" if gpttm is not None else None
    out["__src_ta"]   = "ttm" if tattm is not None else None
    out["__src_fcf"]  = "ttm" if fcf_t is not None else None
    out["__src_cfo"]  = "ttm" if cfo_t is not None else None
    out["__src_ebit"] = "ttm" if ebit_t is not None else None
    out["__src_roic"] = "ttm" if roic_t is not None else None
    out["__src_roa"]  = "ttm" if roa_t is not None else None
    out["__src_nmar"] = "ttm" if nmar_t is not None else None

    # --- COMPANY PROFILE (sector, industry, price, beta, marketCap) ---
    try:
        prof = _http_get(f"https://financialmodelingprep.com/api/v3/profile/{s}")
        p0 = _first_obj(prof)
    except Exception:
        p0 = {}

    # sector/industry para enriquecer VFQ y neutralización
    out["sector"]   = p0.get("sector") or out.get("sector") or "Unknown"
    out["industry"] = p0.get("industry") or out.get("industry")

    # datos útiles para la vista
    if out.get("marketCap") is None:
        out["marketCap"] = _num(p0.get("mktCap"))
    out["price"] = _num(p0.get("price"))
    out["beta"]  = _num(p0.get("beta"))


    # Fallback annual si falta algo crítico
    need_annual = any(
        x is None for x in [out["evToEbitda"], out["grossProfitTTM"], out["totalAssetsTTM"],
                            out["fcf_ttm"], out["cfo_ttm"], out["ebit_ttm"], out["roic"], out["roa"], out["netMargin"]]
    )

    if need_annual:
        # key-metrics annual
        try:
            j = _http_get(f"https://financialmodelingprep.com/api/v3/key-metrics/{s}", params={"period":"annual","limit":4})
            km = j[0] if isinstance(j, list) and j else {}
        except Exception:
            km = {}
        if out["evToEbitda"]     is None: out["evToEbitda"]     = _num(km.get("enterpriseValueOverEBITDA"))
        if out["grossProfitTTM"] is None: out["grossProfitTTM"] = _num(km.get("grossProfit"))
        if out["totalAssetsTTM"] is None: out["totalAssetsTTM"] = _num(km.get("totalAssets"))
        if out["marketCap"]      is None: out["marketCap"]      = _num(km.get("marketCap")) or (market_cap_hint if market_cap_hint else None)

        # ratios annual
        try:
            j = _http_get(f"https://financialmodelingprep.com/api/v3/ratios/{s}", params={"period":"annual","limit":4})
            rr = j[0] if isinstance(j, list) and j else {}
        except Exception:
            rr = {}
        if out["roic"]      is None: out["roic"]      = _num(rr.get("returnOnCapitalEmployed") or rr.get("returnOnInvestedCapital"))
        if out["roa"]       is None: out["roa"]       = _num(rr.get("returnOnAssets"))
        if out["netMargin"] is None: out["netMargin"] = _num(rr.get("netProfitMargin"))

        # cash-flow annual
        if out["cfo_ttm"] is None or out["fcf_ttm"] is None:
            try:
                cf = _http_get(f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{s}", params={"period":"annual","limit":1})
                cf0 = cf[0] if isinstance(cf, list) and cf else {}
            except Exception:
                cf0 = {}
            if out["cfo_ttm"] is None: out["cfo_ttm"] = _num(cf0.get("netCashProvidedByOperatingActivities"))
            if out["fcf_ttm"] is None: out["fcf_ttm"] = _num(cf0.get("freeCashFlow"))

        # income annual
        if out["ebit_ttm"] is None:
            try:
                inc = _http_get(f"https://financialmodelingprep.com/api/v3/income-statement/{s}", params={"period":"annual","limit":1})
                inc0 = inc[0] if isinstance(inc, list) and inc else {}
            except Exception:
                inc0 = {}
            out["ebit_ttm"] = _num(inc0.get("ebit") or inc0.get("operatingIncome"))

    return out

def _coverage_count(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    cols = [c for c in ["evToEbitda","fcf_ttm","cfo_ttm","ebit_ttm",
                        "grossProfitTTM","totalAssetsTTM","roic","roa","netMargin"] if c in df.columns]
    return int(df[cols].notna().sum(axis=1).sum()) if cols else 0

def download_fundamentals(symbols: List[str],
                          market_caps: Dict[str, float] | None = None,
                          cache_key: str | None = None,
                          force: bool = False,
                          max_symbols_per_minute: int = 50) -> pd.DataFrame:
    """
    Descarga mínimos de batalla para VFQ con:
      - reintentos suaves y limitación de tasa
      - evita cachear snapshots sin cobertura
    """
    key = f"fund_{cache_key}" if cache_key else None
    if key and not force:
        dfc = load_df(key)
        if dfc is not None and not dfc.empty:
            return dfc

    rows = []
    mc_map = market_caps or {}
    throttle = max(0.0, 60.0 / max(1, max_symbols_per_minute))
    for i, s in enumerate(symbols):
        if i > 0 and throttle > 0:
            time.sleep(throttle)
        try:
            rec = _fetch_min_battle_fmp(s, market_cap_hint=mc_map.get(s))
            rows.append(rec)
        except Exception as e:
            rows.append({"symbol": s, "__err_fund": str(e)[:180]})

    df = pd.DataFrame(rows).drop_duplicates("symbol")

    # Si literalmente no hay cobertura, intenta un segundo pase con muestra
    if _coverage_count(df) == 0 and len(symbols) > 0:
        sample = list(pd.Series(symbols).drop_duplicates().sample(min(25, len(symbols)), random_state=42))
        rows2 = []
        for s in sample:
            try:
                rows2.append(_fetch_min_battle_fmp(s, market_cap_hint=mc_map.get(s)))
                time.sleep(throttle)
            except Exception as e:
                rows2.append({"symbol": s, "__err_fund": str(e)[:180]})
        df2 = pd.DataFrame(rows2).drop_duplicates("symbol")
        df = df.set_index("symbol").combine_first(df2.set_index("symbol")).reset_index()

    if key and _coverage_count(df) > 0:
        try: save_df(df, key)
        except Exception: pass

    return df

# Flag para progreso Streamlit (opcional)
try:
    import streamlit as st
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# ======================================================================
# GUARDRAILS: descarga (paralela) + aplicación
# ======================================================================

def download_guardrails(symbol: str) -> dict:
    """
    Calcula métricas para guardrails (con fallbacks robustos):
      - ebit_ttm, cfo_ttm, fcf_ttm (profit floor)
      - net_issuance (Δ acciones)
      - asset_growth (y/y)
      - accruals_ta = (NI - CFO)/assets promedio
      - netdebt_ebitda
    """
    sym = (symbol or "").strip().upper()
    out = {"symbol": sym}

    # KEY-METRICS TTM
    try:
        kttm = _http_get(f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{sym}")
        kt0 = _first_obj(kttm)
        out["shares_out_ttm"] = _safe_float(kt0.get("sharesOutstanding"))
        out["net_debt_ttm"]   = _safe_float(kt0.get("netDebtTTM"))
        out["ebitda_ttm"]     = _safe_float(kt0.get("ebitdaTTM"))
    except Exception:
        pass

    # CFO/FCF TTM
    try:
        cfttm = _http_get(f"https://financialmodelingprep.com/api/v3/cash-flow-statement-ttm/{sym}")
        cf0 = _first_obj(cfttm)
        out["cfo_ttm"] = _safe_float(cf0.get("netCashProvidedByOperatingActivitiesTTM"))
        out["fcf_ttm"] = _safe_float(cf0.get("freeCashFlowTTM"))
    except Exception:
        try:
            cf = _http_get(f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{sym}",
                           params={"period": "annual", "limit": 1})
            cf0 = _first_obj(cf)
            out["cfo_ttm"] = _safe_float(cf0.get("netCashProvidedByOperatingActivities"))
            out["fcf_ttm"] = _safe_float(cf0.get("freeCashFlow"))
        except Exception:
            pass

    # EBIT TTM
    try:
        inc_ttm = _http_get(f"https://financialmodelingprep.com/api/v3/income-statement-ttm/{sym}")
        it0 = _first_obj(inc_ttm)
        out["ebit_ttm"] = _safe_float(it0.get("ebitTTM") or it0.get("operatingIncomeTTM"))
    except Exception:
        try:
            inc = _http_get(f"https://financialmodelingprep.com/api/v3/income-statement/{sym}",
                            params={"period": "annual", "limit": 1})
            i0 = _first_obj(inc)
            out["ebit_ttm"] = _safe_float(i0.get("ebit") or i0.get("operatingIncome"))
        except Exception:
            pass

    # Series anuales para growth/accruals/issuance
    try:
        bal = _http_get(f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{sym}",
                        params={"period": "annual", "limit": 5})
    except Exception:
        bal = []
    try:
        inc = _http_get(f"https://financialmodelingprep.com/api/v3/income-statement/{sym}",
                        params={"period": "annual", "limit": 5})
    except Exception:
        inc = []
    try:
        cf = _http_get(f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{sym}",
                       params={"period": "annual", "limit": 5})
    except Exception:
        cf = []
    try:
        km = _http_get(f"https://financialmodelingprep.com/api/v3/key-metrics/{sym}",
                       params={"period": "annual", "limit": 6})
    except Exception:
        km = []

    # Asset growth
    assets = _yr_series(bal, "totalAssets")
    if len(assets) >= 2:
        _, a0 = assets[-2]; _, a1 = assets[-1]
        out["asset_growth"] = (a1 - a0) / a0 if (a0 not in (None, 0)) else None

    # Accruals/TA
    ni = _yr_series(inc, "netIncome")
    cfo = _yr_series(cf, "netCashProvidedByOperatingActivities")
    ta = _yr_series(bal, "totalAssets")
    if len(ni) >= 2 and len(cfo) >= 2 and len(ta) >= 2:
        _, ni1 = ni[-1]; _, cfo1 = cfo[-1]
        _, ta1 = ta[-1]; _, ta0 = ta[-2]
        avg_assets = None
        if ta1 is not None and ta0 is not None:
            avg_assets = (ta1 + ta0) / 2.0
        accruals = None if (ni1 is None or cfo1 is None) else (ni1 - cfo1)
        out["accruals_ta"] = (accruals / avg_assets) if (accruals is not None and avg_assets not in (None, 0)) else None

    # Net issuance
    shares_km = _yr_series(km, "sharesOutstanding")
    shares_bs = _yr_series(bal, "commonStockSharesOutstanding")
    seq = shares_km if len(shares_km) >= 2 else shares_bs
    if len(seq) >= 2:
        _, s0 = seq[-2]; _, s1 = seq[-1]
        out["net_issuance"] = (s1 - s0) / s0 if (s0 not in (None, 0)) else None

    # NetDebt/EBITDA
    nd_eb = None
    if isinstance(km, list) and km:
        for item in reversed(km):
            nd = _safe_float(item.get("netDebt"))
            eb = _safe_float(item.get("ebitda"))
            if nd is not None and eb not in (None, 0):
                nd_eb = nd / eb
                break
    if nd_eb is None:
        try:
            b0 = _first_obj(_http_get(f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{sym}",
                                      params={"period": "annual", "limit": 1}))
            i0 = _first_obj(_http_get(f"https://financialmodelingprep.com/api/v3/income-statement/{sym}",
                                      params={"period": "annual", "limit": 1}))
            total_debt = _safe_float(b0.get("totalDebt")) or _safe_float(b0.get("shortTermDebt"))
            cash_eq = _safe_float(b0.get("cashAndCashEquivalents")) or 0.0
            eb = _safe_float(i0.get("ebitda"))
            if total_debt is not None and eb not in (None, 0):
                nd_eb = (total_debt - (cash_eq or 0.0)) / eb
        except Exception:
            pass

    out["netdebt_ebitda"] = nd_eb
    return out

def _retry_download_guardrails(symbol: str, retries: int = 2, base_sleep: float = 0.6) -> Dict[str, Any]:
    """
    Llama download_guardrails(symbol) con reintentos exponenciales.
    Devuelve siempre un dict con al menos {"symbol": symbol, ...} o {"symbol": symbol, "__err_guard": "..."}.
    """
    for attempt in range(retries + 1):
        try:
            row = download_guardrails(symbol)
            if isinstance(row, dict):
                row.setdefault("symbol", symbol)
                return row
            elif isinstance(row, pd.Series):
                d = row.to_dict(); d.setdefault("symbol", symbol)
                return d
            else:
                return {"symbol": symbol, "__err_guard": f"Unexpected return type: {type(row).__name__}"}
        except Exception as e:
            if attempt < retries:
                time.sleep(base_sleep * (2 ** attempt))
                continue
            return {"symbol": symbol, "__err_guard": str(e)[:180]}

def _norm_symbols(symbols: List[str]) -> List[str]:
    """Normaliza y deduplica preservando orden."""
    seen = set()
    out = []
    for s in symbols:
        if s is None:
            continue
        t = str(s).strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out

def download_guardrails_batch(
    symbols: List[str],
    cache_key: str | None = None,
    force: bool = False,
    *,
    chunk_size: int = 150,
    max_workers: int = 8,
    pause_between_chunks: float = 0.6,
    retries: int = 2
) -> pd.DataFrame:
    """
    Descarga guardrails para muchos símbolos con:
      - chunks para controlar rate limits,
      - concurrencia limitada por chunk,
      - reintentos con backoff por símbolo,
      - caché opcional vía load_df/save_df.
    """
    key = f"guard_{cache_key}" if cache_key else None
    if key and not force:
        dfc = load_df(key)
        if dfc is not None:
            return dfc

    syms = _norm_symbols(symbols)
    if not syms:
        return pd.DataFrame(columns=["symbol"])

    # Progreso en Streamlit si está disponible
    prog = st.progress(0.0) if _HAS_ST else None
    status = st.empty() if _HAS_ST else None

    rows: list[Dict[str, Any]] = []
    total = len(syms)
    processed = 0

    for i in range(0, total, chunk_size):
        chunk = syms[i : i + chunk_size]

        if status:
            status.write(f"Guardrails: procesando {i+1}-{min(i+len(chunk), total)} / {total} símbolos...")

        # Ejecuta en paralelo con límite de workers
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_retry_download_guardrails, s, retries): s for s in chunk}
            for fut in cf.as_completed(futs):
                rows.append(fut.result())
                processed += 1
                if prog:
                    prog.progress(min(processed / total, 1.0))

        # Pausa corta entre chunks para evitar rate limits
        if i + chunk_size < total and pause_between_chunks > 0:
            time.sleep(pause_between_chunks)

    if status:
        status.write("Guardrails: consolidando resultados...")

    df = pd.DataFrame(rows)
    if "symbol" not in df.columns:
        df["symbol"] = syms[:len(df)]
    df = df.drop_duplicates(subset=["symbol"], keep="first")
    order = {s: idx for idx, s in enumerate(syms)}
    df["_ord"] = df["symbol"].map(order)
    df = df.sort_values("_ord").drop(columns=["_ord"])

    if key:
        save_df(df, key)

    if status:
        status.write("Guardrails: listo ✅")
        prog.progress(1.0)

    return df

# ======================================================================
# VFQ clásico (merge universo + fundamentales) y dinámico
# ======================================================================

def _bucket_by_quantiles(s: pd.Series, q: int = 3) -> pd.Series:
    r = s.rank(method="first", na_option="keep")
    try:
        return pd.qcut(r, q, labels=False, duplicates="drop")
    except Exception:
        if r.max() and r.max() > 0:
            pct = r / r.max()
        else:
            pct = r
        return pd.Series(np.select(
            [pct <= 0.33, pct <= 0.66, pct > 0.66],
            [0,1,2],
            default=np.nan
        ), index=s.index)

def build_vfq_scores(df_universe: pd.DataFrame, df_fund: pd.DataFrame,
                     size_buckets: int = 3) -> pd.DataFrame:
    """
    Fusiona universo + fundamentales mínimos y calcula VFQ de forma tolerante a NaNs.
    Devuelve un DF con:
      ['symbol','sector','industry','marketCap_unified','coverage_count',
       'ValueScore','QualityScore','VFQ','VFQ_pct_sector', ...]
    Nota: aplica guardrails "blandos" (anti-junk + liquidez) ANULANDO puntajes fuera de regla.
    """
    dfu = df_universe.copy() if isinstance(df_universe, pd.DataFrame) else pd.DataFrame()
    dff = df_fund.copy()     if isinstance(df_fund, pd.DataFrame)     else pd.DataFrame()

    if dfu.empty or "symbol" not in dfu.columns:
        return pd.DataFrame(columns=["symbol","VFQ","coverage_count"])

    if "symbol" not in dff.columns:
        dff = pd.DataFrame(columns=["symbol"])

    # --- merge base
    df = dfu.merge(dff, on="symbol", how="left").copy()
    df["symbol"] = df["symbol"].astype(str).str.upper()

    # --- coalesce sector/industry (preferir lo ya presente; completar con faltantes)
    for c in ["sector", "industry"]:
        if c not in df.columns:
            df[c] = np.nan
    # normalización simple
    df[["sector","industry"]] = df[["sector","industry"]].astype(str).replace({"None":"", "nan":"", "NaN":""})
    df.loc[df["sector"].eq(""),   "sector"]   = "Unknown"
    df.loc[df["industry"].eq(""), "industry"] = "Unknown"

    # --- market cap unificado (robusto a variantes)
    def to_num(colname: str) -> pd.Series:
        return pd.to_numeric(df[colname], errors="coerce") if colname in df.columns else pd.Series(np.nan, index=df.index)

    mcap = pd.Series(np.nan, index=df.index)
    mcap_candidates = (
        ["marketCap", "marketCap_profile", "marketCap_ev", "marketCap_unified"] +
        [c for c in df.columns if c.lower().startswith("marketcap")]
    )
    for c in mcap_candidates:
        if c in df.columns:
            mcap = mcap.fillna(to_num(c))

    price_series = pd.Series(np.nan, index=df.index)
    for c in [c for c in df.columns if c.lower().startswith("price")]:
        price_series = price_series.fillna(to_num(c))

    shares_series = pd.Series(np.nan, index=df.index)
    shares_candidates = (
        ["sharesOutstanding", "shares_out_ttm"] +
        [c for c in df.columns if c.lower().startswith("sharesoutstanding")]
    )
    for c in shares_candidates:
        if c in df.columns:
            shares_series = shares_series.fillna(to_num(c))

    mcap = mcap.fillna(price_series * shares_series)
    df["marketCap_unified"] = pd.to_numeric(mcap, errors="coerce")

    # --- bucket por tamaño (para ranking intra-sector+tamaño)
    df["size_bucket"] = _bucket_by_quantiles(df["marketCap_unified"], q=size_buckets)
    grp_key = df["sector"].astype(str) + "|" + df["size_bucket"].astype(str)

    # --------- derivadas para Value/Quality ----------
    # Ajusta nombres si tus columnas son 'gross_profit_ttm', 'roic_ttm', etc.
    ev   = to_num("evToEbitda")
    fcf  = to_num("fcf_ttm")
    gp   = to_num("grossProfitTTM")       # <- si usas 'gross_profit_ttm', cambia aquí
    ta   = to_num("totalAssetsTTM")       # <- si usas 'total_assets_ttm', cambia aquí
    roic = to_num("roic")
    roa  = to_num("roa")
    nmar = to_num("netMargin")

    df["inv_ev_ebitda"]       = (1.0 / ev).replace([np.inf, -np.inf], np.nan)
    df["fcf_yield"]           = (fcf / df["marketCap_unified"]).replace([np.inf, -np.inf], np.nan)
    df["gross_profitability"] = (gp  / ta).replace([np.inf, -np.inf], np.nan)

    val_cols = [c for c in ["fcf_yield","inv_ev_ebitda"] if c in df.columns]
    q_cols   = [c for c in ["gross_profitability","roic","roa","netMargin"] if c in df.columns]

    # winsor suave
    for c in val_cols + q_cols:
        df[c] = _winsorize(df[c], 0.01)

    # --- guardrails blandos (ANULA puntajes fuera de regla para que no rankeen)
    # Liquidez
    px   = to_num("price")
    dv3m = to_num("avgDollarVol_3m")  # si no la tienes, crea previamente con rolling price*volume
    mcap = df["marketCap_unified"]

    liq_pass = (px >= 5.0) & (mcap >= 500_000_000.0)
    if not dv3m.isna().all():
        liq_pass = liq_pass & (dv3m >= 2_000_000.0)

    # Anti-junk: pre-revenue biotech y story stocks (PS muy alto con pérdidas)
    sector   = df["sector"].astype(str)
    industry = df["industry"].astype(str)
    # usa tus nombres reales si difieren:
    revenue  = to_num("revenue_ttm")
    gp_ttm   = to_num("gross_profit_ttm") if "gross_profit_ttm" in df.columns else gp
    sales_ntm = to_num("sales_ntm") if "sales_ntm" in df.columns else revenue
    ps = (mcap / sales_ntm.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    biotech_flag = sector.str.contains("health", case=False, na=False) & \
                   industry.str.contains("biotech|drug", case=False, na=False)
    pre_rev = (revenue.fillna(0.0) < 50_000_000.0) | (gp_ttm.fillna(0.0) <= 0.0)
    story_flag = (ps >= 40.0) & (nmar <= 0.0)

    anti_junk_pass = ~((biotech_flag & pre_rev) | story_flag)

    guard_pass = liq_pass & anti_junk_pass

    # --- si no hay campos, devolver frame-base
    fields = val_cols + q_cols
    if len(fields) == 0:
        df["coverage_count"] = 0
        df["ValueScore"] = np.nan
        df["QualityScore"] = np.nan
        df["VFQ"] = np.nan
        # percentil robusto
        try:
            grp_sz = df.groupby("sector")["symbol"].transform("size")
            pct_sec = df.groupby("sector")["VFQ"].rank(pct=True)
            pct_glb = df["VFQ"].rank(pct=True)
            df["VFQ_pct_sector"] = np.where(grp_sz >= 6, pct_sec, pct_glb)
        except Exception:
            df["VFQ_pct_sector"] = df["VFQ"].rank(pct=True)
        df["VFQ_pct_sector"] = pd.to_numeric(df["VFQ_pct_sector"], errors="coerce").clip(0.0, 1.0).fillna(1.0)
        return df

    # anular puntajes en fuera-de-guardrail para que queden abajo
    for c in fields:
        df.loc[~guard_pass, c] = np.nan

    df["coverage_count"] = df[fields].notna().sum(axis=1)

    # ranking intra sector+tamaño (descending = mejor)
    def _rank_group(col: str) -> pd.Series:
        s = pd.to_numeric(df[col], errors="coerce")
        return s.groupby(grp_key).rank(method="average", ascending=False, na_option="bottom")

    df["ValueScore"]   = pd.concat([_rank_group(c) for c in val_cols], axis=1).mean(axis=1) if val_cols else np.nan
    df["QualityScore"] = pd.concat([_rank_group(c) for c in q_cols],  axis=1).mean(axis=1) if q_cols else np.nan
    df["VFQ"]          = pd.concat([df["ValueScore"], df["QualityScore"]], axis=1).mean(axis=1, skipna=True)

    # percentil intra-sector con fallback si el grupo es pequeño
    try:
        grp_sz  = df.groupby("sector")["symbol"].transform("size")
        pct_sec = df.groupby("sector")["VFQ"].rank(pct=True)
        pct_glb = df["VFQ"].rank(pct=True)
        df["VFQ_pct_sector"] = np.where(grp_sz >= 6, pct_sec, pct_glb)
    except Exception:
        df["VFQ_pct_sector"] = df["VFQ"].rank(pct=True)
    df["VFQ_pct_sector"] = pd.to_numeric(df["VFQ_pct_sector"], errors="coerce").clip(0.0, 1.0).fillna(1.0)

    # expone flags para depurar en UI
    df["guard_liquidity"] = liq_pass
    df["guard_anti_junk"] = anti_junk_pass
    df["guard_pass"]      = guard_pass

    return df


# ======================================================================
# Guardrails: aplicación de umbrales
# ======================================================================

def _num_or_nan(d: pd.DataFrame, col: str) -> pd.Series:
    if col not in d.columns:
        return pd.Series(np.nan, index=d.index)
    return pd.to_numeric(d[col], errors="coerce")

# fundamentals.py


def apply_quality_guardrails(
    df_guard: pd.DataFrame,
    *,
    require_profit_floor: bool = True,
    profit_floor_min_hits: int = 2,      # # de hits entre EBIT/CFO/FCF > 0
    max_net_issuance: float = 0.03,      # emisión neta máxima (solo positivos fallan)
    max_asset_growth: float = 0.20,      # |crec. activos y/y| máximo (absoluto)
    max_accruals_ta: float = 0.10,       # |accruals/TA| máximo (absoluto)
    max_netdebt_ebitda: float = 3.0,     # NetDebt/EBITDA máximo
    min_vfq_coverage: int = 1            # cobertura mínima (métricas value+quality disponibles)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (kept, diag):
       - kept: símbolos que pasan guardrails (al menos columna 'symbol')
       - diag: tabla diagnóstica con flags y razones de rechazo/paso
    """
    if df_guard is None or df_guard.empty or "symbol" not in df_guard.columns:
        return pd.DataFrame(columns=["symbol"]), pd.DataFrame(columns=["symbol","reason"])

    g = df_guard.copy()
    g["symbol"] = g["symbol"].astype(str)

    # Coerción numérica defensiva
    def _num(c): 
        return pd.to_numeric(g[c], errors="coerce") if c in g.columns else pd.Series(np.nan, index=g.index)

    ebit = _num("ebit_ttm")
    cfo  = _num("cfo_ttm")
    fcf  = _num("fcf_ttm")

    net_issuance  = _num("net_issuance")
    asset_growth  = _num("asset_growth")
    accruals_ta   = _num("accruals_ta")
    nd_ebitda     = _num("netdebt_ebitda")

    # Cobertura VFQ si no existe (cuenta columnas razonables de V y Q)
    if "coverage_count" in g.columns:
        coverage = pd.to_numeric(g["coverage_count"], errors="coerce").fillna(0).astype(int)
    else:
        cov_cols = [c for c in ["inv_ev_ebitda","fcf_yield","gross_profitability","roic","roa","netMargin"] if c in g.columns]
        coverage = g[cov_cols].notna().sum(axis=1).astype(int) if cov_cols else pd.Series(0, index=g.index)

    # 1) Profit floor (hits en {EBIT, CFO, FCF} > 0)
    hits = (ebit > 0).astype(int) + (cfo > 0).astype(int) + (fcf > 0).astype(int)
    ok_profit = hits >= int(profit_floor_min_hits) if require_profit_floor else pd.Series(True, index=g.index)

    # 2) Emisión neta (solo positivos fallan)
    ok_issuance = (net_issuance.isna()) | (net_issuance <= float(max_net_issuance))

    # 3) Crecimiento de activos (valor absoluto)
    ok_assets = (asset_growth.isna()) | (asset_growth.abs() <= float(max_asset_growth))

    # 4) Accruals/TA (valor absoluto)
    ok_accr = (accruals_ta.isna()) | (accruals_ta.abs() <= float(max_accruals_ta))

    # 5) NetDebt/EBITDA
    ok_ndeb = (nd_ebitda.isna()) | (nd_ebitda <= float(max_netdebt_ebitda))

    # 6) Cobertura mínima
    ok_cov = coverage >= int(min_vfq_coverage)

    pass_all = ok_profit & ok_issuance & ok_assets & ok_accr & ok_ndeb & ok_cov

    # Diagnóstico de razones (primera razón relevante, si falla)
    reasons = []
    for i in range(len(g)):
        if pass_all.iloc[i]:
            reasons.append("OK")
            continue
        r = []
        if require_profit_floor and not ok_profit.iloc[i]:
            r.append(f"profit_floor<{profit_floor_min_hits} (hits={int(hits.iloc[i])})")
        if not ok_cov.iloc[i]:
            r.append(f"low_coverage<{min_vfq_coverage} (cov={int(coverage.iloc[i])})")
        if not ok_issuance.iloc[i]:
            r.append(f"net_issuance>{max_net_issuance:.3f} (val={net_issuance.iloc[i]:.3f})")
        if not ok_assets.iloc[i]:
            r.append(f"|asset_growth|>{max_asset_growth:.2f} (val={asset_growth.iloc[i]:.3f})")
        if not ok_accr.iloc[i]:
            r.append(f"|accruals_ta|>{max_accruals_ta:.2f} (val={accruals_ta.iloc[i]:.3f})")
        if not ok_ndeb.iloc[i]:
            r.append(f"netdebt/EBITDA>{max_netdebt_ebitda:.2f} (val={nd_ebitda.iloc[i]:.3f})")
        reasons.append("; ".join(r) if r else "reject")

    diag = pd.DataFrame({
        "symbol": g["symbol"],
        "profit_hits": hits.astype(int),
        "coverage_count": coverage.astype(int),
        "net_issuance": net_issuance,
        "asset_growth": asset_growth,
        "accruals_ta": accruals_ta,
        "netdebt_ebitda": nd_ebitda,
        "pass_profit": ok_profit,
        "pass_issuance": ok_issuance,
        "pass_assets": ok_assets,
        "pass_accruals": ok_accr,
        "pass_ndebt": ok_ndeb,
        "pass_coverage": ok_cov,
        "pass_all": pass_all,
        "reason": reasons,
    })

    kept = diag.loc[diag["pass_all"], ["symbol"]].reset_index(drop=True)
    if df_guard is None or df_guard.empty or "symbol" not in df_guard.columns:
        empty_kept = pd.DataFrame(columns=["symbol"])
        empty_diag = pd.DataFrame({
            "symbol": pd.Series(dtype=str),
            "profit_hits": pd.Series(dtype="Int64"),
            "coverage_count": pd.Series(dtype="Int64"),
            "net_issuance": pd.Series(dtype=float),
            "asset_growth": pd.Series(dtype=float),
            "accruals_ta": pd.Series(dtype=float),
            "netdebt_ebitda": pd.Series(dtype=float),
            "pass_profit": pd.Series(dtype=bool),
            "pass_issuance": pd.Series(dtype=bool),
            "pass_assets": pd.Series(dtype=bool),
            "pass_accruals": pd.Series(dtype=bool),
            "pass_ndebt": pd.Series(dtype=bool),
            "pass_coverage": pd.Series(dtype=bool),
            "pass_all": pd.Series(dtype=bool),
            "reason": pd.Series(dtype=str),
        })
    return empty_kept, empty_diag


# ======================================================================
# VFQ dinámico (si quieres definir columnas ad-hoc)
# ======================================================================

def build_vfq_scores_dynamic(
    df: pd.DataFrame,
    value_metrics: list[str],
    quality_metrics: list[str],
    w_value: float = 0.5,
    w_quality: float = 0.5,
    method_intra: str = "mean",    # "mean" | "median"
    winsor_p: float = 0.01,
    size_buckets: int = 3,
    group_mode: str = "sector",    # "sector" | "sector|size"
) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    df = df.copy()

    def _numcol(name):
        return pd.to_numeric(df[name], errors="coerce") if name in df.columns else pd.Series(np.nan, index=df.index)

    def _winsor_local(s: pd.Series, p: float):
        s = pd.to_numeric(s, errors="coerce")
        if s.isna().all() or p <= 0:
            return s
        lo, hi = s.quantile(p), s.quantile(1 - p)
        return s.clip(lo, hi)

    # --- Derivadas mínimas si faltan ---
    if "inv_ev_ebitda" in value_metrics and "inv_ev_ebitda" not in df.columns:
        ev = _numcol("evToEbitda")
        df["inv_ev_ebitda"] = (1.0 / ev).replace([np.inf, -np.inf], np.nan)

    if "fcf_yield" in value_metrics and "fcf_yield" not in df.columns:
        df["fcf_yield"] = (_numcol("fcf_ttm") / _numcol("marketCap_unified")).replace([np.inf, -np.inf], np.nan)

    if "gross_profitability" in quality_metrics and "gross_profitability" not in df.columns:
        df["gross_profitability"] = (_numcol("grossProfitTTM") / _numcol("totalAssetsTTM")).replace([np.inf, -np.inf], np.nan)

    V = [c for c in value_metrics if c in df.columns]
    Q = [c for c in quality_metrics if c in df.columns]

    # Winsorizar métricas usadas
    for c in set(V + Q):
        df[c] = _winsor_local(df[c], winsor_p)

    # Cobertura VFQ = nº de métricas disponibles (Value+Quality)
    use_cols = V + Q
    df["coverage_count"] = df[use_cols].notna().sum(axis=1).astype(int) if use_cols else 0

    # --- Size buckets (opcional) ---
    if size_buckets > 1:
        mcap = _numcol("marketCap_unified")
        r = mcap.rank(method="first", na_option="keep")
        try:
            size_bucket = pd.qcut(r, size_buckets, labels=False, duplicates="drop")
        except Exception:
            size_bucket = pd.Series(np.nan, index=df.index)
    else:
        size_bucket = pd.Series(0, index=df.index)

    # --- Clave de grupo ---
    df["sector"] = df.get("sector", "Unknown").fillna("Unknown").astype(str)
    grp_key = df["sector"] if group_mode == "sector" else df["sector"].astype(str) + "|" + size_bucket.astype(str)

    def _rank_group(col):
        s = pd.to_numeric(df[col], errors="coerce")
        # ranks altos = mejor (ascending=False)
        return s.groupby(grp_key).rank(method="average", ascending=False, na_option="bottom")

    def _block_score(cols):
        if not cols:
            return pd.Series(np.nan, index=df.index)
        ranks = pd.concat([_rank_group(c) for c in cols], axis=1)
        if method_intra == "median":
            return ranks.median(axis=1)
        return ranks.mean(axis=1)

    df["ValueScore"]   = _block_score(V)
    df["QualityScore"] = _block_score(Q)

    w_sum = (w_value or 0) + (w_quality or 0)
    if w_sum == 0:
        w_value = w_quality = 0.5
        w_sum = 1.0
    df["VFQ"] = (df["ValueScore"] * w_value + df["QualityScore"] * w_quality) / w_sum

    # Percentil por sector
    try:
        df["VFQ_pct_sector"] = df.groupby("sector")["VFQ"].rank(pct=True)
    except Exception:
        df["VFQ_pct_sector"] = df["VFQ"].rank(pct=True)
    df["VFQ_pct_sector"] = df["VFQ_pct_sector"].clip(0.0, 1.0).fillna(1.0)

    # Devuelve df con columnas listas para filtros/guardrails
    return df


# ----------------------------- Robust VFQ wrapper (appended) -----------------------------
def build_vfq_scores_dynamic2(
    df: pd.DataFrame,
    value_metrics=("inv_ev_ebitda","fcf_yield"),
    quality_metrics=("gross_profitability","roic"),
    w_value: float = 0.5,
    w_quality: float = 0.5,
    method_intra: str = "z",
    winsor_p: float = 0.01,
    size_buckets: int = 3,
    group_mode: str = "sector",
) -> pd.DataFrame:
    # Coerce input
    if isinstance(df, (list, tuple)):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return pd.DataFrame(columns=["symbol","sector","industry","VFQ","VFQ_pct_sector"])
    elif isinstance(df, dict):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return pd.DataFrame(columns=["symbol","sector","industry","VFQ","VFQ_pct_sector"])
    elif not isinstance(df, pd.DataFrame):
        return pd.DataFrame(columns=["symbol","sector","industry","VFQ","VFQ_pct_sector"])

    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol","sector","industry","VFQ","VFQ_pct_sector"])

    work = df.copy()
    if "sector" not in work.columns:
        work["sector"] = "Unknown"
    work["sector"] = work["sector"].astype(str)
    if "industry" not in work.columns:
        work["industry"] = pd.Series([None]*len(work), index=work.index)

    def _prep(col):
        s = pd.to_numeric(work[col], errors="coerce") if col in work.columns else pd.Series(index=work.index, dtype=float)
        s = _winsorize(s, p=winsor_p)
        if isinstance(method_intra, str) and method_intra.lower().startswith("z"):
            return _zscore(s)
        else:
            return _rank_pct(s)

    v_cols = [c for c in (list(value_metrics) if not isinstance(value_metrics, (list, tuple)) else value_metrics) if c in work.columns]
    v_score = pd.concat([_prep(c) for c in v_cols], axis=1).mean(axis=1, skipna=True) if v_cols else pd.Series(0.0, index=work.index)

    q_cols = [c for c in (list(quality_metrics) if not isinstance(quality_metrics, (list, tuple)) else quality_metrics) if c in work.columns]
    q_score = pd.concat([_prep(c) for c in q_cols], axis=1).mean(axis=1, skipna=True) if q_cols else pd.Series(0.0, index=work.index)

    v_weight = float(w_value) if pd.notna(w_value) else 0.5
    q_weight = float(w_quality) if pd.notna(w_quality) else 0.5
    total_w = v_weight + q_weight if (v_weight + q_weight) != 0 else 1.0
    v_weight /= total_w
    q_weight /= total_w

    VFQ = v_weight * v_score + q_weight * q_score

    if "sector" in work.columns:
        VFQ_pct_sector = VFQ.groupby(work["sector"]).rank(pct=True, method="average")
    else:
        VFQ_pct_sector = VFQ.rank(pct=True, method="average")

    out = work[["symbol","sector","industry"]].copy() if all(c in work.columns for c in ["symbol","sector","industry"]) else pd.DataFrame({
        "symbol": work["symbol"] if "symbol" in work.columns else pd.Series([None]*len(work), index=work.index),
        "sector": work["sector"],
        "industry": work["industry"],
    })
    out["VFQ"] = VFQ
    out["VFQ_pct_sector"] = VFQ_pct_sector
    if "coverage_count" in work.columns:
        out["coverage_count"] = pd.to_numeric(work["coverage_count"], errors="coerce")
    return out

# Alias to keep the original app import name
build_vfq_scores_dynamic = build_vfq_scores_dynamic2
