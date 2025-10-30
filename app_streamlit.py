from __future__ import annotations
# --- poner esto ARRIBA DE TODO ---
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "poll"  # o "none" si prefieres desactivar
# ---------------------------------

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Tuple
import altair as alt


# ==================== CONFIG BÁSICO ====================
st.set_page_config(
    page_title="Sistema QVM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS suave
st.markdown("""
<style>
.block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: .2px; }
hr { border: 0; border-top: 1px solid rgba(255,255,255,.08); margin: .6rem 0 1rem 0; }
[data-testid="stDataFrame"] tbody tr:hover { background: rgba(59,130,246,.08) !important; }
[data-testid="stCaptionContainer"] { opacity: .85; }
</style>
""", unsafe_allow_html=True)

# ============== IMPORTS DE TU PIPELINE ==============
from pipeline_factors import build_factor_frame
from dataclasses import dataclass, field
from fundamentals import apply_quality_guardrails as _apply_quality_guardrails
from fundamentals import download_guardrails_batch as _download_guardrails_batch
from scoring import (
    blend_breakout_qvm, build_momentum_proxy
)
from data_io import (
    run_fmp_screener, filter_universe, load_prices_panel, load_benchmark,
    DEFAULT_START, DEFAULT_END
)
from fundamentals import (
    download_fundamentals, build_vfq_scores_dynamic,
    download_guardrails_batch, apply_quality_guardrails
)
from pipeline import (
    apply_trend_filter, enrich_with_breakout,
    market_regime_on
)
from backtests import backtest_many

# NUEVOS IMPORTS (growth-aware)
from factors_growth_aware import compute_qvm_scores, apply_megacap_rules


# ------------------ CACHÉ DE I/O ------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_run_fmp_screener(
    *,
    limit: int,
    mcap_min: float,
    volume_min: int,
    ipo_days: int,
    cache_key: str,
) -> pd.DataFrame:
    """
    Pide universo a FMP con filtros básicos (sin ETFs, sin fondos, activo),
    luego aplica filtros post-request: market cap, volumen y antigüedad IPO.
    """
    df = run_fmp_screener(
        limit=limit,
        mcap_min=mcap_min,
        volume_min=volume_min,
        fetch_profiles=True,
        cache_key=cache_key,
        force=False,
        # Asegúrate que run_fmp_screener ya está mandando:
        #   isEtf=false
        #   isFund=false
        #   isActivelyTrading=true
    )

    # normalizamos nombres probables
    if "marketCap" in df.columns and "market_cap" not in df.columns:
        df["market_cap"] = pd.to_numeric(df["marketCap"], errors="coerce")

    # filtro de market cap adicional (por si la API igual devuelve basura)
    df = df[df["market_cap"] >= float(mcap_min)]

    # filtro de volumen si la API trae "volume"
    if "volume" in df.columns:
        df = df[pd.to_numeric(df["volume"], errors="coerce") >= float(volume_min)]

    # filtro IPO age si trae 'ipoDate'
    if "ipoDate" in df.columns:
        cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=int(ipo_days))
        df["ipoDate"] = pd.to_datetime(df["ipoDate"], errors="coerce", utc=True)
        df = df[df["ipoDate"] < cutoff]

    # asegurar sector
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    df["sector"] = df["sector"].astype(str).replace({"": "Unknown"}).fillna("Unknown")

    # quedarnos con columnas core + lo que ya traiga
    return df.reset_index(drop=True)

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_download_guardrails(symbols: Tuple[str, ...], cache_key: str) -> pd.DataFrame:
    return download_guardrails_batch(list(symbols), cache_key=cache_key, force=False)

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_download_fundamentals(
    symbols: Tuple[str, ...],
    cache_key: str,
    mc_pairs: Tuple[Tuple[str, float], ...] | None = None,
) -> pd.DataFrame:
    mc_map = dict(mc_pairs or ())
    return download_fundamentals(
        list(symbols),
        market_caps=mc_map,          # <-- hint
        cache_key=cache_key,
        force=False
    )

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_load_prices_panel(symbols, start, end, cache_key=""):
    return load_prices_panel(symbols, start, end, cache_key=cache_key, force=False)

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_load_benchmark(bench, start, end):
    return load_benchmark(bench, start, end)

# ------------------ PERF HELPERS ------------------
def perf_summary_from_returns(rets: pd.Series, periods_per_year: int) -> dict:
    r = rets.dropna().astype(float)
    if r.empty:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / periods_per_year if periods_per_year else np.nan
    cagr = eq.iloc[-1]**(1/yrs) - 1 if yrs and yrs > 0 else np.nan
    vol = r.std() * np.sqrt(periods_per_year) if r.std() > 0 else np.nan
    sharpe = (r.mean()*periods_per_year) / r.std() if r.std() > 0 else np.nan
    dd = eq/eq.cummax() - 1
    maxdd = dd.min()
    hit = (r > 0).mean()
    avg_win = r[r > 0].mean() if (r > 0).any() else np.nan
    avg_loss = r[r < 0].mean() if (r < 0).any() else np.nan
    payoff = (avg_win/abs(avg_loss)) if (avg_win and avg_loss) else np.nan
    expct = (hit*avg_win + (1-hit)*avg_loss) if (not np.isnan(hit) and avg_win is not None and avg_loss is not None) else np.nan
    return {
        "CAGR": float(cagr), "Vol_anual": float(vol), "Sharpe": float(sharpe),
        "MaxDD": float(maxdd), "HitRate": float(hit), "AvgWin": float(avg_win),
        "AvgLoss": float(avg_loss), "Payoff": float(payoff), "Expectancy": float(expct),
        "Periodos": int(len(r))
    }


def _enrich_sector_industry(uni_df: pd.DataFrame, src_df: pd.DataFrame) -> pd.DataFrame:
    out = uni_df.copy()
    need_sector = ("sector" not in out.columns) or (out["sector"].isna().mean() > 0.8 if "sector" in out.columns else True)
    have_cols = [c for c in ["sector", "industry"] if c in src_df.columns]
    if need_sector and have_cols:
        map_df = (
            src_df[["symbol"] + have_cols]
            .dropna(subset=["symbol"])
            .drop_duplicates("symbol", keep="last")
        )
        out = out.drop(columns=have_cols, errors="ignore").merge(map_df, on="symbol", how="left")


        # ⬇️ claves: no llamar fillna sobre un string
    if "sector" in out.columns:
        out["sector"] = out["sector"].astype(str).replace({"": "Unknown"}).fillna("Unknown")
    else:
        out["sector"] = "Unknown"

    if "industry" in out.columns:
        out["industry"] = out["industry"].astype(str).fillna("")
    else:
        out["industry"] = ""

    return out

def _as_series(x, index=None):
    import pandas as pd
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x, index=index)

def _ensure_sector_strings(df: pd.DataFrame, sector_col="sector", industry_col="industry") -> pd.DataFrame:
    import numpy as np, pandas as pd
    if sector_col not in df.columns:
        df[sector_col] = pd.Series(["Unknown"] * len(df), index=df.index)
    else:
        s = _as_series(df[sector_col], df.index)
        s = s.astype(str)
        s = s.replace({"": "Unknown"})
        s = s.where(~s.isna(), "Unknown")
        df[sector_col] = s

    if industry_col in df.columns:
        t = _as_series(df[industry_col], df.index)
        df[industry_col] = t.astype(str).where(~t.isna(), "")
    return df

def _as_list(x):
    return x if isinstance(x, (list, tuple, pd.Index, np.ndarray)) else [x]

# ==================== HEADER ====================
l, r = st.columns([0.85, 0.15])
with l:
    st.markdown("<h1 style='margin-bottom:0'>QVM Screener</h1>", unsafe_allow_html=True)
    st.caption("Momentum estructural + Breakout técnico + Value/Quality (VFQ)")
with r:
    st.caption(datetime.now().strftime("Actualizado: %d %b %Y %H:%M"))
st.markdown("<hr/>", unsafe_allow_html=True)

# ------------------ RANK HELPERS ------------------
def _probability_from_percentile(pct: pd.Series, beta: float = 6.0) -> pd.Series:
    s = pd.to_numeric(pct, errors="coerce").fillna(0.5).clip(0, 1)
    return 1.0 / (1.0 + np.exp(-beta * (s - 0.5)))

import numpy as np
import pandas as pd

def normalize_guard_diag(diag: pd.DataFrame, df_guard: pd.DataFrame | None = None) -> pd.DataFrame:
    d = (diag.copy() if isinstance(diag, pd.DataFrame) else pd.DataFrame())
    if d.empty:
        cols = ["symbol","profit_hits","coverage_count","net_issuance","asset_growth",
                "accruals_ta","netdebt_ebitda","pass_profit","pass_issuance",
                "pass_assets","pass_accruals","pass_ndebt","pass_coverage","pass_all","reason"]
        return pd.DataFrame(columns=cols)

    # Asegura 'symbol'
    if "symbol" not in d.columns:
        if d.index.name == "symbol":
            d = d.reset_index()
        elif isinstance(df_guard, pd.DataFrame) and "symbol" in df_guard.columns:
            d["symbol"] = df_guard["symbol"].values[:len(d)]
        else:
            d["symbol"] = pd.Index(range(len(d))).astype(str)

    token_map = {
        "pass_profit":   "profit_floor",
        "pass_issuance": "net_issuance",
        "pass_assets":   "asset_growth",
        "pass_accruals": "accruals_ta",
        "pass_ndebt":    "netdebt_ebitda",
        "pass_coverage": "vfq_coverage",
    }

    # Completa pass_* desde reason si faltan
    has_reason = "reason" in d.columns
    for col, tok in token_map.items():
        if col not in d.columns:
            d[col] = ~d["reason"].fillna("").str.contains(tok) if has_reason else np.nan

    # pass_all si falta
    checks = list(token_map.keys())
    if "pass_all" not in d.columns:
        d["pass_all"] = d[checks].all(axis=1) if all(c in d.columns for c in checks) else False

    # reason si falta
    if "reason" not in d.columns:
        def _mk_reason(row):
            r=[]
            if "pass_profit"   in d.columns and not bool(row.get("pass_profit", True)):     r.append("profit_floor")
            if "pass_issuance" in d.columns and not bool(row.get("pass_issuance", True)):   r.append("net_issuance")
            if "pass_assets"   in d.columns and not bool(row.get("pass_assets", True)):     r.append("asset_growth")
            if "pass_accruals" in d.columns and not bool(row.get("pass_accruals", True)):   r.append("accruals_ta")
            if "pass_ndebt"    in d.columns and not bool(row.get("pass_ndebt", True)):      r.append("netdebt_ebitda")
            if "pass_coverage" in d.columns and not bool(row.get("pass_coverage", True)):   r.append("vfq_coverage")
            return ",".join(r)
        d["reason"] = d.apply(_mk_reason, axis=1)

    return d


# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### ⚙️ Controles")
    preset = st.segmented_control("Preset", options=["Laxo", "Balanceado", "Estricto"], default="Balanceado")

    with st.expander("Universo & Screener", expanded=True):
        limit = st.slider("Límite del universo", 50, 1000, 300, 50)

        min_mcap = st.number_input(
            "MarketCap mínimo (USD)",
            value=5e8,
            step=1e8,
            format="%.0f"
        )

        volume_min = st.number_input(
            "Volumen mínimo diario",
            value=500_000,
            step=50_000,
            format="%.0f"
        )

        ipo_days = st.slider(
            "Antigüedad IPO (días)",
            90, 1500, 365, 30
        )

    with st.expander("Fundamentales & Guardrails", expanded=False):
        min_cov_guard = st.slider("Cobertura VFQ mínima (# métricas)", 1, 4, 2)
        profit_hits = st.slider("Pisos de rentabilidad (hits EBIT/CFO/FCF)", 0, 3, 2)
        max_issuance = st.slider("Net issuance máx.", 0.00, 0.10, 0.03, 0.01)
        max_assets = st.slider("Asset growth |y/y| máx.", 0.00, 0.50, 0.20, 0.01)
        max_accr = st.slider("Accruals/TA | | máx.", 0.00, 0.25, 0.10, 0.01)
        max_ndeb = st.slider("NetDebt/EBITDA máx.", 0.0, 6.0, 3.0, 0.5)

    with st.expander("Técnico — Tendencia & Breakout", expanded=True):
        use_and = st.toggle("MA200 Y Mom 12–1", value=False)
        require_breakout = st.toggle("Exigir Breakout para ENTRY", value=False)
        rvol_th = st.slider("RVOL (20d) mín.", 0.8, 2.5, 1.2, 0.1)
        closepos_th = st.slider("ClosePos mín.", 0.0, 1.0, 0.60, 0.05)
        p52_th = st.slider("Cercanía 52W High", 0.80, 1.00, 0.95, 0.01)
        updown_vol_th = st.slider("Up/Down Vol Ratio (20d)", 0.8, 3.0, 1.2, 0.1)
        min_hits = st.slider("Mínimo checks breakout (K de 4)", 1, 4, 3)
        atr_pct_min = st.slider("ATR pct (6–12m) mín.", 0.0, 1.0, 0.6, 0.05)
        use_rs_slope = st.toggle("Exigir RS slope > 0 (MA20)", value=False)

    with st.expander("Régimen & Fechas", expanded=False):
        bench = st.selectbox("Benchmark", ["SPY", "QQQ", "^GSPC"], index=0)
        risk_on = st.toggle("Exigir mercado Risk-ON", value=True)
        start = st.date_input("Inicio", value=pd.to_datetime(DEFAULT_START).date())
        end = st.date_input("Fin", value=pd.to_datetime(DEFAULT_END).date())

    with st.expander("Ranking avanzado", expanded=False):
        beta_prob = st.slider("Sensibilidad probabilidad (β)", 1.0, 12.0, 6.0, 0.5)
        top_n_show = st.slider("Top N a resaltar", 10, 100, 25, 5)

    st.markdown("---")
    run_btn = st.button("Ejecutar", use_container_width=True)

# Presets (sin pisar cambios del usuario)
if preset == "Laxo":
    rvol_th = min(rvol_th, 1.0); closepos_th = min(closepos_th, 0.55); p52_th = min(p52_th, 0.92); min_hits = min(min_hits, 2)
elif preset == "Estricto":
    rvol_th = max(rvol_th, 1.5); closepos_th = max(closepos_th, 0.65); p52_th = max(p52_th, 0.97); min_hits = max(min_hits, 3)

# cache tag por corrida
cache_tag = f"{int(min_mcap)}_{ipo_days}_{limit}"

# Estado del pipeline
if "pipeline_ready" not in st.session_state:
    st.session_state["pipeline_ready"] = False

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Universo", "Guardrails", "VFQ", "Señales", "QVM (growth-aware)", "Export", "Backtesting"]
)

# ==================== VFQ sidebar extra ====================
with st.sidebar:
    st.markdown("⚙️ Fundamentos (VFQ)")

    # Ajusta estas opciones a las columnas reales que produce tu DF de fundamentales
    # Nombres que sí existen/deriva build_vfq_scores_dynamic
    value_metrics_opts   = ["inv_ev_ebitda", "fcf_yield"]
    quality_metrics_opts = ["gross_profitability", "roic", "roa", "netMargin"]

    sel_value = st.multiselect("Métricas Value", options=value_metrics_opts, default=["inv_ev_ebitda", "fcf_yield"])
    sel_quality = st.multiselect("Métricas Quality", options=quality_metrics_opts, default=["gross_profitability", "roic"])

    c1, c2 = st.columns(2)
    with c1: w_value = st.slider("Peso Value", 0.0, 1.0, 0.5, 0.05)
    with c2: w_quality = st.slider("Peso Quality", 0.0, 1.0, 0.5, 0.05)

    method_intra = st.radio("Agregación intra-bloque", ["mean", "median", "weighted_mean"], index=0, horizontal=True)
    winsor_p = st.slider("Winsor p (cola)", 0.0, 0.10, 0.01, 0.005)
    size_buckets = st.slider("Buckets por tamaño", 1, 5, 3, 1)
    group_mode = st.selectbox("Agrupar por", ["sector", "sector|size"], index=1)
    min_cov = st.slider("Cobertura mín. (# métricas)", 0, 8, 1, 1)
    min_pct = st.slider("VFQ pct (intra-sector) mín.", 0.00, 1.00, 0.00, 0.01)

    st.session_state["min_cov"] = int(min_cov)
    st.session_state["min_pct"] = float(min_pct)

vfq_cfg = dict(
    value_metrics=sel_value,
    quality_metrics=sel_quality,
    w_value=float(w_value),
    w_quality=float(w_quality),
    method_intra=method_intra,
    winsor_p=float(winsor_p),
    size_buckets=int(size_buckets),
    group_mode=group_mode,
)

# ====== Paso 1: UNIVERSO ======
with tab1:
    st.subheader("Universo inicial")

    # ¿Necesitamos refrescar universo?
    # Sí si:
    #   - aún no existe en session_state
    #   - o apretaste "Ejecutar"
    if ("uni" not in st.session_state) or run_btn:
        raw_universe = _cached_run_fmp_screener(
            limit=limit,
            mcap_min=min_mcap,
            volume_min=volume_min,
            ipo_days=ipo_days,
            cache_key=cache_tag,
        )

        # Aseguramos columnas mínimas:
        # symbol (ticker), sector, market_cap
        out = raw_universe.copy()

        if "symbol" not in out.columns:
            # si FMP devolviera "ticker" en vez de "symbol"
            if "ticker" in out.columns:
                out["symbol"] = out["ticker"].astype(str)
            else:
                out["symbol"] = ""

        if "market_cap" not in out.columns:
            if "marketCap" in out.columns:
                out["market_cap"] = pd.to_numeric(out["marketCap"], errors="coerce")
            else:
                out["market_cap"] = np.nan

        if "sector" not in out.columns:
            out["sector"] = "Unknown"
        else:
            s = out["sector"].astype(str)
            s = s.replace({"": "Unknown"})
            s = s.where(~s.isna(), "Unknown")
            out["sector"] = s

        # limpieza final de NaNs raros en symbol
        out = (
            out[["symbol", "sector", "market_cap"]]
            .dropna(subset=["symbol"])
            .reset_index(drop=True)
        )

        # Guardamos en session_state para que otras tabs usen lo mismo
        st.session_state["uni"] = out.copy()

    # ahora leemos lo que tengamos guardado
    uni_df = st.session_state["uni"].copy()

    total_raw = len(uni_df)
    # acá podrías aplicar más filtros (por ejemplo quitar OTC, penny, China ADR si quieres),
    # y contar cuántos sobrevivieron. Por ahora son iguales:
    total_filtrado = len(uni_df)

    c1, c2 = st.columns(2)
    c1.metric("Screener", f"{total_raw}")
    c2.metric("Tras filtros básicos", f"{total_filtrado}")

    st.dataframe(
        uni_df.head(50),
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Esta tabla se guarda como st.session_state['uni'] y alimenta las demás pestañas."
    )



## ====== Paso 2: FUNDAMENTALES & GUARDRAILS ======
with tab2:
    st.subheader("Guardrails")

    uni = st.session_state.get("uni", pd.DataFrame())
    if uni is None or uni.empty or "symbol" not in uni.columns:
        st.info("Primero corre Universo (tab1).")
        st.stop()

    syms = uni["symbol"].dropna().astype(str).unique().tolist()
    if not syms:
        st.info("No hay símbolos en el universo.")
        st.stop()

    # Recalcular todo acá (fuente única)
    df_all = build_factor_frame(syms)

    # injertar sector / market_cap desde universo
    base_cols = ["symbol","sector","market_cap"]
    df_all = (
        df_all.drop(columns=["sector","market_cap"], errors="ignore")
              .merge(
                  uni[[c for c in base_cols if c in uni.columns]],
                  on="symbol",
                  how="left"
              )
    )

    # mascara estricta
    strict_mask = df_all.get("pass_all", False) == True
    kept_raw = (
        df_all.loc[strict_mask, ["symbol"]]
              .drop_duplicates()
              .reset_index(drop=True)
    )

    # guarda para otros tabs si quieres mantener ese flujo
    st.session_state["kept"] = kept_raw
    st.session_state["guard_diag"] = df_all.copy()

    total = len(df_all)
    pasan = int(strict_mask.sum())
    rechaz = total - pasan

    c1, c2, c3 = st.columns(3)
    c1.metric("Pasan guardrails estrictos", f"{pasan}")
    # 'relajado' por ahora lo igualo a pasan; luego lo puedes tunear
    c2.metric("Candidatos saludables (relajado)", f"{pasan}")
    c3.metric("Rechazados totales", f"{rechaz}")

    cols_show = [
        "symbol","sector","pass_all",
        "profit_hits","coverage_count",
        "asset_growth","accruals_ta","netdebt_ebitda",
        "pass_profit","pass_issuance","pass_assets",
        "pass_accruals","pass_ndebt","pass_coverage",
    ]
    cols_show = [c for c in cols_show if c in df_all.columns]

    with st.expander(
        f"Detalle guardrails (estricto): {pasan} / {total}",
        expanded=True
    ):
        st.dataframe(
            df_all[cols_show].sort_values("symbol"),
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "pass_all = pasó TODAS las barreras simultáneamente. "
        "coverage_count = cuánta info fundamental tenemos disponible."
    )

# ====== Paso 3: VFQ ======
# ====== Paso 3: VFQ (PARCHE COMPLETO + UI BONITA) ======


def _fmt_mcap(x):
    try:
        x = float(x)
        if x >= 1e12:  return f"${x/1e12:.2f}T"
        if x >= 1e9:   return f"${x/1e9:.2f}B"
        if x >= 1e6:   return f"${x/1e6:.2f}M"
        return f"${x:,.0f}"
    except Exception:
        return ""

def _numcol(df: pd.DataFrame, col: str) -> pd.Series:
    import pandas as pd
    if col not in df.columns:
        return pd.Series([float("nan")] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


with tab3:
    st.subheader("VFQ (Value / Quality / Flow)")

    kept = st.session_state.get("kept", pd.DataFrame())
    uni  = st.session_state.get("uni",  pd.DataFrame())

    if kept is None or kept.empty or "symbol" not in kept.columns:
        st.warning("No hay símbolos aprobados por Guardrails. Ajusta la pestaña Guardrails.")
        st.stop()

    kept_syms = kept["symbol"].dropna().astype(str).unique().tolist()
    if not kept_syms:
        st.warning("La lista kept está vacía.")
        st.stop()

    # volvemos a pedir la data pero SOLO para los kept (para mantener consistencia)
    df_vfq_all = build_factor_frame(kept_syms)

    # agrego sector / market_cap del universo
    df_vfq_all = (
        df_vfq_all
        .drop(columns=["sector","market_cap"], errors="ignore")
        .merge(
            uni[["symbol","sector","market_cap"]],
            on="symbol",
            how="left"
        )
    )

    # sliders para filtrar VFQ
    c1, c2, c3 = st.columns(3)
    with c1:
        min_quality = st.slider("Min Quality neut.", 0.0, 1.0, 0.3, 0.01)
        min_value   = st.slider("Min Value neut.",   0.0, 1.0, 0.3, 0.01)
        max_ndebt   = st.slider("Max NetDebt/EBITDA", 0.0, 5.0, 2.0, 0.1)
    with c2:
        min_acc_pct = st.slider("Accruals limpios (% mínimo)", 0, 100, 30, 1)
        min_hits    = st.slider("Min hits (breakout hits)",     0, 5,   1, 1)
        min_rvol20  = st.slider("Min RVOL20",                   0.0, 5.0, 1.2, 0.05)
    with c3:
        min_breakout = st.slider("Min BreakoutScore", 0, 100, 50, 1)
        topN_prob    = st.slider("Top N por prob_up", 5, 100, 30, 1)

    mask = pd.Series(True, index=df_vfq_all.index, dtype=bool)

    mask &= df_vfq_all["quality_adj_neut"].fillna(0) >= float(min_quality)
    mask &= df_vfq_all["value_adj_neut"].fillna(0)   >= float(min_value)
    mask &= df_vfq_all["hits"].fillna(0)             >= int(min_hits)
    mask &= df_vfq_all["BreakoutScore"].fillna(0)    >= float(min_breakout)
    mask &= df_vfq_all["RVOL20"].fillna(0)           >= float(min_rvol20)

    mask &= (
        df_vfq_all["netdebt_ebitda"].isna()
        | (df_vfq_all["netdebt_ebitda"] <= float(max_ndebt))
    )

    mask &= (
        df_vfq_all["acc_pct"].isna()
        | (df_vfq_all["acc_pct"] >= float(min_acc_pct))
    )

    df_keep_vfq = df_vfq_all.loc[mask].copy()

    # ranking por prob_up (o BreakoutScore si prob_up no tiene señal)
    if df_keep_vfq["prob_up"].notna().any():
        df_keep_vfq = df_keep_vfq.sort_values("prob_up", ascending=False)
    else:
        df_keep_vfq = df_keep_vfq.sort_values("BreakoutScore", ascending=False)

    vfq_top = df_keep_vfq.head(int(topN_prob)).copy()

    # Mostrar
    st.markdown("### 🟢 Selección VFQ filtrada")
    st.dataframe(
        vfq_top[[c for c in [
            "symbol","netdebt_ebitda","accruals_ta","sector","market_cap",
            "quality_adj_neut","value_adj_neut","acc_pct",
            "hits","BreakoutScore","RVOL20","prob_up"
        ] if c in vfq_top.columns]],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### 🧹 Rechazados por VFQ")
    rejected_syms = sorted(set(kept_syms) - set(df_keep_vfq["symbol"]))
    rej_view = df_vfq_all[df_vfq_all["symbol"].isin(rejected_syms)].copy()

    st.dataframe(
        rej_view[[c for c in [
            "symbol","sector","market_cap",
            "quality_adj_neut","value_adj_neut",
            "netdebt_ebitda","acc_pct","BreakoutScore",
            "hits","RVOL20","prob_up"
        ] if c in rej_view.columns]],
        use_container_width=True,
        hide_index=True
    )




    # Guardar para señales técnicas
    st.session_state["vfq_top"]   = vfq_top[["symbol"]].drop_duplicates()
    st.session_state["vfq_table"] = vfq_top.reset_index(drop=True)
    st.session_state["pipeline_ready"] = True




# ====== Paso 4: SEÑALES (placeholder si tu lógica está en otro módulo) ======
with tab4:
    st.subheader("Señales (Técnico)")

    # Vamos a construir final_df pase lo que pase, sin matar la app:
    final_df = pd.DataFrame()

    try:
        # =========================
        # 1. Traer insumos desde sesión
        # =========================
        vfq_top   = st.session_state.get("vfq_top",   pd.DataFrame())
        vfq_table = st.session_state.get("vfq_table", pd.DataFrame())

        if vfq_top is None or vfq_top.empty or "symbol" not in vfq_top.columns:
            st.warning("No hay lista VFQ priorizada. Corre la pestaña VFQ primero.")
            st.stop()

        syms = (
            vfq_top["symbol"]
            .dropna().astype(str).unique().tolist()
        )
        if not syms:
            st.warning("VFQ no entregó símbolos válidos.")
            st.stop()

        # parámetros técnicos que Tab3 dejó en session_state
        bench_local        = st.session_state.get("bench_local", "SPY")
        use_and_local      = bool(st.session_state.get("use_and_local", True))

        # thresholds de breakout / flujo
        rvol_th_local      = float(st.session_state.get("rvol_th_local",      1.5))
        closepos_th_local  = float(st.session_state.get("closepos_th_local",  0.8))
        p52_th_local       = float(st.session_state.get("p52_th_local",       0.97))
        updown_vol_local   = float(st.session_state.get("updown_vol_local",   1.2))
        min_hits_local     = int(st.session_state.get("min_hits_local",       2))

        use_rs_slope_loc   = bool(st.session_state.get("use_rs_slope_loc",    True))
        risk_on_local      = bool(st.session_state.get("risk_on_local",       True))

        start_local        = st.session_state.get("start_local", "2022-01-01")
        end_local          = st.session_state.get("end_local",   None)
        cache_tag_local    = st.session_state.get("cache_tag_local", "v1")

        # safety: si nos llegaron None o strings vacíos raros, default razonable
        if not start_local or (isinstance(start_local, float) and pd.isna(start_local)):
            start_local = "2022-01-01"
        if isinstance(end_local, float) and pd.isna(end_local):
            end_local = None

        # =========================
        # 2. Cargar precios + señales técnicas
        # =========================
        with st.status("Cargando precios y calculando señales…", expanded=False) as status:
            # panel: precios OHLCV por símbolo
            panel = _cached_load_prices_panel(
                syms,
                start=start_local,
                end=end_local,
                cache_key=cache_tag_local,
            )

            # filtro de tendencia (ej. MA20 > MA50 etc)
            trend_df = apply_trend_filter(
                panel,
                use_and_condition=use_and_local,
            )

            # breakout / flujo: RVOL, cerca del high 52w, etc.
            bo_df = enrich_with_breakout(
                panel,
                rvol_lookback=20,
                rvol_th=float(rvol_th_local),
                closepos_th=float(closepos_th_local),
                p52_th=float(p52_th_local),
                updown_vol_th=float(updown_vol_local),
                bench_series=None,          # benchmark RS se puede enchufar acá si quieres
                min_hits=int(min_hits_local),
                use_rs_slope=bool(use_rs_slope_loc),
                rs_min_slope=0.0,
            )

            # merge técnico
            sig_df = (
                trend_df.merge(bo_df, on="symbol", how="outer")
                        .sort_values("symbol")
                        .reset_index(drop=True)
            )

            # régimen de mercado (risk-on / risk-off)
            bench_df = _cached_load_benchmark(
                bench_local,
                start=start_local,
                end=end_local,
            )
            ok_market = market_regime_on(bench_df, panel)

            # si queremos respetar risk_on, apagamos señales en mercado feo
            if risk_on_local and not ok_market:
                if "signal_trend" in sig_df.columns:
                    sig_df["signal_trend"] = False
                if "signal_breakout" in sig_df.columns:
                    sig_df["signal_breakout"] = False
                sig_df["risk_on"] = False
            else:
                sig_df["risk_on"] = True

            status.update(label="Señales listas", state="complete")

        # Guardar técnico crudo para debugging / export
        st.session_state["signals"] = sig_df

        # =========================
        # 3. Merge fundamental (vfq_table) + técnico (sig_df)
        # =========================
        if vfq_table is None or vfq_table.empty:
            # fallback mínimo: usa sólo símbolo para podernos al menos mostrar técnico
            base_for_merge = vfq_top.drop_duplicates().copy()
        else:
            base_for_merge = vfq_table.copy()

        final_df = base_for_merge.merge(sig_df, on="symbol", how="left")

        # lo guardamos en sesión como "estado maestro" de salida
        st.session_state["final_df"] = final_df

    except Exception as e:
        # Algo tronó arriba (por ejemplo conexión de precios o lo técnico)
        st.error(f"Error calculando señales: {e}")
        # intento rescatar lo último bueno que tengamos
        final_df = st.session_state.get("final_df", pd.DataFrame())

    # =========================
    # 4. UI de ranking final
    # =========================
    if final_df is None or final_df.empty:
        st.info("Sin resultados combinados (fundamental + técnico).")
    else:
        st.markdown("### ✅ Lista final (fundamental + técnico)")

        # columnas candidatas que queremos mostrar si existen
        cols_candidate = [
            "symbol","sector","market_cap",
            "quality_adj_neut","value_adj_neut","acc_pct",
            "netdebt_ebitda","hits","BreakoutScore","RVOL20","prob_up",
            "signal_trend","signal_breakout","risk_on",
            "ClosePos","P52","UDVol20","ATR_pct","rs_ma20_slope"
        ]
        cols_candidate = [c for c in cols_candidate if c in final_df.columns]

        # =========================
        # 5. Ranking heurístico
        # =========================
        ranked = final_df.copy()

        # base de score_rank
        ranked["score_rank"] = 0

        # 1. breakout activo pesa mucho (True -> 1, False -> 0)
        if "signal_breakout" in ranked.columns:
            ranked["score_rank"] = ranked["signal_breakout"].fillna(False).astype(int)

        # 2. luego prob_up (si existe); si no, BreakoutScore
        if "prob_up" in ranked.columns:
            # prob_up puede venir float o NaN
            ranked["score_rank"] = (
                ranked["score_rank"] * 1000
                + pd.to_numeric(ranked["prob_up"], errors="coerce").fillna(0) * 100
            )
        elif "BreakoutScore" in ranked.columns:
            ranked["score_rank"] = (
                ranked["score_rank"] * 1000
                + pd.to_numeric(ranked["BreakoutScore"], errors="coerce").fillna(0)
            )

        # mayor score_rank arriba
        ranked = ranked.sort_values("score_rank", ascending=False)

        st.dataframe(
            ranked[cols_candidate],
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Interpretación rápida:\n"
            "- Arriba = mejor mezcla de fundamentales (quality/value/deuda), flujo/compras (hits, BreakoutScore, RVOL20), "
            "y señal técnica lista para entrar.\n"
            "- signal_breakout=True + risk_on=True = entrada agresiva.\n"
            "- Si risk_on=False, el sistema ve mercado feo y apaga las señales aunque la acción esté buena."
        )



# ====== Paso 5: QVM (growth-aware) ======
# ============ QVM CORE: FundamentalStandardizer ============



@dataclass
class FundamentalStandardizer:
    """
    Normaliza/alia columnas fundamentales y técnicas para que QVM funcione
    sin 'Unknown' de sector ni Series 2D. Incluye coalesce *_vfq → limpio,
    coerción numérica, y derivaciones básicas (market_cap, net_debt, etc.)
    """
    alias: dict = field(default_factory=lambda: {
        # básicos
        "market_cap": ["market_cap","marketCap","marketCap_unified","mkt_cap","Market Cap"],
        "sector":     ["sector","Sector","industry","Industry","gicsSector","GICS_Sector"],
        # value
        "ev":                ["ev","enterpriseValue","EnterpriseValue","EV"],
        "ebitda_ttm":        ["ebitda_ttm","EBITDA_TTM","ebitdaTrailingTwelveMonths","ebitdaTTM"],
        "ebitda_ntm":        ["ebitda_ntm","EBITDA_NTM","ebitdaForward","ebitdaNextTwelveMonths"],
        "gross_profit_ttm":  ["gross_profit_ttm","grossProfitTTM","GrossProfitTTM"],
        "sales_ntm":         ["sales_ntm","revenueNTM","RevenueNTM","salesForward","revenueForward","revenue_ntm"],
        "capex_ttm":         ["capex_ttm","capexTTM","CapExTTM","capitalExpenditureTTM"],
        "sbc_ttm":           ["sbc_ttm","stockBasedCompTTM","shareBasedCompTTM","SBC_TTM"],
        "fcf_ttm":           ["fcf_ttm","freeCashFlowTTM","FCF_TTM"],
        "fcf_5y_median":     ["fcf_5y_median","FCF_5Y_MEDIAN","fcfMedian5Y"],
        # quality / intangible
        "rd_expense_ttm":        ["rd_expense_ttm","researchDevelopmentTTM","R&D_TTM","researchAndDevTTM"],
        "operating_income_ttm":  ["operating_income_ttm","operatingIncomeTTM","OperatingIncomeTTM","ebitTTM"],
        "total_assets_ttm":      ["total_assets_ttm","totalAssetsTTM","TotalAssetsTTM","totalAssets"],
        "net_debt_ttm":          ["net_debt_ttm","netDebtTTM","NetDebtTTM","netDebt"],
        "total_debt_ttm":        ["total_debt_ttm","totalDebtTTM","TotalDebtTTM","totalDebt"],
        "cash_ttm":              ["cash_ttm","cashAndEquivalentsTTM","cashAndCashEquivalentsTTM","cashAndEquivalents"],
        "noa_ttm":               ["noa_ttm","netOperatingAssetsTTM","NOA_TTM"],
        "invested_capital_ttm":  ["invested_capital_ttm","investedCapitalTTM","InvestedCapitalTTM"],
        "current_liabilities_ttm":["current_liabilities_ttm","currentLiabilitiesTTM","CurrentLiabilitiesTTM"],
        "tax_rate":              ["tax_rate","effectiveTaxRate","effectiveTaxRateTTM"],
        "op_margin_hist":        ["op_margin_hist","operatingMarginHistory","opMarginHistory"],
        # técnico
        "BreakoutScore": ["BreakoutScore"],
        "RVOL20":        ["RVOL20"],
        "UDVol20":       ["UDVol20"],
        "hits":          ["hits"],
        "P52":           ["P52"],
        "ClosePos":      ["ClosePos"],
        "momentum_score":["momentum_score","mom","momo","mom_sig","mom_px","momentum_score_prices"],
    })

    def _first_present(self, df: pd.DataFrame, std: str):
        """Toma la primera columna candidata presente; si no está, intenta *_vfq."""
        # ya existe la estándar
        if std in df.columns:
            return df[std]
        # busca alias directos
        for c in self.alias.get(std, []):
            if c in df.columns:
                return df[c]
        # busca versión *_vfq
        cand_vfq = f"{std}_vfq"
        if cand_vfq in df.columns:
            return df[cand_vfq]
        # busca alias con _vfq
        for c in self.alias.get(std, []):
            c_v = f"{c}_vfq"
            if c_v in df.columns:
                return df[c_v]
        # default vacío
        return pd.Series(np.nan, index=df.index)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Elimina columnas duplicadas por nombre (causa Series 2D)
        out = out.loc[:, ~out.columns.duplicated(keep="last")]

        # Asegura 'symbol'
        if "symbol" not in out.columns:
            raise ValueError("Falta columna 'symbol'")

        # Construye y coalesce columnas estándar
        std_cols = [
            # básicos
            "market_cap","sector",
            # value
            "ev","ebitda_ttm","ebitda_ntm","gross_profit_ttm","sales_ntm",
            "capex_ttm","sbc_ttm","fcf_ttm","fcf_5y_median",
            # quality/intangible
            "rd_expense_ttm","operating_income_ttm","total_assets_ttm",
            "net_debt_ttm","total_debt_ttm","cash_ttm","noa_ttm",
            "invested_capital_ttm","current_liabilities_ttm","tax_rate","op_margin_hist",
            # técnico
            "BreakoutScore","RVOL20","UDVol20","hits","P52","ClosePos","momentum_score"
        ]
        for std in std_cols:
            s = self._first_present(out, std)
            # promedia fila a fila si llegó DataFrame (2D)
            if isinstance(s, pd.DataFrame):
                s = s.apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
            out[std] = s

        # Sector: si queda Unknown y hay industry, úsala
        if out["sector"].isna().all() and "industry" in out.columns:
            out["sector"] = out["industry"].astype(str)

        out["sector"] = out["sector"].astype(str).fillna("Unknown")
        # Market cap: coalesce si faltó
        if out["market_cap"].isna().all():
            for alt in ["marketCap_unified","marketCap","mkt_cap"]:
                if alt in df.columns:
                    out["market_cap"] = pd.to_numeric(df[alt], errors="coerce")
                    break

        # Derivación net_debt si falta: total_debt - cash
        if out["net_debt_ttm"].isna().all():
            td = pd.to_numeric(out.get("total_debt_ttm"), errors="coerce")
            cs = pd.to_numeric(out.get("cash_ttm"), errors="coerce")
            out["net_debt_ttm"] = (td - cs)

        # Coerción numérica masiva
        num_cols = [
            "market_cap","ev","ebitda_ttm","ebitda_ntm","gross_profit_ttm","sales_ntm",
            "capex_ttm","sbc_ttm","fcf_ttm","fcf_5y_median","rd_expense_ttm",
            "operating_income_ttm","total_assets_ttm","net_debt_ttm","noa_ttm",
            "invested_capital_ttm","current_liabilities_ttm","tax_rate",
            "BreakoutScore","RVOL20","UDVol20","hits","P52","ClosePos","momentum_score"
        ]
        for c in num_cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")

        # Momentum seguro
        if "momentum_score" not in out.columns or out["momentum_score"].isna().all():
            out["momentum_score"] = 0.0
        else:
            # por si vinieron múltiples fuentes mezcladas
            if isinstance(out["momentum_score"], pd.DataFrame):
                out["momentum_score"] = out["momentum_score"].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
            else:
                out["momentum_score"] = pd.to_numeric(out["momentum_score"], errors="coerce").fillna(0.0)

        # Limpieza final de duplicados de nombre
        out = out.loc[:, ~out.columns.duplicated(keep="last")]
        return out

# =================== TAB 5: QVM (growth-aware) ===================
with tab5:
    st.subheader("QVM (growth-aware) con Guardrails → Técnico → Ranking")

    # --- Parámetros UI ---
    colA, colB, colC = st.columns(3)
    with colA:
        th_q_neut   = st.slider("Min Quality neut.", 0.0, 1.0, 0.0, 0.05)   # puedes subir a 0.2–0.3
        th_v_neut   = st.slider("Min Value neut.",   0.0, 1.0, 0.0, 0.05)
        th_ndebt    = st.slider("Max NetDebt/EBITDA", 0.0, 8.0, 3.0, 0.5)
    with colB:
        th_acc_p    = st.slider("Accruals (NOA) percentil mínimo", 0, 100, 30, 5)
        beta_prob   = st.slider("β prob_up (logit)", 2.0, 12.0, 6.0, 0.5)
        top_n_show  = st.slider("Top N por prob_up", 5, 100, 30, 5)
    with colC:
        th_bo       = st.slider("Min BreakoutScore", 0, 100, 60, 1)
        th_hits     = st.slider("Min hits", 0, 5, 2, 1)
        th_rvol     = st.slider("Min RVOL20", 0.0, 5.0, 1.5, 0.1)

    try:
        # ------------------- Orígenes -------------------
        sig_df        = st.session_state.get("signals", pd.DataFrame())
        vfq_df        = st.session_state.get("vfq", pd.DataFrame())
        uni_df        = st.session_state.get("uni", pd.DataFrame())
        kept_df       = st.session_state.get("kept", pd.DataFrame())
        panel_prices  = st.session_state.get("panel_prices")  # opcional

        if sig_df.empty:
            st.info("Primero corre **Señales**.")
            st.stop()

        # ------------------- BASE desde señales -------------------
        base_keep = ["symbol","sector","industry","marketCap","marketCap_unified",
                     "BreakoutScore","ClosePos","P52","RVOL20","UDVol20","hits","rs_ma20_slope"]
        base_cols = ["symbol"] + [c for c in base_keep if c in sig_df.columns and c != "symbol"]
        base = sig_df[base_cols].drop_duplicates("symbol").copy()

        # Merge VFQ (fundamentales) y UNI (sector/mcap) si existen
        if isinstance(vfq_df, pd.DataFrame) and not vfq_df.empty:
            base = base.merge(vfq_df, on="symbol", how="left", suffixes=("", "_vfq"))
        if isinstance(uni_df, pd.DataFrame) and {"symbol"}.issubset(uni_df.columns):
            extra = [c for c in ["symbol","sector","industry","marketCap","market_cap"] if c in uni_df.columns]
            if len(extra) > 1:
                base = base.merge(uni_df[extra].drop_duplicates("symbol"), on="symbol", how="left", suffixes=("", "_uni"))
        if isinstance(kept_df, pd.DataFrame) and "symbol" in kept_df.columns:
            base = base.merge(kept_df.drop_duplicates("symbol")[["symbol"]], on="symbol", how="right")

        # ------------------- Momentum proxy (coalesce) -------------------
        # 1) desde señales (idempotente)
        mom_sig = build_momentum_proxy(sig_df)
        if isinstance(mom_sig, pd.Series) and not mom_sig.empty:
            base = base.merge(mom_sig.to_frame("mom_sig"), left_on="symbol", right_index=True, how="left")

        # 2) desde precios (si está panel_prices)
        mom_px = None
        try:
            if isinstance(panel_prices, pd.DataFrame):
                df_long = panel_prices.copy()
                if "symbol" not in df_long.columns and "ticker" in df_long.columns:
                    df_long = df_long.rename(columns={"ticker": "symbol"})
                mom_px = build_momentum_proxy(df_long, price_col="close", id_col="symbol", date_col="date")
            elif isinstance(panel_prices, dict):
                frames = []
                for sym, dfp in panel_prices.items():
                    if isinstance(dfp, pd.DataFrame) and "close" in dfp.columns:
                        tmp = dfp.reset_index().rename(columns={"index": "date"} if "date" not in dfp.columns else {})
                        tmp["symbol"] = sym
                        frames.append(tmp[["symbol","date","close"]])
                if frames:
                    df_long = pd.concat(frames, ignore_index=True)
                    mom_px = build_momentum_proxy(df_long, price_col="close", id_col="symbol", date_col="date")
        except Exception:
            mom_px = None

        if isinstance(mom_px, pd.Series) and not mom_px.empty:
            base = base.merge(mom_px.to_frame("mom_px"), left_on="symbol", right_index=True, how="left")

        # Coalesce momentum → 'momentum_score' 1D
        cand_moms = [c for c in ["momentum_score","mom_sig","mom_px","momentum_score_prices"] if c in base.columns]
        if cand_moms:
            base["momentum_score"] = base[cand_moms].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
            base.drop(columns=[c for c in ["mom_sig","mom_px","momentum_score_prices"] if c in base.columns],
                      inplace=True, errors="ignore")
        else:
            base["momentum_score"] = 0.0

        # ------------------- Estandarizar fundamentales (alias/num/sector) -------------------
        if 'FundamentalStandardizer' not in globals():
            st.error("No se encontró FundamentalStandardizer. Asegúrate de haber pegado el QVM CORE arriba.")
            st.stop()

        base = FundamentalStandardizer().fit_transform(base)

        # Si no hay 'sector' pero sí 'industry', úsala como sector
        if base["sector"].eq("Unknown").all() and "industry" in base.columns:
            sec = base["industry"].astype(str).fillna("Unknown")
            base["sector"] = sec

        # ------------------- Cálculo QVM -------------------
        qvm_df = compute_qvm_scores(
            base,
            w_quality=0.40, w_value=0.25, w_momentum=0.35,
            momentum_col="momentum_score",
            sector_col="sector",
            mcap_col="market_cap"
        )

        # Guardrails especiales megacaps
        qvm_df = apply_megacap_rules(
            qvm_df,
            momentum_col="momentum_score",
            quality_col="quality_adj_neut",
            value_col="value_adj_neut"
        )

        # ------------------- Métricas auxiliares (deuda & accruals) -------------------
        def _pct(s: pd.Series) -> pd.Series:
            s = pd.to_numeric(s, errors="coerce")
            return s.rank(pct=True, method="average")

        # percentiles intra-sector para depurar si quieres ver dónde falla
        qvm_df["q_pct"]  = qvm_df.groupby("sector")["quality_adj_neut"].transform(_pct).fillna(0.5)
        qvm_df["v_pct"]  = qvm_df.groupby("sector")["value_adj_neut"].transform(_pct).fillna(0.5)
        qvm_df["vfq_pct"] = 0.6*qvm_df["q_pct"] + 0.4*qvm_df["v_pct"]

        # NetDebt/EBITDA (robusto a numpy.float64)
        net_debt = pd.to_numeric(qvm_df.get("net_debt_ttm"), errors="coerce")
        ebitda_ttm = pd.to_numeric(qvm_df.get("ebitda_ttm", qvm_df.get("ebitda_ntm")), errors="coerce").abs()
        qvm_df["ndebt_ebitda"] = (net_debt / (ebitda_ttm + 1e-9)).replace([np.inf, -np.inf], np.nan)

        # Accruals proxy (percentil de NOA; si falta NOA, no bloquea)
        noa = pd.to_numeric(qvm_df.get("noa_ttm"), errors="coerce")
        if noa.isna().all():
            qvm_df["acc_pct"] = 1.0
        else:
            qvm_df["acc_pct"] = noa.rank(pct=True, method="average")

        # ------------------- 1) Guardrails fundamentales -------------------
        gr = pd.Series(True, index=qvm_df.index, name="pass_guardrails")
        gr &= (pd.to_numeric(qvm_df["quality_adj_neut"], errors="coerce") > th_q_neut)
        gr &= (pd.to_numeric(qvm_df["value_adj_neut"],   errors="coerce") > th_v_neut)
        gr &= (qvm_df["ndebt_ebitda"].isna() | (qvm_df["ndebt_ebitda"] <= th_ndebt))
        gr &= (pd.to_numeric(qvm_df["acc_pct"], errors="coerce") >= (th_acc_p/100.0))
        qvm_df["pass_guardrails"] = gr

        # Tabla de rechazados (para depurar)
        rej_cols = [c for c in ["symbol","sector","market_cap","quality_adj_neut","value_adj_neut",
                                "ndebt_ebitda","acc_pct","BreakoutScore"] if c in qvm_df.columns]
        st.caption("🚧 Rechazados por guardrails")
        st.dataframe(qvm_df.loc[~gr, rej_cols].sort_values(["quality_adj_neut","value_adj_neut"], ascending=True),
                     use_container_width=True, hide_index=True)

        # ------------------- 2) Filtro técnico (entre los que pasan guardrails) -------------------
        tech_ok = pd.Series(True, index=qvm_df.index, name="pass_tech")
        tech_ok &= (pd.to_numeric(qvm_df.get("BreakoutScore"), errors="coerce") >= th_bo)
        tech_ok &= (pd.to_numeric(qvm_df.get("hits"), errors="coerce").fillna(0) >= th_hits)
        tech_ok &= (pd.to_numeric(qvm_df.get("RVOL20"), errors="coerce").fillna(0) >= th_rvol)
        qvm_df["pass_tech"] = tech_ok

        eligibles_pre_tech = qvm_df.loc[gr, ["symbol","sector","market_cap","value_adj_neut",
                                             "quality_adj_neut","qvm_score","ndebt_ebitda","acc_pct","vfq_pct"]]
        st.caption("✅ Elegibles por guardrails (antes del técnico)")
        st.dataframe(eligibles_pre_tech.sort_values(["quality_adj_neut","value_adj_neut","qvm_score"], ascending=False),
                     use_container_width=True, hide_index=True)

        # ------------------- 3) Ranking final -------------------
        def _z(s: pd.Series) -> pd.Series:
            s = pd.to_numeric(s, errors="coerce")
            mu, sd = s.mean(skipna=True), s.std(skipna=True)
            return (s - mu) / (sd if (sd and sd > 0) else 1.0)

        if "BreakoutScore" in qvm_df.columns:
            qvm_z = _z(qvm_df["qvm_score"])
            bo_z  = _z(qvm_df["BreakoutScore"])
            final_alpha = 0.70*qvm_z + 0.30*bo_z
        else:
            final_alpha = qvm_df["qvm_score"].rank(pct=True, method="average")

        qvm_df["final_alpha"]     = final_alpha
        qvm_df["final_alpha_pct"] = pd.to_numeric(final_alpha, errors="coerce").rank(pct=True, method="average")

        pct = qvm_df["final_alpha_pct"].clip(0, 1).fillna(0.5)
        qvm_df["prob_up"] = 1.0 / (1.0 + np.exp(-beta_prob * (pct - 0.5)))

        final_mask = qvm_df["pass_guardrails"] & qvm_df["pass_tech"]
        final_cols = [c for c in [
            "symbol","sector","market_cap","qvm_score","final_alpha","final_alpha_pct","prob_up",
            "value_adj_neut","quality_adj_neut","vfq_pct","momentum_score",
            "BreakoutScore","hits","RVOL20","ClosePos","P52",
            "ndebt_ebitda","acc_pct","mega_exception_ok","quality_too_low"
        ] if c in qvm_df.columns]

        st.metric("Seleccionadas (guardrails + técnico)", int(final_mask.sum()))
        st.dataframe(
            qvm_df.loc[final_mask, final_cols].sort_values(
                ["prob_up","final_alpha","qvm_score"], ascending=False
            ),
            use_container_width=True, hide_index=True
        )

        # ----------- Tabla técnica: ordenar por hits y BreakoutScore -----------
        st.subheader("Orden técnico (hits y BreakoutScore)")
        tech_cols = [c for c in ["symbol","hits","BreakoutScore","RVOL20","ClosePos","P52"] if c in qvm_df.columns]
        tech_table = qvm_df.loc[gr, tech_cols].copy()
        st.dataframe(
            tech_table.sort_values(["hits","BreakoutScore","RVOL20"], ascending=False),
            use_container_width=True, hide_index=True
        )

        # Guarda en sesión
        st.session_state["qvm"] = qvm_df

    except Exception as e:
        st.error(f"Error en QVM growth-aware: {e}")

# ====== Paso 6: EXPORT ======
with tab6:
    st.subheader("Exportar / Guardar ")
    uni_s  = st.session_state.get("uni")
    gdiag  = st.session_state.get("guard_diag")
    vfq_s  = st.session_state.get("vfq")
    sig_s  = st.session_state.get("signals")

    def _dl_btn(df, label, fname):
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            st.download_button(
                label,
                df.to_csv(index=False).encode(),
                file_name=fname,
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.button(label, disabled=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        _dl_btn(uni_s, "Descargar universo (CSV)", "universo.csv")
        _dl_btn(vfq_s, "Descargar VFQ (CSV)", "vfq.csv")
    with c2:
        _dl_btn(gdiag, "Descargar guardrails diag (CSV)", "guardrails_diag.csv")
        _dl_btn(sig_s, "Descargar señales (CSV)", "senales.csv")

# ====== Paso 7: BACKTESTING (placeholder) ======
with tab7:
    st.subheader("Backtesting")

    # ---------- Helpers locales (solo para esta pestaña) ----------
    def _to_panel_dict(panel_prices):
        """Acepta dict {sym: df} o DF largo y retorna dict {sym: df con index datetime y col 'close'}."""
        if isinstance(panel_prices, dict):
            out = {}
            for s, df in panel_prices.items():
                if not isinstance(df, pd.DataFrame) or df.empty or "close" not in df.columns:
                    continue
                dfi = df.copy()
                # asegura índice datetime
                if not isinstance(dfi.index, pd.DatetimeIndex):
                    if "date" in dfi.columns:
                        dfi = dfi.set_index(pd.to_datetime(dfi["date"])).drop(columns=[c for c in ["date"] if c in dfi.columns])
                    else:
                        dfi.index = pd.to_datetime(dfi.index)
                dfi = dfi.sort_index()
                out[s] = dfi[["close"]].dropna()
            return out
        elif isinstance(panel_prices, pd.DataFrame) and not panel_prices.empty:
            df = panel_prices.copy()
            if "symbol" not in df.columns and "ticker" in df.columns:
                df = df.rename(columns={"ticker": "symbol"})
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values(["symbol", "date"])
            out = {}
            for s, grp in df.groupby("symbol"):
                gg = grp.copy()
                if "date" in gg.columns:
                    gg = gg.set_index(gg["date"])
                out[s] = gg[["close"]].dropna()
            return out
        return {}

    def _month_ends_index(df: pd.DataFrame) -> pd.DatetimeIndex:
        return df.resample("M").last().index

    def _daily_returns_from_prices(df: pd.DataFrame) -> pd.Series:
        # espera index datetime y columna 'close'
        px = df["close"].astype(float)
        rets = px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return rets

    def _portfolio_backtest(panel_dict: dict,
                            rank_df: pd.DataFrame,
                            rank_col: str = "final_alpha",
                            top_n: int = 10,
                            cost_bps: int = 10,
                            lag_days: int = 0,
                            target_vol: float | None = None,
                            lev_cap: float = 2.0) -> tuple[pd.Series, pd.DataFrame]:
        """
        Backtest simple de portafolio:
        - Cada fin de mes selecciona Top-N por 'rank_col'
        - Equal weight
        - Aplica lag_days a la ejecución
        - Aplica costos por cambio de pesos (bps)
        - (Opcional) Target de volatilidad diario sobre la serie de portafolio (20d rolling), con tope de leverage
        Devuelve: equity (Serie) y tabla de rebalances (DataFrame con pesos en cada mes).
        """
        # Asegura que rank_df tiene symbol y la columna de ranking
        if "symbol" not in rank_df.columns or rank_col not in rank_df.columns:
            return pd.Series(dtype=float), pd.DataFrame()

        # Construimos calendario de rebalanceo a partir de la intersección de meses disponibles
        # Usamos el universo de símbolos con historial
        if not panel_dict:
            return pd.Series(dtype=float), pd.DataFrame()

        # Calendario mensual común (intersección soft)
        any_sym = next(iter(panel_dict))
        cal = _month_ends_index(panel_dict[any_sym])
        # preferimos el calendario del benchmark si lo tienes; aquí usamos el primero

        lag = pd.Timedelta(days=int(lag_days)) if lag_days else pd.Timedelta(0)

        # Serie de retorno diario por símbolo
        daily_map = {s: _daily_returns_from_prices(df) for s, df in panel_dict.items()}

        # Construir DataFrame de retornos diario alineado
        all_rets = pd.DataFrame({s: sr for s, sr in daily_map.items()}).sort_index().fillna(0.0)
        if all_rets.empty:
            return pd.Series(dtype=float), pd.DataFrame()

        # Rebalances mensuales
        month_ends = all_rets.resample("M").last().index
        weights_hist = []
        port_rets = []

        prev_weights = pd.Series(0.0, index=all_rets.columns)

        for i in range(1, len(month_ends)):
            t0, t1 = month_ends[i-1], month_ends[i]

            # Selección Top-N por ranking (estático por ahora; si tu ranking es dinámico, adaptar a fecha)
            top = (rank_df[["symbol", rank_col]]
                   .dropna()
                   .sort_values(rank_col, ascending=False)
                   .head(int(top_n))["symbol"]
                   .tolist())

            # Pesos equal-weight en seleccionados
            current_weights = pd.Series(0.0, index=all_rets.columns, dtype=float)
            if len(top) > 0:
                w = 1.0 / len(top)
                current_weights.loc[[s for s in top if s in current_weights.index]] = w

            # Turnover y costos (costo proporcional al cambio de peso)
            tw = (current_weights - prev_weights).abs().sum() * 0.5  # convención: media del cambio
            cost_rate = (cost_bps / 1e4) * tw

            # Ventana diaria de [t0+lag, t1+lag]
            sl = all_rets.loc[(all_rets.index > t0 + lag) & (all_rets.index <= t1 + lag)]
            if sl.empty:
                continue

            # Retorno diario de portafolio (pesos estáticos dentro del mes)
            pr = (sl * current_weights).sum(axis=1)

            # Aplica costo SOLO el primer día del bloque
            if len(pr) > 0:
                pr.iloc[0] = pr.iloc[0] - cost_rate

            port_rets.append(pr)
            weights_hist.append(current_weights.rename(t1))
            prev_weights = current_weights

        if not port_rets:
            return pd.Series(dtype=float), pd.DataFrame()

        port_rets = pd.concat(port_rets).sort_index()
        equity = (1.0 + port_rets).cumprod()

        # Volatility targeting (opcional) — 20 días rolling
        if target_vol is not None and target_vol > 0:
            roll = port_rets.rolling(20).std().replace(0.0, np.nan)
            ann_vol = roll * np.sqrt(252)
            lev = (target_vol / ann_vol).clip(lower=0.0, upper=float(lev_cap)).fillna(0.0)
            adj_rets = port_rets * lev
            equity = (1.0 + adj_rets).cumprod()

        weights_table = pd.DataFrame(weights_hist) if weights_hist else pd.DataFrame()
        return equity.rename("Portfolio"), weights_table

    try:
        panel_prices = st.session_state.get("panel_prices")
        qvm_df = st.session_state.get("qvm")
        if panel_prices is None or qvm_df is None or qvm_df.empty:
            st.info("Corre **Señales** y **QVM** antes de backtestear.")
            st.stop()

        # ---------- Controles ----------
        left, right = st.columns([0.55, 0.45])
        with left:
            rank_by = st.selectbox("Criterio de ranking para Top-N", ["final_alpha", "prob_up", "qvm_score"], index=0)
            top_n = st.slider("Top-N (portafolio)", 5, 50, 15, 5)
            use_and_bt = st.toggle("Señal MA200 AND Mom 12-1 (para métricas por símbolo)", value=False)
            rebalance_freq = st.selectbox("Frecuencia rebalance (por símbolo)", ["M", "W"], index=0,
                                          help="Solo para backtest por símbolo (función backtest_many). El portafolio usa mensual fijo en este bloque.")
        with right:
            cost_bps = st.slider("Costos (bps por cambio de peso)", 0, 50, 10, 1)
            lag_days = st.slider("Lag de ejecución (días)", 0, 5, 1, 1)
            enable_target_vol = st.toggle("Target de volatilidad (portafolio)", value=False)
            target_vol = st.number_input("Volatilidad anual objetivo (p.ej. 0.15)", value=0.15, step=0.01, format="%.2f") if enable_target_vol else None
            lev_cap = st.number_input("Límite de apalancamiento", value=2.0, step=0.1, format="%.1f") if enable_target_vol else 2.0

        # ---------- Panel en dict ----------
        panel_dict = _to_panel_dict(panel_prices)
        if not panel_dict:
            st.warning("No hay datos de precios adecuados para backtesting.")
            st.stop()

        # ---------- Backtest por símbolo (usa tu backtests.py) ----------
        try:
            # Selección: los símbolos disponibles intersectados con panel
            avail_syms = [s for s in qvm_df["symbol"].dropna().astype(str).unique().tolist() if s in panel_dict]
            metrics_df, curves = backtest_many(
                panel=panel_dict,
                symbols=avail_syms,
                cost_bps=int(cost_bps),
                lag_days=int(lag_days),
                use_and_condition=bool(use_and_bt),
                rebalance_freq=str(rebalance_freq)
            )

            st.markdown("**Métricas por símbolo (backtest_many)**")
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error en backtest por símbolo: {e}")
            metrics_df, curves = pd.DataFrame(), {}

        st.markdown("---")

        # ---------- Backtest de Portafolio Top-N ----------
        try:
            # rank_df: usa el último QVM disponible (columns: symbol, final_alpha, prob_up, qvm_score ...)
            rank_cols_needed = {"symbol", rank_by}
            if not rank_cols_needed.issubset(qvm_df.columns):
                st.warning(f"No encuentro columnas {rank_cols_needed} en QVM.")
                st.stop()

            # filtra a símbolos con precios
            rank_df = qvm_df.loc[qvm_df["symbol"].isin(panel_dict.keys()), ["symbol", rank_by]].dropna()
            equity, wtable = _portfolio_backtest(
                panel_dict=panel_dict,
                rank_df=rank_df,
                rank_col=rank_by,
                top_n=int(top_n),
                cost_bps=int(cost_bps),
                lag_days=int(lag_days),
                target_vol=(float(target_vol) if enable_target_vol else None),
                lev_cap=float(lev_cap)
            )

            if equity.empty:
                st.warning("No se pudo construir la curva del portafolio (revisa datos).")
            else:
                st.markdown("**Portafolio Top-N (equal-weight)**")
                st.line_chart(equity, use_container_width=True)
                # Resumen rápido
                pr = equity.pct_change().dropna()
                cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (252/len(pr))) - 1 if len(pr) > 0 else 0.0
                vol  = pr.std() * np.sqrt(252) if len(pr) > 0 else 0.0
                shar = (pr.mean()/pr.std()) * np.sqrt(252) if pr.std() > 0 else 0.0
                dd   = (equity / equity.cummax() - 1).min()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("CAGR", f"{cagr:.2%}")
                c2.metric("Vol",  f"{vol:.2%}")
                c3.metric("Sharpe", f"{shar:.2f}")
                c4.metric("MaxDD", f"{dd:.2%}")

                with st.expander("Pesos en cada rebalance (Top-N)", expanded=False):
                    st.dataframe(wtable.fillna(0.0), use_container_width=True)

        except Exception as e:
            st.error(f"Error en backtest de portafolio: {e}")

        st.caption("Tip: el backtest por símbolo usa tu `backtest_many`. El del portafolio aplica Top-N por ranking QVM con equal-weight, costos, lag y target de volatilidad opcional.")

    except Exception as e:
        st.error(f"Error en Backtesting: {e}")
