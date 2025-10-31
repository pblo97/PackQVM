from __future__ import annotations
import altair as alt

# --- poner esto ARRIBA DE TODO ---
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "poll"  # o "none" si prefieres desactivar
# ---------------------------------

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Tuple

# ============== IMPORTS DE TU PIPELINE ==============
from pipeline_factors import build_factor_frame
from fundamentals import (
    download_fundamentals,
    build_vfq_scores_dynamic,
    download_guardrails_batch,
    apply_quality_guardrails,
)
from scoring import (
    blend_breakout_qvm,
    build_momentum_proxy,
)
from data_io import (
    run_fmp_screener,
    filter_universe,
    load_prices_panel,
    load_benchmark,
    DEFAULT_START,
    DEFAULT_END,
)
from pipeline import (
    apply_trend_filter,
    enrich_with_breakout,
    market_regime_on,
)
from backtests import backtest_many

# Opcional (growth-aware). No se usan aún en la UI, pero los dejamos importables.
from factors_growth_aware import (
    compute_qvm_scores,
    apply_megacap_rules,
)

# ==================== CONFIG BÁSICO ====================
st.set_page_config(
    page_title="Sistema QVM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS suave
st.markdown(
    """
<style>
.block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: .2px; }
hr { border: 0; border-top: 1px solid rgba(255,255,255,.08); margin: .6rem 0 1rem 0; }
[data-testid="stDataFrame"] tbody tr:hover {
    background: rgba(59,130,246,.08) !important;
}
[data-testid="stCaptionContainer"] { opacity: .85; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------ HELPERS DE CACHÉ ------------------
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
    1. Pide universo base al screener de FMP con filtros básicos.
    2. Normaliza nombres.
    3. Aplica filtros extra por market cap, volumen y edad IPO.
    """
    df = run_fmp_screener(
        limit=limit,
        mcap_min=mcap_min,
        volume_min=volume_min,
        fetch_profiles=True,
        cache_key=cache_key,
        force=False,
        # IMPORTANTE:
        # run_fmp_screener debe ya mandar:
        #   isEtf=false
        #   isFund=false
        #   isActivelyTrading=true
    )

    if df is None:
        return pd.DataFrame(columns=["symbol", "sector", "market_cap"])

    df = df.copy()

    # market cap normalizado -> "market_cap"
    if "market_cap" not in df.columns:
        if "marketCap" in df.columns:
            df["market_cap"] = pd.to_numeric(df["marketCap"], errors="coerce")
        else:
            df["market_cap"] = np.nan

    # sector seguro
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    df["sector"] = (
        df["sector"]
        .astype(str)
        .replace({"": "Unknown"})
        .fillna("Unknown")
    )

    # volumen numérico
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # ipoDate a datetime
    if "ipoDate" in df.columns:
        df["ipoDate"] = pd.to_datetime(df["ipoDate"], errors="coerce", utc=True)
    else:
        df["ipoDate"] = pd.NaT

    # -------- filtros post-request --------
    # market cap
    df = df[df["market_cap"] >= float(mcap_min)]

    # volumen
    if "volume" in df.columns:
        df = df[df["volume"] >= float(volume_min)]

    # antigüedad IPO
    if df["ipoDate"].notna().any():
        cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=int(ipo_days))
        df = df[df["ipoDate"] < cutoff]

    # columnas mínimas
    core_cols = ["symbol", "sector", "market_cap"]
    for col in core_cols:
        if col not in df.columns:
            if col == "symbol":
                df["symbol"] = ""
            elif col == "sector":
                df["sector"] = "Unknown"
            elif col == "market_cap":
                df["market_cap"] = np.nan

    # limpiar symbol vacío
    df = (
        df[core_cols]
        .dropna(subset=["symbol"])
        .reset_index(drop=True)
    )

    return df


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_download_guardrails(symbols: Tuple[str, ...], cache_key: str) -> pd.DataFrame:
    return download_guardrails_batch(
        list(symbols),
        cache_key=cache_key,
        force=False,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_download_fundamentals(
    symbols: Tuple[str, ...],
    cache_key: str,
    mc_pairs: Tuple[Tuple[str, float], ...] | None = None,
) -> pd.DataFrame:
    """
    Si en algún momento quieres pasarle market caps estimadas para mejorar hints,
    puedes usar mc_pairs=[(sym, mcap), ...].
    """
    mc_map = dict(mc_pairs or ())
    return download_fundamentals(
        list(symbols),
        market_caps=mc_map,
        cache_key=cache_key,
        force=False,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_load_prices_panel(symbols, start, end, cache_key=""):
    return load_prices_panel(
        symbols,
        start,
        end,
        cache_key=cache_key,
        force=False,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_load_benchmark(bench, start, end):
    return load_benchmark(bench, start, end)


# ------------------ MÉTRICAS DE PERFORMANCE ------------------
def perf_summary_from_returns(rets: pd.Series, periods_per_year: int) -> dict:
    r = rets.dropna().astype(float)
    if r.empty:
        return {}

    eq = (1 + r).cumprod()

    yrs = len(r) / periods_per_year if periods_per_year else np.nan
    if yrs and yrs > 0:
        cagr = eq.iloc[-1] ** (1 / yrs) - 1
    else:
        cagr = np.nan

    vol = r.std() * np.sqrt(periods_per_year) if r.std() > 0 else np.nan
    sharpe = (r.mean() * periods_per_year) / r.std() if r.std() > 0 else np.nan

    dd = eq / eq.cummax() - 1
    maxdd = dd.min()

    hit = (r > 0).mean()

    avg_win = r[r > 0].mean() if (r > 0).any() else np.nan
    avg_loss = r[r < 0].mean() if (r < 0).any() else np.nan

    if avg_win and avg_loss:
        payoff = avg_win / abs(avg_loss)
    else:
        payoff = np.nan

    if not np.isnan(hit) and avg_win is not None and avg_loss is not None:
        expct = hit * avg_win + (1 - hit) * avg_loss
    else:
        expct = np.nan

    return {
        "CAGR": float(cagr),
        "Vol_anual": float(vol),
        "Sharpe": float(sharpe),
        "MaxDD": float(maxdd),
        "HitRate": float(hit),
        "AvgWin": float(avg_win),
        "AvgLoss": float(avg_loss),
        "Payoff": float(payoff),
        "Expectancy": float(expct),
        "Periodos": int(len(r)),
    }


# ------------------ NORMALIZADORES / UTILIDADES ------------------
def normalize_guard_diag(diag: pd.DataFrame, df_guard: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Toma el diagnóstico crudo de guardrails (pass_profit, pass_issuance, etc.)
    y garantiza columnas consistentes aunque falten en origen.
    """
    d = diag.copy() if isinstance(diag, pd.DataFrame) else pd.DataFrame()
    if d.empty:
        cols = [
            "symbol",
            "profit_hits",
            "coverage_count",
            "net_issuance",
            "asset_growth",
            "accruals_ta",
            "netdebt_ebitda",
            "pass_profit",
            "pass_issuance",
            "pass_assets",
            "pass_accruals",
            "pass_ndebt",
            "pass_coverage",
            "pass_all",
            "reason",
        ]
        return pd.DataFrame(columns=cols)

    # asegurar 'symbol'
    if "symbol" not in d.columns:
        if d.index.name == "symbol":
            d = d.reset_index()
        elif isinstance(df_guard, pd.DataFrame) and "symbol" in df_guard.columns:
            d["symbol"] = df_guard["symbol"].values[: len(d)]
        else:
            d["symbol"] = pd.Index(range(len(d))).astype(str)

    token_map = {
        "pass_profit": "profit_floor",
        "pass_issuance": "net_issuance",
        "pass_assets": "asset_growth",
        "pass_accruals": "accruals_ta",
        "pass_ndebt": "netdebt_ebitda",
        "pass_coverage": "vfq_coverage",
    }

    has_reason = "reason" in d.columns
    for col, tok in token_map.items():
        if col not in d.columns:
            if has_reason:
                d[col] = ~d["reason"].fillna("").str.contains(tok)
            else:
                d[col] = np.nan

    checks = list(token_map.keys())
    if "pass_all" not in d.columns:
        if all(c in d.columns for c in checks):
            d["pass_all"] = d[checks].all(axis=1)
        else:
            d["pass_all"] = False

    if "reason" not in d.columns:
        def _mk_reason(row):
            r = []
            if "pass_profit" in d.columns and not bool(row.get("pass_profit", True)):
                r.append("profit_floor")
            if "pass_issuance" in d.columns and not bool(row.get("pass_issuance", True)):
                r.append("net_issuance")
            if "pass_assets" in d.columns and not bool(row.get("pass_assets", True)):
                r.append("asset_growth")
            if "pass_accruals" in d.columns and not bool(row.get("pass_accruals", True)):
                r.append("accruals_ta")
            if "pass_ndebt" in d.columns and not bool(row.get("pass_ndebt", True)):
                r.append("netdebt_ebitda")
            if "pass_coverage" in d.columns and not bool(row.get("pass_coverage", True)):
                r.append("vfq_coverage")
            return ",".join(r)

        d["reason"] = d.apply(_mk_reason, axis=1)

    return d


def _fmt_mcap(x):
    """bonito para mostrar market cap"""
    try:
        x = float(x)
        if x >= 1e12:
            return f"${x/1e12:.2f}T"
        if x >= 1e9:
            return f"${x/1e9:.2f}B"
        if x >= 1e6:
            return f"${x/1e6:.2f}M"
        return f"${x:,.0f}"
    except Exception:
        return ""


# ==================== HEADER ====================
l, r = st.columns([0.85, 0.15])
with l:
    st.markdown("<h1 style='margin-bottom:0'>QVM Screener</h1>", unsafe_allow_html=True)
    st.caption("Momentum estructural + Breakout técnico + Value/Quality (VFQ)")
with r:
    st.caption(datetime.now().strftime("Actualizado: %d %b %Y %H:%M"))
st.markdown("<hr/>", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### ⚙️ Controles")
    preset = st.segmented_control(
        "Preset",
        options=["Laxo", "Balanceado", "Estricto"],
        default="Balanceado",
    )

    # ---- Universo & Screener ----
    with st.expander("Universo & Screener", expanded=True):
        limit = st.slider("Límite del universo", 50, 1000, 300, 50)

        min_mcap = st.number_input(
            "MarketCap mínimo (USD)",
            value=5e8,
            step=1e8,
            format="%.0f",
        )

        volume_min = st.number_input(
            "Volumen mínimo diario",
            value=500_000,
            step=50_000,
            format="%.0f",
        )

        ipo_days = st.slider(
            "Antigüedad IPO (días)",
            90,
            1500,
            365,
            30,
        )

    # ---- Fundamentales & Guardrails ----
    with st.expander("Fundamentales & Guardrails", expanded=False):
        min_cov_guard = st.slider(
            "Cobertura VFQ mínima (# métricas)", 1, 4, 2
        )
        profit_hits = st.slider(
            "Pisos de rentabilidad (hits EBIT/CFO/FCF)", 0, 3, 2
        )
        max_issuance = st.slider(
            "Net issuance máx.", 0.00, 0.10, 0.03, 0.01
        )
        max_assets = st.slider(
            "Asset growth |y/y| máx.", 0.00, 0.50, 0.20, 0.01
        )
        max_accr = st.slider(
            "Accruals/TA | | máx.", 0.00, 0.25, 0.10, 0.01
        )
        max_ndeb = st.slider(
            "NetDebt/EBITDA máx.", 0.0, 6.0, 3.0, 0.5
        )

    # ---- Técnico ----
    with st.expander("Técnico — Tendencia & Breakout", expanded=True):
        use_and = st.toggle("MA200 Y Mom 12–1", value=False)
        require_breakout = st.toggle("Exigir Breakout para ENTRY", value=False)
        rvol_th = st.slider("RVOL (20d) mín.", 0.8, 2.5, 1.2, 0.1)
        closepos_th = st.slider("ClosePos mín.", 0.0, 1.0, 0.60, 0.05)
        p52_th = st.slider("Cercanía 52W High", 0.80, 1.00, 0.95, 0.01)
        updown_vol_th = st.slider("Up/Down Vol Ratio (20d)", 0.8, 3.0, 1.2, 0.1)
        min_hits_brk = st.slider("Mínimo checks breakout (K de 4)", 1, 4, 3)
        atr_pct_min = st.slider("ATR pct (6–12m) mín.", 0.0, 1.0, 0.6, 0.05)
        use_rs_slope = st.toggle("Exigir RS slope > 0 (MA20)", value=False)

    # ---- Régimen & Fechas ----
    with st.expander("Régimen & Fechas", expanded=False):
        bench = st.selectbox(
            "Benchmark", ["SPY", "QQQ", "^GSPC"], index=0
        )
        risk_on = st.toggle("Exigir mercado Risk-ON", value=True)
        start = st.date_input(
            "Inicio", value=pd.to_datetime(DEFAULT_START).date()
        )
        end = st.date_input(
            "Fin", value=pd.to_datetime(DEFAULT_END).date()
        )

    # ---- Ranking avanzado ----
    with st.expander("Ranking avanzado", expanded=False):
        beta_prob = st.slider(
            "Sensibilidad probabilidad (β)", 1.0, 12.0, 6.0, 0.5
        )
        top_n_show = st.slider(
            "Top N a resaltar", 10, 100, 25, 5
        )

    st.markdown("---")
    run_btn = st.button("Ejecutar", use_container_width=True)

# Presets que ajustan umbrales técnicos, sin pisar lo que ya moviste manualmente demasiado
if preset == "Laxo":
    rvol_th = min(rvol_th, 1.0)
    closepos_th = min(closepos_th, 0.55)
    p52_th = min(p52_th, 0.92)
    min_hits_brk = min(min_hits_brk, 2)
elif preset == "Estricto":
    rvol_th = max(rvol_th, 1.5)
    closepos_th = max(closepos_th, 0.65)
    p52_th = max(p52_th, 0.97)
    min_hits_brk = max(min_hits_brk, 3)

# cache tag que depende de entradas clave del universo
cache_tag = f"{int(min_mcap)}_{ipo_days}_{limit}_{int(volume_min)}"

# Estado del pipeline
if "pipeline_ready" not in st.session_state:
    st.session_state["pipeline_ready"] = False

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab6, tab7, tab8 = st.tabs(
    [
        "Universo",
        "Guardrails",
        "VFQ",
        "Señales",
        "Export",
        "Backtesting",
        "Tuning"
    ]
)

# ==================== VFQ sidebar extra ====================
with st.sidebar:
    st.markdown("⚙️ Fundamentos (VFQ)")

    value_metrics_opts = ["inv_ev_ebitda", "fcf_yield"]
    quality_metrics_opts = ["gross_profitability", "roic", "roa", "netMargin"]

    sel_value = st.multiselect(
        "Métricas Value",
        options=value_metrics_opts,
        default=["inv_ev_ebitda", "fcf_yield"],
    )

    sel_quality = st.multiselect(
        "Métricas Quality",
        options=quality_metrics_opts,
        default=["gross_profitability", "roic"],
    )

    c1x, c2x = st.columns(2)
    with c1x:
        w_value = st.slider("Peso Value", 0.0, 1.0, 0.5, 0.05)
    with c2x:
        w_quality = st.slider("Peso Quality", 0.0, 1.0, 0.5, 0.05)

    method_intra = st.radio(
        "Agregación intra-bloque",
        ["mean", "median", "weighted_mean"],
        index=0,
        horizontal=True,
    )
    winsor_p = st.slider("Winsor p (cola)", 0.0, 0.10, 0.01, 0.005)
    size_buckets = st.slider("Buckets por tamaño", 1, 5, 3, 1)
    group_mode = st.selectbox(
        "Agrupar por", ["sector", "sector|size"], index=1
    )
    min_cov = st.slider("Cobertura mín. (# métricas)", 0, 8, 1, 1)
    min_pct = st.slider(
        "VFQ pct (intra-sector) mín.", 0.00, 1.00, 0.00, 0.01
    )

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


# ====== TAB 1: UNIVERSO ======
with tab1:
    st.subheader("Universo inicial")

    # refrescamos universo si:
    #   - nunca se creó
    #   - tocaste "Ejecutar"
    need_refresh = ("uni" not in st.session_state) or run_btn

    if need_refresh:
        raw_universe = _cached_run_fmp_screener(
            limit=limit,
            mcap_min=min_mcap,
            volume_min=volume_min,
            ipo_days=ipo_days,
            cache_key=cache_tag,
        )

        # garantizamos columnas core
        out = raw_universe.copy()
        if "symbol" not in out.columns:
            if "ticker" in out.columns:
                out["symbol"] = out["ticker"].astype(str)
            else:
                out["symbol"] = ""

        if "market_cap" not in out.columns:
            if "marketCap" in out.columns:
                out["market_cap"] = pd.to_numeric(
                    out["marketCap"], errors="coerce"
                )
            else:
                out["market_cap"] = np.nan

        if "sector" not in out.columns:
            out["sector"] = "Unknown"
        else:
            s = out["sector"].astype(str)
            s = s.replace({"": "Unknown"})
            s = s.where(~s.isna(), "Unknown")
            out["sector"] = s

        out = (
            out[["symbol", "sector", "market_cap"]]
            .dropna(subset=["symbol"])
            .reset_index(drop=True)
        )

        st.session_state["uni"] = out.copy()

    # leer versión estable en memoria
    uni_df = st.session_state["uni"].copy()

    total_raw = len(uni_df)
    total_filtrado = len(uni_df)  # acá podrías hacer más filtros si quieres

    c1m, c2m = st.columns(2)
    c1m.metric("Screener", f"{total_raw}")
    c2m.metric("Tras filtros básicos", f"{total_filtrado}")

    st.dataframe(
        uni_df.head(50),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Esta tabla vive en st.session_state['uni'] y alimenta las demás pestañas."
    )


# ====== TAB 2: GUARDRAILS ======
with tab2:
    st.subheader("Guardrails")

    uni = st.session_state.get("uni", pd.DataFrame())
    if uni is None or uni.empty or "symbol" not in uni.columns:
        st.info("Primero genera el universo en la pestaña Universo.")
        st.stop()

    syms = (
        uni["symbol"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not syms:
        st.info("No hay símbolos en el universo.")
        st.stop()

    # construimos frame con factores/guardrails para TODO el universo actual
    df_all = build_factor_frame(syms)

    # injertar sector / market_cap desde universo actual
    base_cols = ["symbol", "sector", "market_cap"]
    df_all = (
        df_all.drop(columns=["sector", "market_cap"], errors="ignore")
        .merge(
            uni[[c for c in base_cols if c in uni.columns]],
            on="symbol",
            how="left",
        )
    )

    # máscara 'estricta': pass_all True
    strict_mask = df_all.get("pass_all", False) == True

    kept_raw = (
        df_all.loc[strict_mask, ["symbol"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # guardamos para siguientes tabs
    st.session_state["kept"] = kept_raw
    st.session_state["guard_diag"] = df_all.copy()

    total = len(df_all)
    pasan = int(strict_mask.sum())
    rechaz = total - pasan

    c1g, c2g, c3g = st.columns(3)
    c1g.metric("Pasan guardrails estrictos", f"{pasan}")
    c2g.metric("Candidatos saludables (relajado)", f"{pasan}")  # placeholder
    c3g.metric("Rechazados totales", f"{rechaz}")

    cols_show = [
        "symbol",
        "sector",
        "pass_all",
        "profit_hits",
        "coverage_count",
        "asset_growth",
        "accruals_ta",
        "netdebt_ebitda",
        "pass_profit",
        "pass_issuance",
        "pass_assets",
        "pass_accruals",
        "pass_ndebt",
        "pass_coverage",
    ]
    cols_show = [c for c in cols_show if c in df_all.columns]

    with st.expander(
        f"Detalle guardrails (estricto): {pasan} / {total}",
        expanded=True,
    ):
        st.dataframe(
            df_all[cols_show].sort_values("symbol"),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "pass_all = pasó TODAS las barreras simultáneamente. "
        "coverage_count = cuánta info fundamental tenemos disponible."
    )


# ====== TAB 3: VFQ ======
with tab3:
    st.subheader("VFQ (Value / Quality / Flow)")

    kept = st.session_state.get("kept", pd.DataFrame())
    uni_cur = st.session_state.get("uni", pd.DataFrame())

    if kept is None or kept.empty or "symbol" not in kept.columns:
        st.warning("No hay símbolos aprobados por Guardrails. Ajusta la pestaña Guardrails.")
        st.stop()

    kept_syms = (
        kept["symbol"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if not kept_syms:
        st.warning("La lista kept está vacía.")
        st.stop()

    # reconstruimos factores SOLO de los kept (para consistencia visual)
    df_vfq_all = build_factor_frame(kept_syms)

    # agregamos columnas de universo base
    df_vfq_all = (
        df_vfq_all.drop(columns=["sector", "market_cap"], errors="ignore")
        .merge(
            uni_cur[["symbol", "sector", "market_cap"]],
            on="symbol",
            how="left",
        )
    )

    # sliders VFQ locales a esta vista
    c1v, c2v, c3v = st.columns(3)
    with c1v:
        min_quality = st.slider("Min Quality neut.", 0.0, 1.0, 0.3, 0.01)
        min_value = st.slider("Min Value neut.", 0.0, 1.0, 0.3, 0.01)
        max_ndebt = st.slider("Max NetDebt/EBITDA", 0.0, 5.0, 2.0, 0.1)

    with c2v:
        min_acc_pct = st.slider("Accruals limpios (% mínimo)", 0, 100, 30, 1)
        min_hits_req = st.slider("Min hits (breakout hits)", 0, 5, 1, 1)
        min_rvol20 = st.slider("Min RVOL20", 0.0, 5.0, 1.2, 0.05)

    with c3v:
        min_breakout = st.slider("Min BreakoutScore", 0, 100, 50, 1)
        topN_prob = st.slider("Top N por prob_up", 5, 100, 30, 1)

    # --------- filtros VFQ + técnico ----------
    mask = pd.Series(True, index=df_vfq_all.index, dtype=bool)

    # floors de quality / value
    mask &= (df_vfq_all["quality_adj_neut"].fillna(0) >= float(min_quality))
    mask &= (df_vfq_all["value_adj_neut"].fillna(0)   >= float(min_value))

    # técnica básica: hits, breakoutscore, rvol
    mask &= (df_vfq_all["hits"].fillna(0)            >= int(min_hits_req))
    mask &= (df_vfq_all["BreakoutScore"].fillna(0)   >= float(min_breakout))
    mask &= (df_vfq_all["RVOL20"].fillna(0)          >= float(min_rvol20))

    # endeudamiento
    mask &= (
        df_vfq_all["netdebt_ebitda"].isna()
        | (df_vfq_all["netdebt_ebitda"] <= float(max_ndebt))
    )

    # accruals limpios (% alto es mejor en tu escala actual)
    mask &= (
        df_vfq_all["acc_pct"].isna()
        | (df_vfq_all["acc_pct"] >= float(min_acc_pct))
    )

    df_keep_vfq = df_vfq_all.loc[mask].copy()

    # ranking final: prob_up si existe, si no BreakoutScore
    if df_keep_vfq["prob_up"].notna().any():
        df_keep_vfq = df_keep_vfq.sort_values("prob_up", ascending=False)
    else:
        df_keep_vfq = df_keep_vfq.sort_values("BreakoutScore", ascending=False)

    vfq_top = df_keep_vfq.head(int(topN_prob)).copy()

    st.markdown("### 🟢 Selección VFQ filtrada")
    cols_vfq_show = [
        "symbol",
        "netdebt_ebitda",
        "accruals_ta",
        "sector",
        "market_cap",
        "quality_adj_neut",
        "value_adj_neut",
        "acc_pct",
        "hits",
        "BreakoutScore",
        "RVOL20",
        "prob_up",
    ]
    cols_vfq_show = [c for c in cols_vfq_show if c in vfq_top.columns]

    st.dataframe(
        vfq_top[cols_vfq_show],
        use_container_width=True,
        hide_index=True,
    )

    # ------- rechazados (para watchlist técnica) -------
    st.markdown("### 🧹 Rechazados por VFQ / técnica")
    rejected_syms = sorted(set(kept_syms) - set(df_keep_vfq["symbol"]))
    rej_view = df_vfq_all[df_vfq_all["symbol"].isin(rejected_syms)].copy()

    cols_rej_show = [
        "symbol",
        "sector",
        "market_cap",
        "quality_adj_neut",
        "value_adj_neut",
        "netdebt_ebitda",
        "acc_pct",
        "BreakoutScore",
        "hits",
        "RVOL20",
        "prob_up",
    ]
    cols_rej_show = [c for c in cols_rej_show if c in rej_view.columns]

    st.dataframe(
        rej_view[cols_rej_show],
        use_container_width=True,
        hide_index=True,
    )

    # ------- guardar en session_state para las otras tabs -------
    st.session_state["vfq_top"] = vfq_top[["symbol"]].drop_duplicates()
    st.session_state["vfq_table"] = vfq_top.reset_index(drop=True)
    st.session_state["pipeline_ready"] = True

    st.session_state["vfq_all"] = df_vfq_all.copy()        # todos los kept con métricas
    st.session_state["vfq_keep"] = df_keep_vfq.copy()      # los que pasaron filtros VFQ+técnico
    st.session_state["vfq_rejected"] = rej_view.copy()     # los que quedaron fuera
    st.session_state["vfq_params"] = {
        "min_hits": int(min_hits_req),
        "min_rvol20": float(min_rvol20),
        "min_breakout": float(min_breakout),
        "min_acc_pct": float(min_acc_pct),
        "max_ndebt": float(max_ndebt),
    }


# ====== TAB 4: SEÑALES (placeholder por ahora) ======
with tab4:
    st.subheader("Señales técnicas / Breakout")

    # =========================
    # Recuperar de session_state lo que dejó tab3
    # =========================
    uni_df = st.session_state.get("uni", pd.DataFrame())
    vfq_all = st.session_state.get("vfq_all", pd.DataFrame())
    vfq_keep = st.session_state.get("vfq_keep", pd.DataFrame())
    vfq_rej = st.session_state.get("vfq_rejected", pd.DataFrame())
    vfq_params = st.session_state.get("vfq_params", {})

    if vfq_all is None or vfq_all.empty:
        st.info("Todavía no hay datos técnicos / VFQ. Corre la pestaña VFQ primero.")
        st.stop()

    # thresholds que definiste en tab3 y guardaste en vfq_params
    min_hits_thr = vfq_params.get("min_hits", 1)
    min_breakout_thr = vfq_params.get("min_breakout", 50.0)
    min_rvol20_thr = vfq_params.get("min_rvol20", 1.2)
    require_breakout_flag = vfq_params.get("require_breakout", False)

    # toggle 'risk_on' viene de la sidebar (Régimen & Fechas)
    # si por algún motivo no existe acá en tab4 (scope), lo pisamos a False
    try:
        global_risk_on = bool(risk_on)
    except NameError:
        global_risk_on = False

    # =========================
    # 1. Estado de mercado (proxy amplitud)
    # =========================
    st.markdown("### 1. Estado de mercado")

    # proxy amplitud:
    #   pct_hits_ok        = % tickers cuyo 'hits' >= min_hits_thr
    #   pct_breakout_ok    = % tickers cuyo 'BreakoutScore' >= min_breakout_thr
    #   global_risk_on     = tu semáforo macro/táctico (toggle Risk-ON)

    if "hits" in vfq_all.columns:
        pct_hits_ok = (
            pd.to_numeric(vfq_all["hits"], errors="coerce")
            .fillna(0)
            .ge(min_hits_thr)
            .mean()
        )
    else:
        pct_hits_ok = np.nan

    if "BreakoutScore" in vfq_all.columns:
        pct_breakout_ok = (
            pd.to_numeric(vfq_all["BreakoutScore"], errors="coerce")
            .fillna(0)
            .ge(min_breakout_thr)
            .mean()
        )
    else:
        pct_breakout_ok = np.nan

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "% setups técnicos OK",
        f"{pct_hits_ok*100:.1f}%" if not np.isnan(pct_hits_ok) else "n/d",
        help="≈ % del universo con suficientes 'hits' (checks técnicos cumplidos)."
    )

    c2.metric(
        "% ruptura/momentum OK",
        f"{pct_breakout_ok*100:.1f}%" if not np.isnan(pct_breakout_ok) else "n/d",
        help="≈ % del universo con BreakoutScore ≥ umbral (momentum fuerte / ruptura)."
    )

    c3.metric(
        "Régimen mercado",
        "RISK ON ✅" if global_risk_on else "RISK OFF ⚠️",
        help="Tu switch macro/táctico desde la barra lateral."
    )

    st.caption(
        "- % setups técnicos OK ≈ amplitud de tendencia/volumen en verde según tus 'hits'.\n"
        "- % ruptura/momentum OK ≈ cuántas están rompiendo con fuerza según BreakoutScore.\n"
        "- RISK ON viene de tu toggle (si está OFF, las entradas nuevas son más frágiles).\n"
    )

    st.markdown("---")

    # =========================
    # 2. Checklist técnico actual
    # =========================
    st.markdown("### 2. Checklist técnico activo")

    # Mostramos los thresholds que están gobernando las señales técnicas
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Hits mínimos", f"{min_hits_thr}")
    col_b.metric("BreakoutScore mín.", f"{min_breakout_thr:.1f}")
    col_c.metric("RVOL20 mín.", f"{min_rvol20_thr:.2f}")
    col_d.metric("Requiere breakout?", "Sí" if require_breakout_flag else "No")

    st.caption(
        "Estos parámetros vienen de tu pestaña VFQ / técnico. "
        "Cualquier papel que no cumpla esto, queda fuera."
    )

    st.markdown("---")

    # =========================
    # 3. Watchlist técnica (los que quedaron después de VFQ+técnico)
    # =========================
    st.markdown("### 3. Watchlist técnica (post-VFQ + técnico)")

    if vfq_keep is None or vfq_keep.empty:
        st.warning("Ningún ticker pasó VFQ + técnico con los filtros actuales.")
    else:
        cols_keep_show = [
            "symbol",
            "sector",
            "market_cap",
            "quality_adj_neut",
            "value_adj_neut",
            "acc_pct",
            "hits",
            "BreakoutScore",
            "RVOL20",
            "prob_up",
        ]
        cols_keep_show = [
            c for c in cols_keep_show if c in vfq_keep.columns
        ]

        st.dataframe(
            vfq_keep[cols_keep_show]
                .sort_values(
                    ["prob_up", "BreakoutScore"],
                    ascending=False,
                    na_position="last"
                ),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "Watchlist técnica = candidatos activos que ya pasaron Guardrails y VFQ "
        "y además cumplen señales de tendencia/momentum/volumen."
    )

    st.markdown("---")

    # =========================
    # 4. Rechazados técnicos (fallaron momentum / volumen / hits)
    # =========================
    st.markdown("### 4. Rechazados técnicos")

    if vfq_rej is None or vfq_rej.empty:
        st.info("No hay rechazados técnicos adicionales (o no se guardaron).")
    else:
        cols_rej_show = [
            "symbol",
            "sector",
            "market_cap",
            "quality_adj_neut",
            "value_adj_neut",
            "netdebt_ebitda",
            "acc_pct",
            "hits",
            "BreakoutScore",
            "RVOL20",
            "prob_up",
        ]
        cols_rej_show = [
            c for c in cols_rej_show if c in vfq_rej.columns
        ]

        st.dataframe(
            vfq_rej[cols_rej_show]
                .sort_values(
                    ["BreakoutScore", "prob_up"],
                    ascending=False,
                    na_position="last"
                ),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "Estos tickers pasaron calidad/valor básico pero no cumplen aún las "
        "condiciones mínimas de momentum, volumen o ruptura. Se pueden vigilar "
        "para ver si 'encienden' luego."
    )

    # =========================
    # 5. Nota final
    # =========================
    st.info(
        "Resumen rápido:\n"
        "- El % setups técnicos OK y el % ruptura/momentum OK funcionan como 'amplitud interna' del mercado.\n"
        "- Si ambas amplitudes son altas Y estás en RISK ON ✅, el entorno está listo para tomar trades nuevos.\n"
        "- Si están bajas o estás en RISK OFF ⚠️, reduces agresividad / tamaño de posición."
    )


# ====== TAB 6: EXPORT ======
with tab6:
    st.subheader("Export")
    st.write(
        "Acá puedes hacer st.download_button() del universo filtrado, VFQ_top, etc."
    )


# ====== TAB 7: BACKTESTING ======
with tab7:
    st.subheader("Backtesting")

    # -------------------------------
    # 0. Recuperar candidatos y rango temporal
    # -------------------------------
    # usamos lo que dejó tab3/tab4:
    vfq_keep = st.session_state.get("vfq_keep", pd.DataFrame())
    if vfq_keep is None or vfq_keep.empty or "symbol" not in vfq_keep.columns:
        st.warning(
            "No hay símbolos aprobados en VFQ + técnico. "
            "Corre las pestañas anteriores primero."
        )
        st.stop()

    # universo base para testear
    all_syms_bt = (
        vfq_keep["symbol"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not all_syms_bt:
        st.warning("No hay símbolos válidos para backtest.")
        st.stop()

    # -------------------------------
    # 1. Controles del backtest
    # -------------------------------
    st.markdown("#### Parámetros de simulación")

    c_bt1, c_bt2, c_bt3 = st.columns(3)

    with c_bt1:
        top_n_bt = st.slider(
            "N° máx. de símbolos a probar",
            min_value=5,
            max_value=min(50, len(all_syms_bt)),
            value=min(20, len(all_syms_bt)),
            step=1,
            help="Para no pedir miles de series si el universo está grande."
        )

    with c_bt2:
        cost_bps = st.number_input(
            "Costo de trading (bps por cambio de postura)",
            min_value=0,
            max_value=100,
            value=10,
            step=1,
            help="10 bps = 0.10% cada vez que giras in/out."
        )
        lag_days = st.number_input(
            "Lag ejecución (días)",
            min_value=0,
            max_value=5,
            value=0,
            step=1,
            help="Retraso en la ejecución después de la señal."
        )

    with c_bt3:
        use_and_condition = st.toggle(
            "Exigir MA200 Y Mom12-1 > 0 (no OR)",
            value=False,
            help="Si está apagado: entrar si MA200 o Mom12-1 está OK. "
                 "Si está prendido: exigir ambas."
        )
        rebalance_freq = st.selectbox(
            "Frecuencia rebalance",
            options=["M", "W"],
            index=0,
            help="M = mensual, W = semanal. El modelo asume long-only binario."
        )

    # acotamos el set final a testear
    syms_bt = all_syms_bt[:top_n_bt]

    st.caption(
        f"Testeando {len(syms_bt)} símbolos: {', '.join(syms_bt[:10])}"
        + ("…" if len(syms_bt) > 10 else "")
    )

    # -------------------------------
    # 2. Traer precios históricos
    # -------------------------------
    # usamos las fechas que ya definiste en sidebar ("Régimen & Fechas")
    # si por algún motivo acá no existen start/end en este scope, hacemos fallback
    try:
        start_dt = pd.to_datetime(start).date()
        end_dt   = pd.to_datetime(end).date()
    except NameError:
        start_dt = pd.to_datetime(DEFAULT_START).date()
        end_dt   = pd.to_datetime(DEFAULT_END).date()

    # cargamos panel OHLC para todos los tickers
    # _cached_load_prices_panel debe devolver dict-like:
    #   { "AAPL": df_price, ... }
    # donde df_price.index es datetime y df_price["close"] existe.
    price_panel = _cached_load_prices_panel(
        symbols=syms_bt,
        start=start_dt,
        end=end_dt,
        cache_key=f"bt_{start_dt}_{end_dt}_{len(syms_bt)}"
    )

    if price_panel is None or price_panel == {}:
        st.error("No pude cargar precios históricos para backtest.")
        st.stop()

    # -------------------------------
    # 3. Ejecutar backtest
    # -------------------------------
    # backtest_many viene de tu archivo backtests.py
    bt_metrics, bt_curves = backtest_many(
        panel=price_panel,
        symbols=syms_bt,
        cost_bps=int(cost_bps),
        lag_days=int(lag_days),
        use_and_condition=bool(use_and_condition),
        rebalance_freq=str(rebalance_freq)
    )

    # -------------------------------
    # 4. Mostrar métricas agregadas
    # -------------------------------
    st.markdown("#### Resultados por símbolo")

    if bt_metrics is None or bt_metrics.empty:
        st.warning("No hubo data suficiente para calcular métricas.")
    else:
        # formateo bonito
        show_cols = ["symbol", "CAGR", "Sharpe", "Sortino", "MaxDD", "Turnover", "Trades"]
        show_cols = [c for c in show_cols if c in bt_metrics.columns]

        fmt_df = bt_metrics.copy()

        # porcentajes legibles
        if "CAGR" in fmt_df.columns:
            fmt_df["CAGR"] = (fmt_df["CAGR"] * 100).round(2)
        if "MaxDD" in fmt_df.columns:
            fmt_df["MaxDD"] = (fmt_df["MaxDD"] * 100).round(2)
        if "Turnover" in fmt_df.columns:
            # turnover viene ~0..1. lo pasamos a % promedio por rebalance
            fmt_df["Turnover"] = (fmt_df["Turnover"] * 100).round(2)

        st.dataframe(
            fmt_df[show_cols],
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "- CAGR: rendimiento anualizado del sistema long-only binario.\n"
            "- Sharpe / Sortino: calidad de retornos.\n"
            "- MaxDD: peor drawdown desde pico.\n"
            "- Turnover (%): cuánto cambias postura en promedio cada rebalance.\n"
            "- Trades: cuántas veces el sistema pasó de fuera→dentro o dentro→fuera."
        )

    st.markdown("---")

    # -------------------------------
    # 5. Curvas de equity normalizadas
    # -------------------------------
    st.markdown("#### Curvas de equity normalizadas (1.0 = inicio)")

    if not bt_curves:
        st.info("No hay curvas de equity para graficar.")
    else:
        # bt_curves: dict {sym: Series equity}
        # Las unimos en un DataFrame ancho y luego derretimos para Altair
        eq_df = []
        for sym, curve in bt_curves.items():
            if curve is None or curve.empty:
                continue
            tmp = (
                curve
                .rename("equity")
                .to_frame()
                .reset_index()
                .rename(columns={"index": "date"})
            )
            tmp["symbol"] = sym
            eq_df.append(tmp)

        if len(eq_df) == 0:
            st.info("No se pudo armar data suficiente para el gráfico.")
        else:
            long_eq = pd.concat(eq_df, ignore_index=True)

            # normalizar cada equity a 1.0 en su primer punto
            def _norm_grp(g):
                first_val = g["equity"].iloc[0] if len(g) else np.nan
                if first_val and first_val != 0:
                    g["equity_norm"] = g["equity"] / first_val
                else:
                    g["equity_norm"] = np.nan
                return g

            long_eq = (
                long_eq
                .sort_values(["symbol", "date"])
                .groupby("symbol", group_keys=False)
                .apply(_norm_grp)
            )

            chart = (
                alt.Chart(long_eq)
                .mark_line()
                .encode(
                    x=alt.X("date:T", title="Fecha"),
                    y=alt.Y("equity_norm:Q", title="Equidad normalizada"),
                    color=alt.Color("symbol:N", title="Símbolo"),
                    tooltip=[
                        alt.Tooltip("date:T", title="Fecha"),
                        alt.Tooltip("symbol:N", title="Ticker"),
                        alt.Tooltip("equity_norm:Q", title="Equidad norm.", format=".2f")
                    ]
                )
                .properties(height=320)
                .interactive()
            )

            st.altair_chart(chart, use_container_width=True)

            st.caption(
                "Cada línea = 'estar dentro cuando la señal está ON, en cash cuando está OFF', "
                "reinvertido. Sirve para ver estabilidad comparada de cada ticker."
            )

    st.markdown("---")

    # -------------------------------
    # 6. Nota operativa
    # -------------------------------
    st.info(
        "Cómo leer esto:\n"
        "- Este backtest es súper simple: long-only binario por ticker, sin portfolio construction.\n"
        "- La señal es tendencia/momentum (MA200 y/o Mom 12–1 > 0).\n"
        "- El costo en bps castiga girar la posición.\n"
        "- Lo importante aquí es el orden relativo: ¿qué tickers aguantan bien el swing? "
        "Esos son los candidatos que vale la pena sobreponderar cuando estés en RISK ON."
    )

with tab8:
    import numpy as np
    import pandas as pd
    import streamlit as st

    # --- Defaults por si no existen en tu app ---
    DEFAULT_START = "2020-01-01"
    DEFAULT_END   = pd.Timestamp.today().strftime("%Y-%m-%d")

    # --- Helpers métricas (para no depender de otros módulos) ---
    def _cagr(returns: pd.Series, freq_per_year=12) -> float:
        if returns is None or returns.empty:
            return 0.0
        eq = (1 + returns.fillna(0)).cumprod()
        years = len(returns) / float(freq_per_year)
        if years <= 0 or eq.iloc[-1] <= 0:
            return 0.0
        return eq.iloc[-1] ** (1.0 / years) - 1.0

    def _sharpe(returns: pd.Series, freq_per_year=12) -> float:
        mu = returns.mean() * freq_per_year
        sd = returns.std(ddof=0) * np.sqrt(freq_per_year)
        return 0.0 if sd == 0 or np.isnan(sd) else float(mu / sd)

    def _sortino(returns: pd.Series, freq_per_year=12) -> float:
        dn = returns[returns < 0]
        sd = dn.std(ddof=0) * np.sqrt(freq_per_year)
        mu = returns.mean() * freq_per_year
        return 0.0 if sd == 0 or np.isnan(sd) else float(mu / sd)

    def _maxdd(equity: pd.Series) -> float:
        if equity is None or equity.empty:
            return 0.0
        dd = equity / equity.cummax() - 1.0
        return float(dd.min())

    def _perf_from_rets(rets: pd.Series, periods_per_year: int) -> dict:
        if rets is None or rets.empty:
            return {"CAGR":0.0,"Sharpe":0.0,"Sortino":0.0,"MaxDD":0.0,"Turnover":0.0}
        equity = (1 + rets.fillna(0)).cumprod()
        return {
            "CAGR":   _cagr(rets, periods_per_year),
            "Sharpe": _sharpe(rets, periods_per_year),
            "Sortino":_sortino(rets, periods_per_year),
            "MaxDD":  _maxdd(equity),
        }

    # -------- Loader/Caché de precios (ajusta a tu código de datos) --------
    @st.cache_data(show_spinner=False)
    def _cached_load_prices_panel(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
        """
        Intenta usar un loader existente de tu app. Si no está, muestra error claro.
        Debe devolver: {symbol: DataFrame con index fecha y columna 'close'}.
        """
        # 1) Si ya traes un panel en session_state (por ejemplo desde otra tab)
        panel = st.session_state.get("price_panel")
        if isinstance(panel, dict) and panel:
            # filtra al set pedido
            return {s: df.loc[str(start):str(end)] for s, df in panel.items() if s in symbols and not df.empty}

        # 2) Busca loaders comunes en session_state (inyectados en tu app)
        loader = st.session_state.get("fetch_price_history") or st.session_state.get("load_price_history")
        if callable(loader):
            out = {}
            for s in symbols:
                df = loader(s, start=start, end=end)  # tu función debe existir
                if isinstance(df, pd.DataFrame) and "close" in df.columns and not df.empty:
                    out[s] = df.sort_index()
            return out

        # 3) Si nada existe, forzamos un error explicativo
        raise RuntimeError(
            "No encuentro un loader de precios. Define en st.session_state['fetch_price_history'] "
            "una función (symbol, start, end) -> DataFrame con columna 'close', o provee 'price_panel'."
        )

    # ==================== UI ====================
    st.subheader("🔧 Tuning de umbrales (random search)")

    kept       = st.session_state.get("kept", pd.DataFrame())
    uni_cur    = st.session_state.get("uni", pd.DataFrame())
    df_vfq_all = st.session_state.get("vfq_all", pd.DataFrame())

    if kept.empty or df_vfq_all.empty:
        st.warning("Necesitas correr Guardrails y VFQ antes de tunear.")
        st.stop()

    # Asegura 'acc_pct' si falta (a partir de 'accruals_ta')
    if "acc_pct" not in df_vfq_all.columns and "accruals_ta" in df_vfq_all.columns:
        s = df_vfq_all["accruals_ta"].astype(float)
        # más cerca de 0 es mejor → usar percentil del |accrual| e invertir
        pct = (s.abs().rank(pct=True, method="average"))
        df_vfq_all["acc_pct"] = (1.0 - pct) * 100.0

    # --------- Parámetros de búsqueda ----------
    c1, c2, c3 = st.columns(3)
    with c1:
        n_samples = st.number_input("N° combinaciones aleatorias", 20, 2000, 150, 10)
        cost_bps  = st.number_input("Costos (bps por rebalance)", 0, 100, 10, 1)
        use_and   = st.toggle("Tendencia: MA200 Y Mom12-1>0", value=False)
    with c2:
        start_tune = st.date_input("Inicio tuning", value=pd.to_datetime(DEFAULT_START).date())
        end_tune   = st.date_input("Fin tuning", value=pd.to_datetime(DEFAULT_END).date())
        min_names  = st.number_input("Mín. símbolos por cartera", 5, 200, 15, 1)
    with c3:
        seed = st.number_input("Semilla aleatoria", 0, 10_000, 1234, 1)
        reb_freq = st.selectbox("Frecuencia rebalanceo", ["M","W","Q"], index=0)
        go_btn = st.button("Ejecutar Tuning", use_container_width=True, type="primary")

    # Rangos razonables (puedes ajustarlos)
    ranges = dict(
        min_quality=(0.30, 0.70),
        min_value=(0.30, 0.70),
        min_acc_pct=(40, 85),           # %
        max_ndebt=(1.5, 3.0),
        min_hits_req=(0, 5),            # entero
        min_breakout=(50, 95),
        min_rvol20=(1.00, 2.50),
        topN_prob=(10, 60),
    )

    def _sample_params(rng: np.random.RandomState) -> dict:
        p = {
            "min_quality":  float(np.round(rng.uniform(*ranges["min_quality"]), 2)),
            "min_value":    float(np.round(rng.uniform(*ranges["min_value"]), 2)),
            "min_acc_pct":  int(rng.randint(*ranges["min_acc_pct"])),
            "max_ndebt":    float(np.round(rng.uniform(*ranges["max_ndebt"]), 1)),
            "min_breakout": int(rng.randint(*ranges["min_breakout"])),
            "min_rvol20":   float(np.round(rng.uniform(*ranges["min_rvol20"]), 2)),
            "min_hits_req": int(rng.randint(*ranges["min_hits_req"])),
            "topN_prob":    int(rng.randint(*ranges["topN_prob"])),
        }
        p["topN_prob"] = max(int(min_names), p["topN_prob"])
        return p

    def _rank_and_pick(df: pd.DataFrame, p: dict) -> list[str]:
        """Aplica filtros y rankea por prob_up (o BreakoutScore) devolviendo topN_prob."""
        m = pd.Series(True, index=df.index)
        m &= (df["quality_adj_neut"].fillna(0) >= p["min_quality"])
        m &= (df["value_adj_neut"].fillna(0)   >= p["min_value"])
        m &= (df["hits"].fillna(0)             >= p["min_hits_req"])
        m &= (df["BreakoutScore"].fillna(0)    >= p["min_breakout"])
        m &= (df["RVOL20"].fillna(0)           >= p["min_rvol20"])
        m &= (df["acc_pct"].isna() | (df["acc_pct"] >= p["min_acc_pct"]))
        m &= (df["netdebt_ebitda"].isna() | (df["netdebt_ebitda"] <= p["max_ndebt"]))

        df_f = df.loc[m].copy()
        if df_f.empty:
            return []
        if "prob_up" in df_f.columns and df_f["prob_up"].notna().any():
            df_f = df_f.sort_values("prob_up", ascending=False)
        else:
            df_f = df_f.sort_values("BreakoutScore", ascending=False)

        return (
            df_f["symbol"].dropna().astype(str).unique().tolist()[: int(p["topN_prob"])]
        )

    def _portfolio_metrics_from_curves(curves: dict[str, pd.Series]) -> dict:
        """Cesta igual-ponderada a partir de curvas por símbolo."""
        if not curves:
            return {"CAGR":0,"Sharpe":0,"Sortino":0,"MaxDD":0,"N":0,"Turnover":0}
        eq = pd.DataFrame(curves).dropna(how="all")
        if eq.empty:
            return {"CAGR":0,"Sharpe":0,"Sortino":0,"MaxDD":0,"N":0,"Turnover":0}
        rets = eq.pct_change().mean(axis=1).fillna(0.0)
        perf = _perf_from_rets(rets, {"M":12,"W":52,"Q":4}[reb_freq])
        perf["N"] = int(eq.shape[1])
        return perf

    results, details = [], []

    if go_btn:
        try:
            rng = np.random.RandomState(int(seed))
            pbar = st.progress(0.0, text="Buscando combinaciones…")

            from backtests import backtest_many  # tu módulo agregado

            for i in range(int(n_samples)):
                p = _sample_params(rng)
                picks = _rank_and_pick(df_vfq_all, p)
                pbar.progress((i+1)/float(n_samples), text=f"Eval {i+1}/{n_samples}")

                if len(picks) < int(min_names):
                    continue

                # Precios y backtest
                panel = _cached_load_prices_panel(
                    picks,
                    start=pd.to_datetime(start_tune),
                    end=pd.to_datetime(end_tune),
                )
                if not isinstance(panel, dict) or not panel:
                    continue

                metrics_df, curves = backtest_many(
                    panel=panel,
                    symbols=list(panel.keys()),
                    cost_bps=int(cost_bps),
                    lag_days=0,
                    use_and_condition=bool(use_and),
                    rebalance_freq=reb_freq,
                )
                avg_turn = float(metrics_df["Turnover"].mean()) if isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty else 0.0

                port_perf = _portfolio_metrics_from_curves(curves)
                row = dict(
                    Sharpe=float(port_perf.get("Sharpe",0.0)),
                    Sortino=float(port_perf.get("Sortino",0.0)),
                    CAGR=float(port_perf.get("CAGR",0.0)),
                    MaxDD=float(port_perf.get("MaxDD",0.0)),
                    N=int(port_perf.get("N",0)),
                    Turnover=avg_turn,
                )
                row.update(p)
                results.append(row)
                details.append({"params": p, "picks": picks})

            pbar.empty()

        except Exception as e:
            st.error("Error durante el tuning.")
            st.exception(e)

    if results:
        res_df = pd.DataFrame(results).sort_values(["Sharpe","CAGR"], ascending=False).reset_index(drop=True)
        st.markdown("### 🏁 Top combinaciones")
        st.dataframe(res_df.head(25), use_container_width=True, hide_index=True)

        st.markdown("#### 📌 Detalle de selección")
        idx = st.number_input("Fila (Top-k) a inspeccionar", 0, max(0, len(res_df)-1), 0, 1)
        chosen = res_df.iloc[int(idx)].to_dict()
        st.json({k: chosen[k] for k in [
            "Sharpe","CAGR","Sortino","MaxDD","Turnover","N",
            "min_quality","min_value","min_acc_pct","max_ndebt",
            "min_hits_req","min_breakout","min_rvol20","topN_prob"
        ]})

        picks = details[int(idx)]["picks"]
        st.caption(f"Tickers ({len(picks)}): " + ", ".join(sorted(picks[:120])) + (" …" if len(picks)>120 else ""))

        if st.button("👉 Adoptar este preset", use_container_width=True):
            st.session_state["vfq_best_preset"] = {
                "from_tuning": True,
                "rebalance": reb_freq,
                "use_and": bool(use_and),
                "cost_bps": int(cost_bps),
                "date_range": (str(start_tune), str(end_tune)),
                "params": {k: chosen[k] for k in [
                    "min_quality","min_value","min_acc_pct","max_ndebt",
                    "min_hits_req","min_breakout","min_rvol20","topN_prob"
                ]},
                "metrics": {k: chosen[k] for k in ["Sharpe","CAGR","Sortino","MaxDD","Turnover","N"]},
                "picks": picks,
            }
            st.success("Preset guardado en st.session_state['vfq_best_preset']. Copia estos valores a los sliders de VFQ.")

    st.markdown("---")
    st.caption(
        "Este tuning usa **las métricas VFQ actuales** para filtrar y luego backtestea la cesta en el intervalo elegido. "
        "Es un _proxy_ rápido y puede tener **look-ahead**. Para rigor total, migra a walk-forward con fundamentales históricos."
    )
