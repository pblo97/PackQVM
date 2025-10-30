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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Universo",
        "Guardrails",
        "VFQ",
        "Señales",
        "QVM (growth-aware)",
        "Export",
        "Backtesting",
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
        st.warning(
            "No hay símbolos aprobados por Guardrails. Ajusta la pestaña Guardrails."
        )
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
        min_quality = st.slider(
            "Min Quality neut.", 0.0, 1.0, 0.3, 0.01
        )
        min_value = st.slider(
            "Min Value neut.", 0.0, 1.0, 0.3, 0.01
        )
        max_ndebt = st.slider(
            "Max NetDebt/EBITDA",
            0.0,
            5.0,
            2.0,
            0.1,
        )
    with c2v:
        min_acc_pct = st.slider(
            "Accruals limpios (% mínimo)",
            0,
            100,
            30,
            1,
        )
        min_hits_req = st.slider(
            "Min hits (breakout hits)",
            0,
            5,
            1,
            1,
        )
        min_rvol20 = st.slider(
            "Min RVOL20",
            0.0,
            5.0,
            1.2,
            0.05,
        )
    with c3v:
        min_breakout = st.slider(
            "Min BreakoutScore",
            0,
            100,
            50,
            1,
        )
        topN_prob = st.slider(
            "Top N por prob_up",
            5,
            100,
            30,
            1,
        )

    mask = pd.Series(True, index=df_vfq_all.index, dtype=bool)

    # quality / value floors
    mask &= (
        df_vfq_all["quality_adj_neut"].fillna(0)
        >= float(min_quality)
    )
    mask &= (
        df_vfq_all["value_adj_neut"].fillna(0)
        >= float(min_value)
    )

    # breakout hits, breakoutscore, rvol
    mask &= (
        df_vfq_all["hits"].fillna(0)
        >= int(min_hits_req)
    )
    mask &= (
        df_vfq_all["BreakoutScore"].fillna(0)
        >= float(min_breakout)
    )
    mask &= (
        df_vfq_all["RVOL20"].fillna(0)
        >= float(min_rvol20)
    )

    # endeudamiento
    mask &= (
        df_vfq_all["netdebt_ebitda"].isna()
        | (
            df_vfq_all["netdebt_ebitda"]
            <= float(max_ndebt)
        )
    )

    # accruals limpios (% alto es mejor en tu escala actual)
    mask &= (
        df_vfq_all["acc_pct"].isna()
        | (
            df_vfq_all["acc_pct"]
            >= float(min_acc_pct)
        )
    )

    df_keep_vfq = df_vfq_all.loc[mask].copy()

    # ranking: si hay prob_up úsala, si no, BreakoutScore
    if df_keep_vfq["prob_up"].notna().any():
        df_keep_vfq = df_keep_vfq.sort_values(
            "prob_up", ascending=False
        )
    else:
        df_keep_vfq = df_keep_vfq.sort_values(
            "BreakoutScore", ascending=False
        )

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

    st.markdown("### 🧹 Rechazados por VFQ")
    rejected_syms = sorted(
        set(kept_syms) - set(df_keep_vfq["symbol"])
    )
    rej_view = df_vfq_all[
        df_vfq_all["symbol"].isin(rejected_syms)
    ].copy()

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
    cols_rej_show = [
        c for c in cols_rej_show if c in rej_view.columns
    ]

    st.dataframe(
        rej_view[cols_rej_show],
        use_container_width=True,
        hide_index=True,
    )

    # Guardar para tabs futuras
    st.session_state["vfq_top"] = (
        vfq_top[["symbol"]].drop_duplicates()
    )
    st.session_state["vfq_table"] = vfq_top.reset_index(
        drop=True
    )
    st.session_state["pipeline_ready"] = True


# ====== TAB 4: SEÑALES (placeholder por ahora) ======
with tab4:
    st.subheader("Señales técnicas / Breakout")
    if not st.session_state.get("pipeline_ready", False):
        st.info(
            "Aún no hay pipeline listo. Corre Universo → Guardrails → VFQ."
        )
    else:
        st.write(
            "Acá vas a calcular señal técnica final (tendencia, breakout, RS slope, etc.)."
        )
        st.caption(
            "Todavía placeholder: falta amarrar apply_trend_filter(), enrich_with_breakout(), etc."
        )


# ====== TAB 5: QVM (growth-aware) ======
with tab5:
    st.subheader("QVM (growth-aware)")
    if not st.session_state.get("pipeline_ready", False):
        st.info(
            "Aún no hay datos finales. Termina VFQ primero."
        )
    else:
        st.write(
            "Placeholder QVM growth-aware. Aquí luego aplicas compute_qvm_scores() + apply_megacap_rules()."
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
    st.write(
        "Acá va la simulación histórica con backtest_many(), plus métricas con perf_summary_from_returns()."
    )

