from __future__ import annotations

# --- watcher de archivos (evita recargas agresivas en dev) ---
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "poll"  # o "none" si prefieres desactivar

# ================== IMPORTS BASE ==================
import hashlib
import json
import time
from datetime import datetime
from typing import Tuple
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ============== IMPORTS DE TU PIPELINE ==============
from pipeline_factors import build_factor_frame
from fundamentals import (
    download_fundamentals,
    build_vfq_scores_dynamic,          # (importado si luego lo usas)
    download_guardrails_batch,
    apply_quality_guardrails,          # (importado si luego lo usas)
)
from scoring import (
    blend_breakout_qvm,                # (importado si luego lo usas)
    build_momentum_proxy,              # (importado si luego lo usas)
)
from data_io import (
    run_fmp_screener,
    filter_universe,                   # (importado si luego lo usas)
    load_prices_panel,
    load_benchmark,
    DEFAULT_START,
    DEFAULT_END,
)
from pipeline import (
    apply_trend_filter,                # (importado si luego lo usas)
    enrich_with_breakout,              # (importado si luego lo usas)
    market_regime_on,                  # (importado si luego lo usas)
)
from backtests import backtest_many

# Opcional (growth-aware). No se usan aún en la UI, pero los dejamos importables.
from factors_growth_aware import (
    compute_qvm_scores,                # (importado si luego lo usas)
    apply_megacap_rules,               # (importado si luego lo usas)
)

# ================== KEYS SNAPSHOT ==================
SNAP_KEY  = "vfq_snapshot"
SNAP_META = "vfq_snapshot_meta"

# ============== UTILS UNIVERSO & SNAPSHOT ==============
def _universe_fingerprint(df_universe: pd.DataFrame) -> str:
    """Firma determinista orden-agnóstica del universo por símbolos."""
    syms = (
        df_universe.get("symbol", pd.Series([], dtype=str))
        .dropna()
        .astype(str)
        .sort_values()
    )
    raw = ("|".join(syms.tolist())).encode("utf-8")
    return hashlib.md5(raw).hexdigest()

def compute_vfq_snapshot(uni_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos los factores VFQ sobre el universo dado
    y reinyecta sector/market_cap desde el universo actual.
    """
    universe_syms = (
        uni_df["symbol"].dropna().astype(str).unique().tolist()
    )
    feats = build_factor_frame(universe_syms)
    feats = (
        feats.drop(columns=["sector", "market_cap"], errors="ignore")
        .merge(uni_df[["symbol", "sector", "market_cap"]], on="symbol", how="left")
    )
    return feats

# ------------------ CACHES ------------------
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
    1) Pide universo base a FMP con filtros básicos.
    2) Normaliza columnas core (symbol, sector, market_cap).
    3) Aplica filtros post-request por volumen y antigüedad IPO.
    """
    df = run_fmp_screener(
        limit=limit,
        mcap_min=mcap_min,
        volume_min=volume_min,
        fetch_profiles=True,
        cache_key=cache_key,
        force=False,  # importante para respetar cache_key
    )
    if df is None:
        return pd.DataFrame(columns=["symbol", "sector", "market_cap"])

    df = df.copy()

    # market cap normalizada → market_cap
    if "market_cap" not in df.columns:
        if "marketCap" in df.columns:
            df["market_cap"] = pd.to_numeric(df["marketCap"], errors="coerce")
        else:
            df["market_cap"] = np.nan

    # sector seguro
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    else:
        s = df["sector"].astype(str)
        s = s.replace({"": "Unknown"})
        s = s.where(~s.isna(), "Unknown")
        df["sector"] = s

    # volumen numérico
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # ipoDate a datetime
    if "ipoDate" in df.columns:
        df["ipoDate"] = pd.to_datetime(df["ipoDate"], errors="coerce", utc=True)
    else:
        df["ipoDate"] = pd.NaT

    # -------- filtros post-request --------
    df = df[df["market_cap"] >= float(mcap_min)]
    if "volume" in df.columns:
        df = df[df["volume"] >= float(volume_min)]
    if df["ipoDate"].notna().any():
        cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=int(ipo_days))
        df = df[df["ipoDate"] < cutoff]

    # columnas mínimas
    core_cols = ["symbol", "sector", "market_cap"]
    if "symbol" not in df.columns:
        if "ticker" in df.columns:
            df["symbol"] = df["ticker"].astype(str)
        else:
            df["symbol"] = ""

    out = (
        df[core_cols]
        .dropna(subset=["symbol"])
        .reset_index(drop=True)
    )
    return out

@st.cache_data(show_spinner=False)
def _cached_vfq_snapshot(uni_df: pd.DataFrame, uni_sig: str) -> pd.DataFrame:
    """
    Cachea el snapshot VFQ completo ligado a la firma del universo.
    Cualquier cambio de `uni_sig` invalida este cache automáticamente.
    """
    _ = uni_sig  # se usa solo para invalidar el caché cuando cambia
    return compute_vfq_snapshot(uni_df)

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
    return download_fundamentals(list(symbols), market_caps=mc_map, cache_key=cache_key, force=False)

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_load_prices_panel(symbols, start, end, cache_key=""):
    return load_prices_panel(symbols, start, end, cache_key=cache_key, force=False)

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_load_benchmark(bench, start, end):
    return load_benchmark(bench, start, end)

# ------------------ HELPERS FORMATO ------------------

# ---- Guardrails helpers (QVM) ----
COVERAGE_COLS = ["profit_hits","netdebt_ebitda","accruals_ta","asset_growth","share_issuance"]

def _build_guardrails_base_from_snapshot(snapshot: pd.DataFrame, uni: pd.DataFrame) -> pd.DataFrame:
    """Toma el snapshot VFQ y le injerta sector/mcap + coverage_count; sin filtros."""
    df = (
        snapshot.drop(columns=["sector", "market_cap"], errors="ignore")
        .merge(uni[["symbol","sector","market_cap"]], on="symbol", how="left")
    )
    # coverage_count = cuántas métricas clave existen (no NaN)
    df["coverage_count"] = (
        df[COVERAGE_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .notna()
        .sum(axis=1)
        .astype(int)
    )
    return df

def _fmt_mcap(x):
    try:
        x = float(x)
        if x >= 1e12: return f"${x/1e12:.2f}T"
        if x >= 1e9:  return f"${x/1e9:.2f}B"
        if x >= 1e6:  return f"${x/1e6:.2f}M"
        return f"${x:,.0f}"
    except Exception:
        return ""

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
[data-testid="stDataFrame"] tbody tr:hover { background: rgba(59,130,246,.08) !important; }
[data-testid="stCaptionContainer"] { opacity: .85; }
</style>
""",
    unsafe_allow_html=True,
)

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
            "MarketCap mínimo (USD)", value=5e8, step=1e8, format="%.0f"
        )

        volume_min = st.number_input(
            "Volumen mínimo diario", value=500_000, step=50_000, format="%.0f"
        )

        ipo_days = st.slider("Antigüedad IPO (días)", 90, 1500, 365, 30)

    # ---- Fundamentales & Guardrails ----
    with st.expander("Fundamentales & Guardrails", expanded=False):
        min_cov_guard = st.slider("Cobertura VFQ mínima (# métricas)", 1, 4, 2)
        profit_hits   = st.slider("Pisos de rentabilidad (hits EBIT/CFO/FCF)", 0, 3, 2)
        max_issuance  = st.slider("Net issuance máx.", 0.00, 0.10, 0.03, 0.01)
        max_assets    = st.slider("Asset growth |y/y| máx.", 0.00, 0.50, 0.20, 0.01)
        max_accr      = st.slider("Accruals/TA | | máx.", 0.00, 0.25, 0.10, 0.01)
        max_ndeb      = st.slider("NetDebt/EBITDA máx.", 0.0, 6.0, 3.0, 0.5)

    # ---- Técnico ----
    with st.expander("Técnico — Tendencia & Breakout", expanded=True):
        use_and          = st.toggle("MA200 Y Mom 12–1", value=False)
        require_breakout = st.toggle("Exigir Breakout para ENTRY", value=False)
        rvol_th          = st.slider("RVOL (20d) mín.", 0.8, 2.5, 1.2, 0.1)
        closepos_th      = st.slider("ClosePos mín.", 0.0, 1.0, 0.60, 0.05)
        p52_th           = st.slider("Cercanía 52W High", 0.80, 1.00, 0.95, 0.01)
        updown_vol_th    = st.slider("Up/Down Vol Ratio (20d)", 0.8, 3.0, 1.2, 0.1)
        min_hits_brk     = st.slider("Mínimo checks breakout (K de 4)", 1, 4, 3)
        atr_pct_min      = st.slider("ATR pct (6–12m) mín.", 0.0, 1.0, 0.6, 0.05)
        use_rs_slope     = st.toggle("Exigir RS slope > 0 (MA20)", value=False)

    # ---- Régimen & Fechas ----
    with st.expander("Régimen & Fechas", expanded=False):
        bench   = st.selectbox("Benchmark", ["SPY", "QQQ", "^GSPC"], index=0)
        risk_on = st.toggle("Exigir mercado Risk-ON", value=True)
        start   = st.date_input("Inicio", value=pd.to_datetime(DEFAULT_START).date())
        end     = st.date_input("Fin",    value=pd.to_datetime(DEFAULT_END).date())

    # ---- Ranking avanzado ----
    with st.expander("Ranking avanzado", expanded=False):
        beta_prob   = st.slider("Sensibilidad probabilidad (β)", 1.0, 12.0, 6.0, 0.5)
        top_n_show  = st.slider("Top N a resaltar", 10, 100, 25, 5)

    st.markdown("---")
    run_btn = st.button("Ejecutar", use_container_width=True)

# Presets que ajustan umbrales técnicos, sin pisar lo que ya moviste manualmente demasiado
if preset == "Laxo":
    rvol_th     = min(rvol_th, 1.0)
    closepos_th = min(closepos_th, 0.55)
    p52_th      = min(p52_th, 0.92)
    min_hits_brk = min(min_hits_brk, 2)
elif preset == "Estricto":
    rvol_th     = max(rvol_th, 1.5)
    closepos_th = max(closepos_th, 0.65)
    p52_th      = max(p52_th, 0.97)
    min_hits_brk = max(min_hits_brk, 3)

# Cache tag que depende de entradas clave del universo
cache_tag = f"{int(min_mcap)}_{ipo_days}_{limit}_{int(volume_min)}"

# Estado del pipeline
if "pipeline_ready" not in st.session_state:
    st.session_state["pipeline_ready"] = False

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab6, tab7, tab8 = st.tabs(
    ["Universo", "Guardrails", "VFQ", "Señales", "Export", "Backtesting", "Tuning"]
)

# ==================== VFQ sidebar extra ====================
with st.sidebar:
    st.markdown("⚙️ Fundamentos (VFQ)")

    value_metrics_opts   = ["inv_ev_ebitda", "fcf_yield"]
    quality_metrics_opts = ["gross_profitability", "roic", "roa", "netMargin"]

    sel_value = st.multiselect("Métricas Value", options=value_metrics_opts, default=["inv_ev_ebitda", "fcf_yield"])
    sel_quality = st.multiselect("Métricas Quality", options=quality_metrics_opts, default=["gross_profitability", "roic"])

    c1x, c2x = st.columns(2)
    with c1x:
        w_value = st.slider("Peso Value", 0.0, 1.0, 0.5, 0.05)
    with c2x:
        w_quality = st.slider("Peso Quality", 0.0, 1.0, 0.5, 0.05)

    method_intra = st.radio("Agregación intra-bloque", ["mean", "median", "weighted_mean"], index=0, horizontal=True)
    winsor_p     = st.slider("Winsor p (cola)", 0.0, 0.10, 0.01, 0.005)
    size_buckets = st.slider("Buckets por tamaño", 1, 5, 3, 1)
    group_mode   = st.selectbox("Agrupar por", ["sector", "sector|size"], index=1)
    min_cov      = st.slider("Cobertura mín. (# métricas)", 0, 8, 1, 1)
    min_pct      = st.slider("VFQ pct (intra-sector) mín.", 0.00, 1.00, 0.00, 0.01)

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
        st.session_state["uni"] = raw_universe.copy()

    # leer versión estable en memoria
    uni_df = st.session_state["uni"].copy()

    total_raw = len(uni_df)
    total_filtrado = len(uni_df)  # acá podrías hacer más filtros si quieres

    c1m, c2m = st.columns(2)
    c1m.metric("Screener", f"{total_raw}")
    c2m.metric("Tras filtros básicos", f"{total_filtrado}")

    st.dataframe(uni_df.head(50), hide_index=True, use_container_width=True)
    st.caption("Esta tabla vive en st.session_state['uni'] y alimenta las demás pestañas.")

    # Firma del universo basada en símbolos (orden-agnóstica)
    st.session_state["uni_sig"] = _universe_fingerprint(uni_df)

    # (Opcional y recomendado) Parám. que definen el universo: usa tus controles REALES
    st.session_state["universe_norm_params"] = {
        "n_universe": int(limit),          # slider de tamaño de universo real
        "winsor_p": float(winsor_p),       # slider VFQ
        "buckets": int(size_buckets),      # slider VFQ
        "group_by": group_mode,            # selector VFQ
    }

# ====== TAB 2: GUARDRAILS ======
# ====== TAB 2: GUARDRAILS ======
with tab2:
    st.subheader("Guardrails")

    uni = st.session_state.get("uni", pd.DataFrame())
    if uni is None or uni.empty or "symbol" not in uni.columns:
        st.info("Primero genera el universo en la pestaña Universo.")
        st.stop()

    # 1) Construimos / recuperamos la BASE (snapshot + coverage) una sola vez por universo
    uni_sig = st.session_state.get("uni_sig", "")
    need_rebuild = (
        ("qvm_guard_uni_sig" not in st.session_state) or
        ("qvm_guardrails_base" not in st.session_state) or
        (st.session_state["qvm_guard_uni_sig"] != uni_sig) or
        run_btn  # si apretaste Ejecutar, refrescamos
    )
    if need_rebuild:
        snapshot_vfq = _cached_vfq_snapshot(uni, uni_sig)  # <— usa tu caché ya definido
        base = _build_guardrails_base_from_snapshot(snapshot_vfq, uni)
        st.session_state["qvm_guardrails_base"] = base
        st.session_state["qvm_guard_uni_sig"] = uni_sig

    base = st.session_state["qvm_guardrails_base"].copy()

    # 2) Aplicamos SOLO filtros según sliders (sin recalcular factores)
    # sliders ya definidos en sidebar: min_cov_guard, profit_hits, max_issuance, max_assets, max_accr, max_ndeb
    # (aseguramos tipos y NaNs)
    def _num(s, absval=False):
        s = pd.to_numeric(base[s], errors="coerce")
        return s.abs() if absval else s

    pass_profit   = (_num("profit_hits") >= int(profit_hits))
    pass_issuance = (_num("share_issuance", absval=True) <= float(max_issuance))
    pass_assets   = (_num("asset_growth", absval=True)   <= float(max_assets))
    pass_accruals = (_num("accruals_ta", absval=True)    <= float(max_accr))
    pass_ndebt    = (_num("netdebt_ebitda")              <= float(max_ndeb))
    pass_cover    = (pd.to_numeric(base["coverage_count"], errors="coerce") >= int(min_cov_guard))

    pass_all = pass_profit & pass_issuance & pass_assets & pass_accruals & pass_ndebt & pass_cover

    df_all = base.assign(
        pass_profit=pass_profit.fillna(False),
        pass_issuance=pass_issuance.fillna(False),
        pass_assets=pass_assets.fillna(False),
        pass_accruals=pass_accruals.fillna(False),
        pass_ndebt=pass_ndebt.fillna(False),
        pass_coverage=pass_cover.fillna(False),
        pass_all=pass_all.fillna(False),
    )

    kept_raw = df_all.loc[df_all["pass_all"], ["symbol"]].drop_duplicates().reset_index(drop=True)
    st.session_state["kept"] = kept_raw
    st.session_state["guard_diag"] = df_all.copy()

    total = len(df_all)
    pasan = int(df_all["pass_all"].sum())
    rechaz = total - pasan

    c1g, c2g, c3g = st.columns(3)
    c1g.metric("Pasan guardrails estrictos", f"{pasan}")
    c2g.metric("Candidatos saludables (relajado)", f"{pasan}")  # placeholder
    c3g.metric("Rechazados totales", f"{rechaz}")

    cols_show = [
        "symbol","sector","pass_all","profit_hits","coverage_count",
        "asset_growth","accruals_ta","netdebt_ebitda",
        "pass_profit","pass_issuance","pass_assets","pass_accruals","pass_ndebt","pass_coverage",
    ]
    cols_show = [c for c in cols_show if c in df_all.columns]

    with st.expander(f"Detalle guardrails (estricto): {pasan} / {total}", expanded=True):
        st.dataframe(
            df_all[cols_show].sort_values("symbol"),
            use_container_width=True,
            hide_index=True,
        )

    st.caption("pass_all = pasó TODAS las barreras. coverage_count = cuánta info fundamental tenemos disponible.")

# ====== TAB 3: VFQ ======
with tab3:
    st.subheader("VFQ (Value / Quality / Flow)")

    kept   = st.session_state.get("kept", pd.DataFrame())
    uni_cur = st.session_state.get("uni", pd.DataFrame())

    if kept is None or kept.empty or "symbol" not in kept.columns:
        st.warning("No hay símbolos aprobados por Guardrails. Ajusta la pestaña Guardrails.")
        st.stop()

    kept_syms = kept["symbol"].dropna().astype(str).unique().tolist()
    if not kept_syms:
        st.warning("La lista kept está vacía.")
        st.stop()

    # ------------------------------------------------------------
    # A) 🔒 SNAPSHOT VFQ FIJO con auto-invalidate por cambio de universo
    # ------------------------------------------------------------
    def _kept_signature(kept_syms: list[str], extra: dict | None = None) -> str:
        payload = {"kept": sorted(map(str, kept_syms))}
        if extra:
            payload["extra"] = extra
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.md5(raw).hexdigest()

    snap_key = SNAP_KEY
    meta_key = SNAP_META

    # Botón manual para forzar recálculo
    recalc = st.button("♻️ Recalcular snapshot VFQ (universo)", use_container_width=False)

    # 1) Firma compuesta del universo/kept
    universe_norm_params = st.session_state.get("universe_norm_params", {})
    uni_sig = st.session_state.get("uni_sig", "")
    cur_sig = _kept_signature(kept_syms, extra={"universe": universe_norm_params, "uni_sig": uni_sig})

    # 2) ¿Debemos reconstruir?
    need_rebuild = recalc or (snap_key not in st.session_state)
    if not need_rebuild:
        prev_meta = st.session_state.get(meta_key, {})
        need_rebuild = (prev_meta.get("kept_sig") != cur_sig)

    if need_rebuild:
        # --- Construye snapshot nuevo con el UNIVERSO ACTUAL filtrado a kept ---
        uni_kept = uni_cur[uni_cur["symbol"].astype(str).isin(kept_syms)].copy()

        # 🔐 Cacheado por firma de universo
        df_vfq_all = _cached_vfq_snapshot(uni_kept, uni_sig)

        # Orden determinista (llaves “core”)
        df_vfq_all = df_vfq_all.assign(
            _p=df_vfq_all.get("prob_up", pd.Series(-9e9, index=df_vfq_all.index)).fillna(-9e9),
            _b=df_vfq_all.get("BreakoutScore", pd.Series(-9e9, index=df_vfq_all.index)).fillna(-9e9),
            _q=df_vfq_all.get("quality_adj_neut", pd.Series(-9e9, index=df_vfq_all.index)).fillna(-9e9),
            _v=df_vfq_all.get("value_adj_neut", pd.Series(-9e9, index=df_vfq_all.index)).fillna(-9e9),
        ).sort_values(
            ["_p", "_b", "_q", "_v", "symbol"],
            ascending=[False, False, False, False, True],
            kind="mergesort",
        )

        # Percentil por tamaño (carril mega-cap)
        if "market_cap" in df_vfq_all.columns:
            df_vfq_all["cap_pct"] = df_vfq_all["market_cap"].rank(pct=True)

        st.session_state[snap_key] = df_vfq_all.copy()
        st.session_state[meta_key] = {
            "kept_sig": cur_sig,
            "n_kept": len(kept_syms),
            "ts": time.time(),
        }

    # 3) Consumimos SIEMPRE el snapshot fijo
    df_vfq_all = st.session_state[snap_key].copy()
    meta = st.session_state.get(meta_key, {})
    st.caption(
        f"Snapshot fijo: {meta.get('n_kept','?')} símbolos en kept. "
        f"(ts={meta.get('ts','—')}). Se actualiza solo si cambia el universo/kept o presionas el botón."
    )

    # -------------------------------------
    # B) 🎛️ SLIDERS (+ toggle carril mega-cap)
    # -------------------------------------
    c1v, c2v, c3v = st.columns(3)
    with c1v:
        min_quality = st.slider("Min Quality neut.", 0.0, 1.0, 0.30, 0.01)
        min_value   = st.slider("Min Value neut.",   0.0, 1.0, 0.30, 0.01)
        max_ndebt   = st.slider("Max NetDebt/EBITDA", 0.0, 5.0, 3.0, 0.1)
    with c2v:
        min_acc_pct   = st.slider("Accruals limpios (% mínimo)", 0, 100, 30, 1)
        min_hits_req  = st.slider("Min hits (breakout hits)",     0, 5,  2, 1)
        min_rvol20    = st.slider("Min RVOL20",                   0.0, 5.0, 1.50, 0.05)
    with c3v:
        min_breakout  = st.slider("Min BreakoutScore", 0, 100, 80, 1)
        topN_prob     = st.slider("Top N por prob_up", 5, 100, 30, 1)
        relax_mega    = st.toggle("⚖️ Aflojar técnica para mega-caps (top 10% cap)", value=True)

    # -----------------------------------------------------------------
    # C) 🧪 Filtros sin re-normalizar + orden estable + carril mega-cap
    # -----------------------------------------------------------------
    is_mega = df_vfq_all.get("cap_pct", pd.Series(0, index=df_vfq_all.index)) >= 0.90  # top 10% por market cap

    # Reglas técnicas “size-aware” (si activas el toggle)
    if relax_mega:
        hits_req = np.where(is_mega, np.maximum(1,  min_hits_req-1), min_hits_req)
        rvol_req = np.where(is_mega, np.maximum(1.1, min_rvol20-0.3), min_rvol20)
        brk_req  = np.where(is_mega, np.maximum(60,  min_breakout-10), min_breakout)
    else:
        hits_req = min_hits_req
        rvol_req = min_rvol20
        brk_req  = min_breakout

    m = pd.Series(True, index=df_vfq_all.index, dtype=bool)
    m &= df_vfq_all.get("quality_adj_neut", pd.Series(0, index=df_vfq_all.index)).fillna(0) >= float(min_quality)
    m &= df_vfq_all.get("value_adj_neut",   pd.Series(0, index=df_vfq_all.index)).fillna(0) >= float(min_value)
    m &= (df_vfq_all.get("acc_pct", pd.Series(np.nan, index=df_vfq_all.index)).isna()
          | (df_vfq_all.get("acc_pct").fillna(0) >= float(min_acc_pct)))
    m &= (df_vfq_all.get("netdebt_ebitda", pd.Series(np.nan, index=df_vfq_all.index)).isna()
          | (df_vfq_all.get("netdebt_ebitda").fillna(0) <= float(max_ndebt)))

    # Técnica (ya size-aware si relax_mega=True)
    m &= df_vfq_all.get("hits", pd.Series(0, index=df_vfq_all.index)).fillna(0)          >= hits_req
    m &= df_vfq_all.get("RVOL20", pd.Series(0, index=df_vfq_all.index)).fillna(0)        >= rvol_req
    m &= df_vfq_all.get("BreakoutScore", pd.Series(0, index=df_vfq_all.index)).fillna(0) >= brk_req

    df_keep_vfq = df_vfq_all.loc[m].copy()

    # Orden estable ya viene del snapshot; reforzamos por las llaves de prioridad
    df_keep_vfq = df_keep_vfq.sort_values(
        ["_p", "_b", "_q", "_v", "symbol"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    )

    vfq_top = df_keep_vfq.head(int(topN_prob)).copy()

    # --- Render tablas ---
    st.markdown("### 🟢 Selección VFQ filtrada")
    cols_vfq_show = [
        "symbol", "netdebt_ebitda", "accruals_ta", "sector", "market_cap",
        "quality_adj_neut", "value_adj_neut", "acc_pct",
        "hits", "BreakoutScore", "RVOL20", "prob_up",
    ]
    cols_vfq_show = [c for c in cols_vfq_show if c in vfq_top.columns]
    st.dataframe(vfq_top[cols_vfq_show], use_container_width=True, hide_index=True)

    # Rechazados
    st.markdown("### 🧹 Rechazados por VFQ / técnica")
    rejected_syms = sorted(set(df_vfq_all["symbol"]) - set(df_keep_vfq["symbol"]))
    rej_view = df_vfq_all[df_vfq_all["symbol"].isin(rejected_syms)].copy()
    cols_rej_show = [
        "symbol", "sector", "market_cap", "quality_adj_neut", "value_adj_neut",
        "netdebt_ebitda", "acc_pct", "BreakoutScore", "hits", "RVOL20", "prob_up",
    ]
    cols_rej_show = [c for c in cols_rej_show if c in rej_view.columns]
    st.dataframe(rej_view[cols_rej_show], use_container_width=True, hide_index=True)

    # Guardar en session_state
    st.session_state["vfq_top"]      = vfq_top[["symbol"]].drop_duplicates()
    st.session_state["vfq_table"]    = vfq_top.reset_index(drop=True)
    st.session_state["vfq_all"]      = df_vfq_all.copy()   # snapshot entero (¡fijo!)
    st.session_state["vfq_keep"]     = df_keep_vfq.copy()
    st.session_state["vfq_rejected"] = rej_view.copy()
    st.session_state["vfq_params"]   = {
        "min_hits": int(min_hits_req),
        "min_rvol20": float(min_rvol20),
        "min_breakout": float(min_breakout),
        "min_acc_pct": float(min_acc_pct),
        "max_ndebt": float(max_ndebt),
        "relax_mega": bool(relax_mega),
    }

    # ------------------------------------------------------------
    # D) 🔎 Inspector “¿por qué no sale X?” (al final del tab)
    # ------------------------------------------------------------
    with st.expander("🔎 ¿Por qué no aparece un símbolo?"):
        q = st.text_input("Ticker", "GOOG").strip().upper()
        if q:
            row = df_vfq_all[df_vfq_all["symbol"] == q].head(1)
            if row.empty:
                st.info("No está en el universo kept del snapshot.")
            else:
                r = row.iloc[0]
                # técnica size-aware consistente con el filtro
                _is_mega = bool(r.get("cap_pct", 0) >= 0.90)
                _hits_req = max(1, min_hits_req-1) if (relax_mega and _is_mega) else min_hits_req
                _rvol_req = max(1.1, min_rvol20-0.3) if (relax_mega and _is_mega) else min_rvol20
                _brk_req  = max(60,  min_breakout-10) if (relax_mega and _is_mega) else min_breakout

                checks = {
                    "quality_adj_neut": r.get("quality_adj_neut", 0) >= float(min_quality),
                    "value_adj_neut":   r.get("value_adj_neut", 0)   >= float(min_value),
                    "acc_pct":          (pd.isna(r.get("acc_pct")) or r.get("acc_pct") >= float(min_acc_pct)),
                    "netdebt_ebitda":   (pd.isna(r.get("netdebt_ebitda")) or r.get("netdebt_ebitda") <= float(max_ndebt)),
                    "hits":             r.get("hits", 0)            >= _hits_req,
                    "RVOL20":           r.get("RVOL20", 0)          >= _rvol_req,
                    "BreakoutScore":    r.get("BreakoutScore", 0)   >= _brk_req,
                }
                st.write({k: ("✅" if v else "❌") for k, v in checks.items()})
                st.dataframe(row.T, use_container_width=True)


# ====== TAB 4: SEÑALES (placeholder por ahora) ======
# ====== TAB 4: SEÑALES ======
with tab4:
    st.subheader("Señales técnicas / Breakout")

    # -------------------------
    # Recuperos del Tab 3
    # -------------------------
    uni_df     = st.session_state.get("uni", pd.DataFrame())
    vfq_all    = st.session_state.get("vfq_all", pd.DataFrame())
    vfq_keep   = st.session_state.get("vfq_keep", pd.DataFrame())
    vfq_rej    = st.session_state.get("vfq_rejected", pd.DataFrame())
    vfq_params = st.session_state.get("vfq_params", {})

    if vfq_all is None or vfq_all.empty:
        st.info("Todavía no hay datos técnicos / VFQ. Corre la pestaña VFQ primero.")
        st.stop()

    # Umbrales desde VFQ con fallbacks seguros
    min_hits_thr       = int(vfq_params.get("min_hits", 1))
    min_breakout_thr   = float(vfq_params.get("min_breakout", 50.0))
    min_rvol20_thr     = float(vfq_params.get("min_rvol20", 1.2))
    require_breakout_flag = bool(vfq_params.get("require_breakout", False))

    # Risk-ON: intenta leer de ámbito local; si no, de session_state; si no, False
    try:
        global_risk_on = bool(risk_on)  # del sidebar (scope superior)
    except NameError:
        global_risk_on = bool(st.session_state.get("_risk_on", False))

    # -------------------------
    # 1) Estado de mercado
    # -------------------------
    st.markdown("### 1. Estado de mercado")

    hits_ser = pd.to_numeric(vfq_all.get("hits", pd.Series(dtype=float)), errors="coerce").fillna(0)
    bks_ser  = pd.to_numeric(vfq_all.get("BreakoutScore", pd.Series(dtype=float)), errors="coerce").fillna(0)

    pct_hits_ok = hits_ser.ge(min_hits_thr).mean() if len(hits_ser) else np.nan
    pct_breakout_ok = bks_ser.ge(min_breakout_thr).mean() if len(bks_ser) else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("% setups técnicos OK", f"{pct_hits_ok*100:.1f}%" if not np.isnan(pct_hits_ok) else "n/d",
              help="≈ % del universo con suficientes 'hits' (checks técnicos cumplidos).")
    c2.metric("% ruptura/momentum OK", f"{pct_breakout_ok*100:.1f}%" if not np.isnan(pct_breakout_ok) else "n/d",
              help="≈ % del universo con BreakoutScore ≥ umbral.")
    c3.metric("Régimen mercado", "RISK ON ✅" if global_risk_on else "RISK OFF ⚠️",
              help="Switch macro/táctico desde la barra lateral.")

    st.caption(
        "- % setups técnicos OK ≈ amplitud por 'hits'.  \n"
        "- % ruptura/momentum OK ≈ cuántas rompen con fuerza (BreakoutScore).  \n"
        "- Con RISK ON + amplitud alta → mejor clima para entradas nuevas."
    )
    st.markdown("---")

    # -------------------------
    # 2) Checklist técnico activo
    # -------------------------
    st.markdown("### 2. Checklist técnico activo")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Hits mínimos", f"{min_hits_thr}")
    col_b.metric("BreakoutScore mín.", f"{min_breakout_thr:.1f}")
    col_c.metric("RVOL20 mín.", f"{min_rvol20_thr:.2f}")
    col_d.metric("¿Requiere breakout?", "Sí" if require_breakout_flag else "No")

    st.caption("Parámetros heredados de VFQ/técnico (tab 3).")
    st.markdown("---")

    # -------------------------
    # 3) Watchlist técnica
    # -------------------------
    st.markdown("### 3. Watchlist técnica (post-VFQ + técnico)")
    if vfq_keep is None or vfq_keep.empty:
        st.warning("Ningún ticker pasó VFQ + técnico con los filtros actuales.")
    else:
        cols_keep_show = [
            "symbol","sector","market_cap","quality_adj_neut","value_adj_neut",
            "acc_pct","hits","BreakoutScore","RVOL20","prob_up",
        ]
        cols_keep_show = [c for c in cols_keep_show if c in vfq_keep.columns]
        st.dataframe(
            vfq_keep[cols_keep_show].sort_values(
                ["prob_up","BreakoutScore"], ascending=[False, False], na_position="last"
            ),
            hide_index=True, use_container_width=True
        )

    st.caption(
        "Candidatos que pasaron Guardrails + VFQ y cumplen señales de tendencia/momentum/volumen."
    )
    st.markdown("---")

    # -------------------------
    # 4) Rechazados técnicos
    # -------------------------
    st.markdown("### 4. Rechazados técnicos")
    if vfq_rej is None or vfq_rej.empty:
        st.info("No hay rechazados técnicos adicionales (o no se guardaron).")
    else:
        cols_rej_show = [
            "symbol","sector","market_cap","quality_adj_neut","value_adj_neut",
            "netdebt_ebitda","acc_pct","hits","BreakoutScore","RVOL20","prob_up",
        ]
        cols_rej_show = [c for c in cols_rej_show if c in vfq_rej.columns]
        st.dataframe(
            vfq_rej[cols_rej_show].sort_values(
                ["BreakoutScore","prob_up"], ascending=[False, False], na_position="last"
            ),
            hide_index=True, use_container_width=True
        )

    # -------------------------
    # 5) Nota final
    # -------------------------
    st.info(
        "Guía rápida:\n"
        "- Amplitud alta + RISK ON ✅ → entorno favorable.\n"
        "- Amplitud baja o RISK OFF ⚠️ → reduce agresividad/tamaño."
    )


# ====== TAB 6: EXPORT (placeholder) ======
with tab6:
    st.subheader("Export")
    st.caption("Descarga tus vistas actuales.")
    cex1, cex2, cex3 = st.columns(3)

    # Versiones actuales en memoria
    v_uni  = st.session_state.get("uni", pd.DataFrame())
    v_all  = st.session_state.get("vfq_all", pd.DataFrame())
    v_keep = st.session_state.get("vfq_keep", pd.DataFrame())
    v_rej  = st.session_state.get("vfq_rejected", pd.DataFrame())

    def _csv_bytes(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode("utf-8") if isinstance(df, pd.DataFrame) and not df.empty else b""

    with cex1:
        st.download_button("⬇️ Universo (uni)", data=_csv_bytes(v_uni), file_name="universe.csv", mime="text/csv", use_container_width=True)
    with cex2:
        st.download_button("⬇️ Snapshot VFQ (all)", data=_csv_bytes(v_all), file_name="vfq_all.csv", mime="text/csv", use_container_width=True)
    with cex3:
        st.download_button("⬇️ Selección VFQ (keep)", data=_csv_bytes(v_keep), file_name="vfq_keep.csv", mime="text/csv", use_container_width=True)

    st.download_button("⬇️ Rechazados VFQ/técnico", data=_csv_bytes(v_rej), file_name="vfq_rejected.csv", mime="text/csv", use_container_width=True)


# ====== TAB 7: BACKTESTING ======
with tab7:
    st.subheader("Backtesting")

    vfq_keep = st.session_state.get("vfq_keep", pd.DataFrame())
    if vfq_keep is None or vfq_keep.empty or "symbol" not in vfq_keep.columns:
        st.warning("No hay símbolos aprobados en VFQ + técnico. Corre las pestañas anteriores primero.")
        st.stop()

    all_syms_bt = vfq_keep["symbol"].dropna().astype(str).unique().tolist()
    if not all_syms_bt:
        st.warning("No hay símbolos válidos para backtest.")
        st.stop()

    # ---------- Controles ----------
    st.markdown("#### Parámetros de simulación")
    c_bt1, c_bt2, c_bt3 = st.columns(3)
    with c_bt1:
        top_n_bt = st.slider("N° máx. de símbolos a probar", min_value=5, max_value=min(50, len(all_syms_bt)),
                             value=min(20, len(all_syms_bt)), step=1)
    with c_bt2:
        cost_bps = st.number_input("Costo de trading (bps por cambio de postura)", min_value=0, max_value=100,
                                   value=10, step=1)
        lag_days = st.number_input("Lag ejecución (días)", min_value=0, max_value=5, value=0, step=1)
    with c_bt3:
        use_and_condition = st.toggle("Exigir MA200 Y Mom12-1 > 0", value=False)
        rebalance_freq = st.selectbox("Frecuencia rebalance", options=["M", "W"], index=0)

    syms_bt = all_syms_bt[: int(top_n_bt)]
    st.caption(
        f"Testeando {len(syms_bt)} símbolos: {', '.join(syms_bt[:10])}" + ("…" if len(syms_bt) > 10 else "")
    )

    # ---------- Fechas ----------
    try:
        start_dt = pd.to_datetime(start).date()
        end_dt   = pd.to_datetime(end).date()
    except NameError:
        start_dt = pd.to_datetime(DEFAULT_START).date()
        end_dt   = pd.to_datetime(DEFAULT_END).date()

    # ---------- Precios ----------
    price_panel = _cached_load_prices_panel(
        symbols=syms_bt,
        start=start_dt,
        end=end_dt,
        cache_key=f"bt_{start_dt}_{end_dt}_{len(syms_bt)}"
    )
    if not isinstance(price_panel, dict) or not price_panel:
        st.error("No pude cargar precios históricos para backtest.")
        st.stop()

    # ---------- Backtest ----------
    bt_metrics, bt_curves = backtest_many(  # usa el módulo ya importado arriba
        panel=price_panel,
        symbols=syms_bt,
        cost_bps=int(cost_bps),
        lag_days=int(lag_days),
        use_and_condition=bool(use_and_condition),
        rebalance_freq=str(rebalance_freq)
    )

    # ---------- Métricas ----------
    st.markdown("#### Resultados por símbolo")
    if bt_metrics is None or bt_metrics.empty:
        st.warning("No hubo data suficiente para calcular métricas.")
    else:
        show_cols = [c for c in ["symbol","CAGR","Sharpe","Sortino","MaxDD","Turnover","Trades"] if c in bt_metrics.columns]
        fmt_df = bt_metrics.copy()
        if "CAGR" in fmt_df:  fmt_df["CAGR"]   = (fmt_df["CAGR"] * 100).round(2)
        if "MaxDD" in fmt_df: fmt_df["MaxDD"]  = (fmt_df["MaxDD"] * 100).round(2)
        if "Turnover" in fmt_df: fmt_df["Turnover"] = (fmt_df["Turnover"] * 100).round(2)

        st.dataframe(fmt_df[show_cols], hide_index=True, use_container_width=True)
        st.caption(
            "- CAGR anualizada; MaxDD y Turnover en %.  \n"
            "- Señal simple long-only binaria por ticker."
        )

    st.markdown("---")

    # ---------- Curvas (Altair) ----------
    st.markdown("#### Curvas de equity normalizadas (1.0 = inicio)")
    if not bt_curves:
        st.info("No hay curvas de equity para graficar.")
    else:
        eq_df_list = []
        for sym, curve in bt_curves.items():
            if curve is None or curve.empty:
                continue
            tmp = curve.rename("equity").to_frame().reset_index().rename(columns={"index": "date"})
            tmp["symbol"] = sym
            eq_df_list.append(tmp)

        if not eq_df_list:
            st.info("No se pudo armar data suficiente para el gráfico.")
        else:
            long_eq = pd.concat(eq_df_list, ignore_index=True).sort_values(["symbol","date"])

            def _norm_grp(g):
                first_val = g["equity"].iloc[0] if len(g) else np.nan
                g["equity_norm"] = g["equity"] / first_val if (pd.notna(first_val) and first_val != 0) else np.nan
                return g

            long_eq = long_eq.groupby("symbol", group_keys=False).apply(_norm_grp)

            chart = (
                alt.Chart(long_eq)
                .mark_line()
                .encode(
                    x=alt.X("date:T", title="Fecha"),
                    y=alt.Y("equity_norm:Q", title="Equidad normalizada"),
                    color=alt.Color("symbol:N", title="Símbolo"),
                    tooltip=[alt.Tooltip("date:T", title="Fecha"),
                             alt.Tooltip("symbol:N", title="Ticker"),
                             alt.Tooltip("equity_norm:Q", title="Equidad norm.", format=".2f")]
                )
                .properties(height=320)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption("Cada línea: dentro cuando señal ON, cash cuando OFF; reinvertido.")

# ====== TAB 8: TUNING (Random Search) ======
with tab8:
    import numpy as np
    import pandas as pd

    st.subheader("🔧 Tuning de umbrales (random search)")

    kept       = st.session_state.get("kept", pd.DataFrame())
    df_vfq_all = st.session_state.get("vfq_all", pd.DataFrame())

    if kept is None or kept.empty or df_vfq_all is None or df_vfq_all.empty:
        st.warning("Necesitas correr Guardrails y VFQ antes de tunear.")
        st.stop()

    # Asegura 'acc_pct' si falta (a partir de 'accruals_ta')
    if "acc_pct" not in df_vfq_all.columns and "accruals_ta" in df_vfq_all.columns:
        s = pd.to_numeric(df_vfq_all["accruals_ta"], errors="coerce").astype(float)
        pct = (s.abs().rank(pct=True, method="average"))
        df_vfq_all["acc_pct"] = (1.0 - pct) * 100.0

    # --------- Parámetros de búsqueda ----------
    c1, c2, c3 = st.columns(3)
    with c1:
        n_samples = st.number_input("N° combinaciones aleatorias", 20, 2000, 150, 10)
        cost_bps  = st.number_input("Costos (bps por rebalance)", 0, 100, 10, 1)
        use_and   = st.toggle("Tendencia: MA200 Y Mom12-1>0", value=False)
    with c2:
        try:
            start_tune = st.date_input("Inicio tuning", value=pd.to_datetime(start).date())
            end_tune   = st.date_input("Fin tuning", value=pd.to_datetime(end).date())
        except NameError:
            start_tune = st.date_input("Inicio tuning", value=pd.to_datetime(DEFAULT_START).date())
            end_tune   = st.date_input("Fin tuning", value=pd.to_datetime(DEFAULT_END).date())
        min_names  = st.number_input("Mín. símbolos por cartera", 5, 200, 15, 1)
    with c3:
        seed = st.number_input("Semilla aleatoria", 0, 10_000, 1234, 1)
        reb_freq = st.selectbox("Frecuencia rebalanceo", ["M","W","Q"], index=0)
        go_btn = st.button("Ejecutar Tuning", use_container_width=True, type="primary")

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
        m = pd.Series(True, index=df.index, dtype=bool)
        m &= df.get("quality_adj_neut", pd.Series(0, index=df.index)).fillna(0) >= p["min_quality"]
        m &= df.get("value_adj_neut",   pd.Series(0, index=df.index)).fillna(0) >= p["min_value"]
        m &= df.get("hits", pd.Series(0, index=df.index)).fillna(0)             >= p["min_hits_req"]
        m &= df.get("BreakoutScore", pd.Series(0, index=df.index)).fillna(0)    >= p["min_breakout"]
        m &= df.get("RVOL20", pd.Series(0, index=df.index)).fillna(0)           >= p["min_rvol20"]
        m &= (df.get("acc_pct", pd.Series(np.nan, index=df.index)).isna()
              | (df.get("acc_pct", pd.Series(0, index=df.index)).fillna(0) >= p["min_acc_pct"]))
        m &= (df.get("netdebt_ebitda", pd.Series(np.nan, index=df.index)).isna()
              | (df.get("netdebt_ebitda", pd.Series(0, index=df.index)).fillna(0) <= p["max_ndebt"]))

        df_f = df.loc[m].copy()
        if df_f.empty:
            return []
        rank_col = "prob_up" if ("prob_up" in df_f.columns and df_f["prob_up"].notna().any()) else "BreakoutScore"
        df_f = df_f.sort_values(rank_col, ascending=False)
        return df_f["symbol"].dropna().astype(str).unique().tolist()[: int(p["topN_prob"])]

    def _portfolio_metrics_from_curves(curves: dict[str, pd.Series], freq_code: str) -> dict:
        if not curves:
            return {"CAGR":0,"Sharpe":0,"Sortino":0,"MaxDD":0,"N":0,"Turnover":0}
        eq = pd.DataFrame(curves).dropna(how="all")
        if eq.empty:
            return {"CAGR":0,"Sharpe":0,"Sortino":0,"MaxDD":0,"N":0,"Turnover":0}
        rets = eq.pct_change().mean(axis=1).fillna(0.0)
        periods = {"M":12,"W":52,"Q":4}[freq_code]

        # métricas rápidas:
        mu = rets.mean() * periods
        sd = rets.std(ddof=0) * np.sqrt(periods)
        sharpe = float(mu / sd) if sd else 0.0

        dn = rets[rets < 0]
        sdd = dn.std(ddof=0) * np.sqrt(periods)
        sortino = float(mu / sdd) if sdd else 0.0

        eq_curve = (1 + rets).cumprod()
        years = len(rets) / float(periods) if periods else 0
        cagr = float(eq_curve.iloc[-1] ** (1.0/years) - 1.0) if (years > 0 and eq_curve.iloc[-1] > 0) else 0.0
        dd = (eq_curve / eq_curve.cummax() - 1.0).min()
        maxdd = float(dd) if pd.notna(dd) else 0.0

        return {"CAGR":cagr,"Sharpe":sharpe,"Sortino":sortino,"MaxDD":maxdd,"N":int(eq.shape[1])}

    results, details = [], []

    if go_btn:
        try:
            rng = np.random.RandomState(int(seed))
            pbar = st.progress(0.0, text="Buscando combinaciones…")

            for i in range(int(n_samples)):
                p = _sample_params(rng)
                picks = _rank_and_pick(df_vfq_all, p)
                pbar.progress((i+1)/float(n_samples), text=f"Eval {i+1}/{n_samples}")

                if len(picks) < int(min_names):
                    continue

                panel = _cached_load_prices_panel(
                    symbols=picks,
                    start=pd.to_datetime(start_tune).date(),
                    end=pd.to_datetime(end_tune).date(),
                    cache_key=f"tune_{len(picks)}_{start_tune}_{end_tune}"
                )
                if not isinstance(panel, dict) or not panel:
                    continue

                # Backtest (usa tu función estándar)
                metrics_df, curves = backtest_many(
                    panel=panel,
                    symbols=list(panel.keys()),
                    cost_bps=int(cost_bps),
                    lag_days=0,
                    use_and_condition=bool(use_and),
                    rebalance_freq=str(reb_freq),
                )
                avg_turn = float(metrics_df["Turnover"].mean()) if isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty else 0.0

                port_perf = _portfolio_metrics_from_curves(curves, str(reb_freq))
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
                "rebalance": str(reb_freq),
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
