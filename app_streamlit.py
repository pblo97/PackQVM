"""
QVM Screener - VERSIÓN FINAL INTEGRADA
======================================

Compatible con:
- pipeline_factors.py (refactorizado)
- fundamentals_guardrails.py (nuevo módulo)
- rate_limiter.py (nuevo)
- data_io.py (con rate limiter)

CAMBIOS CLAVE:
1. build_factor_frame() ya devuelve TODO (no más pipelines separados)
2. apply_quality_guardrails() viene del nuevo módulo
3. Single source of truth en PipelineState
"""

from __future__ import annotations
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "poll"

import hashlib
import json
import time
from datetime import datetime
from typing import Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ============================================================================
# IMPORTS DE MÓDULOS REFACTORIZADOS
# ============================================================================

# Pipeline de factores (TODO en uno)
from pipeline_factors import (
    build_factor_frame,      # Función única que trae TODO
    merge_with_universe,     # Helper para reinyectar sector/mcap
)

# Guardrails (módulo separado)
from fundamentals import (
    apply_quality_guardrails,
    GuardrailConfig,
)

# Data I/O (con rate limiter integrado)
from data_io import (
    run_fmp_screener,
    load_prices_panel,
    load_benchmark,
    DEFAULT_START,
    DEFAULT_END,
)

# Backtesting
from backtests import backtest_many

# ============================================================================
# CONFIGURACIÓN Y ESTADO
# ============================================================================

@dataclass(frozen=True)
class PipelineConfig:
    """Configuración que define el universo y guardrails"""
    # Universo
    limit: int
    mcap_min: float
    volume_min: int
    ipo_days: int
    
    # Guardrails
    profit_min_hits: int
    max_issuance: float
    max_asset_growth: float
    max_accruals: float
    max_netdebt_ebitda: float
    min_coverage: int
    
    def signature(self) -> str:
        """Firma única para caché"""
        data = {
            "limit": self.limit,
            "mcap_min": self.mcap_min,
            "volume_min": self.volume_min,
            "ipo_days": self.ipo_days,
        }
        return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    def to_guardrail_config(self) -> GuardrailConfig:
        """Convierte a config de guardrails"""
        return GuardrailConfig(
            profit_min_hits=self.profit_min_hits,
            max_issuance=self.max_issuance,
            max_asset_growth=self.max_asset_growth,
            max_accruals=self.max_accruals,
            max_netdebt_ebitda=self.max_netdebt_ebitda,
            min_coverage=self.min_coverage,
        )


@dataclass
class PipelineState:
    """Estado maestro - Single Source of Truth"""
    df_universe: pd.DataFrame       # Universo crudo desde screener
    df_enriched: pd.DataFrame       # + todos los factores (build_factor_frame)
    df_with_guards: pd.DataFrame    # + flags de guardrails
    df_passed_guards: pd.DataFrame  # Solo los que pasaron
    
    config: PipelineConfig
    timestamp: float
    n_universe: int
    n_passed_guards: int


# ============================================================================
# UTILIDADES
# ============================================================================

def _fmt_mcap(x):
    try:
        x = float(x)
        if x >= 1e12: return f"${x/1e12:.2f}T"
        if x >= 1e9:  return f"${x/1e9:.2f}B"
        if x >= 1e6:  return f"${x/1e6:.2f}M"
        return f"${x:,.0f}"
    except:
        return ""

def _csv_bytes(df: pd.DataFrame) -> bytes:
    if df is None or df.empty:
        return b""
    return df.to_csv(index=False).encode("utf-8")

# ============================================================================
# PIPELINE CORE
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_universe(config: PipelineConfig) -> pd.DataFrame:
    """
    Etapa 1: Screener base desde FMP
    """
    df = run_fmp_screener(
        limit=config.limit,
        mcap_min=config.mcap_min,
        volume_min=config.volume_min,
        fetch_profiles=True,
        cache_key=config.signature(),
        force=False,
    )
    
    # Normalización garantizada
    if "market_cap" not in df.columns:
        df["market_cap"] = pd.to_numeric(df.get("marketCap", np.nan), errors="coerce")
    
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    df["sector"] = df["sector"].fillna("Unknown").astype(str)
    
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    
    if "ipoDate" in df.columns:
        df["ipoDate"] = pd.to_datetime(df["ipoDate"], errors="coerce", utc=True)
    
    # Filtros post-screener
    df = df[df["market_cap"] >= config.mcap_min]
    if "volume" in df.columns:
        df = df[df["volume"] >= config.volume_min]
    
    if df["ipoDate"].notna().any():
        cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=config.ipo_days)
        df = df[df["ipoDate"] < cutoff]
    
    # Columnas mínimas
    keep = ["symbol", "sector", "market_cap", "volume", "ipoDate"]
    keep = [c for c in keep if c in df.columns]
    
    return df[keep].copy().reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def enrich_universe(df_universe: pd.DataFrame, config_sig: str) -> pd.DataFrame:
    """
    Etapa 2: Enriquece con TODOS los factores
    
    IMPORTANTE: build_factor_frame YA trae:
    - Fundamentales (EV/EBITDA, FCF, etc)
    - Guardrails raw (netdebt_ebitda, accruals_ta, etc)
    - profit_hits, coverage_count
    - VFQ (quality_adj_neut, value_adj_neut, acc_pct)
    - Técnico (BreakoutScore, RVOL20, hits, prob_up)
    """
    _ = config_sig  # Para invalidar caché
    
    syms = df_universe["symbol"].dropna().astype(str).unique().tolist()
    if not syms:
        return pd.DataFrame()
    
    # build_factor_frame trae TODO
    df_factors = build_factor_frame(
        tickers=syms,
        use_cache=True,
        cache_key=config_sig[:8],  # Primeros 8 chars de signature
    )
    
    # Reinyectar sector/market_cap del universo actual
    df_enriched = merge_with_universe(df_factors, df_universe)
    
    return df_enriched


def add_guardrail_flags(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Etapa 3: Añade flags de guardrails (usa nuevo módulo)
    """
    guard_config = config.to_guardrail_config()
    
    # apply_quality_guardrails devuelve (kept, diagnostics)
    # Pero nosotros queremos mantener TODAS las filas con flags
    _, diagnostics = apply_quality_guardrails(df, guard_config)
    
    # Merge flags back al DataFrame original
    result = df.merge(
        diagnostics[["symbol", "pass_all", "reason"]],
        on="symbol",
        how="left",
        suffixes=("", "_diag")
    )
    
    # Rellenar pass_all faltantes con False
    result["pass_all"] = result["pass_all"].fillna(False)
    result["reason"] = result["reason"].fillna("no_data")
    
    return result


def filter_by_vfq(
    df: pd.DataFrame,
    min_quality: float,
    min_value: float,
    min_acc_pct: float,
    max_ndebt: float,
    min_hits: int,
    min_rvol20: float,
    min_breakout: float,
    relax_mega: bool = False,
) -> pd.DataFrame:
    """
    Etapa 4: Filtra por VFQ + técnico (solo vista, NO recalcula)
    """
    # Solo trabajamos con los que pasaron guardrails
    df = df[df["pass_all"] == True].copy()
    
    if df.empty:
        return df
    
    # Carril mega-cap
    if "market_cap" in df.columns and relax_mega:
        cap_pct = df["market_cap"].rank(pct=True)
        is_mega = (cap_pct >= 0.90)
    else:
        is_mega = pd.Series(False, index=df.index)
    
    # Ajustes size-aware
    hits_thr = np.where(is_mega, max(1, min_hits - 1), min_hits)
    rvol_thr = np.where(is_mega, max(1.1, min_rvol20 - 0.3), min_rvol20)
    brk_thr = np.where(is_mega, max(60, min_breakout - 10), min_breakout)
    
    # Máscaras
    m = pd.Series(True, index=df.index)
    m &= df.get("quality_adj_neut", 0).fillna(0) >= min_quality
    m &= df.get("value_adj_neut", 0).fillna(0) >= min_value
    m &= (df.get("acc_pct", np.nan).isna() | (df.get("acc_pct", 0).fillna(0) >= min_acc_pct))
    m &= (df.get("netdebt_ebitda", np.nan).isna() | (df.get("netdebt_ebitda", 0).fillna(0) <= max_ndebt))
    
    m &= df.get("hits", 0).fillna(0) >= hits_thr
    m &= df.get("RVOL20", 0).fillna(0) >= rvol_thr
    m &= df.get("BreakoutScore", 0).fillna(0) >= brk_thr
    
    return df[m].copy()


# ============================================================================
# GESTIÓN DE ESTADO
# ============================================================================

def build_pipeline_state(config: PipelineConfig) -> PipelineState:
    """Constructor del estado maestro (ejecuta todo el pipeline)"""
    
    # Etapa 1: Universo
    df_uni = fetch_universe(config)
    
    # Etapa 2: Enriquecimiento (TODO de una vez)
    df_enr = enrich_universe(df_uni, config.signature())
    
    # Etapa 3: Guardrails
    df_guards = add_guardrail_flags(df_enr, config)
    
    # Etapa 4: Solo los que pasaron
    df_passed = df_guards[df_guards["pass_all"] == True].copy()
    
    return PipelineState(
        df_universe=df_uni,
        df_enriched=df_enr,
        df_with_guards=df_guards,
        df_passed_guards=df_passed,
        config=config,
        timestamp=time.time(),
        n_universe=len(df_uni),
        n_passed_guards=int(df_guards["pass_all"].sum()),
    )


def get_or_build_state(config: PipelineConfig) -> PipelineState:
    """Obtiene estado del session_state o lo construye"""
    key = "pipeline_state"
    
    if key not in st.session_state or st.session_state[key].config.signature() != config.signature():
        with st.spinner("🔄 Construyendo pipeline..."):
            state = build_pipeline_state(config)
            st.session_state[key] = state
    
    return st.session_state[key]


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="QVM Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 1.25rem; }
h1, h2, h3 { letter-spacing: .2px; }
hr { border: 0; border-top: 1px solid rgba(255,255,255,.08); margin: .6rem 0 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

l, r = st.columns([0.85, 0.15])
with l:
    st.markdown("<h1 style='margin-bottom:0'>QVM Screener</h1>", unsafe_allow_html=True)
    st.caption("Pipeline unificado: Screener → Factores → Guardrails → VFQ → Señales → Backtest")
with r:
    st.caption(datetime.now().strftime("%d %b %Y %H:%M"))
st.markdown("<hr/>", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Config Base (recarga pipeline)")
    
    with st.expander("🌍 Universo", expanded=True):
        limit = st.slider("Límite", 50, 1000, 300, 50)
        min_mcap = st.number_input("MarketCap mín.", value=5e8, step=1e8, format="%.0f")
        volume_min = st.number_input("Volumen mín.", value=500_000, step=50_000)
        ipo_days = st.slider("Antigüedad IPO (días)", 90, 1500, 365, 30)
    
    with st.expander("🛡️ Guardrails", expanded=True):
        profit_hits = st.slider("Profit hits mín.", 0, 3, 2)
        min_cov = st.slider("Coverage mín.", 1, 4, 2)
        max_issuance = st.slider("Max issuance", 0.00, 0.10, 0.03, 0.01)
        max_assets = st.slider("Max asset growth", 0.00, 0.50, 0.20, 0.01)
        max_accr = st.slider("Max accruals", 0.00, 0.25, 0.10, 0.01)
        max_ndeb = st.slider("Max netdebt/ebitda", 0.0, 6.0, 3.0, 0.5)
    
    st.markdown("---")
    st.markdown("### 🎯 Filtros VFQ (solo vista)")
    
    with st.expander("💎 Value & Quality", expanded=True):
        min_quality = st.slider("Min Quality", 0.0, 1.0, 0.30, 0.01)
        min_value = st.slider("Min Value", 0.0, 1.0, 0.30, 0.01)
        min_acc_pct = st.slider("Min Accruals %", 0, 100, 30, 1)
        max_ndebt_vfq = st.slider("Max NetDebt/EBITDA", 0.0, 5.0, 3.0, 0.1)
    
    with st.expander("📊 Técnico", expanded=True):
        min_hits = st.slider("Min hits", 0, 5, 2, 1)
        min_rvol20 = st.slider("Min RVOL20", 0.0, 5.0, 1.50, 0.05)
        min_breakout = st.slider("Min BreakoutScore", 0, 100, 80, 1)
        relax_mega = st.toggle("Aflojar para mega-caps", value=True)
    
    topN = st.slider("Top N", 5, 100, 30, 5)

# ============================================================================
# CONSTRUIR ESTADO
# ============================================================================

config = PipelineConfig(
    limit=limit,
    mcap_min=min_mcap,
    volume_min=volume_min,
    ipo_days=ipo_days,
    profit_min_hits=profit_hits,
    max_issuance=max_issuance,
    max_asset_growth=max_assets,
    max_accruals=max_accr,
    max_netdebt_ebitda=max_ndeb,
    min_coverage=min_cov,
)

state = get_or_build_state(config)

# Filtrar por VFQ (solo vista)
df_vfq_filtered = filter_by_vfq(
    state.df_passed_guards,
    min_quality=min_quality,
    min_value=min_value,
    min_acc_pct=min_acc_pct,
    max_ndebt=max_ndebt_vfq,
    min_hits=min_hits,
    min_rvol20=min_rvol20,
    min_breakout=min_breakout,
    relax_mega=relax_mega,
)

# Ordenar
if not df_vfq_filtered.empty:
    df_vfq_filtered = df_vfq_filtered.sort_values(
        ["prob_up", "BreakoutScore", "quality_adj_neut"],
        ascending=[False, False, False],
        na_position="last"
    )

df_top = df_vfq_filtered.head(topN)

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Universo",
    "🛡️ Guardrails",
    "💎 VFQ",
    "🎯 Señales",
    "💾 Export",
    "📈 Backtest"
])

# ============================================================================
# TAB 1: UNIVERSO
# ============================================================================

with tab1:
    st.subheader("Universo base")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Screener", f"{state.n_universe}")
    c2.metric("Pasaron guardrails", f"{state.n_passed_guards}")
    c3.metric("Después VFQ", f"{len(df_vfq_filtered)}")
    
    st.dataframe(state.df_universe.head(100), use_container_width=True, hide_index=True)
    st.caption(f"Firma: `{config.signature()}`")

# ============================================================================
# TAB 2: GUARDRAILS
# ============================================================================

with tab2:
    st.subheader("Guardrails")
    
    c1, c2 = st.columns(2)
    c1.metric("✅ Pasaron", f"{state.n_passed_guards}")
    c2.metric("❌ Rechazados", f"{state.n_universe - state.n_passed_guards}")
    
    cols = ["symbol", "sector", "pass_all", "profit_hits", "coverage_count", "reason"]
    cols = [c for c in cols if c in state.df_with_guards.columns]
    
    st.dataframe(
        state.df_with_guards[cols].sort_values("pass_all", ascending=False),
        use_container_width=True,
        hide_index=True
    )

# ============================================================================
# TAB 3: VFQ
# ============================================================================

with tab3:
    st.subheader("VFQ Rankings")
    
    st.info(f"🔍 Filtros activos: Quality≥{min_quality}, Value≥{min_value}, Hits≥{min_hits}")
    
    if df_top.empty:
        st.warning("⚠️ Ningún símbolo pasa los filtros VFQ. Relaja los sliders.")
    else:
        cols = [
            "symbol", "sector", "market_cap",
            "quality_adj_neut", "value_adj_neut", "acc_pct",
            "hits", "BreakoutScore", "RVOL20", "prob_up"
        ]
        cols = [c for c in cols if c in df_top.columns]
        
        st.dataframe(df_top[cols], use_container_width=True, hide_index=True)
    
    with st.expander("🔍 Inspector"):
        q = st.text_input("Símbolo", "AAPL").strip().upper()
        if q:
            row = state.df_with_guards[state.df_with_guards["symbol"] == q]
            if row.empty:
                st.info("No está en el universo")
            else:
                st.dataframe(row.T, use_container_width=True)

# ============================================================================
# TAB 4: SEÑALES
# ============================================================================

with tab4:
    st.subheader("Señales técnicas")
    
    if df_top.empty:
        st.warning("No hay símbolos")
    else:
        avg_brk = df_top.get("BreakoutScore", pd.Series()).mean()
        avg_prob = df_top.get("prob_up", pd.Series()).mean()
        
        c1, c2 = st.columns(2)
        c1.metric("Avg BreakoutScore", f"{avg_brk:.1f}")
        c2.metric("Avg Prob ↑", f"{avg_prob:.2%}")
        
        st.dataframe(df_top[["symbol", "sector", "BreakoutScore", "prob_up", "hits"]])

# ============================================================================
# TAB 5: EXPORT
# ============================================================================

with tab5:
    st.subheader("Export")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.download_button("⬇️ Universo", data=_csv_bytes(state.df_universe), 
                          file_name="universe.csv", mime="text/csv")
    with c2:
        st.download_button("⬇️ Con guardrails", data=_csv_bytes(state.df_with_guards),
                          file_name="guardrails.csv", mime="text/csv")
    with c3:
        st.download_button("⬇️ Top VFQ", data=_csv_bytes(df_top),
                          file_name="vfq_top.csv", mime="text/csv")

# ============================================================================
# TAB 6: BACKTEST
# ============================================================================

with tab6:
    st.subheader("Backtest")
    
    if df_top.empty:
        st.warning("No hay símbolos")
        st.stop()
    
    syms_bt = df_top["symbol"].dropna().astype(str).unique().tolist()[:20]
    
    c1, c2 = st.columns(2)
    with c1:
        cost_bps = st.number_input("Costos (bps)", 0, 100, 10)
        use_and = st.toggle("MA200 Y Mom12-1", value=False)
    with c2:
        start_bt = st.date_input("Inicio", pd.to_datetime(DEFAULT_START).date())
        end_bt = st.date_input("Fin", pd.to_datetime(DEFAULT_END).date())
    
    if st.button("🚀 Ejecutar", type="primary"):
        with st.spinner("Descargando precios..."):
            panel = load_prices_panel(syms_bt, str(start_bt), str(end_bt))
        
        if not panel:
            st.error("No se cargaron precios")
            st.stop()
        
        with st.spinner("Ejecutando backtest..."):
            metrics, curves = backtest_many(
                panel=panel,
                symbols=syms_bt,
                cost_bps=cost_bps,
                use_and_condition=use_and,
                rebalance_freq="M"
            )
        
        if metrics is not None and not metrics.empty:
            st.dataframe(metrics, use_container_width=True, hide_index=True)
            
            if curves:
                eq_list = []
                for sym, eq in curves.items():
                    tmp = eq.rename("equity").to_frame().reset_index()
                    tmp.columns = ["date", "equity"]
                    tmp["symbol"] = sym
                    eq_list.append(tmp)
                
                if eq_list:
                    long_eq = pd.concat(eq_list, ignore_index=True)
                    chart = (
                        alt.Chart(long_eq)
                        .mark_line()
                        .encode(x="date:T", y="equity:Q", color="symbol:N")
                        .properties(height=400)
                        .interactive()
                    )
                    st.altair_chart(chart, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption(f"⏱️ Pipeline: {time.time() - state.timestamp:.2f}s | "
           f"Universo: {state.n_universe} | Guards: {state.n_passed_guards} | VFQ: {len(df_vfq_filtered)}")