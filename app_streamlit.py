# app_streamlit.py
"""
QVM Screener V2 - Con Momentum Real + MA200 + F-Score
======================================================

Implementación académica completa con:
✅ Momentum real desde precios (Jegadeesh & Titman, 1993)
✅ Filtro MA200 obligatorio (Faber, 2007)
✅ Piotroski F-Score (Piotroski, 2000) — Forma B (sin ROE histórico)
✅ Sector-neutral factors
✅ Reglas heurísticas robustas (filtros optimizados)

Módulos requeridos:
- data_fetcher.py: fetch_screener, fetch_fundamentals_batch, fetch_prices
- factor_calculator.py: compute_all_factors (V2 con neutralización)
- momentum_calculator.py: calculate_momentum_batch (12M-1M), etc.
- piotroski_fscore.py: calculate_simplified_fscore_no_roe, filter_by_fscore
- screener_filters.py: FilterConfig, apply_all_filters
- backtest_engine.py: backtest_portfolio, calculate_portfolio_metrics, TradingCosts
"""

from __future__ import annotations
import os
import shutil
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# ----------------------------- Imports proyecto -----------------------------
from data_fetcher import (
    fetch_screener,
    fetch_fundamentals_batch,
    fetch_prices,
)

from factor_calculator import compute_all_factors

from screener_filters import (
    apply_all_filters,
    filter_by_qvm,
    FilterConfig,
)

from backtest_engine import (
    backtest_portfolio,
    calculate_portfolio_metrics,
    TradingCosts,
)

# ⭐ Módulos nuevos / reforzados
from momentum_calculator import (
    calculate_momentum_batch,  # batch(dict<symbol->df_prices>) → df con momentum y MA200
)

# Alias Forma B (sin ROE histórico) para mantener compatibilidad
from piotroski_fscore import (
    calculate_simplified_fscore_no_roe as calculate_simplified_fscore,
    filter_by_fscore,  # disponible si quisieras exponer filtro directo
)

# =============================================================================
# CONFIG INICIAL DE PÁGINA
# =============================================================================

st.set_page_config(
    page_title="QVM Screener V2",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container { padding-top: 0.8rem; padding-bottom: 1.0rem; }
h1, h2, h3 { letter-spacing: .2px; }
hr { border: 0; border-top: 1px solid rgba(255,255,255,.08); margin: .5rem 0; }
.stAlert { padding: 0.5rem 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# CACHÉ / WRAPPERS
# =============================================================================

@st.cache_data(ttl=3600)
def cached_screener(limit: int, mcap_min: float, volume_min: int) -> pd.DataFrame:
    return fetch_screener(limit, mcap_min, volume_min, use_cache=True)

@st.cache_data(ttl=3600)
def cached_fundamentals(symbols: tuple[str, ...]) -> pd.DataFrame:
    return fetch_fundamentals_batch(list(symbols), use_cache=True)

@st.cache_data(ttl=1800)
def cached_prices(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    return fetch_prices(symbol, start, end, use_cache=True)

def clear_cache_disk_and_memory() -> bool:
    """
    Limpia caché en disco usada por data_fetcher (.cache/fmp) y Streamlit cache.
    """
    cache_dir = ".cache/fmp"
    try:
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        st.cache_data.clear()
        return True
    except Exception:
        return False

# =============================================================================
# HEADER
# =============================================================================

c1, c2 = st.columns([0.85, 0.15])
with c1:
    st.markdown("<h1 style='margin-bottom:0'>🚀 QVM Screener V2</h1>", unsafe_allow_html=True)
    st.caption("Quality × Value × Momentum × F-Score | Implementación Académica Completa")
with c2:
    st.caption(datetime.now().strftime("%d %b %Y • %H:%M"))

st.markdown("<hr/>", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuración")

    # Universo
    with st.expander("🌍 Universo", expanded=True):
        limit = st.slider("Tamaño universo", 50, 500, 300, 50)
        mcap_min = st.number_input("Market Cap mín. (USD)", value=2e9, step=1e8, format="%.0f")
        volume_min = st.number_input("Volumen mín. diario", value=1_000_000, step=100_000)

    # Filtros de calidad
    with st.expander("🛡️ Quality Filters", expanded=True):
        min_roe = st.slider("ROE mín.", 0.0, 0.50, 0.15, 0.05)
        min_gm = st.slider("Gross Margin mín.", 0.0, 0.80, 0.30, 0.05)
        req_fcf = st.toggle("Exigir FCF > 0", value=True)
        req_ocf = st.toggle("Exigir Operating CF > 0", value=True)

    # F-Score
    with st.expander("🏆 F-Score (Piotroski)", expanded=True):
        use_fscore = st.toggle("Usar F-Score filter", value=True)
        min_fscore = st.slider("F-Score mínimo", 0, 9, 6, 1) if use_fscore else 0
        if use_fscore:
            st.caption("6-7: Medium quality | 8-9: High quality")

    # Momentum + MA200
    with st.expander("🎯 Momentum + MA200", expanded=True):
        require_ma200 = st.toggle("⭐ Exigir Price > MA200", value=True,
                                  help="Filtro crítico: históricamente reduce drawdowns.")
        if require_ma200:
            st.caption("✅ Solo acciones en uptrend (price > MA200)")
        min_momentum = st.slider("Momentum 12M mín.", -0.50, 0.50, 0.05, 0.05, format="%.2f")

    # Pesos
    with st.expander("💎 Factor Weights", expanded=True):
        w_quality = st.slider("Peso Quality", 0.0, 1.0, 0.35, 0.05)
        w_value = st.slider("Peso Value", 0.0, 1.0, 0.25, 0.05)
        w_momentum = st.slider("Peso Momentum", 0.0, 1.0, 0.25, 0.05)
        w_fscore = st.slider("Peso F-Score", 0.0, 1.0, 0.15, 0.05)
        total_w = max(w_quality + w_value + w_momentum + w_fscore, 1e-9)
        w_quality, w_value, w_momentum, w_fscore = (
            w_quality / total_w, w_value / total_w, w_momentum / total_w, w_fscore / total_w
        )
        st.caption(f"Normalizado: Q={w_quality:.2f} V={w_value:.2f} M={w_momentum:.2f} F={w_fscore:.2f}")

    # Selección final
    with st.expander("📋 Portfolio Selection", expanded=True):
        top_n = st.slider("Top N símbolos", 5, 100, 30, 5)

    st.markdown("---")
    run_btn = st.button("🚀 Ejecutar Pipeline", type="primary", use_container_width=True)

    cA, cB = st.columns(2)
    with cA:
        if st.button("🗑️ Limpiar caché", use_container_width=True):
            if clear_cache_disk_and_memory():
                st.success("✅ Caché limpiada")
            else:
                st.warning("⚠️ No se pudo limpiar toda la caché")
    with cB:
        if st.button("ℹ️ Info", use_container_width=True):
            st.info(
                "- Momentum 12M-1M real\n"
                "- MA200 filter obligatorio (recomendado)\n"
                "- F-Score simplificado (0-9)\n"
                "- Neutralización por sector en factores"
            )

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Pipeline",
    "💎 QVM Rankings",
    "🎯 Momentum + MA200",
    "🏆 F-Score Analysis",
    "📈 Backtest",
    "💾 Export",
])

# =============================================================================
# PIPELINE
# =============================================================================

if run_btn:
    progress = st.progress(0)
    status = st.empty()

    try:
        # 1) Screener
        status.text("📊 Paso 1/8: Descargando universo…")
        progress.progress(10)
        universe = cached_screener(limit, mcap_min, volume_min)
        if universe is None or universe.empty:
            st.error("❌ Screener vacío.")
            st.stop()
        st.session_state["universe"] = universe

        # 2) Fundamentals
        status.text("📊 Paso 2/8: Descargando fundamentales…")
        progress.progress(20)
        fundamentals = cached_fundamentals(tuple(universe["symbol"].tolist()))
        if fundamentals is None or fundamentals.empty:
            st.error("❌ No se obtuvieron fundamentales.")
            st.stop()
        st.session_state["fundamentals"] = fundamentals

        # 3) Filtros de calidad optimizados
        status.text("🛡️ Paso 3/8: Aplicando filtros de calidad…")
        progress.progress(30)
        fcfg = FilterConfig(
            min_roe=min_roe,
            min_gross_margin=min_gm,
            require_positive_fcf=req_fcf,
            require_positive_ocf=req_ocf,
            max_pe=50.0,
            max_ev_ebitda=25.0,
            min_volume=volume_min,
            min_market_cap=mcap_min,
        )
        df_merged = universe.merge(fundamentals, on="symbol", how="left")
        passed_filters, diagnostics = apply_all_filters(df_merged, fcfg)
        st.session_state["passed_filters"] = passed_filters
        st.session_state["diagnostics"] = diagnostics
        if passed_filters.empty:
            st.error("❌ Ningún símbolo pasó los filtros de calidad.")
            st.stop()

        # 4) F-Score
        status.text("🏆 Paso 4/8: Calculando F-Score…")
        progress.progress(40)
        fcols_pref = ["symbol","roa","net_income","total_assets","operating_cf","fcf","capex","roe"]
        base_cols = [c for c in fcols_pref if c in fundamentals.columns] or ["symbol"]
        df_for_fscore = passed_filters.merge(
            fundamentals[base_cols].drop_duplicates("symbol"),
            on="symbol", how="left"
        )
        for c in ["roa","operating_cf","fcf","net_income","total_assets","capex"]:
            if c not in df_for_fscore.columns:
                df_for_fscore[c] = np.nan
        df_for_fscore["fscore"] = calculate_simplified_fscore(df_for_fscore)
        if use_fscore:
            df_fscore_passed = df_for_fscore[df_for_fscore["fscore"] >= min_fscore].copy()
        else:
            df_fscore_passed = df_for_fscore.copy()
        st.session_state["fscore_data"] = df_for_fscore
        st.session_state["fscore_passed"] = df_fscore_passed
        if df_fscore_passed.empty:
            st.error(f"❌ Ningún símbolo con F-Score >= {min_fscore}.")
            st.stop()

        # 5) Precios
        status.text("📈 Paso 5/8: Descargando precios…")
        progress.progress(55)
        symbols_to_fetch = df_fscore_passed["symbol"].dropna().astype(str).unique().tolist()[:limit]
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=2 * 365)).strftime("%Y-%m-%d")  # 2 años
        prices_dict: dict[str, pd.DataFrame] = {}
        step = 20 / max(len(symbols_to_fetch), 1)
        for i, sym in enumerate(symbols_to_fetch, 1):
            try:
                px = cached_prices(sym, start_date, end_date)
                if px is not None and len(px) >= 252:
                    prices_dict[sym] = px
            except Exception:
                pass
            if not (i % 10):
                progress.progress(min(55 + int(i * step), 75))
        if not prices_dict:
            st.error("❌ No se obtuvieron precios suficientes (>=252 sesiones).")
            st.stop()
        st.session_state["prices_dict"] = prices_dict

        # 6) Momentum real + MA200
        status.text("🎯 Paso 6/8: Calculando momentum real + MA200…")
        progress.progress(80)
        momentum_df = calculate_momentum_batch(prices_dict)
        st.session_state["momentum_df"] = momentum_df
        df_with_mom = df_fscore_passed.merge(
            momentum_df[["symbol","momentum_12m1m","momentum_6m","composite_momentum","above_ma200","ma200","trend_strength"]],
            on="symbol", how="left"
        )
        if require_ma200:
            df_with_mom = df_with_mom[df_with_mom["above_ma200"] == True].copy()
        df_with_mom = df_with_mom[df_with_mom["momentum_12m1m"] >= min_momentum].copy()
        if df_with_mom.empty:
            st.error("❌ Ningún símbolo pasó MA200/momentum.")
            st.stop()
        st.session_state["after_momentum"] = df_with_mom

        # 7) Factores QVM (sector-neutral)
        status.text("💎 Paso 7/8: Calculando factores QVM…")
        progress.progress(90)

        # Universo limpio (conservar sector/mcap del universo)
        in_scope = df_with_mom["symbol"].dropna().astype(str).unique().tolist()
        df_universe_clean = (
            universe.loc[:, ["symbol","sector","market_cap"]]
            .drop_duplicates("symbol")
            .merge(pd.DataFrame({"symbol": in_scope}), on="symbol", how="inner")
        )

        needed_fund_cols = ["symbol","ev_ebitda","pb","pe","roe","roic","gross_margin","fcf","operating_cf"]
        df_fundamentals_clean = (
            fundamentals.loc[:, [c for c in needed_fund_cols if c in fundamentals.columns]]
            .drop_duplicates("symbol")
            .merge(pd.DataFrame({"symbol": in_scope}), on="symbol", how="inner")
        )
        for c in needed_fund_cols:
            if c not in df_fundamentals_clean.columns:
                df_fundamentals_clean[c] = np.nan

        # QVM con neutralización; reponderamos (1 - peso F) dentro de Q,V,M
        renorm = max(w_quality + w_value + w_momentum, 1e-9)
        wQ, wV, wM = (w_quality / renorm, w_value / renorm, w_momentum / renorm)

        df_qvm = compute_all_factors(
            df_universe_clean,
            df_fundamentals_clean[needed_fund_cols],
            sector_neutral=True,
            w_quality=wQ,
            w_value=wV,
            w_momentum=wM,
        )

        keep_cols = ["symbol","momentum_12m1m","momentum_6m","composite_momentum","above_ma200","fscore","roe","sector","market_cap"]
        keep_cols = [c for c in keep_cols if c in df_with_mom.columns]
        df_qvm = df_qvm.merge(df_with_mom[keep_cols], on="symbol", how="left")

        # Composite final incorporando F-Score
        df_qvm["qvm_score_corrected"] = (
            w_quality * df_qvm["quality_extended"]
            + w_value * df_qvm["value_score"]
            + w_momentum * df_qvm["composite_momentum"]
            + w_fscore * (df_qvm["fscore"] / 9.0).fillna(0.0)
        )
        df_qvm["final_rank"] = df_qvm["qvm_score_corrected"].rank(pct=True, method="average")

        st.session_state["df_with_factors"] = df_qvm

        # 8) Selección final
        status.text("📋 Paso 8/8: Seleccionando Top portfolio…")
        progress.progress(97)
        portfolio = df_qvm.nlargest(top_n, "qvm_score_corrected").reset_index(drop=True)
        st.session_state["portfolio"] = portfolio

        # Precios del portfolio
        portfolio_prices = {sym: st.session_state["prices_dict"][sym]
                            for sym in portfolio["symbol"] if sym in st.session_state["prices_dict"]}
        st.session_state["portfolio_prices"] = portfolio_prices

        progress.progress(100)
        status.empty()
        progress.empty()

        st.success(f"✅ Pipeline completado: {len(portfolio)} acciones seleccionadas")

        # Métricas rápidas
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1: st.metric("Universo inicial", len(universe))
        with m2: st.metric("Después filtros", len(passed_filters))
        with m3:
            if use_fscore:
                st.metric(f"F-Score ≥ {min_fscore}", len(st.session_state.get("fscore_passed", [])))
            else:
                st.metric("F-Score", "No usado")
        with m4:
            tot_mom = len(st.session_state.get("after_momentum", []))
            tot_before = len(df_fscore_passed)
            rej_ma = max(tot_before - tot_mom, 0)
            st.metric("Rechazados por MA200/Mom", rej_ma)
        with m5: st.metric("Portfolio final", len(portfolio))

    except Exception as e:
        st.error(f"❌ Error en pipeline: {e}")
        import traceback
        with st.expander("🐛 Debug info"):
            st.code(traceback.format_exc())
        st.stop()

# =============================================================================
# TAB 1 — PIPELINE OVERVIEW
# =============================================================================

with tab1:
    st.markdown("### 📊 Pipeline Overview")
    if "portfolio" not in st.session_state:
        st.info("👆 Configura parámetros y ejecuta el pipeline.")
    else:
        funnel = pd.DataFrame({
            "Stage": [
                "1. Universo inicial",
                "2. Filtros calidad",
                "3. F-Score filter",
                "4. Precios válidos",
                "5. MA200 + Momentum",
                "6. Portfolio final",
            ],
            "Count": [
                len(st.session_state.get("universe", [])),
                len(st.session_state.get("passed_filters", [])),
                len(st.session_state.get("fscore_passed", [])),
                len(st.session_state.get("prices_dict", {})),
                len(st.session_state.get("after_momentum", [])),
                len(st.session_state.get("portfolio", [])),
            ],
        })
        chart = alt.Chart(funnel).mark_bar().encode(
            x=alt.X("Count:Q", title="Número de símbolos"),
            y=alt.Y("Stage:N", sort=None, title=""),
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None),
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

        st.markdown("#### 🛡️ Diagnóstico de Filtros")
        diag = st.session_state.get("diagnostics")
        if isinstance(diag, pd.DataFrame) and not diag.empty:
            total = len(diag)
            passed = int(diag["pass_all"].sum())
            rejected = total - passed
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Total evaluados", total)
            with c2: st.metric("Pasaron todos", passed, delta=f"{(passed/total*100):.1f}%")
            with c3: st.metric("Rechazados", rejected, delta=f"-{(rejected/total*100):.1f}%", delta_color="inverse")

            if rejected > 0:
                st.markdown("**Top razones de rechazo:**")
                reasons = diag.loc[~diag["pass_all"], "reason"].value_counts().head(5)
                for r, c in reasons.items():
                    st.write(f"- {r}: {c} ({(c/rejected*100):.1f}%)")

# =============================================================================
# TAB 2 — QVM RANKINGS
# =============================================================================

with tab2:
    st.markdown("### 💎 QVM Rankings")

    def _ensure_sector(df: pd.DataFrame) -> pd.DataFrame:
        """Garantiza columna 'sector' usando alias o el universo original como fallback."""
        pf = df.copy()

        # 1) Alias frecuentes dentro del propio DF
        if "sector" not in pf.columns:
            for cand in ("sectorName", "industry", "industryTitle", "subSector"):
                if cand in pf.columns:
                    pf["sector"] = (
                        pf[cand].astype(str).str.strip().replace({"": "Unknown"}).fillna("Unknown")
                    )
                    break

        # 2) Merge con universo original si aún falta
        if "sector" not in pf.columns:
            uni = st.session_state.get("universe", pd.DataFrame())
            if not uni.empty:
                # Normaliza 'sector' en el universo también
                sector_src = None
                for cand in ("sector", "sectorName", "industry", "industryTitle", "subSector"):
                    if cand in uni.columns:
                        sector_src = cand
                        break
                if sector_src is not None:
                    look = (
                        uni[["symbol", sector_src]].rename(columns={sector_src: "sector"})
                        .dropna(subset=["symbol"])
                        .drop_duplicates("symbol")
                    )
                    look["sector"] = (
                        look["sector"].astype(str).str.strip().replace({"": "Unknown"}).fillna("Unknown")
                    )
                    pf = pf.drop(columns=["sector"], errors="ignore").merge(look, on="symbol", how="left")

        # 3) Último recurso
        if "sector" not in pf.columns:
            pf["sector"] = "Unknown"

        # Limpieza final
        pf["sector"] = pf["sector"].astype(str).str.strip().replace({"": "Unknown"}).fillna("Unknown")
        return pf

    def _safe_metric(label: str, value, fmt: str | None = None):
        """Evita excepciones en métricas cuando faltan columnas."""
        try:
            if value is None:
                st.metric(label, "—")
                return
            st.metric(label, (fmt % value) if fmt else value)
        except Exception:
            st.metric(label, "—")

    if "portfolio" not in st.session_state or st.session_state["portfolio"] is None or len(st.session_state["portfolio"]) == 0:
        st.info("Ejecuta el pipeline primero.")
    else:
        pf = st.session_state["portfolio"].copy()
        pf = _ensure_sector(pf)

        # ===== MÉTRICAS DE CABECERA =====
        a, b, c, d = st.columns(4)
        with a:
            _safe_metric("Portfolio Size", len(pf))
        with b:
            _safe_metric("Avg QVM Score", (pf["qvm_score_corrected"].mean() if "qvm_score_corrected" in pf.columns and len(pf) else None), "%.3f")
        with c:
            _safe_metric("Avg F-Score", (pf["fscore"].mean() if "fscore" in pf.columns and len(pf) else None), "%.1f/9.0")
        with d:
            _safe_metric("Sectores", (pf["sector"].nunique() if "sector" in pf.columns else None))

        st.markdown("---")

        # ===== TABLA PRINCIPAL =====
        cols = [
            "symbol","sector","qvm_score_corrected","final_rank",
            "quality_extended","value_score","composite_momentum","fscore",
            "momentum_12m1m","above_ma200","roe","market_cap",
        ]
        cols = [c for c in cols if c in pf.columns]
        disp = pf[cols].copy()

        rename = {
            "qvm_score_corrected": "QVM Score",
            "final_rank": "Rank %ile",
            "quality_extended": "Quality",
            "value_score": "Value",
            "composite_momentum": "Momentum",
            "fscore": "F-Score",
            "momentum_12m1m": "Mom 12M",
            "above_ma200": "MA200✓",
            "market_cap": "MCap",
        }
        disp.rename(columns={k: v for k, v in rename.items() if k in disp.columns}, inplace=True)

        # Formateo seguro
        if "QVM Score" in disp.columns:
            disp["QVM Score"] = pd.to_numeric(disp["QVM Score"], errors="coerce").round(3)
        if "Rank %ile" in disp.columns:
            disp["Rank %ile"] = (pd.to_numeric(disp["Rank %ile"], errors="coerce") * 100).round(1)
        if "Quality" in disp.columns:
            disp["Quality"] = pd.to_numeric(disp["Quality"], errors="coerce").round(3)
        if "Value" in disp.columns:
            disp["Value"] = pd.to_numeric(disp["Value"], errors="coerce").round(3)
        if "Momentum" in disp.columns:
            disp["Momentum"] = pd.to_numeric(disp["Momentum"], errors="coerce").round(3)
        if "F-Score" in disp.columns:
            disp["F-Score"] = pd.to_numeric(disp["F-Score"], errors="coerce").round(1)
        if "Mom 12M" in disp.columns:
            disp["Mom 12M"] = (pd.to_numeric(disp["Mom 12M"], errors="coerce") * 100).round(1)
        if "roe" in disp.columns:
            disp["roe"] = (pd.to_numeric(disp["roe"], errors="coerce") * 100).round(1)
        if "MCap" in disp.columns:
            disp["MCap"] = (pd.to_numeric(disp["MCap"], errors="coerce") / 1e9).round(2)

        # Mostrar tabla
        st.dataframe(
            disp,
            use_container_width=True,
            height=600,
            column_config={
                "MA200✓": st.column_config.CheckboxColumn(),
                "Rank %ile": st.column_config.ProgressColumn(min_value=0, max_value=100),
                "Mom 12M": st.column_config.NumberColumn(format="%.1f%%"),
                "roe": st.column_config.NumberColumn(format="%.1f%%"),
                "MCap": st.column_config.NumberColumn(format="%.2fB"),
            },
        )

        # ===== DISTRIBUCIÓN POR SECTOR =====
        st.markdown("#### 📊 Distribución por Sector")
        needed = {"sector", "symbol"}
        if needed.issubset(pf.columns):
            # columnas opcionales para promedios (si no están, se omiten)
            agg_dict = {"symbol": "count"}
            if "qvm_score_corrected" in pf.columns:
                agg_dict["qvm_score_corrected"] = "mean"
            if "fscore" in pf.columns:
                agg_dict["fscore"] = "mean"
            if "momentum_12m1m" in pf.columns:
                agg_dict["momentum_12m1m"] = "mean"

            sec = pf.groupby("sector").agg(agg_dict).reset_index()
            # Renombrar amigable
            rename_sec = {"symbol": "Count"}
            if "qvm_score_corrected" in sec.columns:
                rename_sec["qvm_score_corrected"] = "Avg QVM"
            if "fscore" in sec.columns:
                rename_sec["fscore"] = "Avg F-Score"
            if "momentum_12m1m" in sec.columns:
                rename_sec["momentum_12m1m"] = "Avg Mom"
            sec.rename(columns=rename_sec, inplace=True)

            if "Avg Mom" in sec.columns:
                sec["Avg Mom"] = (pd.to_numeric(sec["Avg Mom"], errors="coerce") * 100).round(1)

            st.dataframe(sec.sort_values("Count", ascending=False), use_container_width=True)
        else:
            st.info("No hay columnas suficientes para agrupar por sector.")


# =============================================================================
# TAB 3 — MOMENTUM + MA200
# =============================================================================

with tab3:
    st.markdown("### 🎯 Momentum + MA200 Analysis")
    momdf = st.session_state.get("momentum_df")
    if not isinstance(momdf, pd.DataFrame) or momdf.empty:
        st.info("Ejecuta el pipeline primero.")
    else:
        a, b, c, d = st.columns(4)
        with a:
            pos = int((momdf["momentum_12m1m"] > 0).sum())
            st.metric("Momentum Positivo", f"{pos}/{len(momdf)}", delta=f"{pos/len(momdf)*100:.1f}%")
        with b:
            above = int(momdf["above_ma200"].sum())
            st.metric("Arriba de MA200", f"{above}/{len(momdf)}", delta=f"{above/len(momdf)*100:.1f}%")
        with c:
            st.metric("Momentum Promedio", f"{momdf['momentum_12m1m'].mean():.1%}")
        with d:
            both = int(((momdf["momentum_12m1m"] > 0) & momdf["above_ma200"]).sum())
            st.metric("Ambos ✓", f"{both}/{len(momdf)}", delta=f"{both/len(momdf)*100:.1f}%")

        st.markdown("---")
        st.markdown("#### 📊 Momentum vs MA200 Status")
        plot_df = momdf.copy()
        plot_df["MA200_status"] = plot_df["above_ma200"].map({True:"Above MA200", False:"Below MA200"})
        scatter = alt.Chart(plot_df).mark_circle(size=60).encode(
            x=alt.X("momentum_12m1m:Q", title="12M-1M Momentum", scale=alt.Scale(domain=[-0.5, 1.0])),
            y=alt.Y("composite_momentum:Q", title="Composite Momentum Score"),
            color=alt.Color("MA200_status:N", title="MA200 Status", scale=alt.Scale(scheme="category10")),
            tooltip=["symbol","momentum_12m1m","above_ma200","composite_momentum"],
        ).properties(height=400)
        st.altair_chart(scatter, use_container_width=True)

        st.markdown("#### 📋 Momentum Detail Table")
        cols = ["symbol","momentum_12m1m","momentum_6m","above_ma200","ma200","trend_strength"]
        disp = plot_df[[c for c in cols if c in plot_df.columns]].copy()
        if "momentum_12m1m" in disp.columns:
            disp["momentum_12m1m"] = (disp["momentum_12m1m"] * 100).round(1)
        if "momentum_6m" in disp.columns:
            disp["momentum_6m"] = (disp["momentum_6m"] * 100).round(1)
        if "trend_strength" in disp.columns:
            disp["trend_strength"] = (disp["trend_strength"] * 100).round(0)
        st.dataframe(
            disp,
            use_container_width=True,
            height=420,
            column_config={
                "momentum_12m1m": st.column_config.NumberColumn(format="%.1f%%"),
                "momentum_6m": st.column_config.NumberColumn(format="%.1f%%"),
                "above_ma200": st.column_config.CheckboxColumn(),
                "trend_strength": st.column_config.ProgressColumn(min_value=0, max_value=100),
            },
        )

# =============================================================================
# TAB 4 — F-SCORE ANALYSIS
# =============================================================================

with tab4:
    st.markdown("### 🏆 F-Score Analysis")
    fscore_data = st.session_state.get("fscore_data")
    if not isinstance(fscore_data, pd.DataFrame) or fscore_data.empty or "fscore" not in fscore_data.columns:
        st.info("Ejecuta el pipeline con F-Score habilitado.")
    else:
        avg_f = float(fscore_data["fscore"].mean())
        hi = int((fscore_data["fscore"] >= 8).sum())
        mid = int(((fscore_data["fscore"] >= 6) & (fscore_data["fscore"] < 8)).sum())
        lo = int((fscore_data["fscore"] < 6).sum())
        a, b, c, d = st.columns(4)
        with a: st.metric("F-Score Promedio", f"{avg_f:.1f}/9.0")
        with b: st.metric("High Quality (8-9)", hi)
        with c: st.metric("Medium (6-7)", mid)
        with d: st.metric("Low (0-5)", lo)

        hist = fscore_data["fscore"].round(0).value_counts().sort_index().reset_index()
        hist.columns = ["F-Score","Count"]
        chart = alt.Chart(hist).mark_bar().encode(
            x=alt.X("F-Score:O", title="F-Score"),
            y=alt.Y("Count:Q", title="Número de acciones"),
            color=alt.condition(alt.datum["F-Score"] >= 6, alt.value("steelblue"), alt.value("lightgray")),
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

        st.markdown("#### 📋 Stocks by F-Score Category")
        show_cols = [c for c in ["symbol","fscore","roe","fcf","operating_cf"] if c in fscore_data.columns]
        tbl = fscore_data[show_cols].copy()
        tbl["category"] = pd.cut(tbl["fscore"], bins=[-0.1,5.5,7.5,9.1],
                                 labels=["Low (0-5)","Medium (6-7)","High (8-9)"])
        st.dataframe(tbl.sort_values("fscore", ascending=False), use_container_width=True, height=420)

# =============================================================================
# TAB 5 — BACKTEST
# =============================================================================

with tab5:
    st.markdown("### 📈 Backtest Results")
    if st.button("🚀 Ejecutar Backtest", type="primary"):
        prx = st.session_state.get("portfolio_prices")
        if not isinstance(prx, dict) or not prx:
            st.info("Ejecuta el pipeline primero (no hay precios de portfolio).")
        else:
            with st.spinner("Ejecutando backtest…"):
                costs = TradingCosts(commission_bps=5, slippage_bps=5, market_impact_bps=2)
                metrics, equity_curves = backtest_portfolio(prx, costs=costs, execution_lag_days=1)
                port_metrics = calculate_portfolio_metrics(equity_curves, costs)
                st.session_state["backtest_metrics"] = metrics
                st.session_state["portfolio_metrics"] = port_metrics
                st.session_state["equity_curves"] = equity_curves

    if isinstance(st.session_state.get("portfolio_metrics"), dict):
        pm = st.session_state["portfolio_metrics"]
        a, b, c, d, e = st.columns(5)
        with a: st.metric("CAGR", f"{pm['CAGR']:.2%}")
        with b: st.metric("Sharpe", f"{pm['Sharpe']:.2f}")
        with c: st.metric("Sortino", f"{pm['Sortino']:.2f}")
        with d: st.metric("Max Drawdown", f"{pm['MaxDD']:.2%}")
        with e: st.metric("Calmar", f"{pm['Calmar']:.2f}")

        st.markdown("---")
        st.markdown("#### 📈 Portfolio Equity Curve")
        eq = st.session_state["equity_curves"]
        eq_df = pd.DataFrame(eq)
        port_eq = eq_df.mean(axis=1)
        plot_df = pd.DataFrame({"Date": port_eq.index, "Equity": port_eq.values})
        curve = alt.Chart(plot_df).mark_line(color="steelblue").encode(
            x="Date:T", y=alt.Y("Equity:Q", title="Portfolio Value ($)"),
            tooltip=["Date:T","Equity:Q"]
        ).properties(height=400)
        st.altair_chart(curve, use_container_width=True)

        st.markdown("#### 📊 Individual Stock Performance")
        st.dataframe(st.session_state["backtest_metrics"], use_container_width=True, height=420)

# =============================================================================
# TAB 6 — EXPORT
# =============================================================================

with tab6:
    st.markdown("### 💾 Export Data")
    pf = st.session_state.get("portfolio")
    if not isinstance(pf, pd.DataFrame) or pf.empty:
        st.info("Ejecuta el pipeline primero.")
    else:
        csv = pf.to_csv(index=False)
        st.download_button(
            label="⬇️ Descargar Portfolio (CSV)",
            data=csv,
            file_name=f"qvm_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown("**Preview:**")
        st.dataframe(pf.head(10), use_container_width=True)

        if isinstance(st.session_state.get("backtest_metrics"), pd.DataFrame):
            st.markdown("---")
            bm = st.session_state["backtest_metrics"]
            csv_b = bm.to_csv(index=False)
            st.download_button(
                label="⬇️ Descargar Backtest Metrics (CSV)",
                data=csv_b,
                file_name=f"qvm_backtest_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.caption(
    "**QVM Screener V2** | Implementación académica con Momentum Real + MA200 + F-Score  \n"
    "Basado en: Jegadeesh & Titman (1993), Faber (2007), Piotroski (2000), Asness et al. (2019)"
)
