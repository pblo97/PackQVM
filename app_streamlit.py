"""
QVM Screener - Aplicación Final Limpia
======================================

Stack completo desde cero:
- data_fetcher.py: API + caché
- factor_calculator.py: QVM según bibliografía
- screener_filters.py: Guardrails simples
- backtest_engine.py: Backtest básico
- app_streamlit.py: Esta UI

TODO funciona con flujo limpio de datos.
"""

from __future__ import annotations
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# Imports de nuestros módulos limpios
from data_fetcher import (
    fetch_screener,
    fetch_fundamentals_batch,
    fetch_prices,
    clear_cache,
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
)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

st.set_page_config(
    page_title="QVM Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: .2px; }
hr { border: 0; border-top: 1px solid rgba(255,255,255,.08); margin: .5rem 0; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# CACHÉ DE ESTADO
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def cached_screener(limit: int, mcap_min: float, volume_min: int):
    """Caché del screener"""
    return fetch_screener(limit, mcap_min, volume_min, use_cache=True)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fundamentals(symbols: tuple):
    """Caché de fundamentales"""
    return fetch_fundamentals_batch(list(symbols), use_cache=True)


# ============================================================================
# HEADER
# ============================================================================

col1, col2 = st.columns([0.85, 0.15])
with col1:
    st.markdown("<h1 style='margin-bottom:0'>QVM Screener</h1>", unsafe_allow_html=True)
    st.caption("Quality × Value × Momentum | Limpio desde cero")
with col2:
    st.caption(datetime.now().strftime("%d %b %Y %H:%M"))

st.markdown("<hr/>", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR - CONFIGURACIÓN
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    
    # UNIVERSO
    with st.expander("🌍 Universo", expanded=True):
        limit = st.slider("Tamaño universo", 50, 500, 200, 50)
        mcap_min = st.number_input("Market Cap mín. (USD)", value=5e8, step=1e8, format="%.0f")
        volume_min = st.number_input("Volumen mín. diario", value=500_000, step=100_000)
    
    # FILTROS
    with st.expander("🛡️ Guardrails", expanded=True):
        min_roe = st.slider("ROE mín.", 0.0, 0.50, 0.10, 0.05)
        min_gm = st.slider("Gross Margin mín.", 0.0, 0.80, 0.20, 0.05)
        req_fcf = st.toggle("Exigir FCF > 0", value=True)
        req_ocf = st.toggle("Exigir Operating CF > 0", value=True)
    
    # QVM
    with st.expander("💎 QVM Weights", expanded=True):
        w_quality = st.slider("Peso Quality", 0.0, 1.0, 0.40, 0.05)
        w_value = st.slider("Peso Value", 0.0, 1.0, 0.30, 0.05)
        w_momentum = st.slider("Peso Momentum", 0.0, 1.0, 0.30, 0.05)
        
        # Normalizar pesos
        total_w = w_quality + w_value + w_momentum
        if total_w > 0:
            w_quality /= total_w
            w_value /= total_w
            w_momentum /= total_w
    
    # FILTRO FINAL
    with st.expander("🎯 Selección Final", expanded=True):
        min_qvm_rank = st.slider("QVM Percentil mín.", 0.0, 1.0, 0.50, 0.05)
        top_n = st.slider("Top N símbolos", 5, 100, 30, 5)
    
    st.markdown("---")
    
    run_btn = st.button("🚀 Ejecutar", type="primary", use_container_width=True)
    
    if st.button("🗑️ Limpiar caché", use_container_width=True):
        clear_cache()
        st.cache_data.clear()
        st.success("Caché limpiado")


# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Pipeline",
    "💎 QVM Rankings",
    "🛡️ Guardrails",
    "📈 Backtest",
    "💾 Export"
])


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

if "pipeline_executed" not in st.session_state:
    st.session_state["pipeline_executed"] = False

if run_btn or not st.session_state["pipeline_executed"]:
    
    with st.spinner("🔄 Ejecutando pipeline..."):
        start_time = time.time()
        
        # -------------------------
        # ETAPA 1: SCREENER
        # -------------------------
        with st.spinner("1️⃣ Descargando universo..."):
            df_universe = cached_screener(limit, mcap_min, volume_min)
        
        if df_universe.empty:
            st.error("❌ No se pudo descargar el universo")
            st.stop()
        
        st.session_state["df_universe"] = df_universe
        
        # -------------------------
        # ETAPA 2: FUNDAMENTALES
        # -------------------------
        with st.spinner("2️⃣ Descargando fundamentales..."):
            symbols = tuple(df_universe["symbol"].tolist())
            df_fund = cached_fundamentals(symbols)
        
        st.session_state["df_fund"] = df_fund
        
        # -------------------------
        # ETAPA 3: FACTORES QVM
        # -------------------------
        with st.spinner("3️⃣ Calculando factores QVM..."):
            df_qvm = compute_all_factors(df_universe, df_fund)
        
        st.session_state["df_qvm"] = df_qvm
        
        # -------------------------
        # ETAPA 4: GUARDRAILS
        # -------------------------
        with st.spinner("4️⃣ Aplicando guardrails..."):
            filter_config = FilterConfig(
                min_roe=min_roe,
                min_gross_margin=min_gm,
                require_positive_fcf=req_fcf,
                require_positive_ocf=req_ocf,
            )
            passed, diagnostics = apply_all_filters(df_qvm, filter_config)
        
        st.session_state["passed"] = passed
        st.session_state["diagnostics"] = diagnostics
        
        # -------------------------
        # ETAPA 5: FILTRO QVM
        # -------------------------
        with st.spinner("5️⃣ Aplicando filtro QVM..."):
            df_qvm_passed = df_qvm.merge(passed, on="symbol", how="inner")
            df_final = filter_by_qvm(df_qvm_passed, min_qvm_rank, top_n)
        
        st.session_state["df_final"] = df_final
        
        elapsed = time.time() - start_time
        st.session_state["pipeline_time"] = elapsed
        st.session_state["pipeline_executed"] = True
        
        st.success(f"✅ Pipeline completado en {elapsed:.1f}s")


# ============================================================================
# TAB 1: PIPELINE
# ============================================================================

with tab1:
    st.subheader("Pipeline de Datos")
    
    if st.session_state.get("pipeline_executed"):
        df_uni = st.session_state["df_universe"]
        df_qvm = st.session_state["df_qvm"]
        passed = st.session_state["passed"]
        df_final = st.session_state["df_final"]
        
        # Métricas
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Universo inicial", f"{len(df_uni)}")
        c2.metric("Con QVM calculado", f"{len(df_qvm)}")
        c3.metric("Pasaron guardrails", f"{len(passed)}")
        c4.metric("Top N final", f"{len(df_final)}")
        
        st.markdown("---")
        
        # Preview universo
        st.markdown("#### 📊 Universo inicial")
        st.dataframe(df_uni.head(50), use_container_width=True, hide_index=True)
        
        st.markdown("#### ⏱️ Tiempos")
        st.caption(f"Pipeline: {st.session_state['pipeline_time']:.2f}s")
    
    else:
        st.info("👈 Presiona 'Ejecutar' en el sidebar para comenzar")


# ============================================================================
# TAB 2: QVM RANKINGS
# ============================================================================

with tab2:
    st.subheader("Rankings QVM")
    
    if st.session_state.get("pipeline_executed"):
        df_final = st.session_state["df_final"]
        
        if df_final.empty:
            st.warning("⚠️ No hay símbolos que pasen los filtros")
        else:
            st.markdown(f"### 🏆 Top {len(df_final)} Símbolos")
            
            # Métricas agregadas
            avg_qvm = df_final["qvm_rank"].mean()
            avg_quality = df_final["quality_score"].mean()
            avg_value = df_final["value_score"].mean()
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg QVM Rank", f"{avg_qvm:.2%}")
            c2.metric("Avg Quality", f"{avg_quality:.2%}")
            c3.metric("Avg Value", f"{avg_value:.2%}")
            
            st.markdown("---")
            
            # Tabla principal
            display_cols = [
                "symbol", "sector", "market_cap",
                "qvm_rank", "qvm_score",
                "quality_score", "value_score", "momentum_score"
            ]
            display_cols = [c for c in display_cols if c in df_final.columns]
            
            st.dataframe(
                df_final[display_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "market_cap": st.column_config.NumberColumn(
                        "Market Cap",
                        format="$%.2fB",
                    ),
                    "qvm_rank": st.column_config.ProgressColumn(
                        "QVM Rank",
                        min_value=0,
                        max_value=1,
                    ),
                }
            )
            
            # Gráfico de dispersión
            st.markdown("### 📊 Quality vs Value")
            
            chart_data = df_final[["symbol", "quality_score", "value_score"]].copy()
            
            scatter = alt.Chart(chart_data).mark_circle(size=100).encode(
                x=alt.X("quality_score:Q", title="Quality Score"),
                y=alt.Y("value_score:Q", title="Value Score"),
                tooltip=["symbol", "quality_score", "value_score"],
            ).properties(height=400).interactive()
            
            st.altair_chart(scatter, use_container_width=True)
    
    else:
        st.info("Ejecuta el pipeline primero")


# ============================================================================
# TAB 3: GUARDRAILS
# ============================================================================

with tab3:
    st.subheader("Diagnóstico de Guardrails")
    
    if st.session_state.get("pipeline_executed"):
        diagnostics = st.session_state["diagnostics"]
        passed = st.session_state["passed"]
        
        total = len(diagnostics)
        n_passed = len(passed)
        n_failed = total - n_passed
        
        c1, c2 = st.columns(2)
        c1.metric("✅ Pasaron", f"{n_passed}")
        c2.metric("❌ Rechazados", f"{n_failed}")
        
        st.markdown("---")
        
        # Tabla de diagnóstico
        st.dataframe(
            diagnostics.sort_values("pass_all", ascending=False),
            use_container_width=True,
            hide_index=True
        )
        
        # Razones de rechazo
        if n_failed > 0:
            st.markdown("#### 📋 Razones de rechazo")
            
            failed = diagnostics[diagnostics["pass_all"] == False]
            reasons = failed["reason"].value_counts()
            
            st.bar_chart(reasons)
    
    else:
        st.info("Ejecuta el pipeline primero")


# ============================================================================
# TAB 4: BACKTEST
# ============================================================================

with tab4:
    st.subheader("Backtest")
    
    if st.session_state.get("pipeline_executed"):
        df_final = st.session_state["df_final"]
        
        if df_final.empty:
            st.warning("No hay símbolos para backtest")
        else:
            st.markdown("#### ⚙️ Configuración")
            
            c1, c2 = st.columns(2)
            with c1:
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=365*3)  # 3 años
                
                bt_start = st.date_input("Inicio", value=start_date)
                bt_end = st.date_input("Fin", value=end_date)
            
            with c2:
                bt_symbols = df_final["symbol"].head(20).tolist()
                st.caption(f"Testeando {len(bt_symbols)} símbolos")
            
            if st.button("🚀 Ejecutar Backtest", type="primary"):
                with st.spinner("Descargando precios..."):
                    prices_dict = {}
                    
                    progress = st.progress(0.0)
                    for i, symbol in enumerate(bt_symbols):
                        try:
                            df_prices = fetch_prices(
                                symbol,
                                str(bt_start),
                                str(bt_end)
                            )
                            if df_prices is not None:
                                prices_dict[symbol] = df_prices
                        except Exception:
                            pass
                        
                        progress.progress((i+1) / len(bt_symbols))
                
                if not prices_dict:
                    st.error("No se pudieron descargar precios")
                else:
                    with st.spinner("Ejecutando backtest..."):
                        metrics, curves = backtest_portfolio(prices_dict)
                    
                    if metrics.empty:
                        st.warning("No hay resultados de backtest")
                    else:
                        st.markdown("#### 📊 Resultados por símbolo")
                        st.dataframe(metrics, use_container_width=True, hide_index=True)
                        
                        # Portfolio metrics
                        port_metrics = calculate_portfolio_metrics(curves)
                        
                        st.markdown("#### 💼 Portfolio (equal-weight)")
                        pm_col1, pm_col2, pm_col3 = st.columns(3)
                        pm_col1.metric("CAGR", f"{port_metrics['CAGR']:.2%}")
                        pm_col2.metric("Sharpe", f"{port_metrics['Sharpe']:.2f}")
                        pm_col3.metric("Max DD", f"{port_metrics['MaxDD']:.2%}")
                        
                        # Equity curves
                        if curves:
                            st.markdown("#### 📈 Equity Curves")
                            
                            # Preparar data para Altair
                            eq_list = []
                            for sym, eq in curves.items():
                                tmp = eq.reset_index()
                                tmp.columns = ["date", "equity"]
                                tmp["symbol"] = sym
                                eq_list.append(tmp)
                            
                            eq_df = pd.concat(eq_list, ignore_index=True)
                            
                            chart = alt.Chart(eq_df).mark_line().encode(
                                x="date:T",
                                y="equity:Q",
                                color="symbol:N",
                                tooltip=["date:T", "symbol:N", "equity:Q"]
                            ).properties(height=400).interactive()
                            
                            st.altair_chart(chart, use_container_width=True)
    
    else:
        st.info("Ejecuta el pipeline primero")


# ============================================================================
# TAB 5: EXPORT
# ============================================================================

with tab5:
    st.subheader("Exportar Datos")
    
    if st.session_state.get("pipeline_executed"):
        df_final = st.session_state["df_final"]
        diagnostics = st.session_state["diagnostics"]
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.download_button(
                "⬇️ Top Símbolos (CSV)",
                data=df_final.to_csv(index=False).encode("utf-8"),
                file_name="qvm_top.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with c2:
            st.download_button(
                "⬇️ Diagnóstico Completo (CSV)",
                data=diagnostics.to_csv(index=False).encode("utf-8"),
                file_name="qvm_diagnostics.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    else:
        st.info("Ejecuta el pipeline primero")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("💡 QVM Screener | Arquitectura limpia | Flujo unidireccional de datos")
