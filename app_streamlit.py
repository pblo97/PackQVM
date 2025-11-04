"""
QVM Screener V2 - Con Momentum Real + MA200 + F-Score
======================================================

Implementación académica completa con:
✅ Momentum real desde precios (Jegadeesh & Titman, 1993)
✅ Filtro MA200 obligatorio (Faber, 2007)
✅ Piotroski F-Score (Piotroski, 2000)
✅ Sector-neutral factors
✅ Reglas heurísticas robustas

Módulos:
- data_fetcher.py: API + caché
- factor_calculator.py: Factores QVM
- momentum_calculator.py: Momentum real + MA200 ⭐ NUEVO
- piotroski_fscore.py: F-Score de 9 puntos ⭐ NUEVO
- screener_filters.py: Guardrails
- backtest_engine.py: Backtest (corregido)
"""

from __future__ import annotations
import os
import shutil
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# Imports de módulos base (ajustados)
from data_fetcher import (
    fetch_screener,
    fetch_fundamentals_batch,
    fetch_prices_daily,   # ← usamos daily con lookback
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

# ⭐ Imports de módulos NUEVOS
from momentum_calculator import (
    integrate_real_momentum,
    filter_above_ma200,
    calculate_momentum_batch,
)
# Alias a la Forma B (sin ROE) para mantener el nombre usado en la app
from piotroski_fscore import (
    filter_by_fscore,
    calculate_simplified_fscore_no_roe as calculate_simplified_fscore,
)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

st.set_page_config(
    page_title="QVM Screener V2",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: .2px; }
hr { border: 0; border-top: 1px solid rgba(255,255,255,.08); margin: .5rem 0; }
.stAlert { padding: 0.5rem 1rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# CACHÉ DE ESTADO (wrappers simplificados)
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def cached_screener(limit: int, mcap_min: float, volume_min: int):
    """Caché del screener (memo en sesión; data_fetcher ya usa caché a disco)."""
    return fetch_screener(limit=limit, mcap_min=mcap_min, volume_min=volume_min)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_fundamentals(symbols: tuple):
    """Caché de fundamentales."""
    return fetch_fundamentals_batch(list(symbols))

@st.cache_data(ttl=1800, show_spinner=False)
def cached_prices(symbol: str, start: str, end: str):
    """
    Caché de precios individuales.
    Ignora start/end y pide ~800 días (~3 años bursátiles) con fetch_prices_daily.
    """
    return fetch_prices_daily(symbol, lookback_days=800)

def clear_cache():
    """
    Limpia caché en disco usada por data_fetcher (.cache/fmp) y la memo de Streamlit.
    """
    cache_dir = ".cache/fmp"
    try:
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        return True
    except Exception:
        return False


# ============================================================================
# HEADER
# ============================================================================

col1, col2 = st.columns([0.85, 0.15])
with col1:
    st.markdown("<h1 style='margin-bottom:0'>🚀 QVM Screener V2</h1>", unsafe_allow_html=True)
    st.caption("Quality × Value × Momentum × F-Score | Implementación Académica Completa")
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
        limit = st.slider("Tamaño universo", 50, 500, 300, 50)
        mcap_min = st.number_input(
            "Market Cap mín. (USD)", 
            value=2e9,  # ⭐ Aumentado a $2B
            step=1e8, 
            format="%.0f"
        )
        volume_min = st.number_input(
            "Volumen mín. diario", 
            value=1_000_000,  # ⭐ Aumentado a 1M
            step=100_000
        )
    
    # FILTROS DE CALIDAD
    with st.expander("🛡️ Quality Filters", expanded=True):
        min_roe = st.slider(
            "ROE mín.", 
            0.0, 0.50, 
            0.15,  # ⭐ 15%
            0.05
        )
        min_gm = st.slider(
            "Gross Margin mín.", 
            0.0, 0.80, 
            0.30,  # ⭐ 30%
            0.05
        )
        req_fcf = st.toggle("Exigir FCF > 0", value=True)
        req_ocf = st.toggle("Exigir Operating CF > 0", value=True)
    
    # ⭐ F-SCORE (NUEVO)
    with st.expander("🏆 F-Score (Piotroski)", expanded=True):
        use_fscore = st.toggle("Usar F-Score filter", value=True)
        if use_fscore:
            min_fscore = st.slider("F-Score mínimo", 0, 9, 6, 1)
            st.caption("6-7: Medium quality | 8-9: High quality")
    
    # ⭐ MOMENTUM + MA200 (NUEVO)
    with st.expander("🎯 Momentum + MA200", expanded=True):
        require_ma200 = st.toggle(
            "⭐ Exigir Price > MA200", 
            value=True,
            help="Filtro crítico: reduce drawdowns 50%+"
        )
        if require_ma200:
            st.caption("✅ Solo acciones en uptrend")
        else:
            st.warning("⚠️ MA200 filter desactivado (no recomendado)")
        
        min_momentum = st.slider(
            "Momentum 12M mín.", 
            -0.50, 0.50, 
            0.05,  # ⭐ 5% mínimo
            0.05,
            format="%.2f"
        )
    
    # QVM WEIGHTS
    with st.expander("💎 Factor Weights", expanded=True):
        w_quality = st.slider("Peso Quality", 0.0, 1.0, 0.35, 0.05)
        w_value = st.slider("Peso Value", 0.0, 1.0, 0.25, 0.05)
        w_momentum = st.slider("Peso Momentum", 0.0, 1.0, 0.25, 0.05)
        w_fscore = st.slider("Peso F-Score", 0.0, 1.0, 0.15, 0.05)
        
        # Normalizar pesos
        total_w = w_quality + w_value + w_momentum + w_fscore
        if total_w > 0:
            w_quality /= total_w
            w_value /= total_w
            w_momentum /= total_w
            w_fscore /= total_w
        
        st.caption(f"Normalizado: Q={w_quality:.2f} V={w_value:.2f} M={w_momentum:.2f} F={w_fscore:.2f}")
    
    # SELECCIÓN FINAL
    with st.expander("📋 Portfolio Selection", expanded=True):
        top_n = st.slider("Top N símbolos", 5, 100, 30, 5)
    
    st.markdown("---")
    
    # BOTONES
    run_btn = st.button("🚀 Ejecutar Pipeline", type="primary", use_container_width=True)
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🗑️ Limpiar caché", use_container_width=True):
            cleared = clear_cache()
            st.cache_data.clear()
            if cleared:
                st.success("✅ Caché limpiado")
            else:
                st.warning("⚠️ No se pudo limpiar toda la caché en disco")
    
    with col_btn2:
        if st.button("ℹ️ Info", use_container_width=True):
            st.info("""
            **V2 Features:**
            - ✅ Momentum real (12M-1M)
            - ✅ MA200 filter obligatorio
            - ✅ F-Score de Piotroski (Forma B simplificada)
            - ✅ Sector-neutral factors
            """)


# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Pipeline",
    "💎 QVM Rankings",
    "🎯 Momentum + MA200",
    "🏆 F-Score Analysis",
    "📈 Backtest",
    "💾 Export"
])


# ============================================================================
# EJECUCIÓN DEL PIPELINE
# ============================================================================

if run_btn:
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # ---------------------------------------------------------------------
        # PASO 1: SCREENER
        # ---------------------------------------------------------------------
        status_text.text("📊 Paso 1/8: Descargando universo...")
        progress_bar.progress(10)
        
        universe = cached_screener(limit, mcap_min, volume_min)
        
        if universe.empty:
            st.error("❌ No se encontraron símbolos en el screener")
            st.stop()
        
        st.session_state['universe'] = universe
        
        # ---------------------------------------------------------------------
        # PASO 2: FUNDAMENTALS
        # ---------------------------------------------------------------------
        status_text.text("📊 Paso 2/8: Descargando fundamentales...")
        progress_bar.progress(20)
        
        fundamentals = cached_fundamentals(tuple(universe['symbol'].tolist()))
        
        if fundamentals.empty:
            st.error("❌ No se pudieron obtener fundamentales")
            st.stop()
        
        st.session_state['fundamentals'] = fundamentals
        
        # ---------------------------------------------------------------------
        # PASO 3: FILTROS DE CALIDAD
        # ---------------------------------------------------------------------
        status_text.text("🛡️ Paso 3/8: Aplicando filtros de calidad...")
        progress_bar.progress(30)
        
        filter_config = FilterConfig(
            min_roe=min_roe,
            min_gross_margin=min_gm,
            require_positive_fcf=req_fcf,
            require_positive_ocf=req_ocf,
            max_pe=50.0,            # ⭐ Conservador
            max_ev_ebitda=25.0,
            min_volume=volume_min,
            min_market_cap=mcap_min,
        )
        
        df_merged = universe.merge(fundamentals, on='symbol', how='left')
        passed_filters, diagnostics = apply_all_filters(df_merged, filter_config)
        
        st.session_state['passed_filters'] = passed_filters
        st.session_state['diagnostics'] = diagnostics
        
        if passed_filters.empty:
            st.error("❌ Ningún símbolo pasó los filtros de calidad")
            st.stop()
        
        # ---------------------------------------------------------------------
        # PASO 4: F-SCORE (Forma B ya alias)
        # ---------------------------------------------------------------------
        if use_fscore:
            status_text.text("🏆 Paso 4/8: Calculando F-Score...")
            progress_bar.progress(40)
            
            df_for_fscore = passed_filters.merge(fundamentals, on='symbol', how='left')
            df_for_fscore['fscore'] = calculate_simplified_fscore(df_for_fscore)
            
            df_fscore_passed = df_for_fscore[df_for_fscore['fscore'] >= min_fscore].copy()
            
            st.session_state['fscore_data'] = df_for_fscore
            st.session_state['fscore_passed'] = df_fscore_passed
            
            if df_fscore_passed.empty:
                st.error(f"❌ Ningún símbolo con F-Score >= {min_fscore}")
                st.stop()
        else:
            df_fscore_passed = passed_filters.copy()
            st.session_state['fscore_passed'] = df_fscore_passed
        
        # ---------------------------------------------------------------------
        # PASO 5: DESCARGAR PRECIOS
        # ---------------------------------------------------------------------
        status_text.text("📈 Paso 5/8: Descargando precios históricos...")
        progress_bar.progress(50)
        
        symbols_to_fetch = df_fscore_passed['symbol'].tolist()[:limit]  # Limitar para no explotar API
        
        prices_dict = {}
        failed_count = 0
        
        # Fechas no usadas por cached_prices (se dejan por compatibilidad)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")  # 2 años
        
        progress_step = 30 / max(len(symbols_to_fetch), 1)
        
        for i, symbol in enumerate(symbols_to_fetch):
            try:
                prices = cached_prices(symbol, start_date, end_date)
                if prices is not None and len(prices) >= 252:
                    prices_dict[symbol] = prices
                else:
                    failed_count += 1
            except Exception:
                failed_count += 1
            
            if (i + 1) % 10 == 0:
                progress_bar.progress(min(50 + int((i + 1) * progress_step), 80))
        
        st.session_state['prices_dict'] = prices_dict
        
        if not prices_dict:
            st.error("❌ No se pudieron descargar precios")
            st.stop()
        
        # ---------------------------------------------------------------------
        # PASO 6: MOMENTUM + MA200 FILTER
        # ---------------------------------------------------------------------
        status_text.text("🎯 Paso 6/8: Calculando momentum real + MA200...")
        progress_bar.progress(80)
        
        momentum_df = calculate_momentum_batch(prices_dict)
        st.session_state['momentum_df'] = momentum_df
        
        # Merge con datos existentes
        df_with_momentum = df_fscore_passed.merge(
            momentum_df[['symbol', 'momentum_12m1m', 'above_ma200', 'composite_momentum']],
            on='symbol',
            how='left'
        )
        
        # ⭐ Filtro MA200 (crítico)
        if require_ma200:
            df_final = df_with_momentum[df_with_momentum['above_ma200'] == True].copy()
        else:
            df_final = df_with_momentum.copy()
        
        # Filtro momentum mínimo
        df_final = df_final[df_final['momentum_12m1m'] >= min_momentum].copy()
        
        st.session_state['after_momentum'] = df_final
        
        if df_final.empty:
            st.error("❌ Ningún símbolo pasó filtro MA200/momentum")
            st.stop()
        
        # ---------------------------------------------------------------------
        # PASO 7: CALCULAR FACTORES QVM
        # ---------------------------------------------------------------------
       # PASO 7: CALCULAR FACTORES QVM
# ---------------------------------------------------------------------
        status_text.text("💎 Paso 7/8: Calculando factores QVM...")
        progress_bar.progress(90)

        # --- Reinyectar SIEMPRE universo y fundamentales aquí ---
        symbols_in_scope = df_final["symbol"].dropna().astype(str).unique().tolist()

        # Universo limpio (aseguramos sector y market_cap desde el universo original)
        df_universe_clean = (
            universe.loc[:, ["symbol", "sector", "market_cap"]]
            .drop_duplicates("symbol")
            .merge(pd.DataFrame({"symbol": symbols_in_scope}), on="symbol", how="inner")
        )

        # Fundamentales limpios SOLO para los símbolos vigentes
        needed_fund_cols = ["symbol","ev_ebitda","pb","pe","roe","roic","gross_margin","fcf","operating_cf"]
        df_fundamentals_clean = (
            fundamentals.loc[:, [c for c in needed_fund_cols if c in fundamentals.columns]]
            .drop_duplicates("symbol")
            .merge(pd.DataFrame({"symbol": symbols_in_scope}), on="symbol", how="inner")
        )

        # Asegurar que existan todas las columnas esperadas (si falta alguna, crearla en NaN)
        for col in needed_fund_cols:
            if col not in df_fundamentals_clean.columns:
                df_fundamentals_clean[col] = np.nan

        # Calcular QVM con neutralización por sector
        df_with_factors = compute_all_factors(
            df_universe_clean,
            df_fundamentals_clean[needed_fund_cols],
            sector_neutral=True,
            w_quality=w_quality * (1 - w_fscore),  # Ajuste por peso F-Score
            w_value=w_value * (1 - w_fscore),
            w_momentum=w_momentum * (1 - w_fscore),
        )

        # Merge con momentum y F-Score reales
        cols_keep_from_final = ["symbol","momentum_12m1m","above_ma200","composite_momentum","fscore","roe","sector","market_cap"]
        cols_keep_from_final = [c for c in cols_keep_from_final if c in df_final.columns]
        df_with_factors = df_with_factors.merge(
            df_final[cols_keep_from_final],
            on="symbol",
            how="left"
        )

        # Recalcular composite score final con MOM real y F-Score
        df_with_factors["qvm_score_corrected"] = (
            w_quality * df_with_factors["quality_extended"] +
            w_value   * df_with_factors["value_score"] +
            w_momentum* df_with_factors["composite_momentum"] +
            w_fscore  * (df_with_factors["fscore"] / 9.0)
        )

        df_with_factors["final_rank"] = df_with_factors["qvm_score_corrected"].rank(pct=True, method="average")
        st.session_state["df_with_factors"] = df_with_factors
        # ---------------------------------------------------------------------
        # PASO 8: SELECCIÓN FINAL
        # ---------------------------------------------------------------------
        status_text.text("📋 Paso 8/8: Seleccionando Top portfolio...")
        progress_bar.progress(95)
        
        portfolio = df_with_factors.nlargest(top_n, 'qvm_score_corrected')
        st.session_state['portfolio'] = portfolio
        
        # Filtrar precios solo de portfolio
        portfolio_prices = {
            sym: prices_dict[sym] 
            for sym in portfolio['symbol'] 
            if sym in prices_dict
        }
        st.session_state['portfolio_prices'] = portfolio_prices
        
        # ---------------------------------------------------------------------
        # COMPLETADO
        # ---------------------------------------------------------------------
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        st.success(f"✅ Pipeline completado: {len(portfolio)} acciones seleccionadas")
        
        # Mostrar métricas clave
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Universo inicial", len(universe))
        
        with col2:
            st.metric("Después filtros", len(passed_filters))
        
        with col3:
            if use_fscore:
                st.metric(f"F-Score >= {min_fscore}", len(st.session_state.get('fscore_passed', [])))
            else:
                st.metric("F-Score", "No usado", delta_color="off")
        
        with col4:
            if require_ma200:
                rejection_rate = 1 - (len(df_final) / len(df_with_momentum))
                st.metric("MA200 Filter", f"{rejection_rate:.0%} rechazado")
            else:
                st.metric("MA200", "No usado", delta_color="off")
        
        with col5:
            st.metric("Portfolio final", len(portfolio))
    
    except Exception as e:
        st.error(f"❌ Error en pipeline: {str(e)}")
        import traceback
        with st.expander("🐛 Debug info"):
            st.code(traceback.format_exc())
        st.stop()


# ============================================================================
# TAB 1: PIPELINE OVERVIEW
# ============================================================================

with tab1:
    st.markdown("### 📊 Pipeline Overview")
    
    if 'portfolio' not in st.session_state:
        st.info("👆 Configura parámetros en sidebar y haz click en 'Ejecutar Pipeline'")
    else:
        # Funnel visualization
        st.markdown("#### 🔍 Funnel de Selección")
        
        funnel_data = pd.DataFrame({
            'Stage': [
                '1. Universo inicial',
                '2. Filtros calidad',
                '3. F-Score filter',
                '4. Precios válidos',
                '5. MA200 + Momentum',
                '6. Portfolio final'
            ],
            'Count': [
                len(st.session_state.get('universe', [])),
                len(st.session_state.get('passed_filters', [])),
                len(st.session_state.get('fscore_passed', [])),
                len(st.session_state.get('prices_dict', {})),
                len(st.session_state.get('after_momentum', [])),
                len(st.session_state.get('portfolio', []))
            ]
        })
        
        chart = alt.Chart(funnel_data).mark_bar().encode(
            x=alt.X('Count:Q', title='Número de Símbolos'),
            y=alt.Y('Stage:N', sort=None, title=''),
            color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues'), legend=None)
        ).properties(height=300)
        
        st.altair_chart(chart, use_container_width=True)
        
        # Tabla de diagnostics
        st.markdown("#### 🛡️ Diagnóstico de Filtros")
        
        if 'diagnostics' in st.session_state:
            diag = st.session_state['diagnostics']
            
            # Contar rechazos
            total = len(diag)
            passed = diag['pass_all'].sum()
            rejected = total - passed
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total evaluados", total)
            with col2:
                st.metric("Pasaron todos", passed, delta=f"{passed/total*100:.1f}%")
            with col3:
                st.metric("Rechazados", rejected, delta=f"-{rejected/total*100:.1f}%", delta_color="inverse")
            
            # Razones de rechazo
            if rejected > 0:
                rejection_reasons = diag[~diag['pass_all']]['reason'].value_counts()
                
                st.markdown("**Top razones de rechazo:**")
                for reason, count in rejection_reasons.head(5).items():
                    st.write(f"- {reason}: {count} ({count/rejected*100:.1f}%)")


# ============================================================================
# TAB 2: QVM RANKINGS
# ============================================================================

with tab2:
    st.markdown("### 💎 QVM Rankings")
    
    if 'portfolio' not in st.session_state:
        st.info("Ejecuta el pipeline primero")
    else:
        portfolio = st.session_state['portfolio']
        
        # Métricas generales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Portfolio Size", len(portfolio))
        with col2:
            st.metric("Avg QVM Score", f"{portfolio['qvm_score_corrected'].mean():.3f}")
        with col3:
            st.metric("Avg F-Score", f"{portfolio['fscore'].mean():.1f}/9.0")
        with col4:
            st.metric("Sectores", portfolio['sector'].nunique())
        
        st.markdown("---")
        
        # Tabla principal
        display_cols = [
            'symbol', 'sector', 'qvm_score_corrected', 'final_rank',
            'quality_extended', 'value_score', 'composite_momentum', 'fscore',
            'momentum_12m1m', 'above_ma200', 'roe', 'market_cap'
        ]
        
        display_df = portfolio[display_cols].copy()
        display_df = display_df.rename(columns={
            'qvm_score_corrected': 'QVM Score',
            'final_rank': 'Rank %ile',
            'quality_extended': 'Quality',
            'value_score': 'Value',
            'composite_momentum': 'Momentum',
            'fscore': 'F-Score',
            'momentum_12m1m': 'Mom 12M',
            'above_ma200': 'MA200✓',
            'market_cap': 'MCap'
        })
        
        # Format
        display_df['QVM Score'] = display_df['QVM Score'].round(3)
        display_df['Rank %ile'] = (display_df['Rank %ile'] * 100).round(1)
        display_df['Quality'] = display_df['Quality'].round(3)
        display_df['Value'] = display_df['Value'].round(3)
        display_df['Momentum'] = display_df['Momentum'].round(3)
        display_df['F-Score'] = display_df['F-Score'].round(1)
        display_df['Mom 12M'] = (display_df['Mom 12M'] * 100).round(1)
        display_df['roe'] = (display_df['roe'] * 100).round(1)
        display_df['MCap'] = (display_df['MCap'] / 1e9).round(2)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=600,
            column_config={
                'MA200✓': st.column_config.CheckboxColumn(),
                'Rank %ile': st.column_config.ProgressColumn(min_value=0, max_value=100),
                'Mom 12M': st.column_config.NumberColumn(format="%.1f%%"),
                'roe': st.column_config.NumberColumn(format="%.1f%%"),
                'MCap': st.column_config.NumberColumn(format="%.2fB"),
            }
        )
        
        # Distribución por sector
        st.markdown("#### 📊 Distribución por Sector")
        
        sector_dist = portfolio.groupby('sector').agg({
            'symbol': 'count',
            'qvm_score_corrected': 'mean',
            'fscore': 'mean',
            'momentum_12m1m': 'mean'
        }).reset_index()
        
        sector_dist.columns = ['Sector', 'Count', 'Avg QVM', 'Avg F-Score', 'Avg Mom']
        sector_dist = sector_dist.sort_values('Count', ascending=False)
        
        st.dataframe(sector_dist, use_container_width=True)


# ============================================================================
# TAB 3: MOMENTUM + MA200
# ============================================================================

with tab3:
    st.markdown("### 🎯 Momentum + MA200 Analysis")
    
    if 'momentum_df' not in st.session_state:
        st.info("Ejecuta el pipeline primero")
    else:
        momentum_df = st.session_state['momentum_df']
        
        # Métricas globales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            n_positive = (momentum_df['momentum_12m1m'] > 0).sum()
            st.metric(
                "Momentum Positivo", 
                f"{n_positive}/{len(momentum_df)}",
                delta=f"{n_positive/len(momentum_df)*100:.1f}%"
            )
        
        with col2:
            n_above = momentum_df['above_ma200'].sum()
            st.metric(
                "Arriba de MA200",
                f"{n_above}/{len(momentum_df)}",
                delta=f"{n_above/len(momentum_df)*100:.1f}%"
            )
        
        with col3:
            avg_mom = momentum_df['momentum_12m1m'].mean()
            st.metric(
                "Momentum Promedio",
                f"{avg_mom:.1%}"
            )
        
        with col4:
            n_both = ((momentum_df['momentum_12m1m'] > 0) & momentum_df['above_ma200']).sum()
            st.metric(
                "Ambos ✓",
                f"{n_both}/{len(momentum_df)}",
                delta=f"{n_both/len(momentum_df)*100:.1f}%"
            )
        
        st.markdown("---")
        
        # Scatter plot: Momentum vs Trend
        st.markdown("#### 📊 Momentum vs MA200 Status")
        
        plot_df = momentum_df.copy()
        plot_df['MA200_status'] = plot_df['above_ma200'].map({True: 'Above MA200', False: 'Below MA200'})
        
        scatter = alt.Chart(plot_df).mark_circle(size=60).encode(
            x=alt.X('momentum_12m1m:Q', title='12M-1M Momentum', scale=alt.Scale(domain=[-0.5, 1.0])),
            y=alt.Y('composite_momentum:Q', title='Composite Momentum Score'),
            color=alt.Color('MA200_status:N', title='MA200 Status', scale=alt.Scale(scheme='category10')),
            tooltip=['symbol', 'momentum_12m1m', 'above_ma200', 'composite_momentum']
        ).properties(height=400)
        
        st.altair_chart(scatter, use_container_width=True)
        
        # Tabla detallada
        st.markdown("#### 📋 Momentum Detail Table")
        
        momentum_display = momentum_df[['symbol', 'momentum_12m1m', 'momentum_6m', 
                                         'above_ma200', 'ma200', 'trend_strength']].copy()
        
        momentum_display['momentum_12m1m'] = (momentum_display['momentum_12m1m'] * 100).round(1)
        momentum_display['momentum_6m'] = (momentum_display['momentum_6m'] * 100).round(1)
        momentum_display['ma200'] = momentum_display['ma200'].round(2)
        momentum_display['trend_strength'] = (momentum_display['trend_strength'] * 100).round(0)
        
        st.dataframe(
            momentum_display,
            use_container_width=True,
            height=400,
            column_config={
                'momentum_12m1m': st.column_config.NumberColumn(format="%.1f%%"),
                'momentum_6m': st.column_config.NumberColumn(format="%.1f%%"),
                'above_ma200': st.column_config.CheckboxColumn(),
                'trend_strength': st.column_config.ProgressColumn(min_value=0, max_value=100),
            }
        )


# ============================================================================
# TAB 4: F-SCORE ANALYSIS
# ============================================================================

with tab4:
    st.markdown("### 🏆 F-Score Analysis")
    
    if 'fscore_data' not in st.session_state:
        st.info("Ejecuta el pipeline con F-Score habilitado")
    else:
        fscore_data = st.session_state['fscore_data']
        
        # Distribución de F-Score
        st.markdown("#### 📊 F-Score Distribution")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_fscore = fscore_data['fscore'].mean()
            st.metric("F-Score Promedio", f"{avg_fscore:.1f}/9.0")
        
        with col2:
            high_quality = (fscore_data['fscore'] >= 8).sum()
            st.metric("High Quality (8-9)", high_quality)
        
        with col3:
            medium_quality = ((fscore_data['fscore'] >= 6) & (fscore_data['fscore'] < 8)).sum()
            st.metric("Medium Quality (6-7)", medium_quality)
        
        with col4:
            low_quality = (fscore_data['fscore'] < 6).sum()
            st.metric("Low Quality (0-5)", low_quality)
        
        # Histograma
        hist_df = fscore_data['fscore'].value_counts().reset_index()
        hist_df.columns = ['F-Score', 'Count']
        
        hist_chart = alt.Chart(hist_df).mark_bar().encode(
            x=alt.X('F-Score:O', title='F-Score'),
            y=alt.Y('Count:Q', title='Number of Stocks'),
            color=alt.condition(
                alt.datum['F-Score'] >= 6,
                alt.value('steelblue'),
                alt.value('lightgray')
            )
        ).properties(height=300)
        
        st.altair_chart(hist_chart, use_container_width=True)
        
        # Tabla por categoría
        st.markdown("#### 📋 Stocks by F-Score Category")
        
        fscore_display = fscore_data[['symbol', 'fscore', 'roe', 'fcf', 'operating_cf']].copy()
        fscore_display['category'] = pd.cut(
            fscore_display['fscore'],
            bins=[0, 5.5, 7.5, 9],
            labels=['Low (0-5)', 'Medium (6-7)', 'High (8-9)']
        )
        
        fscore_display = fscore_display.sort_values('fscore', ascending=False)
        
        st.dataframe(fscore_display, use_container_width=True, height=400)


# ============================================================================
# TAB 5: BACKTEST
# ============================================================================

with tab5:
    st.markdown("### 📈 Backtest Results")
    
    if 'portfolio_prices' not in st.session_state:
        st.info("Ejecuta el pipeline primero")
    else:
        if st.button("🚀 Ejecutar Backtest", type="primary"):
            with st.spinner("Ejecutando backtest..."):
                
                portfolio_prices = st.session_state['portfolio_prices']
                
                # Costos de trading
                costs = TradingCosts(
                    commission_bps=5,
                    slippage_bps=5,
                    market_impact_bps=2,
                )
                
                # Backtest
                metrics, equity_curves = backtest_portfolio(
                    portfolio_prices,
                    costs=costs,
                    execution_lag_days=1,
                )
                
                # Portfolio metrics
                port_metrics = calculate_portfolio_metrics(equity_curves, costs)
                
                st.session_state['backtest_metrics'] = metrics
                st.session_state['portfolio_metrics'] = port_metrics
                st.session_state['equity_curves'] = equity_curves
        
        if 'portfolio_metrics' in st.session_state:
            port_metrics = st.session_state['portfolio_metrics']
            
            # Métricas principales
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("CAGR", f"{port_metrics['CAGR']:.2%}")
            with col2:
                st.metric("Sharpe Ratio", f"{port_metrics['Sharpe']:.2f}")
            with col3:
                st.metric("Sortino Ratio", f"{port_metrics['Sortino']:.2f}")
            with col4:
                st.metric("Max Drawdown", f"{port_metrics['MaxDD']:.2%}")
            with col5:
                st.metric("Calmar Ratio", f"{port_metrics['Calmar']:.2f}")
            
            st.markdown("---")
            
            # Equity curve
            st.markdown("#### 📈 Portfolio Equity Curve")
            
            equity_curves = st.session_state['equity_curves']
            
            # Combinar curvas
            equity_df = pd.DataFrame(equity_curves)
            portfolio_equity = equity_df.mean(axis=1)
            
            equity_plot_df = pd.DataFrame({
                'Date': portfolio_equity.index,
                'Equity': portfolio_equity.values
            })
            
            equity_chart = alt.Chart(equity_plot_df).mark_line(color='steelblue').encode(
                x='Date:T',
                y=alt.Y('Equity:Q', title='Portfolio Value ($)'),
                tooltip=['Date:T', 'Equity:Q']
            ).properties(height=400)
            
            st.altair_chart(equity_chart, use_container_width=True)
            
            # Tabla de métricas individuales
            st.markdown("#### 📊 Individual Stock Performance")
            
            backtest_metrics = st.session_state['backtest_metrics']
            st.dataframe(
                backtest_metrics,
                use_container_width=True,
                height=400
            )


# ============================================================================
# TAB 6: EXPORT
# ============================================================================

with tab6:
    st.markdown("### 💾 Export Data")
    
    if 'portfolio' not in st.session_state:
        st.info("Ejecuta el pipeline primero")
    else:
        st.markdown("#### 📋 Portfolio Export")
        
        portfolio = st.session_state['portfolio']
        
        # CSV export
        csv = portfolio.to_csv(index=False)
        
        st.download_button(
            label="⬇️ Descargar Portfolio (CSV)",
            data=csv,
            file_name=f"qvm_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Preview
        st.markdown("**Preview:**")
        st.dataframe(portfolio.head(10), use_container_width=True)
        
        st.markdown("---")
        
        # Backtest metrics export
        if 'backtest_metrics' in st.session_state:
            st.markdown("#### 📈 Backtest Metrics Export")
            
            metrics = st.session_state['backtest_metrics']
            metrics_csv = metrics.to_csv(index=False)
            
            st.download_button(
                label="⬇️ Descargar Backtest Metrics (CSV)",
                data=metrics_csv,
                file_name=f"qvm_backtest_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("""
**QVM Screener V2** | Implementación académica con Momentum Real + MA200 + F-Score  
Basado en: Jegadeesh & Titman (1993), Faber (2007), Piotroski (2000), Asness et al. (2019)
""")
