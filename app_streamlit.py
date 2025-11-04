"""
QVM Screener App V2 - Con Piotroski Real y Parámetros Ajustables
================================================================

Interfaz Streamlit con:
✅ Piotroski Score real (9 checks completos)
✅ Quality-Value Score sin multicolinealidad
✅ Sliders para ajustar TODOS los parámetros
✅ Análisis por pasos con checks y visualizaciones
✅ Tablas interactivas y exportables
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Imports del pipeline
from qvm_pipeline_v2 import (
    run_qvm_pipeline_v2,
    QVMConfig,
    analyze_portfolio_v2,
)


# ============================================================================
# CONFIGURACIÓN DE LA APP
# ============================================================================

st.set_page_config(
    page_title="QVM Screener V2 - Piotroski Real",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    .stAlert { padding: 0.6rem 1rem; margin-bottom: 0.5rem; }
    h1 { font-size: 2.5rem !important; letter-spacing: -0.5px; }
    h2 { font-size: 1.8rem !important; margin-top: 1.5rem; }
    h3 { font-size: 1.4rem !important; }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HEADER
# ============================================================================

st.title("🎯 QVM Screener V2")
st.markdown("**Quality-Value Momentum Strategy** con Piotroski Score Real y Zero Multicolinealidad")

st.divider()


# ============================================================================
# SIDEBAR - PARÁMETROS AJUSTABLES
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuración")

    st.subheader("📊 Universo")

    universe_size = st.slider(
        "Tamaño del universo",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Número de stocks a considerar inicialmente"
    )

    min_market_cap = st.slider(
        "Market Cap mínimo ($B)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Market cap mínimo en miles de millones"
    )

    min_volume = st.slider(
        "Volumen diario mínimo (K)",
        min_value=100,
        max_value=2000,
        value=500,
        step=100,
        help="Volumen diario mínimo en miles de acciones"
    )

    st.divider()

    st.subheader("🎯 Quality-Value Weights")

    st.info("Los pesos se normalizarán automáticamente para sumar 100%")

    w_quality = st.slider(
        "📈 Quality (Piotroski)",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.05,
        help="Peso del Piotroski Score"
    )

    w_value = st.slider(
        "💰 Value (Multiples)",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help="Peso de los múltiplos de valoración"
    )

    w_fcf_yield = st.slider(
        "💵 FCF Yield",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.05,
        help="Peso del FCF Yield"
    )

    w_momentum = st.slider(
        "🚀 Momentum",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.05,
        help="Peso del momentum (actualmente placeholder)"
    )

    # Mostrar suma de pesos
    total_weight = w_quality + w_value + w_fcf_yield + w_momentum
    st.caption(f"Total: {total_weight:.2f} (se normalizará a 1.0)")

    st.divider()

    st.subheader("🔍 Filtros")

    min_piotroski = st.slider(
        "Piotroski Score mínimo",
        min_value=0,
        max_value=9,
        value=5,
        step=1,
        help="Mínimo Piotroski Score (0-9). Recomendado: 6+"
    )

    min_qv_score = st.slider(
        "QV Score mínimo",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.05,
        help="Mínimo Quality-Value Score (0-1). Recomendado: 0.5+"
    )

    max_pe = st.slider(
        "P/E máximo",
        min_value=10.0,
        max_value=100.0,
        value=50.0,
        step=5.0,
        help="P/E máximo permitido"
    )

    max_ev_ebitda = st.slider(
        "EV/EBITDA máximo",
        min_value=5.0,
        max_value=50.0,
        value=25.0,
        step=5.0,
        help="EV/EBITDA máximo permitido"
    )

    require_positive_fcf = st.checkbox(
        "Requerir FCF positivo",
        value=True,
        help="Solo incluir empresas con Free Cash Flow positivo"
    )

    st.divider()

    st.subheader("📋 Portfolio")

    portfolio_size = st.slider(
        "Tamaño del portfolio",
        min_value=10,
        max_value=50,
        value=30,
        step=5,
        help="Número de stocks en el portfolio final"
    )

    st.divider()

    # Botón de ejecución
    run_button = st.button("🚀 Ejecutar Screening", type="primary", use_container_width=True)

    # Botón para limpiar caché
    if st.button("🗑️ Limpiar Caché", use_container_width=True):
        st.cache_data.clear()
        st.success("Caché limpiado!")


# ============================================================================
# CREAR CONFIGURACIÓN
# ============================================================================

config = QVMConfig(
    universe_size=universe_size,
    min_market_cap=min_market_cap * 1e9,  # Convertir a dólares
    min_volume=min_volume * 1000,          # Convertir a unidades
    w_quality=w_quality,
    w_value=w_value,
    w_fcf_yield=w_fcf_yield,
    w_momentum=w_momentum,
    min_piotroski_score=min_piotroski,
    min_qv_score=min_qv_score,
    max_pe=max_pe,
    max_ev_ebitda=max_ev_ebitda,
    require_positive_fcf=require_positive_fcf,
    portfolio_size=portfolio_size,
)


# ============================================================================
# MAIN APP
# ============================================================================

# Mostrar configuración actual
with st.expander("📋 Ver Configuración Completa", expanded=False):
    config_df = pd.DataFrame([config.to_dict()]).T
    config_df.columns = ['Valor']
    st.dataframe(config_df, use_container_width=True)


# Ejecutar pipeline
if run_button or st.session_state.get('results') is not None:

    if run_button:
        # Limpiar resultados anteriores
        st.session_state.results = None

    if st.session_state.get('results') is None:
        with st.spinner("🔄 Ejecutando pipeline..."):
            try:
                results = run_qvm_pipeline_v2(config=config, verbose=False)
                st.session_state.results = results
            except Exception as e:
                st.error(f"❌ Error al ejecutar pipeline: {str(e)}")
                st.stop()

    results = st.session_state.results

    # Verificar si hubo error
    if 'error' in results:
        st.error(f"❌ Pipeline failed: {results['error']}")
        st.stop()

    # ========================================================================
    # RESULTADOS
    # ========================================================================

    st.success(f"✅ Pipeline completado exitosamente!")

    # ------------------------------------------------------------------------
    # ANÁLISIS POR PASOS
    # ------------------------------------------------------------------------
    st.header("📊 Análisis por Pasos")

    steps = results.get('steps', [])

    if steps:
        # Crear visualización de funnel
        funnel_data = []
        for i, step in enumerate(steps):
            funnel_data.append({
                'Step': f"{step.name}\n{step.description}",
                'Count': step.output_count,
                'Stage': i + 1
            })

        funnel_df = pd.DataFrame(funnel_data)

        # Gráfico de funnel
        fig_funnel = px.funnel(
            funnel_df,
            x='Count',
            y='Step',
            title="Pipeline Funnel - Stocks en Cada Paso",
            labels={'Count': 'Número de Stocks'},
        )
        fig_funnel.update_layout(height=400)
        st.plotly_chart(fig_funnel, use_container_width=True)

        # Tabla de detalles por paso
        st.subheader("Detalles por Paso")

        for step in steps:
            status_emoji = "✅" if step.success else "❌"
            pass_rate = step.get_pass_rate() * 100

            with st.expander(f"{status_emoji} {step.name}: {step.output_count} stocks ({pass_rate:.1f}%)"):
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Input", step.input_count)
                    st.metric("Output", step.output_count)

                with col2:
                    st.metric("Pass Rate", f"{pass_rate:.1f}%")
                    st.metric("Rejected", step.input_count - step.output_count)

                # Métricas adicionales
                if step.metrics:
                    st.markdown("**📊 Métricas:**")
                    metrics_df = pd.DataFrame([step.metrics]).T
                    metrics_df.columns = ['Valor']
                    st.dataframe(metrics_df, use_container_width=True)

                # Warnings
                if step.warnings:
                    st.warning("⚠️ Warnings:")
                    for warning in step.warnings:
                        st.write(f"- {warning}")

    st.divider()

    # ------------------------------------------------------------------------
    # PORTFOLIO FINAL
    # ------------------------------------------------------------------------
    st.header("📋 Portfolio Final")

    portfolio = results.get('portfolio')

    if portfolio is not None and not portfolio.empty:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Stocks Seleccionados",
                len(portfolio)
            )

        with col2:
            avg_piotroski = portfolio['piotroski_score'].mean()
            st.metric(
                "Piotroski Promedio",
                f"{avg_piotroski:.1f}/9"
            )

        with col3:
            avg_qv = portfolio['qv_score'].mean()
            st.metric(
                "QV Score Promedio",
                f"{avg_qv:.3f}"
            )

        with col4:
            n_sectors = portfolio['sector'].nunique()
            st.metric(
                "Sectores Únicos",
                n_sectors
            )

        st.divider()

        # Análisis detallado
        st.subheader("🏆 Top Stocks")

        analysis = analyze_portfolio_v2(results, n_top=portfolio_size)

        if not analysis.empty:
            # Formatear para mostrar
            display_df = analysis.copy()

            # Formatear columnas numéricas
            for col in display_df.columns:
                if 'score' in col.lower() or 'component' in col.lower():
                    if display_df[col].dtype in [float, np.float64]:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
                elif col in ['roic', 'roe', 'gross_margin', 'fcf_yield']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
                elif col in ['pe', 'pb', 'ev_ebitda']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")

            st.dataframe(
                display_df,
                use_container_width=True,
                height=600
            )

            # Botón de descarga
            csv = analysis.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Portfolio (CSV)",
                data=csv,
                file_name=f"qvm_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

        st.divider()

        # ------------------------------------------------------------------------
        # VISUALIZACIONES
        # ------------------------------------------------------------------------
        st.header("📈 Visualizaciones")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Score Distribution",
            "Sector Analysis",
            "Piotroski Components",
            "Valuation Metrics"
        ])

        with tab1:
            st.subheader("Distribución de QV Score")

            fig_score = px.histogram(
                portfolio,
                x='qv_score',
                nbins=20,
                title="Distribución de Quality-Value Score",
                labels={'qv_score': 'QV Score', 'count': 'Frecuencia'},
            )
            st.plotly_chart(fig_score, use_container_width=True)

            # Scatter plot: Quality vs Value
            st.subheader("Quality vs Value")

            fig_scatter = px.scatter(
                portfolio,
                x='value_score_component',
                y='quality_score_component',
                size='market_cap',
                color='sector',
                hover_data=['symbol', 'piotroski_score', 'qv_score'],
                title="Quality Score vs Value Score",
                labels={
                    'value_score_component': 'Value Score',
                    'quality_score_component': 'Quality Score (Piotroski Normalizado)'
                }
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with tab2:
            st.subheader("Distribución por Sector")

            sector_counts = portfolio['sector'].value_counts()

            fig_sector = px.pie(
                values=sector_counts.values,
                names=sector_counts.index,
                title="Portfolio por Sector",
            )
            st.plotly_chart(fig_sector, use_container_width=True)

            # Piotroski por sector
            st.subheader("Piotroski Score por Sector")

            fig_box = px.box(
                portfolio,
                x='sector',
                y='piotroski_score',
                title="Distribución de Piotroski Score por Sector",
                labels={'piotroski_score': 'Piotroski Score', 'sector': 'Sector'}
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with tab3:
            st.subheader("Piotroski Score Distribution")

            piotroski_counts = portfolio['piotroski_score'].value_counts().sort_index()

            fig_piotroski = px.bar(
                x=piotroski_counts.index,
                y=piotroski_counts.values,
                title="Distribución de Piotroski Score",
                labels={'x': 'Piotroski Score', 'y': 'Cantidad'},
            )
            st.plotly_chart(fig_piotroski, use_container_width=True)

            # Mostrar estadísticas de componentes si están disponibles
            piotroski_cols = [c for c in portfolio.columns if 'positive' in c or 'delta_' in c or 'accruals' in c]

            if piotroski_cols:
                st.subheader("Piotroski Components Analysis")

                components_summary = portfolio[piotroski_cols].mean().sort_values(ascending=False)

                fig_components = px.bar(
                    x=components_summary.values * 100,
                    y=components_summary.index,
                    orientation='h',
                    title="% de Stocks que Pasan cada Check de Piotroski",
                    labels={'x': '% Pass Rate', 'y': 'Component'}
                )
                st.plotly_chart(fig_components, use_container_width=True)

        with tab4:
            st.subheader("Métricas de Valoración")

            valuation_cols = []
            for col in ['pe', 'pb', 'ev_ebitda']:
                if col in portfolio.columns:
                    valuation_cols.append(col)

            if valuation_cols:
                col1, col2 = st.columns(2)

                for i, col in enumerate(valuation_cols):
                    with col1 if i % 2 == 0 else col2:
                        fig = px.histogram(
                            portfolio,
                            x=col,
                            title=f"Distribución de {col.upper()}",
                            labels={col: col.upper(), 'count': 'Frecuencia'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No hay stocks en el portfolio final.")

else:
    # Instrucciones iniciales
    st.info("""
    👈 **Ajusta los parámetros en la barra lateral y presiona "Ejecutar Screening"**

    **Guía de Parámetros:**

    - **Quality (Piotroski)**: Mide salud financiera operacional (9 checks)
    - **Value (Multiples)**: Mide cuán barato está el stock (EV/EBITDA, P/E, P/B)
    - **FCF Yield**: Free Cash Flow / Market Cap (mayor es mejor)
    - **Momentum**: Retornos históricos (actualmente placeholder)

    **Piotroski Score:**
    - 8-9: Excelente calidad (STRONG BUY)
    - 6-7: Buena calidad (BUY)
    - 4-5: Calidad media (HOLD)
    - 0-3: Baja calidad (AVOID)

    **QV Score:**
    - > 0.70: Muy atractivo (STRONG BUY)
    - 0.50-0.70: Atractivo (BUY)
    - 0.30-0.50: Neutral (HOLD)
    - < 0.30: No atractivo (AVOID)
    """)


# ============================================================================
# FOOTER
# ============================================================================

st.divider()

st.caption("""
**QVM Screener V2** | Powered by Financial Modeling Prep API
- Piotroski (2000): Value Investing
- Asness et al. (2019): Quality Minus Junk
- Fama & French (1992, 2015): Multi-factor Models
""")
