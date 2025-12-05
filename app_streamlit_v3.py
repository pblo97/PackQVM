"""
QVM Screener App V3 - COMPLETO con MA200, Backtest y Métricas Avanzadas
========================================================================

Interfaz Streamlit con:
✅ Piotroski Score real (9 checks completos)
✅ Quality-Value Score sin multicolinealidad
✅ MA200 Filter (Faber 2007)
✅ Momentum 12M-1M
✅ Filtros 52w high y volumen relativo
✅ Métricas avanzadas: EBIT/EV, FCF/EV, ROIC>WACC
✅ Backtest integrado con métricas de performance
✅ Sliders para ajustar TODOS los parámetros
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Imports del pipeline V3
from qvm_pipeline_v3 import (
    run_qvm_pipeline_v3,
    QVMConfigV3,
    analyze_portfolio_v3,
)


# ============================================================================
# CONFIGURACIÓN DE LA APP
# ============================================================================

st.set_page_config(
    page_title="QVM Screener V3 - Completo",
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

st.title("🎯 QVM Screener V3 - COMPLETO")
st.markdown("**Quality-Value Momentum Strategy** con MA200, Backtest y Métricas Avanzadas")

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
        max_value=1000,
        value=800,
        step=50,
        help="Número de stocks a considerar inicialmente (ampliado a 1000)"
    )

    min_market_cap = st.slider(
        "Market Cap mínimo ($B)",
        min_value=0.5,
        max_value=50.0,
        value=5.0,
        step=0.5,
        help="Market cap mínimo en miles de millones. Recomendado: 5B+ para evitar micro-caps"
    )

    # Opción rápida: Large Cap only
    large_cap_only = st.checkbox(
        "🏢 Solo Large Caps ($10B+)",
        value=False,
        help="Filtrar solo empresas grandes (market cap >= $10B)"
    )

    if large_cap_only:
        min_market_cap = max(min_market_cap, 10.0)
        st.info(f"✅ Filtro Large Cap activo: Min Market Cap ajustado a ${min_market_cap:.1f}B")

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
        value=0.35,  # Reducido de 0.40 para evitar overlap con FCF
        step=0.05,
        help="Peso del Piotroski Score"
    )

    w_value = st.slider(
        "💰 Value (Multiples)",
        min_value=0.0,
        max_value=1.0,
        value=0.40,  # Aumentado de 0.35 (mayor peso en valoración)
        step=0.05,
        help="Peso de los múltiplos de valoración (EV/EBITDA, P/B, P/E)"
    )

    w_fcf_yield = st.slider(
        "💵 FCF Yield",
        min_value=0.0,
        max_value=1.0,
        value=0.10,  # Reducido de 0.15 para minimizar overlap con Piotroski
        step=0.05,
        help="Peso del FCF Yield (tiene overlap parcial con Piotroski CFO)"
    )

    w_momentum = st.slider(
        "🚀 Momentum",
        min_value=0.0,
        max_value=1.0,
        value=0.15,  # Aumentado de 0.10 según Jegadeesh & Titman (1993)
        step=0.05,
        help="Peso del momentum 12M-1M (Jegadeesh & Titman 1993)"
    )

    # Mostrar suma de pesos
    total_weight = w_quality + w_value + w_fcf_yield + w_momentum
    st.caption(f"Total: {total_weight:.2f} (se normalizará a 1.0)")

    st.divider()

    st.subheader("🔍 Filtros Básicos")

    col1, col2 = st.columns(2)

    with col1:
        min_piotroski = st.slider(
            "Piotroski Score mínimo",
            min_value=0,
            max_value=9,
            value=7,
            step=1,
            help="Mínimo Piotroski Score (0-9). 7+ = alta calidad"
        )

    with col2:
        high_quality_only = st.checkbox(
            "✨ Solo Alta Calidad",
            value=False,
            help="Piotroski >= 8 (empresas excelentes)"
        )
        if high_quality_only:
            min_piotroski = max(min_piotroski, 8)

    min_qv_score = st.slider(
        "QV Score mínimo",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="Mínimo Quality-Value Score (0-1). 0.5+ recomendado"
    )

    max_pe = st.slider(
        "P/E máximo",
        min_value=10.0,
        max_value=100.0,
        value=40.0,
        step=5.0,
        help="P/E máximo permitido"
    )

    max_ev_ebitda = st.slider(
        "EV/EBITDA máximo",
        min_value=5.0,
        max_value=50.0,
        value=20.0,
        step=5.0,
        help="EV/EBITDA máximo permitido"
    )

    require_positive_fcf = st.checkbox(
        "Requerir FCF positivo",
        value=True,
        help="Solo incluir empresas con Free Cash Flow positivo"
    )

    min_roic = st.slider(
        "ROIC mínimo (%)",
        min_value=0,
        max_value=50,
        value=10,
        step=5,
        help="ROIC mínimo requerido (Return on Invested Capital)"
    ) / 100.0  # Convertir de % a decimal

    st.divider()

    st.subheader("🚀 Filtros Avanzados (NUEVO)")

    # MA200 Filter
    require_above_ma200 = st.checkbox(
        "✅ Filtro MA200 (Faber 2007)",
        value=True,
        help="Solo incluir stocks por encima de su MA de 200 días"
    )

    # Momentum
    min_momentum_12m = st.slider(
        "Momentum 12M mínimo (%)",
        min_value=-20,
        max_value=50,
        value=10,
        step=5,
        help="Retorno mínimo 12M-1M (Jegadeesh & Titman 1993)"
    ) / 100.0  # Convertir de % a decimal

    # 52-Week High
    require_near_52w_high = st.checkbox(
        "Filtro 52w High",
        value=False,
        help="Solo stocks cerca de su máximo de 52 semanas"
    )

    min_pct_from_52w_high = st.slider(
        "% mínimo del 52w high",
        min_value=50,
        max_value=100,
        value=80,
        step=5,
        help="Precio actual >= X% del máximo de 52 semanas",
        disabled=not require_near_52w_high
    ) / 100.0  # Convertir de % a decimal

    # ROIC > WACC
    require_roic_above_wacc = st.checkbox(
        "✅ Requerir ROIC > WACC",
        value=True,
        help="Solo empresas que crean valor (ROIC > Costo de Capital)"
    )

    st.divider()

    st.subheader("⚡ Breakouts y Volumen (NUEVO)")

    st.info("Filtros de momentum técnico: breakouts de niveles previos confirmados con volumen")

    # Breakout filters
    enable_breakout_filter = st.checkbox(
        "🚀 Filtro de Breakout",
        value=False,
        help="Requiere breakout de niveles técnicos (52w, 3M o 20D)"
    )

    require_breakout_confirmed = st.checkbox(
        "✅ Solo Breakouts Confirmados",
        value=False,
        help="Breakout + volumen >1.5x promedio (señal más fuerte)",
        disabled=not enable_breakout_filter
    )

    require_breakout_strong = st.checkbox(
        "💪 Solo Breakouts Fuertes",
        value=False,
        help="Breakout + volumen >2x promedio (señal muy fuerte)",
        disabled=not enable_breakout_filter
    )

    # Volume filters
    enable_volume_surge_filter = st.checkbox(
        "📊 Filtro de Surge de Volumen",
        value=False,
        help="Requiere volumen >2x promedio (sin breakout necesario)"
    )

    with st.expander("ℹ️ Sobre Breakouts y Volumen"):
        st.markdown("""
        **Tipos de Breakout:**
        - **52w High**: Precio rompe máximo de 52 semanas
        - **3M High**: Precio rompe máximo de 3 meses (60 días)
        - **20D High**: Precio rompe máximo de 20 días (consolidación)

        **Confirmación con Volumen:**
        - **Confirmado**: Breakout + volumen >1.5x promedio
        - **Fuerte**: Breakout + volumen >2x promedio

        **Literatura:**
        - George & Hwang (2004): "52-week high momentum"
        - Lee & Swaminathan (2000): "Price momentum and trading volume"
        """)

    st.divider()

    st.subheader("💎 Métricas Avanzadas Valoración")

    min_ebit_ev = st.slider(
        "EBIT/EV mínimo",
        min_value=0.0,
        max_value=0.20,
        value=0.08,
        step=0.01,
        format="%.2f",
        help="Earnings Yield mínimo (EBIT/Enterprise Value)"
    )

    max_fcf_ev = st.slider(
        "FCF/EV máximo",
        min_value=0.05,
        max_value=0.30,
        value=0.15,
        step=0.01,
        format="%.2f",
        help="Free Cash Flow Yield máximo permitido"
    )

    st.divider()

    st.subheader("🎓 Mejoras Académicas V3.1 (NUEVO)")

    st.info("Funcionalidades avanzadas basadas en literatura académica reciente")

    enable_earnings_quality = st.checkbox(
        "✅ Earnings Quality Filter (Sloan 1996)",
        value=True,
        help="Filtra empresas con accruals altos (posible manipulación de earnings)"
    )

    enable_red_flags = st.checkbox(
        "🚩 Red Flags Detection",
        value=True,
        help="Detecta dilución excesiva, pérdidas recurrentes, etc."
    )

    enable_reversal_filter = st.checkbox(
        "📉 Short-Term Reversal Filter (Jegadeesh 1990)",
        value=True,
        help="Evita stocks que cayeron >8% last week (mean reversion)"
    )

    use_enhanced_value_score = st.checkbox(
        "💎 Enhanced Value Score (7 métricas vs 3)",
        value=True,
        help="Usa EV/EBITDA, EV/EBIT, EV/FCF, P/B, P/E, P/Sales, Shareholder Yield"
    )

    # Opcionales (más avanzados) - Definir defaults primero
    enable_fundamental_momentum = False
    enable_sector_relative = False

    with st.expander("⚙️ Opcionales (Avanzado)"):
        enable_fundamental_momentum = st.checkbox(
            "📈 Fundamental Momentum (Piotroski & So 2012)",
            value=False,
            help="Requiere datos históricos multi-year (puede ser lento)"
        )

        enable_sector_relative = st.checkbox(
            "🎯 Sector Relative Momentum",
            value=False,
            help="Solo selecciona stocks que outperforman su sector"
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

    st.subheader("📊 Backtest")

    backtest_enabled = st.checkbox(
        "✅ Ejecutar Backtest",
        value=True,
        help="Backtest histórico del portfolio"
    )

    backtest_start = st.date_input(
        "Fecha inicio",
        value=datetime(2020, 1, 1),
        help="Fecha de inicio del backtest"
    )

    backtest_end = st.date_input(
        "Fecha fin",
        value=datetime(2024, 12, 31),
        help="Fecha de fin del backtest"
    )

    rebalance_freq = st.selectbox(
        "Frecuencia de rebalanceo",
        options=["Q", "M", "Y"],
        index=0,
        format_func=lambda x: {"Q": "Trimestral", "M": "Mensual", "Y": "Anual"}[x],
        help="Con qué frecuencia rebalancear el portfolio"
    )

    with st.expander("⚙️ Costos de Trading"):
        commission_bps = st.number_input(
            "Comisión (bps)",
            min_value=0,
            max_value=50,
            value=5,
            help="Comisión por transacción en basis points"
        )

        slippage_bps = st.number_input(
            "Slippage (bps)",
            min_value=0,
            max_value=50,
            value=5,
            help="Slippage esperado en basis points"
        )

        market_impact_bps = st.number_input(
            "Market Impact (bps)",
            min_value=0,
            max_value=50,
            value=2,
            help="Impacto de mercado en basis points"
        )

    st.divider()

    st.subheader("💎 Risk Management (FASE 1)")

    st.info("Sistema de stop loss, take profit y position sizing basado en literatura académica 2014-2020")

    enable_risk_management = st.checkbox(
        "✅ Activar Risk Management",
        value=True,
        help="Calcula stop loss, take profit y position sizing para cada stock"
    )

    if enable_risk_management:
        with st.expander("⚙️ Configuración de Stop Loss"):
            use_volatility_stop = st.checkbox(
                "Volatility-Based Stop (Kaminski & Lo 2014)",
                value=True,
                help="Stop loss basado en volatilidad realizada (2σ = 95% CI)"
            )

            volatility_stop_confidence = st.slider(
                "Confidence Level (σ)",
                min_value=1.0,
                max_value=3.0,
                value=2.0,
                step=0.5,
                help="2.0 = 95% CI, 2.5 = 99% CI",
                disabled=not use_volatility_stop
            )

            use_trailing_stop = st.checkbox(
                "Trailing Stop (Han et al. 2016)",
                value=True,
                help="Stop loss dinámico que sigue al precio"
            )

            trailing_stop_method = st.selectbox(
                "Método de Trailing Stop",
                options=['ATR', 'FIXED', 'CHANDELIER'],
                index=0,
                help="ATR = basado en Average True Range, FIXED = % fijo, CHANDELIER = Chandelier Exit",
                disabled=not use_trailing_stop
            )

            trailing_atr_multiplier = st.slider(
                "ATR Multiplier",
                min_value=1.5,
                max_value=4.0,
                value=2.5,
                step=0.5,
                help="Multiplicador del ATR (2-3x recomendado)",
                disabled=not use_trailing_stop or trailing_stop_method != 'ATR'
            )

        with st.expander("⚙️ Configuración de Take Profit"):
            use_take_profit = st.checkbox(
                "Risk-Reward Take Profit (Harris & Yilmaz 2019)",
                value=True,
                help="Take profit basado en ratio risk-reward óptimo"
            )

            risk_reward_ratio = st.slider(
                "Risk-Reward Ratio",
                min_value=1.5,
                max_value=4.0,
                value=2.5,
                step=0.5,
                help="2.5:1 = ganas 2.5× lo que arriesgas (recomendado 2-3×)",
                disabled=not use_take_profit
            )

        with st.expander("⚙️ Position Sizing"):
            use_volatility_sizing = st.checkbox(
                "Volatility-Managed Sizing (Moreira & Muir 2017)",
                value=True,
                help="Ajusta tamaño de posición según volatilidad (+50% Sharpe ratio)"
            )

            target_volatility = st.slider(
                "Target Volatility (%)",
                min_value=5,
                max_value=25,
                value=15,
                step=5,
                help="Volatilidad objetivo anual (15% recomendado)",
                disabled=not use_volatility_sizing
            ) / 100.0

            max_position_size = st.slider(
                "Max Position Size (%)",
                min_value=5,
                max_value=30,
                value=20,
                step=5,
                help="Tamaño máximo por posición",
                disabled=not use_volatility_sizing
            ) / 100.0

            use_kelly = st.checkbox(
                "Kelly Criterion (Rotando & Thorp 2018)",
                value=False,
                help="Position sizing usando Kelly Criterion (requiere win rate histórico)"
            )

    else:
        # Defaults cuando está deshabilitado
        use_volatility_stop = True
        volatility_stop_confidence = 2.0
        use_trailing_stop = True
        trailing_stop_method = 'ATR'
        trailing_atr_multiplier = 2.5
        use_take_profit = True
        risk_reward_ratio = 2.5
        use_volatility_sizing = True
        target_volatility = 0.15
        max_position_size = 0.20
        use_kelly = False

    st.divider()

    st.subheader("🤖 ML Integration (FASE 3)")

    st.info("Machine Learning para ranking de stocks basado en Gu et al. (2020)")

    enable_ml_ranking = st.checkbox(
        "✅ Activar ML Ranking",
        value=True,
        help="Usa ML + QV score para ranking (cambia selección de stocks)"
    )

    if enable_ml_ranking:
        ml_rank_weight = st.slider(
            "ML Weight (%)",
            min_value=0,
            max_value=50,
            value=30,
            step=5,
            help="% de ML en hybrid score (resto es QV score). 30% = 30% ML + 70% QV"
        ) / 100.0

        with st.expander("⚙️ Features a Usar"):
            use_technical_features = st.checkbox(
                "Technical Features (MA ratios, RSI, 52w)",
                value=True,
                help="MA ratios, RSI, distancia de 52w high/low"
            )

            use_momentum_features = st.checkbox(
                "Momentum Features (multi-horizon returns)",
                value=True,
                help="Returns en 5d, 20d, 60d, 120d, 252d + aceleración"
            )

            use_volatility_features = st.checkbox(
                "Volatility Features (realized vol, downside vol)",
                value=True,
                help="Volatilidad realizada, downside vol, volatility change"
            )

            use_volume_features = st.checkbox(
                "Volume Features (turnover, volume momentum)",
                value=True,
                help="Volume ratio, volume momentum"
            )

    else:
        # Defaults
        ml_rank_weight = 0.30
        use_technical_features = True
        use_momentum_features = True
        use_volatility_features = True
        use_volume_features = True

    st.divider()

    st.subheader("🔄 Advanced Exits (FASE 2)")

    st.info("Exits adaptativos: regime-based stops, percentile targets, time exits")

    enable_advanced_exits = st.checkbox(
        "✅ Activar Advanced Exits",
        value=True,
        help="Ajusta stops/targets según régimen de mercado y tiempo"
    )

    if enable_advanced_exits:
        with st.expander("⚙️ Regime-Based Stops"):
            use_regime_stops = st.checkbox(
                "Regime-Based Stops (Nystrup et al. 2020)",
                value=True,
                help="Ajusta stops según volatilidad del mercado"
            )

            regime_lookback = st.slider(
                "Regime Lookback (días)",
                min_value=20,
                max_value=120,
                value=60,
                step=10,
                help="Días para detectar régimen de volatilidad",
                disabled=not use_regime_stops
            )

            high_vol_multiplier = st.slider(
                "High Vol Multiplier",
                min_value=1.0,
                max_value=2.0,
                value=1.5,
                step=0.1,
                help="Multiplicador de stops en alta volatilidad (evitar whipsaws)",
                disabled=not use_regime_stops
            )

            low_vol_multiplier = st.slider(
                "Low Vol Multiplier",
                min_value=0.5,
                max_value=1.0,
                value=0.8,
                step=0.1,
                help="Multiplicador de stops en baja volatilidad (proteger ganancias)",
                disabled=not use_regime_stops
            )

        with st.expander("⚙️ Percentile Targets & Time Exits"):
            use_percentile_targets = st.checkbox(
                "Percentile Targets (Lopez de Prado 2020)",
                value=True,
                help="Target basado en distribución empírica de retornos"
            )

            target_percentile = st.slider(
                "Target Percentile",
                min_value=60,
                max_value=90,
                value=75,
                step=5,
                help="75 = conservador, 90 = agresivo",
                disabled=not use_percentile_targets
            )

            use_time_exits = st.checkbox(
                "Time-Based Exits (Harvey & Liu 2021)",
                value=True,
                help="Exit forzado después de X días"
            )

            max_holding_days = st.slider(
                "Max Holding Days",
                min_value=30,
                max_value=180,
                value=90,
                step=10,
                help="Días máximo de holding",
                disabled=not use_time_exits
            )

            use_profit_lock = st.checkbox(
                "Profit Lock (Trailing TP)",
                value=True,
                help="Activa trailing TP después de ganancia significativa"
            )

            profit_lock_threshold = st.slider(
                "Profit Lock Threshold (%)",
                min_value=10,
                max_value=30,
                value=15,
                step=5,
                help="Activa trailing TP a este % de ganancia",
                disabled=not use_profit_lock
            ) / 100.0

    else:
        # Defaults
        use_regime_stops = True
        regime_lookback = 60
        high_vol_multiplier = 1.5
        low_vol_multiplier = 0.8
        use_percentile_targets = True
        target_percentile = 75
        use_time_exits = True
        max_holding_days = 90
        use_profit_lock = True
        profit_lock_threshold = 0.15

    st.divider()

    st.subheader("💾 Gestión de Datos")

    use_price_cache = st.checkbox(
        "Usar caché de precios",
        value=True,
        help="Si está desmarcado, descarga datos de precios frescos (más lento pero datos actualizados). ⚠️ Si ves stocks que no deberían pasar los filtros MA200, DESMARCA esta opción."
    )

    if not use_price_cache:
        st.warning("⚠️ Cache deshabilitado: Se descargarán datos frescos (esto puede tomar más tiempo)")

    st.divider()

    # Botón de ejecución
    run_button = st.button("🚀 Ejecutar Screening V3", type="primary", use_container_width=True)

    # Botón para limpiar caché
    if st.button("🗑️ Limpiar Caché", use_container_width=True):
        st.cache_data.clear()
        st.success("Caché limpiado!")


# ============================================================================
# CREAR CONFIGURACIÓN V3
# ============================================================================

config = QVMConfigV3(
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
    min_roic=min_roic,
    require_above_ma200=require_above_ma200,
    min_momentum_12m=min_momentum_12m,
    require_near_52w_high=require_near_52w_high,
    min_pct_from_52w_high=min_pct_from_52w_high,
    require_roic_above_wacc=require_roic_above_wacc,
    # Breakouts y Volumen (NUEVO)
    enable_breakout_filter=enable_breakout_filter,
    require_breakout_confirmed=require_breakout_confirmed,
    require_breakout_strong=require_breakout_strong,
    enable_volume_surge_filter=enable_volume_surge_filter,
    min_ebit_ev=min_ebit_ev,
    max_fcf_ev=max_fcf_ev,
    portfolio_size=portfolio_size,
    backtest_enabled=backtest_enabled,
    backtest_start=backtest_start.strftime('%Y-%m-%d'),
    backtest_end=backtest_end.strftime('%Y-%m-%d'),
    rebalance_freq=rebalance_freq,
    commission_bps=commission_bps,
    slippage_bps=slippage_bps,
    market_impact_bps=market_impact_bps,
    # Gestión de Datos
    use_price_cache=use_price_cache,
    # Mejoras V3.1
    enable_earnings_quality=enable_earnings_quality,
    enable_red_flags=enable_red_flags,
    enable_reversal_filter=enable_reversal_filter,
    use_enhanced_value_score=use_enhanced_value_score,
    enable_fundamental_momentum=enable_fundamental_momentum,
    enable_sector_relative=enable_sector_relative,
    # Risk Management (FASE 1)
    enable_risk_management=enable_risk_management,
    use_volatility_stop=use_volatility_stop,
    volatility_stop_confidence=volatility_stop_confidence,
    use_trailing_stop=use_trailing_stop,
    trailing_stop_method=trailing_stop_method,
    trailing_atr_multiplier=trailing_atr_multiplier,
    use_take_profit=use_take_profit,
    risk_reward_ratio=risk_reward_ratio,
    use_volatility_sizing=use_volatility_sizing,
    target_volatility=target_volatility,
    max_position_size=max_position_size,
    use_kelly=use_kelly,
    # ML Integration (FASE 3)
    enable_ml_ranking=enable_ml_ranking,
    ml_rank_weight=ml_rank_weight,
    use_technical_features=use_technical_features,
    use_momentum_features=use_momentum_features,
    use_volatility_features=use_volatility_features,
    use_volume_features=use_volume_features,
    # Advanced Exits (FASE 2)
    enable_advanced_exits=enable_advanced_exits,
    use_regime_stops=use_regime_stops,
    regime_lookback=regime_lookback,
    high_vol_multiplier=high_vol_multiplier,
    low_vol_multiplier=low_vol_multiplier,
    use_percentile_targets=use_percentile_targets,
    target_percentile=target_percentile,
    use_time_exits=use_time_exits,
    max_holding_days=max_holding_days,
    use_profit_lock=use_profit_lock,
    profit_lock_threshold=profit_lock_threshold,
)


# ============================================================================
# MAIN APP
# ============================================================================

# Mostrar configuración actual
with st.expander("📋 Ver Configuración Completa V3", expanded=False):
    config_df = pd.DataFrame([config.to_dict()]).T
    config_df.columns = ['Valor']
    st.dataframe(config_df, use_container_width=True)


# Ejecutar pipeline
if run_button or st.session_state.get('results') is not None:

    if run_button:
        # Limpiar resultados anteriores
        st.session_state.results = None

    if st.session_state.get('results') is None:
        with st.spinner("🔄 Ejecutando Pipeline V3... (esto puede tomar varios minutos)"):
            try:
                results = run_qvm_pipeline_v3(config=config, verbose=False)
                st.session_state.results = results
            except Exception as e:
                st.error(f"❌ Error al ejecutar pipeline: {str(e)}")
                st.exception(e)
                st.stop()

    results = st.session_state.results

    # Verificar si hubo error
    if 'error' in results:
        st.error(f"❌ Pipeline failed: {results['error']}")
        st.stop()

    # ========================================================================
    # RESULTADOS
    # ========================================================================

    st.success(f"✅ Pipeline V3 completado exitosamente!")

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
            title="Pipeline V3 Funnel - Stocks en Cada Paso",
            labels={'Count': 'Número de Stocks'},
        )
        fig_funnel.update_layout(height=500)
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
    # BACKTEST RESULTS (NUEVO)
    # ------------------------------------------------------------------------
    if results.get('backtest') is not None:
        st.header("📊 Resultados del Backtest")

        backtest = results['backtest']
        pm = backtest['portfolio_metrics']

        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            cagr = pm['CAGR']
            st.metric(
                "CAGR",
                f"{cagr:.2%}",
                delta=f"{cagr:.2%}" if cagr > 0 else None
            )

        with col2:
            sharpe = pm['Sharpe']
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.2f}",
                delta="Excelente" if sharpe > 1.5 else ("Bueno" if sharpe > 1.0 else "Medio")
            )

        with col3:
            sortino = pm['Sortino']
            st.metric(
                "Sortino Ratio",
                f"{sortino:.2f}",
                delta="Excelente" if sortino > 2.0 else ("Bueno" if sortino > 1.5 else "Medio")
            )

        with col4:
            max_dd = pm['MaxDD']
            st.metric(
                "Max Drawdown",
                f"{max_dd:.2%}",
                delta=f"{max_dd:.2%}",
                delta_color="inverse"
            )

        # Curva de equity
        if 'equity_curves' in backtest and backtest['equity_curves']:
            st.subheader("Curva de Equity del Portfolio")

            equity_curves = backtest['equity_curves']

            # Calcular curva promedio del portfolio (equal-weight)
            try:
                # Convertir dict de Series a DataFrame
                ec_df = pd.concat(equity_curves.values(), axis=1)
                ec_df.columns = list(equity_curves.keys())
                ec_df = ec_df.dropna(how='all').ffill().dropna(how='all')

                # Equity promedio (equal-weight)
                portfolio_curve = ec_df.mean(axis=1)

                # Crear gráfico de equity curve
                fig_equity = go.Figure()

                # Agregar curva del portfolio
                fig_equity.add_trace(go.Scatter(
                    x=portfolio_curve.index,
                    y=portfolio_curve.values,
                    mode='lines',
                    name='Portfolio (Equal Weight)',
                    line=dict(color='gold', width=3)
                ))

                # Opcionalmente, agregar curvas individuales (semi-transparentes)
                show_individual = st.checkbox("Mostrar stocks individuales", value=False)
                if show_individual:
                    for symbol, curve in equity_curves.items():
                        fig_equity.add_trace(go.Scatter(
                            x=curve.index,
                            y=curve.values,
                            mode='lines',
                            name=symbol,
                            line=dict(width=1),
                            opacity=0.3
                        ))

                fig_equity.update_layout(
                    title="Portfolio Equity Curve (Equal-Weighted)",
                    xaxis_title="Fecha",
                    yaxis_title="Valor Normalizado (Base 1.0)",
                    height=500,
                    hovermode='x unified'
                )

                st.plotly_chart(fig_equity, use_container_width=True)

            except Exception as e:
                st.warning(f"No se pudo generar la equity curve: {str(e)}")
        else:
            st.info("No hay datos de equity curve disponibles. Verifica que el backtest se ejecutó correctamente.")

        # Estadísticas detalladas
        with st.expander("📊 Estadísticas Detalladas del Backtest"):
            stats_df = pd.DataFrame([pm]).T
            stats_df.columns = ['Valor']
            st.dataframe(stats_df, use_container_width=True)

        st.divider()

    # ------------------------------------------------------------------------
    # PORTFOLIO FINAL
    # ------------------------------------------------------------------------
    st.header("📋 Portfolio Final")

    portfolio = results.get('portfolio')

    if portfolio is not None and not portfolio.empty:
        col1, col2, col3, col4, col5 = st.columns(5)

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
            if 'momentum_12m' in portfolio.columns:
                avg_momentum = portfolio['momentum_12m'].mean()
                st.metric(
                    "Momentum Promedio",
                    f"{avg_momentum:.1%}"
                )

        with col5:
            n_sectors = portfolio['sector'].nunique()
            st.metric(
                "Sectores Únicos",
                n_sectors
            )

        st.divider()

        # Análisis detallado
        st.subheader("🏆 Top Stocks")

        analysis = analyze_portfolio_v3(results, n_top=portfolio_size)

        if not analysis.empty:
            # Formatear para mostrar
            display_df = analysis.copy()

            # Formatear columnas numéricas
            for col in display_df.columns:
                if 'score' in col.lower() or 'component' in col.lower():
                    if display_df[col].dtype in [float, np.float64]:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
                elif col in ['roic', 'roe', 'gross_margin', 'fcf_yield', 'momentum_12m', 'ebit_ev', 'fcf_ev', 'pct_from_52w_high']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
                elif col in ['pe', 'pb', 'ev_ebitda']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
                elif col == 'above_ma200':
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: "✅" if x else "❌")
                elif col == 'roic_above_wacc':
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: "✅" if x else "❌")
                # Risk Management columns
                elif col in ['entry_price', 'stop_loss', 'take_profit']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "-")
                elif col in ['position_size_pct', 'risk_pct', 'reward_pct']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")
                elif col == 'rr_ratio':
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}:1" if pd.notna(x) else "-")

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
                file_name=f"qvm_portfolio_v3_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

        st.divider()

        # ------------------------------------------------------------------------
        # VISUALIZACIONES
        # ------------------------------------------------------------------------
        st.header("📈 Visualizaciones")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Score Distribution",
            "Sector Analysis",
            "Piotroski Components",
            "Valuation Metrics",
            "Momentum & MA200",
            "Risk Management"
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

            # Verificar si las columnas existen
            if 'value_score_component' in portfolio.columns and 'quality_score_component' in portfolio.columns:
                try:
                    # Filtrar datos válidos
                    valid_scatter = portfolio[
                        portfolio['value_score_component'].notna() &
                        portfolio['quality_score_component'].notna()
                    ].copy()

                    if not valid_scatter.empty and len(valid_scatter) > 0:
                        hover_cols = ['symbol']
                        if 'piotroski_score' in valid_scatter.columns:
                            hover_cols.append('piotroski_score')
                        if 'qv_score' in valid_scatter.columns:
                            hover_cols.append('qv_score')

                        fig_scatter = px.scatter(
                            valid_scatter,
                            x='value_score_component',
                            y='quality_score_component',
                            size='market_cap' if 'market_cap' in valid_scatter.columns else None,
                            color='sector' if 'sector' in valid_scatter.columns else None,
                            hover_data=hover_cols,
                            title="Quality Score vs Value Score",
                            labels={
                                'value_score_component': 'Value Score',
                                'quality_score_component': 'Quality Score (Piotroski Normalizado)'
                            }
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    else:
                        st.info("No hay suficientes datos válidos para el scatter plot Quality vs Value.")
                except Exception as e:
                    st.warning(f"No se pudo generar el gráfico Quality vs Value: {str(e)}")
            else:
                # Usar Piotroski como proxy de Quality
                if 'piotroski_score' in portfolio.columns:
                    try:
                        # Determinar columna de valoración
                        val_col = None
                        if 'ev_ebitda' in portfolio.columns:
                            val_col = 'ev_ebitda'
                        elif 'pe' in portfolio.columns:
                            val_col = 'pe'

                        if val_col:
                            valid_scatter = portfolio[
                                portfolio['piotroski_score'].notna() &
                                portfolio[val_col].notna()
                            ].copy()

                            if not valid_scatter.empty and len(valid_scatter) > 0:
                                hover_cols = ['symbol']
                                if 'qv_score' in valid_scatter.columns:
                                    hover_cols.append('qv_score')

                                fig_scatter = px.scatter(
                                    valid_scatter,
                                    x=val_col,
                                    y='piotroski_score',
                                    size='market_cap' if 'market_cap' in valid_scatter.columns else None,
                                    color='sector' if 'sector' in valid_scatter.columns else None,
                                    hover_data=hover_cols,
                                    title="Piotroski Score vs Valuation",
                                    labels={
                                        'piotroski_score': 'Piotroski Score (Quality)',
                                        val_col: val_col.upper().replace('_', '/')
                                    }
                                )
                                st.plotly_chart(fig_scatter, use_container_width=True)
                            else:
                                st.info("No hay suficientes datos válidos para el scatter plot.")
                        else:
                            st.info("No hay métricas de valoración disponibles para el scatter plot.")
                    except Exception as e:
                        st.warning(f"No se pudo generar el gráfico alternativo: {str(e)}")
                else:
                    st.info("Componentes de score no disponibles para visualización.")

        with tab2:
            st.subheader("Distribución por Sector")

            if 'sector' in portfolio.columns:
                try:
                    sector_counts = portfolio['sector'].value_counts()

                    if not sector_counts.empty:
                        fig_sector = px.pie(
                            values=sector_counts.values,
                            names=sector_counts.index,
                            title="Portfolio por Sector",
                        )
                        st.plotly_chart(fig_sector, use_container_width=True)
                    else:
                        st.info("No hay datos de sector disponibles.")
                except Exception as e:
                    st.warning(f"No se pudo generar el gráfico de sectores: {str(e)}")
            else:
                st.info("La columna 'sector' no está disponible en el portfolio.")

            # Piotroski por sector
            st.subheader("Piotroski Score por Sector")

            if 'sector' in portfolio.columns and 'piotroski_score' in portfolio.columns:
                try:
                    valid_data = portfolio[
                        portfolio['sector'].notna() &
                        portfolio['piotroski_score'].notna()
                    ]

                    if not valid_data.empty and len(valid_data) > 0:
                        fig_box = px.box(
                            valid_data,
                            x='sector',
                            y='piotroski_score',
                            title="Distribución de Piotroski Score por Sector",
                            labels={'piotroski_score': 'Piotroski Score', 'sector': 'Sector'}
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                    else:
                        st.info("No hay suficientes datos para el gráfico de Piotroski por sector.")
                except Exception as e:
                    st.warning(f"No se pudo generar el box plot: {str(e)}")
            else:
                st.info("Las columnas necesarias para Piotroski por sector no están disponibles.")

        with tab3:
            st.subheader("Piotroski Score Distribution")

            if 'piotroski_score' in portfolio.columns:
                try:
                    valid_piotroski = portfolio[portfolio['piotroski_score'].notna()]

                    if not valid_piotroski.empty:
                        piotroski_counts = valid_piotroski['piotroski_score'].value_counts().sort_index()

                        fig_piotroski = px.bar(
                            x=piotroski_counts.index,
                            y=piotroski_counts.values,
                            title="Distribución de Piotroski Score",
                            labels={'x': 'Piotroski Score', 'y': 'Cantidad'},
                        )
                        st.plotly_chart(fig_piotroski, use_container_width=True)
                    else:
                        st.info("No hay datos de Piotroski Score disponibles.")
                except Exception as e:
                    st.warning(f"No se pudo generar el gráfico de Piotroski: {str(e)}")
            else:
                st.info("La columna 'piotroski_score' no está disponible.")

            # Mostrar estadísticas de componentes si están disponibles
            piotroski_cols = [c for c in portfolio.columns if 'positive' in c or 'delta_' in c or 'accruals' in c]

            if piotroski_cols:
                try:
                    st.subheader("Piotroski Components Analysis")

                    components_summary = portfolio[piotroski_cols].mean().sort_values(ascending=False)

                    if not components_summary.empty:
                        fig_components = px.bar(
                            x=components_summary.values * 100,
                            y=components_summary.index,
                            orientation='h',
                            title="% de Stocks que Pasan cada Check de Piotroski",
                            labels={'x': '% Pass Rate', 'y': 'Component'}
                        )
                        st.plotly_chart(fig_components, use_container_width=True)
                except Exception as e:
                    st.warning(f"No se pudo generar el análisis de componentes: {str(e)}")

        with tab4:
            st.subheader("Métricas de Valoración")

            valuation_cols = []
            for col in ['pe', 'pb', 'ev_ebitda', 'ebit_ev', 'fcf_ev']:
                if col in portfolio.columns:
                    valuation_cols.append(col)

            if valuation_cols:
                col1, col2 = st.columns(2)

                for i, col in enumerate(valuation_cols):
                    with col1 if i % 2 == 0 else col2:
                        # Filtrar valores válidos para evitar errores
                        valid_data = portfolio[portfolio[col].notna()]
                        if not valid_data.empty:
                            fig = px.histogram(
                                valid_data,
                                x=col,
                                title=f"Distribución de {col.upper()}",
                                labels={col: col.upper(), 'count': 'Frecuencia'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"No hay datos válidos para {col.upper()}")
            else:
                st.info("No hay métricas de valoración disponibles para visualizar.")

        with tab5:
            st.subheader("Momentum & MA200 Analysis")

            has_momentum = 'momentum_12m' in portfolio.columns
            has_ma200 = 'above_ma200' in portfolio.columns

            if not has_momentum and not has_ma200:
                st.info("⚠️ Los datos de Momentum y MA200 no están disponibles. Asegúrate de que el pipeline descargue precios históricos correctamente.")
            else:
                if has_momentum:
                    # Histograma de momentum
                    valid_momentum = portfolio[portfolio['momentum_12m'].notna()]
                    if not valid_momentum.empty:
                        fig_momentum = px.histogram(
                            valid_momentum,
                            x='momentum_12m',
                            nbins=20,
                            title="Distribución de Momentum 12M",
                            labels={'momentum_12m': 'Momentum 12M', 'count': 'Frecuencia'}
                        )
                        st.plotly_chart(fig_momentum, use_container_width=True)

                # MA200 status
                if has_ma200:
                    ma200_counts = portfolio['above_ma200'].value_counts()

                    fig_ma200 = px.pie(
                        values=ma200_counts.values,
                        names=['Por encima MA200' if x else 'Por debajo MA200' for x in ma200_counts.index],
                        title="Distribución MA200",
                    )
                    st.plotly_chart(fig_ma200, use_container_width=True)

                # Scatter: Momentum vs QV Score
                if has_momentum and 'qv_score' in portfolio.columns:
                    valid_data = portfolio[portfolio['momentum_12m'].notna() & portfolio['qv_score'].notna()]
                    if not valid_data.empty:
                        fig_momentum_qv = px.scatter(
                            valid_data,
                            x='momentum_12m',
                            y='qv_score',
                            size='market_cap' if 'market_cap' in portfolio.columns else None,
                            color='above_ma200' if has_ma200 else 'sector',
                            hover_data=['symbol', 'piotroski_score'] if 'piotroski_score' in portfolio.columns else ['symbol'],
                            title="Momentum vs QV Score",
                            labels={
                                'momentum_12m': 'Momentum 12M',
                                'qv_score': 'QV Score'
                            }
                        )
                        st.plotly_chart(fig_momentum_qv, use_container_width=True)

        with tab6:
            st.subheader("💎 Risk Management Analysis (FASE 1)")

            # Verificar si hay datos de risk management
            has_risk_data = all(col in portfolio.columns for col in ['stop_loss', 'take_profit', 'position_size_pct'])

            if not has_risk_data:
                st.info("⚠️ Los datos de Risk Management no están disponibles. Activa 'Risk Management' en la barra lateral y vuelve a ejecutar.")
            else:
                # Resumen de métricas de risk
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    avg_position = portfolio['position_size_pct'].mean()
                    st.metric(
                        "Position Size Promedio",
                        f"{avg_position:.1f}%",
                        help="Tamaño promedio de posición recomendado"
                    )

                with col2:
                    avg_rr = portfolio['rr_ratio'].mean()
                    st.metric(
                        "R:R Ratio Promedio",
                        f"{avg_rr:.2f}:1",
                        delta="Excelente" if avg_rr > 2.5 else ("Bueno" if avg_rr > 2.0 else "Medio")
                    )

                with col3:
                    avg_risk = portfolio['risk_pct'].mean()
                    st.metric(
                        "Risk Promedio",
                        f"{avg_risk:.2f}%",
                        help="Pérdida potencial promedio por posición"
                    )

                with col4:
                    avg_reward = portfolio['reward_pct'].mean()
                    st.metric(
                        "Reward Promedio",
                        f"{avg_reward:.2f}%",
                        help="Ganancia potencial promedio por posición"
                    )

                st.divider()

                # Distribución de Position Size
                st.subheader("Distribución de Position Size")
                valid_position = portfolio[portfolio['position_size_pct'].notna()]
                if not valid_position.empty:
                    fig_position = px.histogram(
                        valid_position,
                        x='position_size_pct',
                        nbins=20,
                        title="Distribución de Position Size (%)",
                        labels={'position_size_pct': 'Position Size (%)', 'count': 'Frecuencia'}
                    )
                    st.plotly_chart(fig_position, use_container_width=True)

                # Risk-Reward Scatter
                st.subheader("Risk vs Reward")
                valid_rr = portfolio[portfolio['risk_pct'].notna() & portfolio['reward_pct'].notna()]
                if not valid_rr.empty:
                    fig_rr = px.scatter(
                        valid_rr,
                        x='risk_pct',
                        y='reward_pct',
                        size='position_size_pct',
                        color='rr_ratio',
                        hover_data=['symbol', 'qv_score'],
                        title="Risk vs Reward (tamaño = position size, color = R:R ratio)",
                        labels={
                            'risk_pct': 'Risk (%)',
                            'reward_pct': 'Reward (%)',
                            'rr_ratio': 'R:R Ratio'
                        },
                        color_continuous_scale='RdYlGn'
                    )
                    # Agregar línea de referencia para R:R ratios
                    fig_rr.add_trace(go.Scatter(
                        x=[0, valid_rr['risk_pct'].max()],
                        y=[0, valid_rr['risk_pct'].max() * 2.5],
                        mode='lines',
                        name='2.5:1 R:R',
                        line=dict(dash='dash', color='gray')
                    ))
                    st.plotly_chart(fig_rr, use_container_width=True)

                # Entry vs Stop/Take Profit
                st.subheader("Entry Price vs Stop Loss & Take Profit")

                col1, col2 = st.columns(2)

                with col1:
                    # Stop Loss levels
                    if 'entry_price' in portfolio.columns and 'stop_loss' in portfolio.columns:
                        valid_stops = portfolio[portfolio['entry_price'].notna() & portfolio['stop_loss'].notna()].head(10)
                        if not valid_stops.empty:
                            stop_data = []
                            for _, row in valid_stops.iterrows():
                                stop_data.append({
                                    'Symbol': row['symbol'],
                                    'Price': row['entry_price'],
                                    'Type': 'Entry'
                                })
                                stop_data.append({
                                    'Symbol': row['symbol'],
                                    'Price': row['stop_loss'],
                                    'Type': 'Stop Loss'
                                })

                            stop_df = pd.DataFrame(stop_data)
                            fig_stops = px.bar(
                                stop_df,
                                x='Symbol',
                                y='Price',
                                color='Type',
                                barmode='group',
                                title="Top 10: Entry vs Stop Loss",
                                labels={'Price': 'Price ($)'}
                            )
                            st.plotly_chart(fig_stops, use_container_width=True)

                with col2:
                    # Take Profit levels
                    if 'entry_price' in portfolio.columns and 'take_profit' in portfolio.columns:
                        valid_tp = portfolio[portfolio['entry_price'].notna() & portfolio['take_profit'].notna()].head(10)
                        if not valid_tp.empty:
                            tp_data = []
                            for _, row in valid_tp.iterrows():
                                tp_data.append({
                                    'Symbol': row['symbol'],
                                    'Price': row['entry_price'],
                                    'Type': 'Entry'
                                })
                                tp_data.append({
                                    'Symbol': row['symbol'],
                                    'Price': row['take_profit'],
                                    'Type': 'Take Profit'
                                })

                            tp_df = pd.DataFrame(tp_data)
                            fig_tp = px.bar(
                                tp_df,
                                x='Symbol',
                                y='Price',
                                color='Type',
                                barmode='group',
                                title="Top 10: Entry vs Take Profit",
                                labels={'Price': 'Price ($)'}
                            )
                            st.plotly_chart(fig_tp, use_container_width=True)

                # Portfolio Risk Summary
                st.subheader("📊 Portfolio Risk Summary")

                if 'position_size_pct' in portfolio.columns and 'risk_pct' in portfolio.columns:
                    # Calcular portfolio risk total
                    total_capital = 100  # Asumiendo 100% de capital
                    portfolio_risk = (portfolio['position_size_pct'] / 100 * portfolio['risk_pct'] / 100 * total_capital).sum()

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Total Portfolio Risk",
                            f"{portfolio_risk:.2f}%",
                            help="Suma de riesgo ponderado por position size"
                        )

                    with col2:
                        total_positions = len(portfolio[portfolio['position_size_pct'].notna()])
                        st.metric(
                            "Posiciones con Risk Data",
                            total_positions
                        )

                    with col3:
                        if total_positions > 0:
                            avg_risk_per_position = portfolio_risk / total_positions
                            st.metric(
                                "Risk por Posición",
                                f"{avg_risk_per_position:.2f}%"
                            )

    else:
        st.warning("No hay stocks en el portfolio final.")

else:
    # Instrucciones iniciales
    st.info("""
    👈 **Ajusta los parámetros en la barra lateral y presiona "Ejecutar Screening V3"**

    **NUEVAS CARACTERÍSTICAS V3:**

    - ✅ **MA200 Filter**: Filtro de tendencia de Faber (2007) - solo stocks por encima de MA de 200 días
    - ✅ **Momentum 12M-1M**: Jegadeesh & Titman (1993) - retornos de 12 meses excluyendo el último mes
    - ✅ **52-Week High**: Filtro de precio cerca del máximo de 52 semanas
    - ✅ **EBIT/EV & FCF/EV**: Métricas avanzadas de valoración normalizadas
    - ✅ **ROIC > WACC**: Filtro de creación de valor (ROIC > Costo de Capital)
    - ✅ **Backtest Integrado**: Performance histórica con CAGR, Sharpe, Sortino, Max DD

    **Guía de Parámetros:**

    - **Quality (Piotroski)**: Mide salud financiera operacional (9 checks)
    - **Value (Multiples)**: Mide cuán barato está el stock (EV/EBITDA, P/E, P/B)
    - **FCF Yield**: Free Cash Flow / Market Cap (mayor es mejor)
    - **Momentum**: Retornos 12M-1M (momentum excluyendo reversión de corto plazo)

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

    **MA200 Filter (Faber 2007):**
    - Precio > MA200: Tendencia alcista (BUY)
    - Precio < MA200: Tendencia bajista (AVOID)

    **ROIC > WACC:**
    - ROIC > WACC: Crea valor para accionistas
    - ROIC < WACC: Destruye valor (AVOID)
    """)


# ============================================================================
# FOOTER
# ============================================================================

st.divider()

st.caption("""
**QVM Screener V3** | Powered by Financial Modeling Prep API

Bibliografía:
- Piotroski (2000): Value Investing - The Use of Historical Financial Statement Information
- Asness et al. (2019): Quality Minus Junk
- Fama & French (1992, 2015): Multi-factor Models
- Faber (2007): A Quantitative Approach to Tactical Asset Allocation
- Jegadeesh & Titman (1993): Returns to Buying Winners and Selling Losers
""")
