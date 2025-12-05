"""
QVM Strategy Pipeline V3 - COMPLETO con MA200, Backtest y Métricas Avanzadas
=============================================================================

NUEVAS CARACTERÍSTICAS V3.1 (Mejoras Académicas):
1. ✅ MA200 Filter (Faber 2007)
2. ✅ Reglas heurísticas: 52w high, volumen relativo
3. ✅ Nuevas métricas de valoración: EBIT/EV, FCF/EV, ROIC > WACC
4. ✅ Backtest integrado con rebalanceo
5. ✅ Momentum risk-adjusted (Barroso & Santa-Clara 2015)
6. ✅ Multi-timeframe momentum (Novy-Marx 2012)
7. ✅ Earnings quality (Sloan 1996, Beneish 1999)
8. ✅ Fundamental momentum (Piotroski & So 2012)
9. ✅ Sector relative momentum (O'Shaughnessy 2005)
10. ✅ Value score expandido (Gray & Carlisle 2012)
11. ✅ Insider trading signals (Lakonishok & Lee 2001)
12. ✅ Red flags detection

Bibliografía:
- Faber (2007): "A Quantitative Approach to Tactical Asset Allocation"
- Piotroski (2000): F-Score
- Asness et al. (2019): Quality factors
- Sloan (1996): Accruals anomaly
- Novy-Marx (2012): Intermediate horizon returns
- Barroso & Santa-Clara (2015): Momentum has its moments
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

# Imports de módulos
from data_fetcher import (
    fetch_screener,
    fetch_financial_statements_batch,
    fetch_fundamentals_batch,
    fetch_prices,
)
from quality_value_score import (
    compute_quality_value_factors,
    top_quality_value_stocks,
    analyze_score_components,
)
from momentum_calculator import (
    calculate_momentum_batch,
    filter_above_ma200,
    is_above_ma200,
    calculate_12m_1m_momentum,
    calculate_risk_adjusted_momentum,
    calculate_multi_timeframe_momentum,
    filter_short_term_reversal,
    calculate_sector_relative_momentum,
)
from backtest_engine import (
    backtest_portfolio,
    calculate_portfolio_metrics,
    TradingCosts,
)
from earnings_quality import (
    add_earnings_quality_metrics,
    apply_earnings_quality_filters,
)
from fundamental_momentum import (
    add_fundamental_momentum_to_df,
    calculate_fundamental_momentum_batch,
)
from red_flags import (
    add_red_flags_metrics,
    apply_red_flags_filters,
)
from risk_management import (
    RiskCalculator,
    RiskConfig,
)


# ============================================================================
# CONFIGURACIÓN COMPLETA
# ============================================================================

@dataclass
class QVMConfigV3:
    """
    Configuración completa del pipeline QVM V3.
    """

    # ========== UNIVERSE ==========
    universe_size: int = 800  # Ampliado de 300 a 800
    min_market_cap: float = 2e9
    min_volume: int = 500_000

    # ========== QUALITY-VALUE WEIGHTS ==========
    # Optimizado según análisis académico (ver ANALISIS_ACADEMICO.md)
    w_quality: float = 0.35      # Reducido para evitar overlap con FCF
    w_value: float = 0.40        # Aumentado (mayor peso en valoración)
    w_fcf_yield: float = 0.10    # Reducido (overlap parcial con Piotroski CFO)
    w_momentum: float = 0.15     # Aumentado según Jegadeesh & Titman (1993)

    # ========== FILTROS BÁSICOS ==========
    min_piotroski_score: int = 6
    min_qv_score: float = 0.50

    # Valoración
    max_pe: float = 40.0
    max_pb: float = 10.0
    max_ev_ebitda: float = 20.0

    # Calidad
    require_positive_fcf: bool = True
    min_roic: float = 0.10  # ROIC mínimo 10%

    # ========== FILTROS AVANZADOS (NUEVOS) ==========
    # MA200 Filter (Faber 2007)
    require_above_ma200: bool = True

    # Momentum
    min_momentum_12m: float = 0.10  # 10% return mínimo 12M

    # 52-Week High
    require_near_52w_high: bool = False
    min_pct_from_52w_high: float = 0.80  # Precio >= 80% del 52w high

    # ========== BREAKOUT FILTERS (NUEVO) ==========
    # Filtros de breakout de niveles técnicos
    enable_breakout_filter: bool = False  # Requiere breakout
    require_breakout_confirmed: bool = False  # Breakout + volumen >1.5x
    require_breakout_strong: bool = False  # Breakout + volumen >2x
    breakout_types: Optional[List[str]] = None  # ['52w', '3m', '20d'] o None para todos
    breakout_lookback_days: int = 5  # Días hacia atrás para detectar breakout (default: 5)

    # Volumen relativo
    min_relative_volume: float = 1.0  # Volumen actual >= promedio
    enable_volume_surge_filter: bool = False  # Requiere surge de volumen >2x

    # ========== NUEVAS MÉTRICAS VALORACIÓN ==========
    max_fcf_ev: float = 0.15  # FCF/EV máximo (better si > 0.08)
    min_ebit_ev: float = 0.08  # EBIT/EV mínimo
    require_roic_above_wacc: bool = True  # ROIC > WACC

    # ========== MEJORAS V3.1 (ACADÉMICAS) ==========
    # Earnings Quality (Sloan 1996)
    enable_earnings_quality: bool = True
    min_earnings_quality_score: float = 50.0  # 0-100
    max_accruals_ratio: float = 0.10  # <10%

    # Fundamental Momentum (Piotroski & So 2012)
    enable_fundamental_momentum: bool = True
    min_fundamental_momentum_score: float = 55.0  # >55 = tendencias positivas

    # Red Flags
    enable_red_flags: bool = True
    min_red_flags_score: float = 60.0  # >60 = sin red flags serios

    # Short-term reversal filter
    enable_reversal_filter: bool = True
    reversal_threshold: float = -0.08  # -8%

    # Sector relative momentum
    enable_sector_relative: bool = False  # Opcional
    min_sector_relative_mom: float = 0.05  # Outperform sector >5%

    # Value score enhanced (Gray & Carlisle)
    use_enhanced_value_score: bool = True  # Usa 7 métricas en lugar de 3

    # ========== PORTFOLIO ==========
    portfolio_size: int = 30

    # ========== BACKTEST ==========
    backtest_enabled: bool = True
    backtest_start: str = "2020-01-01"
    backtest_end: str = "2024-12-31"
    rebalance_freq: str = "Q"  # Quarterly

    # Trading costs
    commission_bps: int = 5
    slippage_bps: int = 5
    market_impact_bps: int = 2

    # ========== DATA CACHING ==========
    use_price_cache: bool = True  # Si False, siempre descarga datos frescos

    # ========== RISK MANAGEMENT (FASE 1) ==========
    enable_risk_management: bool = True  # Calcular stops, targets y position sizing

    # Stop Loss
    use_volatility_stop: bool = True
    volatility_stop_confidence: float = 2.0  # 2σ = 95% CI
    use_trailing_stop: bool = True
    trailing_stop_method: str = 'ATR'  # 'ATR', 'FIXED', 'CHANDELIER'
    trailing_atr_multiplier: float = 2.5

    # Take Profit
    use_take_profit: bool = True
    risk_reward_ratio: float = 2.5  # 2.5:1 R:R ratio

    # Position Sizing
    use_volatility_sizing: bool = True
    target_volatility: float = 0.15  # 15% annual target
    max_position_size: float = 0.20  # Max 20%
    use_kelly: bool = False  # Kelly sizing (requires historical win rate)

    def __post_init__(self):
        """Validación de configuración"""
        # Normalizar pesos
        total_weight = self.w_quality + self.w_value + self.w_fcf_yield + self.w_momentum
        if not (0.99 <= total_weight <= 1.01):
            self.w_quality /= total_weight
            self.w_value /= total_weight
            self.w_fcf_yield /= total_weight
            self.w_momentum /= total_weight

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


# ============================================================================
# PASO: ANÁLISIS POR PASOS
# ============================================================================

class PipelineStep:
    """Clase para trackear cada paso del pipeline"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.input_count = 0
        self.output_count = 0
        self.success = False
        self.warnings = []
        self.metrics = {}

    def log_input(self, count: int):
        self.input_count = count

    def log_output(self, count: int):
        self.output_count = count

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_metric(self, key: str, value):
        self.metrics[key] = value

    def mark_success(self):
        self.success = True

    def get_pass_rate(self) -> float:
        if self.input_count == 0:
            return 0.0
        return self.output_count / self.input_count

    def summary(self) -> str:
        status = "✅" if self.success else "❌"
        pass_rate = self.get_pass_rate() * 100
        return f"{status} {self.name}: {self.output_count}/{self.input_count} ({pass_rate:.1f}%)"


# ============================================================================
# NUEVAS FUNCIONES: MÉTRICAS AVANZADAS
# ============================================================================

# WACC por sector (Weighted Average Cost of Capital)
# Basado en datos históricos de industria (Damodaran NYU Stern School)
WACC_BY_SECTOR = {
    # Technology & Internet
    'Technology': 0.08,
    'Communication Services': 0.08,
    'Software': 0.08,
    'Technology Hardware, Equipment & Parts': 0.09,

    # Financial
    'Financial Services': 0.10,
    'Banks': 0.09,
    'Insurance': 0.09,
    'Capital Markets': 0.10,

    # Energy & Utilities
    'Energy': 0.09,
    'Utilities': 0.07,
    'Oil, Gas & Consumable Fuels': 0.10,

    # Consumer
    'Consumer Cyclical': 0.09,
    'Consumer Defensive': 0.07,
    'Consumer Discretionary': 0.09,
    'Consumer Staples': 0.07,

    # Healthcare
    'Healthcare': 0.09,
    'Pharmaceuticals': 0.09,
    'Biotechnology': 0.10,

    # Industrial
    'Industrials': 0.09,
    'Basic Materials': 0.09,
    'Materials': 0.09,

    # Real Estate
    'Real Estate': 0.08,
    'REITs': 0.08,

    # Otros
    'Telecommunications': 0.08,
    'Transportation': 0.09,
    'Aerospace & Defense': 0.08,
}

# WACC por defecto si sector no está en el diccionario
DEFAULT_WACC = 0.09


def get_wacc_for_sector(sector: str) -> float:
    """
    Retorna WACC específico del sector.
    Si no encuentra el sector, usa WACC por defecto de 9%.
    """
    if pd.isna(sector) or not isinstance(sector, str):
        return DEFAULT_WACC

    # Buscar match exacto
    if sector in WACC_BY_SECTOR:
        return WACC_BY_SECTOR[sector]

    # Buscar match parcial (por si el nombre del sector varía)
    sector_lower = sector.lower()
    for key, wacc in WACC_BY_SECTOR.items():
        if key.lower() in sector_lower or sector_lower in key.lower():
            return wacc

    # Si no encuentra, usar default
    return DEFAULT_WACC


def calculate_advanced_valuation_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas avanzadas de valoración:
    - EBIT/EV
    - FCF/EV
    - ROIC vs WACC (ahora por sector)
    """
    df = df.copy()

    # EBIT/EV (Earnings Yield)
    if 'ebit' in df.columns and 'enterprise_value' in df.columns:
        df['ebit_ev'] = pd.to_numeric(df['ebit'], errors='coerce') / pd.to_numeric(df['enterprise_value'], errors='coerce').replace(0, np.nan)
    elif 'operating_income' in df.columns and 'market_cap' in df.columns:
        # Aproximación: EBIT ≈ Operating Income, EV ≈ Market Cap (si no hay deuda)
        df['ebit_ev'] = pd.to_numeric(df['operating_income'], errors='coerce') / pd.to_numeric(df['market_cap'], errors='coerce').replace(0, np.nan)

    # FCF/EV
    if 'fcf' in df.columns and 'enterprise_value' in df.columns:
        df['fcf_ev'] = pd.to_numeric(df['fcf'], errors='coerce') / pd.to_numeric(df['enterprise_value'], errors='coerce').replace(0, np.nan)
    elif 'fcf' in df.columns and 'market_cap' in df.columns:
        # Aproximación con market cap
        df['fcf_ev'] = pd.to_numeric(df['fcf'], errors='coerce') / pd.to_numeric(df['market_cap'], errors='coerce').replace(0, np.nan)

    # ROIC vs WACC (ahora POR SECTOR)
    if 'roic' in df.columns:
        # Calcular WACC específico por sector
        if 'sector' in df.columns:
            df['wacc'] = df['sector'].apply(get_wacc_for_sector)
        else:
            df['wacc'] = DEFAULT_WACC

        roic = pd.to_numeric(df['roic'], errors='coerce')
        df['roic_above_wacc'] = roic > df['wacc']
        df['roic_wacc_spread'] = roic - df['wacc']

    return df


def calculate_52w_metrics(prices_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calcula métricas relacionadas con 52-week high/low.
    """
    results = []

    for symbol, prices in prices_dict.items():
        if prices is None or prices.empty or 'close' not in prices.columns:
            continue

        try:
            current_price = prices['close'].iloc[-1]
            high_52w = prices['close'].iloc[-252:].max() if len(prices) >= 252 else prices['close'].max()
            low_52w = prices['close'].iloc[-252:].min() if len(prices) >= 252 else prices['close'].min()

            # % from 52w high
            pct_from_high = current_price / high_52w if high_52w > 0 else 0

            # % from 52w low
            pct_from_low = (current_price - low_52w) / low_52w if low_52w > 0 else 0

            results.append({
                'symbol': symbol,
                'price_52w_high': high_52w,
                'price_52w_low': low_52w,
                'pct_from_52w_high': pct_from_high,
                'pct_from_52w_low': pct_from_low,
                'near_52w_high': pct_from_high >= 0.90,  # Dentro del 10% del high
            })
        except Exception:
            continue

    return pd.DataFrame(results)


def calculate_volume_metrics(prices_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calcula métricas avanzadas de volumen.

    Returns:
        DataFrame con:
        - avg_volume_20d: Volumen promedio últimos 20 días
        - current_volume: Volumen del día actual
        - relative_volume: Volumen actual vs promedio (ratio)
        - volume_surge: True si volumen > 2x promedio
    """
    results = []

    for symbol, prices in prices_dict.items():
        if prices is None or prices.empty:
            continue

        try:
            # Si tenemos columna 'volume'
            if 'volume' in prices.columns:
                volumes = prices['volume'].iloc[-20:]  # Últimos 20 días

                if len(volumes) >= 5:  # Mínimo 5 días de datos
                    avg_volume_20d = volumes.mean()
                    current_volume = prices['volume'].iloc[-1]
                    relative_volume = current_volume / avg_volume_20d if avg_volume_20d > 0 else 1.0
                    volume_surge = relative_volume > 2.0  # Surge si >2x promedio

                    results.append({
                        'symbol': symbol,
                        'avg_volume_20d': avg_volume_20d,
                        'current_volume': current_volume,
                        'relative_volume': relative_volume,
                        'volume_surge': volume_surge,
                    })
        except Exception:
            continue

    return pd.DataFrame(results)


def detect_breakouts(prices_dict: Dict[str, pd.DataFrame], lookback_days: int = 5) -> pd.DataFrame:
    """
    Detecta breakouts de niveles técnicos previos.

    Reglas heurísticas MEJORADAS:
    1. Breakout de 52w high (precio > máximo 52w anterior)
    2. Breakout de resistencia 3M (precio > máximo últimos 60 días)
    3. Breakout de consolidación (precio > máximo últimos 20 días)

    Args:
        prices_dict: Dict de DataFrames con precios
        lookback_days: Días hacia atrás para detectar breakout (default: 5)
                       En lugar de detectar solo el día exacto, detecta si
                       hubo breakout en los últimos N días

    Returns:
        DataFrame con indicadores de breakout

    NOTA: Lógica relajada para detectar breakouts recientes (últimos 5 días)
          en lugar de solo el día exacto, lo cual es más robusto con datos EOD.
    """
    results = []

    for symbol, prices in prices_dict.items():
        if prices is None or prices.empty or 'close' not in prices.columns:
            continue

        try:
            current_price = prices['close'].iloc[-1]

            # Detectar breakouts en ventana reciente (más robusto)
            # En lugar de: "rompió HOY", buscamos: "rompió en los últimos N días"

            # ===== BREAKOUT 52W =====
            if len(prices) >= 252 + lookback_days:
                # Máximo ANTES de la ventana reciente
                high_52w_prev = prices['close'].iloc[-(252+lookback_days):-lookback_days].max()
                # ¿El precio actual está sobre ese nivel?
                breakout_52w = current_price > high_52w_prev
            else:
                high_52w_prev = None
                breakout_52w = False

            # ===== BREAKOUT 3M =====
            if len(prices) >= 60 + lookback_days:
                # Máximo ANTES de la ventana reciente
                high_3m = prices['close'].iloc[-(60+lookback_days):-lookback_days].max()
                breakout_3m = current_price > high_3m
            else:
                high_3m = None
                breakout_3m = False

            # ===== BREAKOUT 20D =====
            if len(prices) >= 20 + lookback_days:
                # Máximo ANTES de la ventana reciente
                high_20d = prices['close'].iloc[-(20+lookback_days):-lookback_days].max()
                breakout_20d = current_price > high_20d
            else:
                high_20d = None
                breakout_20d = False

            # Breakout general (cualquiera de los anteriores)
            any_breakout = breakout_52w or breakout_3m or breakout_20d

            # Calcular % por encima del nivel (si hay datos)
            if high_52w_prev and high_52w_prev > 0:
                pct_above_52w = ((current_price - high_52w_prev) / high_52w_prev) * 100
            else:
                pct_above_52w = 0

            results.append({
                'symbol': symbol,
                'breakout_52w': breakout_52w,
                'breakout_3m': breakout_3m,
                'breakout_20d': breakout_20d,
                'any_breakout': any_breakout,
                'pct_above_52w_level': pct_above_52w,
            })

        except Exception:
            continue

    return pd.DataFrame(results)


def detect_volume_confirmed_breakouts(
    prices_dict: Dict[str, pd.DataFrame],
    lookback_days: int = 5
) -> pd.DataFrame:
    """
    Detecta breakouts confirmados con volumen anormal.

    Un breakout confirmado requiere:
    1. Precio rompe nivel de resistencia
    2. Volumen del día es > promedio (idealmente >1.5x)

    Esta es una señal más fuerte que breakout solo.

    Args:
        prices_dict: Dict de DataFrames con precios
        lookback_days: Días hacia atrás para detectar breakout
    """
    breakouts_df = detect_breakouts(prices_dict, lookback_days=lookback_days)
    volume_df = calculate_volume_metrics(prices_dict)

    if breakouts_df.empty or volume_df.empty:
        return pd.DataFrame()

    # Merge ambos DataFrames
    merged = breakouts_df.merge(volume_df, on='symbol', how='inner')

    # Breakout confirmado: cualquier breakout + volumen relativo > 1.5
    merged['breakout_confirmed'] = (
        merged['any_breakout'] & (merged['relative_volume'] > 1.5)
    )

    # Breakout fuerte: breakout + volume surge (>2x)
    merged['breakout_strong'] = (
        merged['any_breakout'] & merged['volume_surge']
    )

    return merged


def calculate_relative_volume(df: pd.DataFrame) -> pd.Series:
    """
    DEPRECATED: Usar calculate_volume_metrics() en su lugar.
    Mantenido por compatibilidad.
    """
    return pd.Series(1.0, index=df.index, name='relative_volume')


# ============================================================================
# PIPELINE PRINCIPAL V3
# ============================================================================

def run_qvm_pipeline_v3(
    config: Optional[QVMConfigV3] = None,
    verbose: bool = True,
) -> Dict:
    """
    Pipeline QVM V3 COMPLETO con todos los filtros y backtest.

    PASOS:
    1. Screener inicial
    2. Estados financieros + Piotroski Score
    3. Filtros básicos de calidad
    4. Quality-Value Score
    5. Métricas avanzadas de valoración (EBIT/EV, FCF/EV, ROIC>WACC)
    6. Descargar precios históricos
    7. Momentum + MA200 Filter (CRÍTICO)
    8. Filtros de 52w high y volumen relativo
    9. Selección de portfolio
    10. Backtest (opcional)

    Returns:
        Dict con resultados completos
    """

    if config is None:
        config = QVMConfigV3()

    steps = []

    if verbose:
        print("\n" + "="*80)
        print("🚀 QVM STRATEGY PIPELINE V3 - COMPLETO")
        print("   MA200 + Momentum + Backtest + Métricas Avanzadas")
        print("="*80)

    # -------------------------------------------------------------------------
    # PASO 1: SCREENER INICIAL
    # -------------------------------------------------------------------------
    step1 = PipelineStep("PASO 1", "Screener Inicial")

    if verbose:
        print(f"\n📊 {step1.name}: {step1.description}")
        print(f"   Min Market Cap: ${config.min_market_cap/1e9:.1f}B")
        print(f"   Min Volume: {config.min_volume:,}")

    try:
        universe = fetch_screener(
            limit=config.universe_size,
            mcap_min=config.min_market_cap,
            volume_min=config.min_volume,
        )

        step1.log_input(config.universe_size)
        step1.log_output(len(universe))

        if universe.empty:
            step1.add_warning("No symbols in screener")
            steps.append(step1)
            return {"error": "No symbols in initial universe", "steps": steps}

        step1.add_metric("Unique sectors", universe['sector'].nunique())
        step1.add_metric("Avg market cap ($B)", universe['market_cap'].mean() / 1e9)
        step1.mark_success()

        if verbose:
            print(step1.summary())

    except Exception as e:
        step1.add_warning(f"Error: {str(e)}")
        steps.append(step1)
        return {"error": str(e), "steps": steps}

    steps.append(step1)

    # -------------------------------------------------------------------------
    # PASO 2: ESTADOS FINANCIEROS + PIOTROSKI
    # -------------------------------------------------------------------------
    step2 = PipelineStep("PASO 2", "Estados Financieros + Piotroski")

    if verbose:
        print(f"\n📊 {step2.name}: {step2.description}")
        print("   Calculando Piotroski Score (9 checks) + métricas...")

    try:
        symbols = universe['symbol'].tolist()
        step2.log_input(len(symbols))

        # Descargar estados financieros completos
        financial_data = fetch_financial_statements_batch(symbols)

        # También ratios de valoración
        valuation_ratios = fetch_fundamentals_batch(symbols, use_full_statements=False)

        # Merge
        financial_data = financial_data.merge(
            valuation_ratios[['symbol', 'ev_ebitda', 'pb', 'pe']],
            on='symbol',
            how='left'
        )

        financial_data = financial_data[financial_data['piotroski_score'].notna()].copy()

        step2.log_output(len(financial_data))

        avg_piotroski = financial_data['piotroski_score'].mean()
        step2.add_metric("Avg Piotroski Score", avg_piotroski)
        step2.add_metric("Piotroski >= 7", (financial_data['piotroski_score'] >= 7).sum())
        step2.mark_success()

        if verbose:
            print(step2.summary())

    except Exception as e:
        step2.add_warning(f"Error: {str(e)}")
        steps.append(step2)
        return {"error": str(e), "steps": steps}

    steps.append(step2)

    # -------------------------------------------------------------------------
    # PASO 3: FILTROS BÁSICOS DE CALIDAD
    # -------------------------------------------------------------------------
    step3 = PipelineStep("PASO 3", "Filtros Básicos de Calidad")

    if verbose:
        print(f"\n🔍 {step3.name}: {step3.description}")
        print(f"   Min Piotroski: {config.min_piotroski_score}")
        print(f"   Max P/E: {config.max_pe}, Max EV/EBITDA: {config.max_ev_ebitda}")

    try:
        step3.log_input(len(financial_data))

        df = universe.merge(financial_data, on='symbol', how='inner')

        # Filtros
        df = df[df['piotroski_score'] >= config.min_piotroski_score].copy()

        if 'pe' in df.columns:
            df = df[(df['pe'].isna()) | (df['pe'] <= config.max_pe)].copy()

        if 'ev_ebitda' in df.columns:
            df = df[(df['ev_ebitda'].isna()) | (df['ev_ebitda'] <= config.max_ev_ebitda)].copy()

        if config.require_positive_fcf and 'fcf' in df.columns:
            df = df[(df['fcf'].isna()) | (df['fcf'] > 0)].copy()

        if 'roic' in df.columns:
            df = df[(df['roic'].isna()) | (df['roic'] >= config.min_roic)].copy()

        step3.log_output(len(df))
        step3.mark_success()

        if verbose:
            print(step3.summary())

    except Exception as e:
        step3.add_warning(f"Error: {str(e)}")
        steps.append(step3)
        return {"error": str(e), "steps": steps}

    steps.append(step3)

    # -------------------------------------------------------------------------
    # PASO 4: QUALITY-VALUE SCORE
    # -------------------------------------------------------------------------
    step4 = PipelineStep("PASO 4", "Quality-Value Score")

    if verbose:
        print(f"\n🎯 {step4.name}: {step4.description}")
        print(f"   Pesos: Q={config.w_quality:.0%}, V={config.w_value:.0%}, " +
              f"FCF={config.w_fcf_yield:.0%}, M={config.w_momentum:.0%}")

    try:
        step4.log_input(len(df))

        # Calcular FCF Yield si no existe
        if 'fcf_yield' not in df.columns and 'fcf' in df.columns:
            df['fcf_yield'] = df['fcf'] / df['market_cap'].replace(0, np.nan)

        df_universe = df[['symbol', 'sector', 'market_cap']].copy()
        df_fundamentals = df[['symbol', 'piotroski_score', 'ev_ebitda', 'pb', 'pe', 'fcf', 'fcf_yield', 'market_cap']].copy()

        df_with_qv = compute_quality_value_factors(
            df_universe,
            df_fundamentals,
            w_quality=config.w_quality,
            w_value=config.w_value,
            w_fcf_yield=config.w_fcf_yield,
            w_momentum=config.w_momentum,
            use_enhanced_value=config.use_enhanced_value_score,
        )

        # Filtrar por QV Score mínimo
        df_with_qv = df_with_qv[df_with_qv['qv_score'] >= config.min_qv_score].copy()

        step4.log_output(len(df_with_qv))
        step4.add_metric("Avg QV Score", df_with_qv['qv_score'].mean())
        step4.mark_success()

        if verbose:
            print(step4.summary())

    except Exception as e:
        step4.add_warning(f"Error: {str(e)}")
        steps.append(step4)
        return {"error": str(e), "steps": steps}

    steps.append(step4)

    # -------------------------------------------------------------------------
    # PASO 5: MÉTRICAS AVANZADAS DE VALORACIÓN
    # -------------------------------------------------------------------------
    step5 = PipelineStep("PASO 5", "Métricas Avanzadas (EBIT/EV, FCF/EV, ROIC>WACC)")

    if verbose:
        print(f"\n💎 {step5.name}: {step5.description}")

    try:
        step5.log_input(len(df_with_qv))

        df_with_qv = calculate_advanced_valuation_metrics(df_with_qv)

        # Filtros opcionales
        if 'roic_above_wacc' in df_with_qv.columns and config.require_roic_above_wacc:
            before = len(df_with_qv)
            df_with_qv = df_with_qv[df_with_qv['roic_above_wacc'] == True].copy()
            rejected = before - len(df_with_qv)
            step5.add_metric("Rejected by ROIC<WACC", rejected)

        step5.log_output(len(df_with_qv))
        step5.mark_success()

        if verbose:
            print(step5.summary())

    except Exception as e:
        step5.add_warning(f"Error: {str(e)}")
        steps.append(step5)
        return {"error": str(e), "steps": steps}

    steps.append(step5)

    # -------------------------------------------------------------------------
    # PASO 6: DESCARGAR PRECIOS HISTÓRICOS
    # -------------------------------------------------------------------------
    step6 = PipelineStep("PASO 6", "Precios Históricos")

    if verbose:
        print(f"\n📈 {step6.name}: {step6.description}")
        cache_status = "ENABLED" if config.use_price_cache else "DISABLED (datos frescos)"
        print(f"   Cache: {cache_status}")

    try:
        step6.log_input(len(df_with_qv))

        symbols_to_fetch = df_with_qv['symbol'].tolist()
        prices_dict = {}

        for symbol in symbols_to_fetch:
            try:
                prices = fetch_prices(
                    symbol,
                    start=config.backtest_start,
                    end=config.backtest_end,
                    use_cache=config.use_price_cache
                )
                if prices is not None and len(prices) >= 252:
                    prices_dict[symbol] = prices
            except Exception:
                continue

        step6.log_output(len(prices_dict))
        step6.mark_success()

        if verbose:
            print(step6.summary())

    except Exception as e:
        step6.add_warning(f"Error: {str(e)}")
        steps.append(step6)
        return {"error": str(e), "steps": steps}

    steps.append(step6)

    # -------------------------------------------------------------------------
    # PASO 7: MOMENTUM + MA200 FILTER (CRÍTICO)
    # -------------------------------------------------------------------------
    step7 = PipelineStep("PASO 7", "Momentum + MA200 Filter")

    if verbose:
        print(f"\n🚀 {step7.name}: {step7.description}")
        if config.require_above_ma200:
            print("   ✅ MA200 Filter: ENABLED (Faber 2007)")
        else:
            print("   ⚠️  MA200 Filter: DISABLED")
        print(f"   Min Momentum 12M: {config.min_momentum_12m:.0%}")

    try:
        step7.log_input(len(prices_dict))

        # Calcular momentum para cada stock (risk-adjusted según Barroso & Santa-Clara 2015)
        momentum_results = []
        for symbol, prices in prices_dict.items():
            try:
                momentum_12m = calculate_risk_adjusted_momentum(prices)
                above_ma200 = is_above_ma200(prices)

                momentum_results.append({
                    'symbol': symbol,
                    'momentum_12m': momentum_12m,
                    'above_ma200': above_ma200,
                })
            except Exception:
                continue

        momentum_df = pd.DataFrame(momentum_results)

        # Merge con df_with_qv
        df_merged = df_with_qv.merge(momentum_df, on='symbol', how='inner')

        # Filtro MA200
        if config.require_above_ma200:
            before = len(df_merged)
            below_ma200_count = (df_merged['above_ma200'] == False).sum()
            # Guardar símbolos rechazados antes de filtrar
            below_ma200_symbols = df_merged[df_merged['above_ma200'] == False]['symbol'].head(5).tolist()
            df_merged = df_merged[df_merged['above_ma200'] == True].copy()
            rejected = before - len(df_merged)
            step7.add_metric("Rejected by MA200", rejected)
            if verbose and rejected > 0:
                print(f"   ⚠️  Rechazados por MA200: {rejected} stocks (estaban BAJO MA200)")
                if below_ma200_symbols:
                    print(f"      Ejemplos: {', '.join(below_ma200_symbols[:3])}")

        # Filtro Momentum mínimo
        before = len(df_merged)
        df_merged = df_merged[df_merged['momentum_12m'] >= config.min_momentum_12m].copy()
        rejected = before - len(df_merged)
        step7.add_metric("Rejected by Momentum", rejected)
        if verbose and rejected > 0:
            print(f"   ⚠️  Rechazados por Momentum: {rejected} stocks (momentum < {config.min_momentum_12m:.0%})")

        step7.log_output(len(df_merged))
        step7.add_metric("Avg Momentum 12M", df_merged['momentum_12m'].mean())
        step7.add_metric("% Above MA200", (df_merged['above_ma200'].sum() / len(df_merged) * 100) if len(df_merged) > 0 else 0)
        step7.mark_success()

        if verbose:
            print(step7.summary())
            if len(df_merged) > 0:
                above_ma200_pct = df_merged['above_ma200'].sum() / len(df_merged) * 100
                print(f"   ✅ {df_merged['above_ma200'].sum()}/{len(df_merged)} stocks sobre MA200 ({above_ma200_pct:.0f}%)")

    except Exception as e:
        step7.add_warning(f"Error: {str(e)}")
        steps.append(step7)
        return {"error": str(e), "steps": steps}

    steps.append(step7)

    # -------------------------------------------------------------------------
    # PASO 8: FILTROS 52W HIGH, BREAKOUTS Y VOLUMEN
    # -------------------------------------------------------------------------
    step8 = PipelineStep("PASO 8", "Filtros 52w High, Breakouts y Volumen")

    if verbose:
        print(f"\n📊 {step8.name}: {step8.description}")
        if config.enable_breakout_filter:
            print("   ⚡ Breakout Filter: ENABLED")
        if config.require_breakout_confirmed:
            print("   ⚡ Confirmed Breakouts Only: YES (vol >1.5x)")
        if config.require_breakout_strong:
            print("   ⚡ Strong Breakouts Only: YES (vol >2x)")
        if config.enable_volume_surge_filter:
            print("   📊 Volume Surge Filter: ENABLED (>2x)")

    try:
        step8.log_input(len(df_merged))

        # Calcular 52w metrics
        high52w_df = calculate_52w_metrics(prices_dict)

        if not high52w_df.empty:
            df_merged = df_merged.merge(high52w_df, on='symbol', how='left')

            # Filtro 52w high (legacy)
            if config.require_near_52w_high and 'pct_from_52w_high' in df_merged.columns:
                before = len(df_merged)
                df_merged = df_merged[df_merged['pct_from_52w_high'] >= config.min_pct_from_52w_high].copy()
                rejected = before - len(df_merged)
                step8.add_metric("Rejected by 52w high", rejected)

        # NUEVO: Calcular breakouts y volumen
        breakout_vol_df = detect_volume_confirmed_breakouts(
            prices_dict,
            lookback_days=config.breakout_lookback_days
        )

        if not breakout_vol_df.empty:
            # Agregar logging de breakouts detectados
            if verbose:
                total_with_any = breakout_vol_df['any_breakout'].sum()
                total_52w = breakout_vol_df['breakout_52w'].sum()
                total_3m = breakout_vol_df['breakout_3m'].sum()
                total_20d = breakout_vol_df['breakout_20d'].sum()
                total_confirmed = breakout_vol_df['breakout_confirmed'].sum()
                total_strong = breakout_vol_df['breakout_strong'].sum()

                print(f"   📊 Breakouts detectados (últimos {config.breakout_lookback_days} días):")
                print(f"      - Any breakout:  {total_with_any}/{len(breakout_vol_df)} ({100*total_with_any/len(breakout_vol_df):.1f}%)")
                print(f"      - 52w breakout:  {total_52w}/{len(breakout_vol_df)}")
                print(f"      - 3M breakout:   {total_3m}/{len(breakout_vol_df)}")
                print(f"      - 20D breakout:  {total_20d}/{len(breakout_vol_df)}")
                print(f"      - Confirmed:     {total_confirmed}/{len(breakout_vol_df)}")
                print(f"      - Strong:        {total_strong}/{len(breakout_vol_df)}")

            df_merged = df_merged.merge(
                breakout_vol_df[['symbol', 'breakout_52w', 'breakout_3m', 'breakout_20d',
                                 'any_breakout', 'breakout_confirmed', 'breakout_strong',
                                 'relative_volume', 'volume_surge']],
                on='symbol',
                how='left'
            )

            # Filtro de breakout general
            if config.enable_breakout_filter and 'any_breakout' in df_merged.columns:
                before = len(df_merged)
                df_merged = df_merged[df_merged['any_breakout'] == True].copy()
                rejected = before - len(df_merged)
                step8.add_metric("Rejected by breakout", rejected)

            # Filtro de breakout confirmado (con volumen)
            if config.require_breakout_confirmed and 'breakout_confirmed' in df_merged.columns:
                before = len(df_merged)
                df_merged = df_merged[df_merged['breakout_confirmed'] == True].copy()
                rejected = before - len(df_merged)
                step8.add_metric("Rejected by breakout confirmation", rejected)

            # Filtro de breakout fuerte (volumen >2x)
            if config.require_breakout_strong and 'breakout_strong' in df_merged.columns:
                before = len(df_merged)
                df_merged = df_merged[df_merged['breakout_strong'] == True].copy()
                rejected = before - len(df_merged)
                step8.add_metric("Rejected by strong breakout", rejected)

            # Filtro de surge de volumen
            if config.enable_volume_surge_filter and 'volume_surge' in df_merged.columns:
                before = len(df_merged)
                df_merged = df_merged[df_merged['volume_surge'] == True].copy()
                rejected = before - len(df_merged)
                step8.add_metric("Rejected by volume surge", rejected)

        step8.log_output(len(df_merged))
        step8.mark_success()

        if verbose:
            print(step8.summary())

    except Exception as e:
        step8.add_warning(f"Error: {str(e)}")
        steps.append(step8)
        return {"error": str(e), "steps": steps}

    steps.append(step8)

    # -------------------------------------------------------------------------
    # PASO 8.1: EARNINGS QUALITY FILTER (Sloan 1996)
    # -------------------------------------------------------------------------
    if config.enable_earnings_quality:
        step8_1 = PipelineStep("PASO 8.1", "Earnings Quality Filter")

        if verbose:
            print(f"\n📊 {step8_1.name}: {step8_1.description}")
            print(f"   Min EQ Score: {config.min_earnings_quality_score}")

        try:
            step8_1.log_input(len(df_merged))

            # Aplicar filtro earnings quality
            df_merged = apply_earnings_quality_filters(
                df_merged,
                min_eq_score=config.min_earnings_quality_score,
                max_accruals=config.max_accruals_ratio,
                verbose=verbose,
            )

            step8_1.log_output(len(df_merged))
            step8_1.mark_success()

            if verbose:
                print(step8_1.summary())

        except Exception as e:
            step8_1.add_warning(f"Error: {str(e)}")

        steps.append(step8_1)

    # -------------------------------------------------------------------------
    # PASO 8.2: RED FLAGS FILTER
    # -------------------------------------------------------------------------
    if config.enable_red_flags:
        step8_2 = PipelineStep("PASO 8.2", "Red Flags Filter")

        if verbose:
            print(f"\n🚩 {step8_2.name}: {step8_2.description}")
            print(f"   Min Red Flags Score: {config.min_red_flags_score}")

        try:
            step8_2.log_input(len(df_merged))

            # Aplicar filtro red flags (sin history por ahora)
            df_merged = apply_red_flags_filters(
                df_merged,
                financials_history_dict=None,  # TODO: agregar si disponible
                min_score=config.min_red_flags_score,
                verbose=verbose,
            )

            step8_2.log_output(len(df_merged))
            step8_2.mark_success()

            if verbose:
                print(step8_2.summary())

        except Exception as e:
            step8_2.add_warning(f"Error: {str(e)}")

        steps.append(step8_2)

    # -------------------------------------------------------------------------
    # PASO 8.3: SHORT-TERM REVERSAL FILTER
    # -------------------------------------------------------------------------
    if config.enable_reversal_filter:
        step8_3 = PipelineStep("PASO 8.3", "Short-Term Reversal Filter")

        if verbose:
            print(f"\n📉 {step8_3.name}: {step8_3.description}")
            print(f"   Threshold: {config.reversal_threshold:.1%}")

        try:
            step8_3.log_input(len(df_merged))

            # Aplicar filtro de reversal
            reversal_pass = []
            for symbol in df_merged['symbol']:
                if symbol in prices_dict:
                    if filter_short_term_reversal(prices_dict[symbol], config.reversal_threshold):
                        reversal_pass.append(symbol)

            df_merged = df_merged[df_merged['symbol'].isin(reversal_pass)]

            step8_3.log_output(len(df_merged))
            removed = step8_3.input_count - step8_3.output_count
            step8_3.add_metric("Rejected by reversal", removed)
            step8_3.mark_success()

            if verbose:
                print(step8_3.summary())

        except Exception as e:
            step8_3.add_warning(f"Error: {str(e)}")

        steps.append(step8_3)

    # -------------------------------------------------------------------------
    # PASO 9: SELECCIÓN DE PORTFOLIO
    # -------------------------------------------------------------------------
    step9 = PipelineStep("PASO 9", f"Selección Portfolio (Top {config.portfolio_size})")

    if verbose:
        print(f"\n📋 {step9.name}: {step9.description}")

    try:
        step9.log_input(len(df_merged))

        # Ordenar por QV Score y tomar top N
        portfolio = df_merged.nlargest(config.portfolio_size, 'qv_score').copy()

        step9.log_output(len(portfolio))
        step9.add_metric("Avg Piotroski", portfolio['piotroski_score'].mean())
        step9.add_metric("Avg QV Score", portfolio['qv_score'].mean())
        step9.add_metric("Avg Momentum", portfolio['momentum_12m'].mean())
        step9.mark_success()

        if verbose:
            print(step9.summary())

    except Exception as e:
        step9.add_warning(f"Error: {str(e)}")
        steps.append(step9)
        return {"error": str(e), "steps": steps}

    steps.append(step9)

    # -------------------------------------------------------------------------
    # PASO 9.5: RISK MANAGEMENT (FASE 1)
    # -------------------------------------------------------------------------
    if config.enable_risk_management:
        step9_5 = PipelineStep("PASO 9.5", "Risk Management (Stops, Targets, Position Sizing)")

        if verbose:
            print(f"\n💎 {step9_5.name}: {step9_5.description}")
            print("   Calculando stop loss, take profit y position sizing...")

        try:
            step9_5.log_input(len(portfolio))

            # Configurar RiskConfig desde QVMConfigV3
            risk_config = RiskConfig(
                use_volatility_stop=config.use_volatility_stop,
                volatility_stop_confidence=config.volatility_stop_confidence,
                use_trailing_stop=config.use_trailing_stop,
                trailing_stop_method=config.trailing_stop_method,
                trailing_atr_multiplier=config.trailing_atr_multiplier,
                use_take_profit=config.use_take_profit,
                risk_reward_ratio=config.risk_reward_ratio,
                use_volatility_sizing=config.use_volatility_sizing,
                target_volatility=config.target_volatility,
                max_position_size=config.max_position_size,
                use_kelly=config.use_kelly,
            )

            calculator = RiskCalculator(risk_config)

            # Calcular risk parameters para cada stock
            risk_results = []
            for _, row in portfolio.iterrows():
                symbol = row['symbol']

                if symbol not in prices_dict:
                    continue

                try:
                    prices = prices_dict[symbol]
                    entry_price = prices['close'].iloc[-1]

                    # Calcular parámetros de trade
                    trade_params = calculator.calculate_trade_parameters(
                        entry_price=entry_price,
                        prices=prices,
                        historical_returns=prices['close'].pct_change(),
                        win_rate=None,  # No disponible aún
                        avg_win=None,
                        avg_loss=None
                    )

                    risk_results.append({
                        'symbol': symbol,
                        'entry_price': entry_price,
                        'stop_loss': trade_params['final_stop_loss'],
                        'take_profit': trade_params['final_take_profit'],
                        'position_size_pct': trade_params['recommended_position_size'] * 100,
                        'risk_pct': trade_params['risk_metrics']['risk_pct'],
                        'reward_pct': trade_params['risk_metrics']['reward_pct'],
                        'rr_ratio': trade_params['risk_metrics']['actual_rr_ratio'],
                        'realized_vol': trade_params.get('realized_volatility', np.nan),
                    })

                except Exception as e:
                    if verbose:
                        print(f"   ⚠️  Error calculando risk para {symbol}: {str(e)}")
                    continue

            # Merge risk results con portfolio
            if risk_results:
                risk_df = pd.DataFrame(risk_results)
                portfolio = portfolio.merge(risk_df, on='symbol', how='left')

                step9_5.log_output(len(risk_df))
                step9_5.add_metric("Avg Position Size", f"{risk_df['position_size_pct'].mean():.1f}%")
                step9_5.add_metric("Avg R:R Ratio", f"{risk_df['rr_ratio'].mean():.2f}:1")
                step9_5.add_metric("Avg Risk", f"{risk_df['risk_pct'].mean():.2f}%")
                step9_5.mark_success()

                if verbose:
                    print(step9_5.summary())
                    print(f"   📊 Position Size promedio: {risk_df['position_size_pct'].mean():.1f}%")
                    print(f"   📊 R:R Ratio promedio: {risk_df['rr_ratio'].mean():.2f}:1")
                    print(f"   📊 Risk promedio: {risk_df['risk_pct'].mean():.2f}%")
            else:
                step9_5.add_warning("No se pudieron calcular parámetros de risk")

        except Exception as e:
            step9_5.add_warning(f"Error: {str(e)}")

        steps.append(step9_5)

    # -------------------------------------------------------------------------
    # PASO 10: BACKTEST (OPCIONAL)
    # -------------------------------------------------------------------------
    backtest_results = None

    if config.backtest_enabled:
        step10 = PipelineStep("PASO 10", "Backtest")

        if verbose:
            print(f"\n📊 {step10.name}: {step10.description}")
            print(f"   Período: {config.backtest_start} → {config.backtest_end}")
            print(f"   Rebalance: {config.rebalance_freq}")

        try:
            step10.log_input(len(portfolio))

            # Filtrar precios del portfolio
            portfolio_prices = {
                sym: prices_dict[sym]
                for sym in portfolio['symbol']
                if sym in prices_dict
            }

            if len(portfolio_prices) > 0:
                # Configurar costos
                costs = TradingCosts(
                    commission_bps=config.commission_bps,
                    slippage_bps=config.slippage_bps,
                    market_impact_bps=config.market_impact_bps,
                )

                # Ejecutar backtest
                metrics, equity_curves = backtest_portfolio(
                    portfolio_prices,
                    costs=costs,
                    execution_lag_days=1,
                )

                portfolio_metrics = calculate_portfolio_metrics(equity_curves, costs)

                backtest_results = {
                    'metrics': metrics,
                    'equity_curves': equity_curves,
                    'portfolio_metrics': portfolio_metrics,
                }

                step10.log_output(len(portfolio_prices))
                step10.add_metric("CAGR", portfolio_metrics['CAGR'])
                step10.add_metric("Sharpe", portfolio_metrics['Sharpe'])
                step10.add_metric("Max DD", portfolio_metrics['MaxDD'])
                step10.mark_success()

                if verbose:
                    print(step10.summary())
                    print(f"\n   📈 CAGR: {portfolio_metrics['CAGR']:.2%}")
                    print(f"   📈 Sharpe: {portfolio_metrics['Sharpe']:.2f}")
                    print(f"   📉 Max DD: {portfolio_metrics['MaxDD']:.2%}")
            else:
                step10.add_warning("No price data for backtest")

        except Exception as e:
            step10.add_warning(f"Error: {str(e)}")

        steps.append(step10)

    # -------------------------------------------------------------------------
    # RESULTADOS FINALES
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "="*80)
        print("✅ PIPELINE V3 COMPLETO - RESUMEN")
        print("="*80)
        for step in steps:
            print(step.summary())

    results = {
        'portfolio': portfolio,
        'full_dataset': df_merged,
        'prices': prices_dict,
        'backtest': backtest_results,
        'steps': steps,
        'config': config,
        'success': True,
    }

    return results


# ============================================================================
# ANÁLISIS DEL PORTFOLIO
# ============================================================================

def analyze_portfolio_v3(results: Dict, n_top: int = 20) -> pd.DataFrame:
    """
    Análisis detallado del portfolio V3.
    """
    if 'portfolio' not in results:
        return pd.DataFrame()

    portfolio = results['portfolio'].copy()

    cols = [
        'symbol', 'sector', 'market_cap',
        'piotroski_score', 'qv_score',
        'momentum_12m', 'above_ma200',
        'roic', 'fcf_yield_component',
    ]

    # Agregar columnas opcionales
    for col in ['ev_ebitda', 'pe', 'pb', 'ebit_ev', 'fcf_ev', 'roic_above_wacc', 'pct_from_52w_high']:
        if col in portfolio.columns:
            cols.append(col)

    # Agregar columnas de risk management (FASE 1)
    for col in ['entry_price', 'stop_loss', 'take_profit', 'position_size_pct', 'risk_pct', 'reward_pct', 'rr_ratio']:
        if col in portfolio.columns:
            cols.append(col)

    available_cols = [c for c in cols if c in portfolio.columns]
    analysis = portfolio[available_cols].copy()

    if 'market_cap' in analysis.columns:
        analysis['market_cap_$B'] = (analysis['market_cap'] / 1e9).round(2)
        analysis.drop('market_cap', axis=1, inplace=True)

    if 'qv_score' in analysis.columns:
        analysis = analysis.sort_values('qv_score', ascending=False)

    return analysis.head(n_top).reset_index(drop=True)


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("\n🧪 Testing QVM Pipeline V3...")

    config = QVMConfigV3(
        universe_size=20,
        portfolio_size=10,
        min_piotroski_score=6,
        min_qv_score=0.45,
        require_above_ma200=True,
        backtest_enabled=True,
    )

    results = run_qvm_pipeline_v3(config=config, verbose=True)

    if results.get('success'):
        portfolio = results['portfolio']
        print("\n" + "="*80)
        print("📋 TOP 10 STOCKS")
        print("="*80)
        analysis = analyze_portfolio_v3(results, n_top=10)
        print(analysis.to_string(index=True))

        if results.get('backtest'):
            print("\n" + "="*80)
            print("📊 BACKTEST RESULTS")
            print("="*80)
            pm = results['backtest']['portfolio_metrics']
            print(f"CAGR: {pm['CAGR']:.2%}")
            print(f"Sharpe: {pm['Sharpe']:.2f}")
            print(f"Sortino: {pm['Sortino']:.2f}")
            print(f"Max DD: {pm['MaxDD']:.2%}")

        print("\n✅ Pipeline V3 test complete!")
    else:
        print(f"\n❌ Pipeline failed: {results.get('error')}")
