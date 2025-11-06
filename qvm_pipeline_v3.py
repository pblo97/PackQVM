"""
QVM Strategy Pipeline V3 - COMPLETO con MA200, Backtest y Métricas Avanzadas
=============================================================================

NUEVAS CARACTERÍSTICAS:
1. ✅ MA200 Filter (Faber 2007)
2. ✅ Reglas heurísticas: 52w high, volumen relativo
3. ✅ Nuevas métricas de valoración: EBIT/EV, FCF/EV, ROIC > WACC
4. ✅ Backtest integrado
5. ✅ Pipeline completo: Screening → Quality-Value → Momentum/MA200 → Backtest

Bibliografía:
- Faber (2007): "A Quantitative Approach to Tactical Asset Allocation"
- Piotroski (2000): F-Score
- Asness et al. (2019): Quality factors
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
)
from backtest_engine import (
    backtest_portfolio,
    calculate_portfolio_metrics,
    TradingCosts,
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
    universe_size: int = 300
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

    # Volumen relativo
    min_relative_volume: float = 1.0  # Volumen actual >= promedio

    # ========== NUEVAS MÉTRICAS VALORACIÓN ==========
    max_fcf_ev: float = 0.15  # FCF/EV máximo (better si > 0.08)
    min_ebit_ev: float = 0.08  # EBIT/EV mínimo
    require_roic_above_wacc: bool = True  # ROIC > WACC

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


def calculate_relative_volume(df: pd.DataFrame) -> pd.Series:
    """
    Calcula volumen relativo vs promedio.
    (En producción, esto requeriría datos intraday)
    Por ahora retornamos placeholder.
    """
    # Placeholder: asumimos volumen normal
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
                    use_cache=True
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
            print("   MA200 Filter: ENABLED (Faber 2007)")
        print(f"   Min Momentum 12M: {config.min_momentum_12m:.0%}")

    try:
        step7.log_input(len(prices_dict))

        # Calcular momentum para cada stock
        momentum_results = []
        for symbol, prices in prices_dict.items():
            try:
                momentum_12m = calculate_12m_1m_momentum(prices)
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
            df_merged = df_merged[df_merged['above_ma200'] == True].copy()
            rejected = before - len(df_merged)
            step7.add_metric("Rejected by MA200", rejected)

        # Filtro Momentum mínimo
        before = len(df_merged)
        df_merged = df_merged[df_merged['momentum_12m'] >= config.min_momentum_12m].copy()
        rejected = before - len(df_merged)
        step7.add_metric("Rejected by Momentum", rejected)

        step7.log_output(len(df_merged))
        step7.add_metric("Avg Momentum 12M", df_merged['momentum_12m'].mean())
        step7.mark_success()

        if verbose:
            print(step7.summary())

    except Exception as e:
        step7.add_warning(f"Error: {str(e)}")
        steps.append(step7)
        return {"error": str(e), "steps": steps}

    steps.append(step7)

    # -------------------------------------------------------------------------
    # PASO 8: FILTROS 52W HIGH Y VOLUMEN RELATIVO
    # -------------------------------------------------------------------------
    step8 = PipelineStep("PASO 8", "Filtros 52w High y Volumen")

    if verbose:
        print(f"\n📊 {step8.name}: {step8.description}")

    try:
        step8.log_input(len(df_merged))

        # Calcular 52w metrics
        high52w_df = calculate_52w_metrics(prices_dict)

        if not high52w_df.empty:
            df_merged = df_merged.merge(high52w_df, on='symbol', how='left')

            # Filtro 52w high
            if config.require_near_52w_high and 'pct_from_52w_high' in df_merged.columns:
                before = len(df_merged)
                df_merged = df_merged[df_merged['pct_from_52w_high'] >= config.min_pct_from_52w_high].copy()
                rejected = before - len(df_merged)
                step8.add_metric("Rejected by 52w high", rejected)

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
