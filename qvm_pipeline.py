"""
QVM Strategy Pipeline - VERSIÓN OPTIMIZADA ACADÉMICA
====================================================

Implementación completa basada en:
- Jegadeesh & Titman (1993): Momentum real
- Faber (2007): Filtro MA200 obligatorio  
- Piotroski (2000): F-Score de calidad
- Asness et al. (2019): Quality minus junk
- Sector-neutral factors

CAMBIOS CRÍTICOS vs versión anterior:
1. ✅ Momentum REAL calculado desde precios (no placeholder)
2. ✅ Filtro MA200 ANTES de selección (no opcional)
3. ✅ Reglas heurísticas ROBUSTAS (ROE>15%, Margin>30%)
4. ✅ F-Score OBLIGATORIO (min 6/9)
5. ✅ Pesos balanceados correctamente
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

# Imports de módulos
from data_fetcher import (
    fetch_screener,
    fetch_fundamentals_batch,
    fetch_prices,
)
from factor_calculator import compute_all_factors
from momentum_calculator import (
    calculate_momentum_batch,
    filter_above_ma200,
)
from piotroski_fscore import calculate_simplified_fscore
from screener_filters import FilterConfig, apply_all_filters
from backtest_engine import (
    backtest_portfolio,
    calculate_portfolio_metrics,
    TradingCosts,
)


# ============================================================================
# CONFIGURACIÓN ACADÉMICA ROBUSTA
# ============================================================================

@dataclass
class AcademicConfig:
    """
    Configuración basada en literatura académica.
    
    Valores calibrados según:
    - Novy-Marx (2013): Quality thresholds
    - Piotroski (2000): F-Score requirements
    - Faber (2007): MA200 filter
    """
    
    # ========== UNIVERSE ==========
    universe_size: int = 500
    min_market_cap: float = 2e9          # $2B+ (mid-cap+)
    min_volume: int = 1_000_000          # 1M daily volume
    
    # ========== QUALITY FILTERS (ROBUSTOS) ==========
    # CRÍTICO: Umbrales más altos que versión anterior
    min_roe: float = 0.15                # 15% ROE (quality threshold)
    min_roic: float = 0.12               # 12% ROIC  
    min_gross_margin: float = 0.30       # 30% margen (empresas con moat)
    require_positive_fcf: bool = True
    require_positive_ocf: bool = True
    
    # ========== VALUATION ==========
    max_pe: float = 40.0                 # P/E <= 40 (más estricto)
    max_ev_ebitda: float = 20.0          # EV/EBITDA <= 20 (más estricto)
    
    # ========== MOMENTUM & TREND (OBLIGATORIO) ==========
    require_above_ma200: bool = True     # ⭐ SIEMPRE True
    min_momentum_12m: float = 0.10       # 10% retorno 12M mínimo
    
    # ========== F-SCORE (OBLIGATORIO) ==========
    min_fscore: int = 6                  # F-Score >= 6 (quality check)
    
    # ========== PORTFOLIO ==========
    portfolio_size: int = 30             # Top 30 stocks
    
    # ========== FACTOR WEIGHTS (BALANCEADOS) ==========
    # Total debe sumar 1.0
    w_quality: float = 0.30              # Quality (incluye profitability)
    w_value: float = 0.25                # Value  
    w_momentum: float = 0.30             # Momentum (REAL desde precios)
    w_fscore: float = 0.15               # F-Score (confirma quality)
    
    # ========== BACKTEST ==========
    backtest_start: str = "2020-01-01"
    backtest_end: str = "2024-12-31"
    rebalance_freq: str = "M"            # Monthly rebalancing
    
    # ========== COSTOS ==========
    commission_bps: int = 5
    slippage_bps: int = 5
    market_impact_bps: int = 2
    
    def __post_init__(self):
        """Validación de configuración"""
        # Verificar que pesos sumen ~1.0
        total_weight = self.w_quality + self.w_value + self.w_momentum + self.w_fscore
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(f"Factor weights must sum to 1.0, got {total_weight}")
        
        # MA200 debe ser obligatorio
        if not self.require_above_ma200:
            raise ValueError("MA200 filter MUST be True (academic requirement)")


# ============================================================================
# FUNCIONES DE DIAGNÓSTICO
# ============================================================================

def print_funnel_stats(stage: str, count: int, total: int):
    """Imprime estadísticas del funnel de selección"""
    pct = (count / total * 100) if total > 0 else 0
    print(f"  {stage:30s}: {count:4d} ({pct:5.1f}%)")


def print_portfolio_characteristics(portfolio: pd.DataFrame):
    """Imprime características del portfolio final"""
    print("\n" + "="*70)
    print("📊 CARACTERÍSTICAS DEL PORTFOLIO")
    print("="*70)
    
    print(f"\n  Tamaño: {len(portfolio)} acciones")
    print(f"  Sectores únicos: {portfolio['sector'].nunique()}")
    
    print("\n  📈 QUALITY METRICS:")
    print(f"    F-Score promedio: {portfolio['fscore'].mean():.1f}/9.0")
    print(f"    ROE promedio: {portfolio['roe'].mean():.1%}")
    print(f"    ROIC promedio: {portfolio.get('roic', pd.Series([0])).mean():.1%}")
    print(f"    Gross Margin promedio: {portfolio['gross_margin'].mean():.1%}")
    
    print("\n  💰 VALUATION:")
    print(f"    P/E mediano: {portfolio['pe'].median():.1f}")
    print(f"    EV/EBITDA mediano: {portfolio['ev_ebitda'].median():.1f}")
    
    print("\n  🚀 MOMENTUM:")
    print(f"    Momentum 12M promedio: {portfolio['momentum_12m1m'].mean():.1%}")
    print(f"    % Price > MA200: {portfolio['above_ma200'].mean():.0%}")
    
    print("\n  🏢 DISTRIBUCIÓN POR SECTOR:")
    sector_dist = portfolio['sector'].value_counts()
    for sector, count in sector_dist.items():
        print(f"    {sector:25s}: {count:2d} ({count/len(portfolio)*100:.0f}%)")


# ============================================================================
# PIPELINE PRINCIPAL (OPTIMIZADO)
# ============================================================================

def run_optimized_qvm_strategy(
    config: Optional[AcademicConfig] = None,
    verbose: bool = True,
) -> Dict:
    """
    Ejecuta estrategia QVM OPTIMIZADA con todas las mejoras académicas.
    
    ORDEN DEL PIPELINE (CRÍTICO):
    1. Screener inicial (market cap, volume)
    2. Fundamentals download
    3. Filtros de calidad ROBUSTOS (ROE>15%, Margin>30%, etc)
    4. F-Score >= 6 (Piotroski)
    5. Precios históricos
    6. Momentum REAL + MA200 filter (OBLIGATORIO)
    7. Factores QVM (sector-neutral)
    8. Composite score con pesos balanceados
    9. Top N selection
    10. Backtest
    
    Args:
        config: Configuración (usa default si None)
        verbose: Imprimir progreso
        
    Returns:
        Dict con resultados completos
    """
    
    if config is None:
        config = AcademicConfig()
    
    if verbose:
        print("\n" + "="*70)
        print("🚀 QVM STRATEGY - VERSIÓN OPTIMIZADA ACADÉMICA")
        print("="*70)
        print(f"   Basado en: Jegadeesh & Titman (1993), Faber (2007),")
        print(f"              Piotroski (2000), Asness et al. (2019)")
        print("="*70)
    
    total_initial = config.universe_size
    
    # -------------------------------------------------------------------------
    # PASO 1: SCREENER INICIAL
    # -------------------------------------------------------------------------
    if verbose:
        print("\n📊 PASO 1/10: Screener inicial")
        print(f"  Min Market Cap: ${config.min_market_cap/1e9:.1f}B")
        print(f"  Min Volume: {config.min_volume:,}")
    
    universe = fetch_screener(
        limit=config.universe_size,
        mcap_min=config.min_market_cap,
        volume_min=config.min_volume,
        use_cache=True,
    )
    
    if universe.empty:
        return {"error": "No symbols in initial universe"}
    
    if verbose:
        print_funnel_stats("Universe inicial", len(universe), total_initial)
    
    # -------------------------------------------------------------------------
    # PASO 2: FUNDAMENTALS
    # -------------------------------------------------------------------------
    if verbose:
        print("\n📊 PASO 2/10: Descargando fundamentales")
    
    fundamentals = fetch_fundamentals_batch(
        universe['symbol'].tolist(),
        use_cache=True,
    )
    
    if verbose:
        print_funnel_stats("Con datos fundamentales", len(fundamentals), total_initial)
    
    # -------------------------------------------------------------------------
    # PASO 3: FILTROS DE CALIDAD ROBUSTOS
    # -------------------------------------------------------------------------
    if verbose:
        print("\n🔍 PASO 3/10: Filtros de calidad ROBUSTOS")
        print(f"  Min ROE: {config.min_roe:.0%}")
        print(f"  Min Gross Margin: {config.min_gross_margin:.0%}")
        print(f"  Max P/E: {config.max_pe}")
        print(f"  Max EV/EBITDA: {config.max_ev_ebitda}")
    
    filter_config = FilterConfig(
        min_roe=config.min_roe,
        min_gross_margin=config.min_gross_margin,
        require_positive_fcf=config.require_positive_fcf,
        require_positive_ocf=config.require_positive_ocf,
        max_pe=config.max_pe,
        max_ev_ebitda=config.max_ev_ebitda,
        min_volume=config.min_volume,
        min_market_cap=config.min_market_cap,
    )
    
    df_merged = universe.merge(fundamentals, on='symbol', how='left')
    passed_filters, diagnostics = apply_all_filters(df_merged, filter_config)
    
    if verbose:
        print_funnel_stats("Pasaron filtros calidad", len(passed_filters), total_initial)
        
        # Mostrar razones de rechazo
        rejection_reasons = diagnostics[diagnostics['pass_all'] == False]['reason'].value_counts()
        if len(rejection_reasons) > 0:
            print("\n  ❌ Top razones de rechazo:")
            for reason, count in rejection_reasons.head(3).items():
                print(f"    {reason}: {count}")
    
    if passed_filters.empty:
        return {"error": "No symbols passed quality filters"}
    
    # -------------------------------------------------------------------------
    # PASO 4: F-SCORE (OBLIGATORIO)
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n🏆 PASO 4/10: F-Score >= {config.min_fscore} (Piotroski)")
    
    df_for_fscore = passed_filters.merge(fundamentals, on='symbol', how='left')
    df_for_fscore['fscore'] = calculate_simplified_fscore(df_for_fscore)
    
    # FILTRO OBLIGATORIO: F-Score >= min_fscore
    df_fscore_passed = df_for_fscore[
        df_for_fscore['fscore'] >= config.min_fscore
    ].copy()
    
    if verbose:
        print_funnel_stats(f"F-Score >= {config.min_fscore}", len(df_fscore_passed), total_initial)
        print(f"  📊 F-Score promedio: {df_fscore_passed['fscore'].mean():.1f}/9.0")
        print(f"  📊 F-Score mediano: {df_fscore_passed['fscore'].median():.1f}/9.0")
    
    if df_fscore_passed.empty:
        return {"error": f"No symbols with F-Score >= {config.min_fscore}"}
    
    # -------------------------------------------------------------------------
    # PASO 5: DESCARGAR PRECIOS HISTÓRICOS
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n📈 PASO 5/10: Descargando precios ({config.backtest_start} - {config.backtest_end})")
    
    symbols_to_fetch = df_fscore_passed['symbol'].tolist()
    prices_dict = {}
    failed = 0
    
    for i, symbol in enumerate(symbols_to_fetch):
        if verbose and (i + 1) % 50 == 0:
            print(f"  Progreso: {i+1}/{len(symbols_to_fetch)}")
        
        try:
            prices = fetch_prices(
                symbol,
                start=config.backtest_start,
                end=config.backtest_end,
            )
            
            # Requiere mínimo 1 año de datos
            if prices is not None and len(prices) >= 252:
                prices_dict[symbol] = prices
            else:
                failed += 1
        except Exception:
            failed += 1
    
    if verbose:
        print_funnel_stats("Con precios históricos", len(prices_dict), total_initial)
        if failed > 0:
            print(f"  ⚠️  {failed} símbolos sin datos suficientes")
    
    if len(prices_dict) == 0:
        return {"error": "No price data available"}
    
    # -------------------------------------------------------------------------
    # PASO 6: MOMENTUM REAL + MA200 FILTER (CRÍTICO)
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n🚀 PASO 6/10: Momentum REAL + Filtro MA200 (OBLIGATORIO)")
    
    # Calcular momentum desde precios
    momentum_df = calculate_momentum_batch(prices_dict)
    
    # Merge con df_fscore_passed
    df_with_momentum = df_fscore_passed.merge(
        momentum_df[['symbol', 'momentum_12m1m', 'above_ma200', 'composite_momentum']],
        on='symbol',
        how='inner'  # Solo los que tienen momentum
    )
    
    # FILTRO OBLIGATORIO: Price > MA200
    if config.require_above_ma200:
        df_ma200_passed = df_with_momentum[
            df_with_momentum['above_ma200'] == True
        ].copy()
        
        if verbose:
            print_funnel_stats("Price > MA200", len(df_ma200_passed), total_initial)
            rejected_ma200 = len(df_with_momentum) - len(df_ma200_passed)
            print(f"  ❌ Rechazados por MA200: {rejected_ma200}")
    else:
        df_ma200_passed = df_with_momentum.copy()
    
    # FILTRO ADICIONAL: Momentum mínimo
    df_final = df_ma200_passed[
        df_ma200_passed['momentum_12m1m'] >= config.min_momentum_12m
    ].copy()
    
    if verbose:
        print_funnel_stats(f"Momentum >= {config.min_momentum_12m:.0%}", len(df_final), total_initial)
        print(f"  📊 Momentum 12M promedio: {df_final['momentum_12m1m'].mean():.1%}")
    
    if df_final.empty:
        return {"error": "No symbols passed MA200 + Momentum filters"}
    
    # -------------------------------------------------------------------------
    # PASO 7: CALCULAR FACTORES QVM (SECTOR-NEUTRAL)
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n🧮 PASO 7/10: Calculando factores QVM (sector-neutral)")
    
    df_universe_clean = df_final[['symbol', 'sector', 'market_cap']].copy()
    df_fundamentals_clean = df_final[[
        'symbol', 'ev_ebitda', 'pb', 'pe', 
        'roe', 'roic', 'gross_margin', 
        'fcf', 'operating_cf'
    ]].copy()
    
    # Calcular factores (sector-neutral=True)
    df_with_factors = compute_all_factors(
        df_universe_clean,
        df_fundamentals_clean,
        sector_neutral=True,  # CRÍTICO para comparar tech vs tech, no tech vs utilities
        w_quality=config.w_quality / (config.w_quality + config.w_value + config.w_momentum),
        w_value=config.w_value / (config.w_quality + config.w_value + config.w_momentum),
        w_momentum=config.w_momentum / (config.w_quality + config.w_value + config.w_momentum),
    )
    
    # Merge con momentum real y fscore
    df_with_factors = df_with_factors.merge(
        df_final[[
            'symbol', 'momentum_12m1m', 'above_ma200', 
            'composite_momentum', 'fscore'
        ]],
        on='symbol',
        how='left'
    )
    
    if verbose:
        print(f"  ✅ Factores calculados para {len(df_with_factors)} símbolos")
    
    # -------------------------------------------------------------------------
    # PASO 8: COMPOSITE SCORE FINAL (CON PESOS BALANCEADOS)
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n🎯 PASO 8/10: Composite score final")
        print(f"  Pesos: Quality={config.w_quality:.0%}, Value={config.w_value:.0%}, "
              f"Momentum={config.w_momentum:.0%}, F-Score={config.w_fscore:.0%}")
    
    # SCORE FINAL usando momentum REAL (no placeholder)
    df_with_factors['composite_score'] = (
        config.w_quality * df_with_factors['quality_extended'] +
        config.w_value * df_with_factors['value_score'] +
        config.w_momentum * df_with_factors['composite_momentum'] +  # ⭐ REAL momentum
        config.w_fscore * (df_with_factors['fscore'] / 9.0)  # Normalizado 0-1
    )
    
    # Ranking final
    df_with_factors['final_rank'] = df_with_factors['composite_score'].rank(
        pct=True, 
        method='average'
    )
    
    # -------------------------------------------------------------------------
    # PASO 9: SELECCIÓN DE PORTFOLIO (TOP N)
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n📋 PASO 9/10: Seleccionando Top {config.portfolio_size} stocks")
    
    portfolio = df_with_factors.nlargest(
        config.portfolio_size, 
        'composite_score'
    ).copy()
    
    if verbose:
        print_funnel_stats("Portfolio final", len(portfolio), total_initial)
        print_portfolio_characteristics(portfolio)
    
    # -------------------------------------------------------------------------
    # PASO 10: BACKTEST
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n📊 PASO 10/10: Backtesting portfolio")
    
    # Filtrar precios
    portfolio_prices = {
        sym: prices_dict[sym] 
        for sym in portfolio['symbol'] 
        if sym in prices_dict
    }
    
    # Costos realistas
    costs = TradingCosts(
        commission_bps=config.commission_bps,
        slippage_bps=config.slippage_bps,
        market_impact_bps=config.market_impact_bps,
    )
    
    # Backtest
    metrics, equity_curves = backtest_portfolio(
        portfolio_prices,
        costs=costs,
        execution_lag_days=1,
    )
    
    # Métricas de portfolio
    port_metrics = calculate_portfolio_metrics(equity_curves, costs)
    
    if verbose:
        print("\n" + "="*70)
        print("📈 RESULTADOS BACKTEST")
        print("="*70)
        print(f"  CAGR: {port_metrics['CAGR']:.2%}")
        print(f"  Sharpe: {port_metrics['Sharpe']:.2f}")
        print(f"  Sortino: {port_metrics['Sortino']:.2f}")
        print(f"  Max Drawdown: {port_metrics['MaxDD']:.2%}")
        print(f"  Calmar: {port_metrics['Calmar']:.2f}")
        print(f"  Costos: {costs.round_trip_bps} bps round-trip")
    
    # -------------------------------------------------------------------------
    # RESULTADOS
    # -------------------------------------------------------------------------
    
    results = {
        'portfolio': portfolio,
        'backtest_metrics': metrics,
        'portfolio_metrics': port_metrics,
        'equity_curves': equity_curves,
        'funnel_stats': {
            'universe_initial': len(universe),
            'with_fundamentals': len(fundamentals),
            'passed_quality_filters': len(passed_filters),
            'passed_fscore': len(df_fscore_passed),
            'with_prices': len(prices_dict),
            'passed_ma200': len(df_ma200_passed) if config.require_above_ma200 else len(df_with_momentum),
            'passed_momentum': len(df_final),
            'final_portfolio': len(portfolio),
        },
        'config': config,
    }
    
    if verbose:
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETO - QVM OPTIMIZADO")
        print("="*70)
    
    return results


# ============================================================================
# ANÁLISIS Y COMPARACIÓN
# ============================================================================

def analyze_portfolio(results: Dict) -> pd.DataFrame:
    """
    Genera análisis detallado del portfolio.
    """
    portfolio = results['portfolio']
    
    analysis = portfolio[[
        'symbol', 'sector', 'market_cap',
        'fscore', 'composite_score', 'final_rank',
        'momentum_12m1m', 'roe', 'gross_margin',
        'pe', 'ev_ebitda', 'above_ma200'
    ]].copy()
    
    analysis = analysis.sort_values('composite_score', ascending=False)
    analysis = analysis.reset_index(drop=True)
    
    # Formatting
    analysis['market_cap'] = (analysis['market_cap'] / 1e9).round(1)  # en $B
    analysis = analysis.rename(columns={'market_cap': 'mcap_$b'})
    
    return analysis


def compare_with_spy(
    results: Dict,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compara portfolio con SPY benchmark.
    """
    from data_fetcher import fetch_prices
    from backtest_engine import backtest_single_symbol
    
    config = results['config']
    
    spy_prices = fetch_prices('SPY', config.backtest_start, config.backtest_end)
    
    if spy_prices is None:
        return pd.DataFrame()
    
    spy_result = backtest_single_symbol(
        spy_prices,
        'SPY',
        costs=TradingCosts(1, 1, 0),  # ETF costs
    )
    
    port = results['portfolio_metrics']
    
    comparison = pd.DataFrame({
        'Metric': ['CAGR', 'Sharpe', 'Sortino', 'Max DD', 'Calmar'],
        'QVM Portfolio': [
            f"{port['CAGR']:.2%}",
            f"{port['Sharpe']:.2f}",
            f"{port['Sortino']:.2f}",
            f"{port['MaxDD']:.2%}",
            f"{port['Calmar']:.2f}",
        ],
        'SPY Benchmark': [
            f"{spy_result.cagr:.2%}",
            f"{spy_result.sharpe:.2f}",
            f"{spy_result.sortino:.2f}",
            f"{spy_result.max_dd:.2%}",
            f"{spy_result.calmar:.2f}",
        ],
        'Outperformance': [
            f"{(port['CAGR'] - spy_result.cagr):.2%}",
            f"{(port['Sharpe'] - spy_result.sharpe):+.2f}",
            f"{(port['Sortino'] - spy_result.sortino):+.2f}",
            f"{(port['MaxDD'] - spy_result.max_dd):.2%}",
            f"{(port['Calmar'] - spy_result.calmar):+.2f}",
        ],
    })
    
    if verbose:
        print("\n" + "="*70)
        print("📊 COMPARACIÓN CON SPY")
        print("="*70)
        print(comparison.to_string(index=False))
    
    return comparison


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n🧪 Testing QVM Pipeline Optimized...")
    
    # Usar configuración default
    config = AcademicConfig()
    
    # Ejecutar estrategia
    results = run_optimized_qvm_strategy(
        config=config,
        verbose=True,
    )
    
    if 'error' in results:
        print(f"\n❌ Error: {results['error']}")
    else:
        # Análisis
        analysis = analyze_portfolio(results)
        print("\n" + "="*70)
        print("📋 TOP 10 STOCKS DEL PORTFOLIO")
        print("="*70)
        print(analysis.head(10).to_string(index=False))
        
        # Comparación con SPY
        comparison = compare_with_spy(results, verbose=True)
        
        print("\n✅ Pipeline optimizado completo!")
