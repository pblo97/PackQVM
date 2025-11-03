"""
QVM Strategy Pipeline - Implementación Correcta
================================================

Este pipeline integra TODAS las mejoras académicas:
1. ✅ Momentum REAL desde precios (Jegadeesh & Titman, 1993)
2. ✅ Filtro MA200 obligatorio (Faber, 2007)
3. ✅ Piotroski F-Score (Piotroski, 2000)
4. ✅ Sector-neutral factors
5. ✅ Reglas heurísticas robustas

USO:
```python
from qvm_pipeline_corrected import run_qvm_strategy

results = run_qvm_strategy(
    universe_size=500,
    portfolio_size=30,
    require_ma200=True,  # ⭐ CRÍTICO
    min_fscore=6,
)
```
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta

# Imports de módulos custom
from data_fetcher import (
    fetch_screener,
    fetch_fundamentals_batch,
    fetch_prices,
)
from factor_calculator import compute_all_factors
from momentum_calculator import (
    integrate_real_momentum,
    filter_above_ma200,
    calculate_momentum_batch,
)
from piotroski_fscore import (
    filter_by_fscore,
    calculate_simplified_fscore,
)
from screener_filters import (
    apply_all_filters,
    FilterConfig,
)
from backtest_engine import (
    backtest_portfolio,
    calculate_portfolio_metrics,
    TradingCosts,
)


# ============================================================================
# CONFIGURACIÓN ROBUSTA
# ============================================================================

class RobustConfig:
    """Configuración basada en literatura académica"""
    
    # Universe
    universe_size: int = 500
    min_market_cap: float = 2e9        # $2B (mid-cap+)
    min_volume: int = 1_000_000        # 1M daily volume
    
    # Quality filters (Piotroski-style)
    min_roe: float = 0.15              # 15% ROE minimum
    min_gross_margin: float = 0.30     # 30% margin
    require_positive_fcf: bool = True
    require_positive_ocf: bool = True
    
    # Valuation (avoid extremes)
    max_pe: float = 50.0               # P/E <= 50
    max_ev_ebitda: float = 25.0        # EV/EBITDA <= 25
    
    # Momentum & Trend (CRÍTICO)
    require_above_ma200: bool = True   # ⭐ OBLIGATORIO
    min_momentum_12m: float = 0.05     # 5% retorno 12M mínimo
    
    # F-Score
    min_fscore: int = 6                # F-Score >= 6
    use_simplified_fscore: bool = True # Si no hay históricos
    
    # Portfolio
    portfolio_size: int = 30           # Top 30 stocks
    
    # Factor weights
    w_quality: float = 0.35
    w_value: float = 0.25
    w_momentum: float = 0.25
    w_fscore: float = 0.15             # Nuevo: F-Score weight
    
    # Backtest
    backtest_start: str = "2020-01-01"
    backtest_end: str = "2024-12-31"
    rebalance_freq: str = "M"          # Monthly
    

# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def run_qvm_strategy(
    universe_size: int = 500,
    portfolio_size: int = 30,
    require_ma200: bool = True,
    min_fscore: int = 6,
    verbose: bool = True,
) -> Dict:
    """
    Ejecuta estrategia QVM completa con TODAS las mejoras.
    
    Args:
        universe_size: Tamaño del universo inicial
        portfolio_size: Número de acciones en portfolio
        require_ma200: Si True, filtro MA200 obligatorio (RECOMENDADO)
        min_fscore: F-Score mínimo (6-9)
        verbose: Print progress
    
    Returns:
        Dict con resultados completos
    """
    
    if verbose:
        print("="*70)
        print("🚀 QVM STRATEGY - IMPLEMENTACIÓN ACADÉMICA CORRECTA")
        print("="*70)
    
    config = RobustConfig()
    
    # -------------------------------------------------------------------------
    # PASO 1: SCREENER INICIAL
    # -------------------------------------------------------------------------
    if verbose:
        print("\n📊 PASO 1: Screener inicial...")
    
    universe = fetch_screener(
        limit=universe_size,
        mcap_min=config.min_market_cap,
        volume_min=config.min_volume,
        use_cache=True,
    )
    
    if verbose:
        print(f"  ✅ {len(universe)} símbolos en universo inicial")
    
    if universe.empty:
        return {"error": "No symbols in universe"}
    
    # -------------------------------------------------------------------------
    # PASO 2: FUNDAMENTALS
    # -------------------------------------------------------------------------
    if verbose:
        print("\n📊 PASO 2: Descargando fundamentales...")
    
    fundamentals = fetch_fundamentals_batch(
        universe['symbol'].tolist(),
        use_cache=True,
    )
    
    if verbose:
        print(f"  ✅ {len(fundamentals)} símbolos con datos fundamentales")
    
    # -------------------------------------------------------------------------
    # PASO 3: FILTROS DE CALIDAD BÁSICOS
    # -------------------------------------------------------------------------
    if verbose:
        print("\n🔍 PASO 3: Aplicando filtros de calidad...")
    
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
    
    # Merge universe + fundamentals
    df_merged = universe.merge(fundamentals, on='symbol', how='left')
    
    # Aplicar filtros
    passed_filters, diagnostics = apply_all_filters(df_merged, filter_config)
    
    if verbose:
        print(f"  ✅ {len(passed_filters)} símbolos pasaron filtros de calidad")
        print(f"  ❌ Rechazados: {len(df_merged) - len(passed_filters)}")
    
    if passed_filters.empty:
        return {"error": "No symbols passed quality filters"}
    
    # -------------------------------------------------------------------------
    # PASO 4: F-SCORE (Piotroski)
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n🏆 PASO 4: Calculando F-Score (min={min_fscore})...")
    
    # Merge con fundamentals para calcular F-Score
    df_for_fscore = passed_filters.merge(fundamentals, on='symbol', how='left')
    
    # Calcular F-Score simplificado (si no hay históricos)
    df_for_fscore['fscore'] = calculate_simplified_fscore(df_for_fscore)
    
    # Filtrar por F-Score
    df_fscore_passed = df_for_fscore[df_for_fscore['fscore'] >= min_fscore].copy()
    
    if verbose:
        print(f"  ✅ {len(df_fscore_passed)} símbolos con F-Score >= {min_fscore}")
        avg_fscore = df_fscore_passed['fscore'].mean()
        print(f"  📊 F-Score promedio: {avg_fscore:.1f}/9.0")
    
    if df_fscore_passed.empty:
        return {"error": f"No symbols with F-Score >= {min_fscore}"}
    
    # -------------------------------------------------------------------------
    # PASO 5: DESCARGAR PRECIOS
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n📈 PASO 5: Descargando precios históricos...")
    
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
            
            if prices is not None and len(prices) >= 252:  # Mínimo 1 año
                prices_dict[symbol] = prices
            else:
                failed += 1
        
        except Exception:
            failed += 1
    
    if verbose:
        print(f"  ✅ {len(prices_dict)} símbolos con precios válidos")
        print(f"  ❌ {failed} símbolos sin precios suficientes")
    
    if not prices_dict:
        return {"error": "No price data available"}
    
    # -------------------------------------------------------------------------
    # PASO 6: MOMENTUM REAL + MA200 FILTER (⭐ CRÍTICO)
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n🎯 PASO 6: Calculando Momentum Real + MA200 Filter...")
    
    # Calcular momentum batch
    momentum_df = calculate_momentum_batch(prices_dict)
    
    if verbose:
        n_positive_mom = (momentum_df['momentum_12m1m'] > 0).sum()
        n_above_ma200 = momentum_df['above_ma200'].sum()
        print(f"  📊 Momentum positivo (12M-1M): {n_positive_mom}/{len(momentum_df)}")
        print(f"  📊 Precio > MA200: {n_above_ma200}/{len(momentum_df)}")
    
    # Merge momentum con datos
    df_with_momentum = df_fscore_passed.merge(
        momentum_df[['symbol', 'momentum_12m1m', 'above_ma200', 'composite_momentum']],
        on='symbol',
        how='left'
    )
    
    # ⭐ FILTRO CRÍTICO: Solo mantener si Price > MA200
    if require_ma200:
        df_final = df_with_momentum[df_with_momentum['above_ma200'] == True].copy()
        
        if verbose:
            print(f"\n  ⭐ MA200 FILTER APLICADO:")
            print(f"     Antes: {len(df_with_momentum)} símbolos")
            print(f"     Después: {len(df_final)} símbolos")
            print(f"     Rechazados: {len(df_with_momentum) - len(df_final)}")
    else:
        df_final = df_with_momentum.copy()
        if verbose:
            print(f"\n  ⚠️  MA200 Filter DESACTIVADO (no recomendado)")
    
    if df_final.empty:
        return {"error": "No symbols passed MA200 filter"}
    
    # -------------------------------------------------------------------------
    # PASO 7: CALCULAR FACTORES QVM
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n🧮 PASO 7: Calculando factores QVM...")
    
    # Preparar universo y fundamentals para compute_all_factors
    df_universe_clean = df_final[['symbol', 'sector', 'market_cap']].copy()
    df_fundamentals_clean = df_final[['symbol', 'ev_ebitda', 'pb', 'pe', 
                                        'roe', 'roic', 'gross_margin', 
                                        'fcf', 'operating_cf']].copy()
    
    # Calcular factores (sector-neutral)
    df_with_factors = compute_all_factors(
        df_universe_clean,
        df_fundamentals_clean,
        sector_neutral=True,
        w_quality=config.w_quality,
        w_value=config.w_value,
        w_momentum=config.w_momentum,
    )
    
    # Merge con momentum real
    df_with_factors = df_with_factors.merge(
        df_final[['symbol', 'momentum_12m1m', 'above_ma200', 'composite_momentum', 'fscore']],
        on='symbol',
        how='left'
    )
    
    # -------------------------------------------------------------------------
    # PASO 8: COMPOSITE SCORE FINAL
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n🎯 PASO 8: Calculando composite score final...")
    
    # Recalcular QVM con momentum REAL
    df_with_factors['qvm_score_corrected'] = (
        config.w_quality * df_with_factors['quality_extended'] +
        config.w_value * df_with_factors['value_score'] +
        config.w_momentum * df_with_factors['composite_momentum'] +
        config.w_fscore * (df_with_factors['fscore'] / 9.0)  # Normalizado
    )
    
    # Ranking final
    df_with_factors['final_rank'] = df_with_factors['qvm_score_corrected'].rank(
        pct=True, method='average'
    )
    
    # -------------------------------------------------------------------------
    # PASO 9: SELECCIÓN DE PORTFOLIO
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n📋 PASO 9: Seleccionando Top {portfolio_size} stocks...")
    
    # Top N
    portfolio = df_with_factors.nlargest(portfolio_size, 'qvm_score_corrected')
    
    if verbose:
        print(f"\n✅ PORTFOLIO FINAL:")
        print(f"   Tamaño: {len(portfolio)} acciones")
        print(f"   Sectores: {portfolio['sector'].nunique()}")
        print(f"   F-Score promedio: {portfolio['fscore'].mean():.1f}")
        print(f"   ROE promedio: {portfolio['roe'].mean():.1%}")
        print(f"   Momentum 12M promedio: {portfolio['momentum_12m1m'].mean():.1%}")
        print(f"   100% con Price > MA200: {portfolio['above_ma200'].all()}")
    
    # -------------------------------------------------------------------------
    # PASO 10: BACKTEST
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n📊 PASO 10: Backtesting portfolio...")
    
    # Filtrar precios solo de portfolio
    portfolio_prices = {
        sym: prices_dict[sym] 
        for sym in portfolio['symbol'] 
        if sym in prices_dict
    }
    
    # Backtest con costos realistas
    costs = TradingCosts(
        commission_bps=5,
        slippage_bps=5,
        market_impact_bps=2,
    )
    
    metrics, equity_curves = backtest_portfolio(
        portfolio_prices,
        costs=costs,
        execution_lag_days=1,
    )
    
    # Métricas de portfolio
    port_metrics = calculate_portfolio_metrics(equity_curves, costs)
    
    if verbose:
        print(f"\n📈 RESULTADOS BACKTEST:")
        print(f"   CAGR: {port_metrics['CAGR']:.2%}")
        print(f"   Sharpe: {port_metrics['Sharpe']:.2f}")
        print(f"   Sortino: {port_metrics['Sortino']:.2f}")
        print(f"   Max Drawdown: {port_metrics['MaxDD']:.2%}")
        print(f"   Calmar: {port_metrics['Calmar']:.2f}")
        print(f"   Costos: {costs.round_trip_bps} bps round-trip")
    
    # -------------------------------------------------------------------------
    # RETURN RESULTS
    # -------------------------------------------------------------------------
    
    results = {
        'portfolio': portfolio,
        'backtest_metrics': metrics,
        'portfolio_metrics': port_metrics,
        'equity_curves': equity_curves,
        'diagnostics': {
            'universe_size': len(universe),
            'after_quality_filters': len(passed_filters),
            'after_fscore': len(df_fscore_passed),
            'after_ma200': len(df_final),
            'final_portfolio': len(portfolio),
        },
        'config': config,
    }
    
    if verbose:
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETO")
        print("="*70)
    
    return results


# ============================================================================
# ANÁLISIS DE RESULTADOS
# ============================================================================

def analyze_results(results: Dict) -> pd.DataFrame:
    """
    Genera análisis detallado de resultados.
    """
    portfolio = results['portfolio']
    
    analysis = pd.DataFrame({
        'Symbol': portfolio['symbol'],
        'Sector': portfolio['sector'],
        'F-Score': portfolio['fscore'].round(1),
        'QVM Score': portfolio['qvm_score_corrected'].round(3),
        'Momentum 12M': portfolio['momentum_12m1m'].round(3),
        'ROE': portfolio['roe'].round(3),
        'Above MA200': portfolio['above_ma200'],
    })
    
    analysis = analysis.sort_values('QVM Score', ascending=False)
    
    return analysis


# ============================================================================
# COMPARACIÓN CON BENCHMARK
# ============================================================================

def compare_with_benchmark(
    results: Dict,
    benchmark_symbol: str = 'SPY',
) -> pd.DataFrame:
    """
    Compara portfolio con benchmark (SPY).
    """
    from data_fetcher import fetch_prices
    from backtest_engine import backtest_single_symbol
    
    # Backtest benchmark
    spy_prices = fetch_prices(
        benchmark_symbol,
        start=results['config'].backtest_start,
        end=results['config'].backtest_end,
    )
    
    if spy_prices is None:
        return pd.DataFrame()
    
    spy_result = backtest_single_symbol(
        spy_prices,
        benchmark_symbol,
        costs=TradingCosts(1, 1, 0),  # ETF tiene costos mínimos
    )
    
    port_metrics = results['portfolio_metrics']
    
    comparison = pd.DataFrame({
        'Metric': ['CAGR', 'Sharpe', 'Sortino', 'Max DD', 'Calmar'],
        'Portfolio (QVM)': [
            port_metrics['CAGR'],
            port_metrics['Sharpe'],
            port_metrics['Sortino'],
            port_metrics['MaxDD'],
            port_metrics['Calmar'],
        ],
        'Benchmark (SPY)': [
            spy_result.cagr,
            spy_result.sharpe,
            spy_result.sortino,
            spy_result.max_dd,
            spy_result.calmar,
        ],
    })
    
    comparison['Alpha'] = comparison['Portfolio (QVM)'] - comparison['Benchmark (SPY)']
    
    return comparison


# ============================================================================
# MAIN (para testing)
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing QVM Pipeline Corrected...")
    
    # Ejecutar estrategia completa
    results = run_qvm_strategy(
        universe_size=200,  # Reducido para test
        portfolio_size=20,
        require_ma200=True,  # ⭐ CRÍTICO
        min_fscore=6,
        verbose=True,
    )
    
    if 'error' in results:
        print(f"\n❌ Error: {results['error']}")
    else:
        # Análisis
        print("\n" + "="*70)
        print("📊 ANÁLISIS DE PORTFOLIO")
        print("="*70)
        
        analysis = analyze_results(results)
        print("\n", analysis.to_string(index=False))
        
        # Comparación con SPY
        print("\n" + "="*70)
        print("📈 COMPARACIÓN CON SPY")
        print("="*70)
        
        comparison = compare_with_benchmark(results)
        if not comparison.empty:
            print("\n", comparison.to_string(index=False))
        
        print("\n✅ Pipeline test complete!")
