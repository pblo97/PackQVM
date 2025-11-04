"""
QVM Strategy Pipeline - VERSIÓN OPTIMIZADA ACADÉMICA (Actualizada)
==================================================================

Cambios clave respecto a tu borrador:
- Alias correcto del F-Score: usamos la forma simplificada sin ROE
- Normalización estricta de símbolos (strip + upper) antes de pedir precios
- Homogeneización de columnas de precios (date/close) por si el endpoint cambia nombres
- Asserts y prints de diagnóstico por etapa para detectar rápido el "cuello de botella"
- Merges y columnas verificados para que el flujo no se corte
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from data_fetcher import (
    fetch_screener,
    fetch_fundamentals_batch,
    fetch_prices,
)
from factor_calculator import compute_all_factors
from momentum_calculator import (
    calculate_momentum_batch,
)
# ⚠️ Alias consistente con tu módulo piotroski_fscore.py
from piotroski_fscore import calculate_simplified_fscore_no_roe as calculate_simplified_fscore
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
    # Universo
    universe_size: int = 500
    min_market_cap: float = 2e9
    min_volume: int = 1_000_000

    # Quality (robusto)
    min_roe: float = 0.15
    min_roic: float = 0.12
    min_gross_margin: float = 0.30
    require_positive_fcf: bool = True
    require_positive_ocf: bool = True

    # Valuation
    max_pe: float = 40.0
    max_ev_ebitda: float = 20.0

    # Momentum/Trend
    require_above_ma200: bool = True
    min_momentum_12m: float = 0.10

    # F-Score
    min_fscore: int = 6

    # Portfolio
    portfolio_size: int = 30

    # Pesos (suman 1.0)
    w_quality: float = 0.30
    w_value: float = 0.25
    w_momentum: float = 0.30
    w_fscore: float = 0.15

    # Backtest
    backtest_start: str = "2020-01-01"
    backtest_end: str = "2024-12-31"

    # Costos
    commission_bps: int = 5
    slippage_bps: int = 5
    market_impact_bps: int = 2

    def __post_init__(self):
        total_weight = self.w_quality + self.w_value + self.w_momentum + self.w_fscore
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(f"Factor weights must sum to 1.0, got {total_weight}")
        if not self.require_above_ma200:
            raise ValueError("MA200 filter MUST be True (academic requirement)")


# ============================================================================
# UTILIDADES DE LOG/DIAGNÓSTICO
# ============================================================================

def _print_stage(label: str, count: int, total: int):
    pct = (count / total * 100) if total else 0.0
    print(f"  {label:32s}: {count:5d} ({pct:5.1f}%)")

def _ensure_price_columns(px: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas de precios a ['date','close'] orden ascendente."""
    if px is None or px.empty:
        return pd.DataFrame(columns=["date", "close"])
    df = px.copy()
    df = df.rename(columns={"Date": "date", "Close": "close", "adjClose": "close", "Adj Close": "close"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date", "close"]).sort_values("date")[["date", "close"]].reset_index(drop=True)


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def run_optimized_qvm_strategy(
    config: Optional[AcademicConfig] = None,
    verbose: bool = True,
) -> Dict:
    if config is None:
        config = AcademicConfig()

    if verbose:
        print("\n" + "=" * 76)
        print("🚀 QVM STRATEGY - VERSIÓN OPTIMIZADA ACADÉMICA (Actualizada)")
        print("=" * 76)

    total_initial = config.universe_size

    # ---------------- PASO 1: Screener ----------------
    if verbose:
        print("\n📊 PASO 1/10: Screener inicial")
        print(f"   Min Market Cap: ${config.min_market_cap/1e9:.1f}B | Min Volume: {config.min_volume:,}")

    universe = fetch_screener(
        limit=config.universe_size,
        mcap_min=config.min_market_cap,
        volume_min=config.min_volume,
        use_cache=True,
    )
    if universe.empty:
        return {"error": "No symbols in initial universe"}
    assert {"symbol", "sector", "market_cap"} <= set(universe.columns), \
        f"Screener incompleto: cols={universe.columns.tolist()}"
    _print_stage("Universe inicial", len(universe), total_initial)

    # ---------------- PASO 2: Fundamentals ----------------
    if verbose:
        print("\n📊 PASO 2/10: Descargando fundamentales (TTM)")

    fundamentals = fetch_fundamentals_batch(universe["symbol"].tolist(), use_cache=True)
    need_fund = {"symbol", "ev_ebitda", "pb", "pe", "roe", "roic", "gross_margin", "fcf", "operating_cf"}
    missing = need_fund - set(fundamentals.columns)
    if missing and verbose:
        print("⚠️ Fundamentals faltantes:", missing)
    _print_stage("Con fundamentales", len(fundamentals), total_initial)

    # ---------------- PASO 3: Filtros de calidad ----------------
    if verbose:
        print("\n🔍 PASO 3/10: Filtros de calidad (robustos)")
        print(f"   ROE>={config.min_roe:.0%}, GM>={config.min_gross_margin:.0%}, "
              f"PE<={config.max_pe}, EV/EBITDA<={config.max_ev_ebitda}, OCF/FCF>0")

    filt_cfg = FilterConfig(
        min_roe=config.min_roe,
        min_gross_margin=config.min_gross_margin,
        require_positive_fcf=config.require_positive_fcf,
        require_positive_ocf=config.require_positive_ocf,
        max_pe=config.max_pe,
        max_ev_ebitda=config.max_ev_ebitda,
        min_volume=config.min_volume,
        min_market_cap=config.min_market_cap,
    )
    df_merged = universe.merge(fundamentals, on="symbol", how="left")
    passed_filters, diagnostics = apply_all_filters(df_merged, filt_cfg)
    _print_stage("Pasan filtros calidad", len(passed_filters), total_initial)

    if passed_filters.empty:
        return {"error": "No symbols passed quality filters"}

    # ---------------- PASO 4: F-Score ----------------
    if verbose:
        print(f"\n🏆 PASO 4/10: F-Score (>= {config.min_fscore})")

    df_for_fscore = passed_filters.merge(fundamentals, on="symbol", how="left")
    df_for_fscore["fscore"] = calculate_simplified_fscore(df_for_fscore)
    df_fscore_passed = df_for_fscore[df_for_fscore["fscore"] >= config.min_fscore].copy()
    _print_stage("F-Score >= umbral", len(df_fscore_passed), total_initial)

    if df_fscore_passed.empty:
        return {"error": f"No symbols with F-Score >= {config.min_fscore}"}

    # ---------------- PASO 5: Precios ----------------
    if verbose:
        print(f"\n📈 PASO 5/10: Descargando precios {config.backtest_start} → {config.backtest_end}")

    symbols_to_fetch = (
        df_fscore_passed["symbol"].dropna().astype(str).str.strip().str.upper().unique().tolist()
    )
    prices_dict: Dict[str, pd.DataFrame] = {}
    failed = 0

    for i, sym in enumerate(symbols_to_fetch, 1):
        try:
            px = fetch_prices(sym, start=config.backtest_start, end=config.backtest_end, use_cache=True)
            px = _ensure_price_columns(px)
            if len(px) >= 252:
                prices_dict[sym] = px
            else:
                failed += 1
        except Exception:
            failed += 1
        if verbose and (i % 50 == 0):
            print(f"   · Progreso precios: {i}/{len(symbols_to_fetch)}")

    _print_stage("Con precios históricos", len(prices_dict), total_initial)
    if failed and verbose:
        print(f"   ⚠️ Sin datos suficientes: {failed} símbolos")
    if not prices_dict:
        return {"error": "No price data available"}

    # ---------------- PASO 6: Momentum REAL + MA200 ----------------
    if verbose:
        print("\n🚀 PASO 6/10: Momentum REAL + Filtro MA200")

    momentum_df = calculate_momentum_batch(prices_dict)
    for col in ["symbol", "momentum_12m1m", "above_ma200", "composite_momentum"]:
        assert col in momentum_df.columns, f"Momentum DF sin columna requerida: {col}"

    df_with_momentum = df_fscore_passed.merge(
        momentum_df[["symbol", "momentum_12m1m", "above_ma200", "composite_momentum"]],
        on="symbol",
        how="inner",
    )
    if config.require_above_ma200:
        df_ma200_passed = df_with_momentum[df_with_momentum["above_ma200"] == True].copy()
    else:
        df_ma200_passed = df_with_momentum.copy()
    _print_stage("Price > MA200", len(df_ma200_passed), total_initial)

    df_final = df_ma200_passed[df_ma200_passed["momentum_12m1m"] >= config.min_momentum_12m].copy()
    _print_stage(f"Momentum >= {config.min_momentum_12m:.0%}", len(df_final), total_initial)
    if df_final.empty:
        return {"error": "No symbols passed MA200 + Momentum filters"}

    # ---------------- PASO 7: Factores QVM (sector-neutral) ----------------
    if verbose:
        print("\n🧮 PASO 7/10: Factores QVM (sector-neutral)")

    keep_uni_cols = ["symbol", "sector", "market_cap"]
    df_universe_clean = df_final[[c for c in keep_uni_cols if c in df_final.columns]].drop_duplicates("symbol")

    keep_fund_cols = ["symbol", "ev_ebitda", "pb", "pe", "roe", "roic", "gross_margin", "fcf", "operating_cf"]
    df_fundamentals_clean = df_final[[c for c in keep_fund_cols if c in df_final.columns]].drop_duplicates("symbol")

    df_with_factors = compute_all_factors(
        df_universe_clean,
        df_fundamentals_clean,
        sector_neutral=True,
        # Normalizamos pesos internos (sólo Q/V/M) para que sumen 1
        w_quality=config.w_quality / (config.w_quality + config.w_value + config.w_momentum),
        w_value=config.w_value / (config.w_quality + config.w_value + config.w_momentum),
        w_momentum=config.w_momentum / (config.w_quality + config.w_value + config.w_momentum),
    )
    for col in ["value_score", "quality_score", "quality_extended"]:
        assert col in df_with_factors.columns, f"Faltó {col} en compute_all_factors()"

    # Adjunta momentum real y fscore
    df_with_factors = df_with_factors.merge(
        df_final[["symbol", "momentum_12m1m", "above_ma200", "composite_momentum", "fscore"]],
        on="symbol",
        how="left",
    )
    if verbose:
        print(f"   ✅ Factores calculados para {len(df_with_factors)} símbolos")

    # ---------------- PASO 8: Composite score final ----------------
    if verbose:
        print("\n🎯 PASO 8/10: Composite final (Q/V/M + F-Score)")

    df_with_factors["composite_score"] = (
        config.w_quality * df_with_factors["quality_extended"]
        + config.w_value * df_with_factors["value_score"]
        + config.w_momentum * df_with_factors["composite_momentum"]
        + config.w_fscore * (df_with_factors["fscore"] / 9.0)
    )
    df_with_factors["final_rank"] = df_with_factors["composite_score"].rank(pct=True, method="average")

    # ---------------- PASO 9: Selección TOP N ----------------
    if verbose:
        print(f"\n📋 PASO 9/10: Selección Top {config.portfolio_size}")

    portfolio = df_with_factors.nlargest(config.portfolio_size, "composite_score").copy()
    _print_stage("Portfolio final", len(portfolio), total_initial)

    # ---------------- PASO 10: Backtest ----------------
    if verbose:
        print("\n📊 PASO 10/10: Backtest")

    portfolio_prices = {sym: prices_dict[sym] for sym in portfolio["symbol"] if sym in prices_dict}

    costs = TradingCosts(
        commission_bps=config.commission_bps,
        slippage_bps=config.slippage_bps,
        market_impact_bps=config.market_impact_bps,
    )

    metrics, equity_curves = backtest_portfolio(
        portfolio_prices,
        costs=costs,
        execution_lag_days=1,
    )
    port_metrics = calculate_portfolio_metrics(equity_curves, costs)

    if verbose:
        print("\n" + "=" * 76)
        print("📈 RESULTADOS BACKTEST")
        print("=" * 76)
        print(f"  CAGR:        {port_metrics['CAGR']:.2%}")
        print(f"  Sharpe:      {port_metrics['Sharpe']:.2f}")
        print(f"  Sortino:     {port_metrics['Sortino']:.2f}")
        print(f"  Max Drawdown:{port_metrics['MaxDD']:.2%}")
        print(f"  Calmar:      {port_metrics['Calmar']:.2f}")

    return {
        "portfolio": portfolio,
        "backtest_metrics": metrics,
        "portfolio_metrics": port_metrics,
        "equity_curves": equity_curves,
        "config": config,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    cfg = AcademicConfig()
    res = run_optimized_qvm_strategy(cfg, verbose=True)
    if "error" in res:
        print("\n❌ Error:", res["error"])
    else:
        print("\n✅ Pipeline ejecutado correctamente. Top 5:")
        cols_show = [
            "symbol", "sector", "market_cap", "fscore",
            "composite_score", "final_rank",
            "momentum_12m1m", "roe", "gross_margin", "pe", "ev_ebitda", "above_ma200",
        ]
        df_show = res["portfolio"][ [c for c in cols_show if c in res["portfolio"].columns] ].copy()
        df_show.loc[:, "market_cap"] = (df_show["market_cap"] / 1e9).round(2)
        print(df_show.head(5).to_string(index=False))
