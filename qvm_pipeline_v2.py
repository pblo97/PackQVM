"""
QVM Strategy Pipeline V2 - Con Piotroski Real y Quality-Value Score
====================================================================

MEJORAS vs V1:
1. ✅ Piotroski Score REAL calculado desde estados financieros completos (9 checks)
2. ✅ Quality-Value Score SIN multicolinealidad
3. ✅ ROIC, FCF Yield calculados correctamente
4. ✅ Análisis por pasos con checks y validaciones
5. ✅ Parámetros ajustables para optimización

Bibliografía:
- Piotroski (2000): F-Score calculation
- Asness, Frazzini & Pedersen (2019): Quality factors
- Fama & French (1992, 2015): Value factors
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Imports de módulos
from data_fetcher import (
    fetch_screener,
    fetch_financial_statements_batch,
    fetch_prices_daily,
)
from quality_value_score import (
    compute_quality_value_factors,
    top_quality_value_stocks,
    analyze_score_components,
)


# ============================================================================
# CONFIGURACIÓN AJUSTABLE
# ============================================================================

@dataclass
class QVMConfig:
    """
    Configuración del pipeline QVM con parámetros ajustables.
    """

    # ========== UNIVERSE ==========
    universe_size: int = 300
    min_market_cap: float = 2e9          # $2B+ (mid-cap+)
    min_volume: int = 500_000            # 500K daily volume

    # ========== QUALITY-VALUE WEIGHTS ==========
    # Pesos del Quality-Value Score (deben sumar 1.0)
    w_quality: float = 0.40              # Peso de Piotroski Score
    w_value: float = 0.35                # Peso de múltiplos valoración
    w_fcf_yield: float = 0.15            # Peso de FCF Yield
    w_momentum: float = 0.10             # Peso de Momentum

    # ========== FILTROS BÁSICOS ==========
    min_piotroski_score: int = 5         # Mínimo Piotroski (0-9)
    min_qv_score: float = 0.40           # Mínimo QV Score (0-1)

    # Filtros de valoración
    max_pe: float = 50.0                 # P/E máximo
    max_pb: float = 10.0                 # P/B máximo
    max_ev_ebitda: float = 25.0          # EV/EBITDA máximo

    # Filtros de calidad
    require_positive_fcf: bool = True    # Requerir FCF > 0

    # ========== PORTFOLIO ==========
    portfolio_size: int = 30             # Top N stocks

    def __post_init__(self):
        """Validación de configuración"""
        # Verificar que pesos sumen 1.0
        total_weight = self.w_quality + self.w_value + self.w_fcf_yield + self.w_momentum
        if not (0.99 <= total_weight <= 1.01):
            # Normalizar automáticamente
            self.w_quality /= total_weight
            self.w_value /= total_weight
            self.w_fcf_yield /= total_weight
            self.w_momentum /= total_weight

    def to_dict(self) -> Dict:
        """Convierte config a dict para serialización"""
        return {k: v for k, v in self.__dict__.items()}


# ============================================================================
# ANÁLISIS POR PASOS CON CHECKS
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
        summary = f"{status} {self.name}: {self.output_count}/{self.input_count} ({pass_rate:.1f}%)"

        if self.warnings:
            summary += f"\n   ⚠️  {len(self.warnings)} warnings"

        if self.metrics:
            summary += "\n   📊 Metrics:"
            for k, v in self.metrics.items():
                if isinstance(v, float):
                    summary += f"\n      {k}: {v:.2f}"
                else:
                    summary += f"\n      {k}: {v}"

        return summary


# ============================================================================
# PIPELINE PRINCIPAL V2
# ============================================================================

def run_qvm_pipeline_v2(
    config: Optional[QVMConfig] = None,
    verbose: bool = True,
) -> Dict:
    """
    Pipeline QVM V2 con Piotroski Score real y Quality-Value Score sin multicolinealidad.

    PASOS:
    1. Screener inicial (market cap, volume)
    2. Descarga de estados financieros completos
    3. Cálculo de Piotroski Score (9 checks)
    4. Cálculo de ROIC, FCF Yield, métricas avanzadas
    5. Filtros básicos de calidad
    6. Cálculo de Quality-Value Score
    7. Filtros finales (min QV score)
    8. Selección de portfolio (Top N)

    Returns:
        Dict con resultados completos + análisis por paso
    """

    if config is None:
        config = QVMConfig()

    steps = []

    if verbose:
        print("\n" + "="*80)
        print("🚀 QVM STRATEGY PIPELINE V2")
        print("   Con Piotroski Score Real + Quality-Value Score (sin multicolinealidad)")
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
            step1.add_warning("No symbols found in screener")
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
    # PASO 2: ESTADOS FINANCIEROS COMPLETOS + PIOTROSKI
    # -------------------------------------------------------------------------
    step2 = PipelineStep("PASO 2", "Estados Financieros + Piotroski Score")

    if verbose:
        print(f"\n📊 {step2.name}: {step2.description}")
        print("   Descargando income statement, balance sheet, cash flow...")
        print("   Calculando Piotroski Score (9 checks) + ROIC + FCF Yield...")

    try:
        symbols = universe['symbol'].tolist()
        step2.log_input(len(symbols))

        # Descargar estados financieros y calcular Piotroski + métricas
        financial_data = fetch_financial_statements_batch(symbols)

        # También obtener ratios de valoración (pe, pb, ev_ebitda)
        from data_fetcher import fetch_fundamentals_batch
        valuation_ratios = fetch_fundamentals_batch(symbols, use_full_statements=False)

        # Merge financial data con ratios
        financial_data = financial_data.merge(
            valuation_ratios[['symbol', 'ev_ebitda', 'pb', 'pe']],
            on='symbol',
            how='left'
        )

        # Filtrar los que tienen datos
        financial_data = financial_data[financial_data['piotroski_score'].notna()].copy()

        step2.log_output(len(financial_data))

        if financial_data.empty:
            step2.add_warning("No financial data available")
            steps.append(step2)
            return {"error": "No financial data", "steps": steps}

        # Métricas de calidad
        avg_piotroski = financial_data['piotroski_score'].mean()
        median_piotroski = financial_data['piotroski_score'].median()

        step2.add_metric("Avg Piotroski Score", avg_piotroski)
        step2.add_metric("Median Piotroski Score", median_piotroski)
        step2.add_metric("Piotroski >= 7 (High Quality)", (financial_data['piotroski_score'] >= 7).sum())

        # Distribución de Piotroski
        piotroski_dist = financial_data['piotroski_score'].value_counts().sort_index()

        step2.mark_success()

        if verbose:
            print(step2.summary())
            print(f"\n   📈 Distribución de Piotroski Score:")
            for score, count in piotroski_dist.items():
                bar = "█" * int(count / len(financial_data) * 50)
                print(f"      Score {int(score)}: {count:3d} {bar}")

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
        print(f"   Min Piotroski Score: {config.min_piotroski_score}")
        print(f"   Max P/E: {config.max_pe}")
        print(f"   Max EV/EBITDA: {config.max_ev_ebitda}")
        if config.require_positive_fcf:
            print("   Require FCF > 0")

    try:
        step3.log_input(len(financial_data))

        # Merge con universe para tener market_cap
        df = universe.merge(financial_data, on='symbol', how='inner')

        # Filtro 1: Piotroski mínimo
        df = df[df['piotroski_score'] >= config.min_piotroski_score].copy()
        rejected_piotroski = step3.input_count - len(df)

        # Filtro 2: Valoración
        if 'pe' in df.columns:
            before = len(df)
            df = df[(df['pe'].isna()) | (df['pe'] <= config.max_pe)].copy()
            rejected_pe = before - len(df)
        else:
            rejected_pe = 0

        if 'ev_ebitda' in df.columns:
            before = len(df)
            df = df[(df['ev_ebitda'].isna()) | (df['ev_ebitda'] <= config.max_ev_ebitda)].copy()
            rejected_ev = before - len(df)
        else:
            rejected_ev = 0

        # Filtro 3: FCF positivo
        if config.require_positive_fcf and 'fcf' in df.columns:
            before = len(df)
            df = df[(df['fcf'].isna()) | (df['fcf'] > 0)].copy()
            rejected_fcf = before - len(df)
        else:
            rejected_fcf = 0

        step3.log_output(len(df))

        step3.add_metric("Rejected by Piotroski", rejected_piotroski)
        step3.add_metric("Rejected by P/E", rejected_pe)
        step3.add_metric("Rejected by EV/EBITDA", rejected_ev)
        step3.add_metric("Rejected by FCF", rejected_fcf)

        if len(df) == 0:
            step3.add_warning("No stocks passed quality filters")
            steps.append(step3)
            return {"error": "No stocks passed filters", "steps": steps}

        step3.mark_success()

        if verbose:
            print(step3.summary())

    except Exception as e:
        step3.add_warning(f"Error: {str(e)}")
        steps.append(step3)
        return {"error": str(e), "steps": steps}

    steps.append(step3)

    # -------------------------------------------------------------------------
    # PASO 4: QUALITY-VALUE SCORE (SIN MULTICOLINEALIDAD)
    # -------------------------------------------------------------------------
    step4 = PipelineStep("PASO 4", "Quality-Value Score (sin multicolinealidad)")

    if verbose:
        print(f"\n🎯 {step4.name}: {step4.description}")
        print(f"   Pesos: Quality={config.w_quality:.0%}, Value={config.w_value:.0%}, "
              f"FCF Yield={config.w_fcf_yield:.0%}, Momentum={config.w_momentum:.0%}")

    try:
        step4.log_input(len(df))

        # Calcular FCF Yield si no existe
        if 'fcf_yield' not in df.columns and 'fcf' in df.columns and 'market_cap' in df.columns:
            df['fcf_yield'] = df['fcf'] / df['market_cap'].replace(0, np.nan)

        # Preparar datos
        df_universe = df[['symbol', 'sector', 'market_cap']].copy()
        df_fundamentals = df[[
            'symbol', 'piotroski_score',
            'ev_ebitda', 'pb', 'pe',
            'fcf', 'fcf_yield', 'market_cap'
        ]].copy()

        # Calcular Quality-Value Score
        df_with_qv_score = compute_quality_value_factors(
            df_universe,
            df_fundamentals,
            w_quality=config.w_quality,
            w_value=config.w_value,
            w_fcf_yield=config.w_fcf_yield,
            w_momentum=config.w_momentum,
        )

        step4.log_output(len(df_with_qv_score))

        # Métricas del score
        avg_qv_score = df_with_qv_score['qv_score'].mean()
        median_qv_score = df_with_qv_score['qv_score'].median()

        step4.add_metric("Avg QV Score", avg_qv_score)
        step4.add_metric("Median QV Score", median_qv_score)
        step4.add_metric("QV Score >= 0.7 (Strong)", (df_with_qv_score['qv_score'] >= 0.7).sum())
        step4.mark_success()

        if verbose:
            print(step4.summary())

            # Análisis de componentes
            print("\n   📊 Score Component Analysis:")
            stats = analyze_score_components(df_with_qv_score)
            if not stats.empty:
                for col in ['quality_score_component', 'value_score_component', 'qv_score']:
                    if col in stats.columns:
                        print(f"      {col}:")
                        print(f"         Mean: {stats[col]['mean']:.3f}, Std: {stats[col]['std']:.3f}")

    except Exception as e:
        step4.add_warning(f"Error: {str(e)}")
        steps.append(step4)
        return {"error": str(e), "steps": steps}

    steps.append(step4)

    # -------------------------------------------------------------------------
    # PASO 5: FILTRO POR QV SCORE MÍNIMO
    # -------------------------------------------------------------------------
    step5 = PipelineStep("PASO 5", f"Filtro QV Score >= {config.min_qv_score}")

    if verbose:
        print(f"\n🔍 {step5.name}: {step5.description}")

    try:
        step5.log_input(len(df_with_qv_score))

        df_filtered = df_with_qv_score[
            df_with_qv_score['qv_score'] >= config.min_qv_score
        ].copy()

        step5.log_output(len(df_filtered))

        if len(df_filtered) == 0:
            step5.add_warning(f"No stocks with QV Score >= {config.min_qv_score}")
            steps.append(step5)
            return {"error": f"No stocks pass QV score filter", "steps": steps}

        step5.mark_success()

        if verbose:
            print(step5.summary())

    except Exception as e:
        step5.add_warning(f"Error: {str(e)}")
        steps.append(step5)
        return {"error": str(e), "steps": steps}

    steps.append(step5)

    # -------------------------------------------------------------------------
    # PASO 6: SELECCIÓN DE PORTFOLIO (TOP N)
    # -------------------------------------------------------------------------
    step6 = PipelineStep("PASO 6", f"Selección Portfolio (Top {config.portfolio_size})")

    if verbose:
        print(f"\n📋 {step6.name}: {step6.description}")

    try:
        step6.log_input(len(df_filtered))

        # Ordenar por QV Score y tomar top N
        portfolio = df_filtered.nlargest(config.portfolio_size, 'qv_score').copy()

        step6.log_output(len(portfolio))

        # Estadísticas del portfolio
        avg_piotroski_port = portfolio['piotroski_score'].mean()
        avg_qv_port = portfolio['qv_score'].mean()

        step6.add_metric("Avg Piotroski (Portfolio)", avg_piotroski_port)
        step6.add_metric("Avg QV Score (Portfolio)", avg_qv_port)
        step6.add_metric("Unique Sectors", portfolio['sector'].nunique())
        step6.mark_success()

        if verbose:
            print(step6.summary())

            # Distribución por sector
            print("\n   🏢 Sector Distribution:")
            sector_dist = portfolio['sector'].value_counts()
            for sector, count in sector_dist.items():
                pct = count / len(portfolio) * 100
                print(f"      {sector:25s}: {count:2d} ({pct:.0f}%)")

    except Exception as e:
        step6.add_warning(f"Error: {str(e)}")
        steps.append(step6)
        return {"error": str(e), "steps": steps}

    steps.append(step6)

    # -------------------------------------------------------------------------
    # RESULTADOS FINALES
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETO - RESUMEN")
        print("="*80)
        for step in steps:
            pass_rate = step.get_pass_rate() * 100
            status = "✅" if step.success else "❌"
            print(f"{status} {step.name}: {step.output_count:4d} stocks ({pass_rate:5.1f}% pass rate)")

    results = {
        'portfolio': portfolio,
        'full_dataset': df_with_qv_score,
        'steps': steps,
        'config': config,
        'success': True,
    }

    return results


# ============================================================================
# ANÁLISIS DEL PORTFOLIO
# ============================================================================

def analyze_portfolio_v2(results: Dict, n_top: int = 20) -> pd.DataFrame:
    """
    Análisis detallado del portfolio final.
    """
    if 'portfolio' not in results:
        return pd.DataFrame()

    portfolio = results['portfolio'].copy()

    # Columnas relevantes
    cols = [
        'symbol', 'sector', 'market_cap',
        'piotroski_score', 'qv_score', 'qv_rank',
        'quality_score_component', 'value_score_component',
        'fcf_yield_component',
    ]

    # Incluir columnas opcionales
    for col in ['ev_ebitda', 'pe', 'pb', 'roic', 'roe', 'gross_margin', 'fcf']:
        if col in portfolio.columns:
            cols.append(col)

    available_cols = [c for c in cols if c in portfolio.columns]
    analysis = portfolio[available_cols].copy()

    # Formatear market cap a $B
    if 'market_cap' in analysis.columns:
        analysis['market_cap_$B'] = (analysis['market_cap'] / 1e9).round(2)
        analysis.drop('market_cap', axis=1, inplace=True)

    # Ordenar por QV Score
    if 'qv_score' in analysis.columns:
        analysis = analysis.sort_values('qv_score', ascending=False)

    return analysis.head(n_top).reset_index(drop=True)


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("\n🧪 Testing QVM Pipeline V2...")

    # Configuración de prueba (pequeña para testing)
    test_config = QVMConfig(
        universe_size=50,
        portfolio_size=10,
        min_piotroski_score=6,
        min_qv_score=0.45,
    )

    # Ejecutar pipeline
    results = run_qvm_pipeline_v2(
        config=test_config,
        verbose=True,
    )

    if results.get('success'):
        # Mostrar portfolio
        print("\n" + "="*80)
        print("📋 TOP 10 STOCKS")
        print("="*80)
        analysis = analyze_portfolio_v2(results, n_top=10)
        print(analysis.to_string(index=True))

        print("\n✅ Pipeline V2 test complete!")
    else:
        print(f"\n❌ Pipeline failed: {results.get('error')}")
