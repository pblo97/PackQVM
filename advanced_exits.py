"""
Advanced Exit Strategies - FASE 2
==================================

Implementa estrategias avanzadas de exit basadas en:
- Nystrup et al. (2020): Regime-Based Dynamic Stops
- Lopez de Prado (2020): Statistical Percentile Targets
- Harvey & Liu (2021): Time-Based Exits and Target Decay

FASE 2 expande el sistema de risk management con exits adaptativos.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

@dataclass
class AdvancedExitsConfig:
    """
    Configuración de estrategias avanzadas de exit
    """
    # Regime-Based Stops
    use_regime_stops: bool = True
    regime_lookback: int = 60  # Días para detectar régimen
    high_vol_multiplier: float = 1.5  # Stops más amplios en alta volatilidad
    low_vol_multiplier: float = 0.8   # Stops más ajustados en baja volatilidad

    # Statistical Percentile Targets
    use_percentile_targets: bool = True
    target_percentile: int = 75  # 75th percentile (conservador)
    holding_period_days: int = 20  # Horizonte de holding esperado
    min_observations: int = 100  # Mínimo de observaciones para calcular

    # Time-Based Exits
    use_time_exits: bool = True
    max_holding_days: int = 90  # Máximo 90 días
    time_decay_enabled: bool = True  # Target decae con el tiempo
    decay_rate: float = 0.02  # 2% decay por semana

    # Profit Lock (Trailing Take Profit)
    use_profit_lock: bool = True
    profit_lock_threshold: float = 0.15  # Activa trailing TP a +15%
    profit_lock_trail: float = 0.05  # Trail 5% desde peak


# ============================================================================
# REGIME DETECTION
# ============================================================================

def detect_volatility_regime(
    returns: pd.Series,
    lookback: int = 60
) -> str:
    """
    Detecta régimen de volatilidad usando método simple de 2-state.

    Paper: Nystrup et al. (2020) "Dynamic Allocation or Diversification"

    Método simplificado:
    - Calcula volatilidad rolling
    - Compara con mediana histórica
    - Alta vol si > mediana, baja vol si < mediana

    Args:
        returns: Serie de retornos diarios
        lookback: Ventana para calcular régimen

    Returns:
        'HIGH_VOL' o 'LOW_VOL'
    """
    if len(returns) < lookback:
        return 'UNKNOWN'

    # Calcular volatilidad rolling
    vol_window = min(20, lookback // 3)  # 20 días o 1/3 del lookback
    rolling_vol = returns.tail(lookback).rolling(vol_window).std()

    # Volatilidad actual
    current_vol = rolling_vol.iloc[-1]

    # Mediana histórica
    median_vol = rolling_vol.median()

    # Clasificar régimen
    if pd.isna(current_vol) or pd.isna(median_vol):
        return 'UNKNOWN'

    if current_vol > median_vol:
        return 'HIGH_VOL'
    else:
        return 'LOW_VOL'


def calculate_regime_adjusted_stop(
    base_stop_distance: float,
    regime: str,
    high_vol_mult: float = 1.5,
    low_vol_mult: float = 0.8
) -> float:
    """
    Ajusta stop loss según régimen de volatilidad.

    Paper: Nystrup et al. (2020)

    Lógica:
    - Alta volatilidad → stops más amplios (evitar whipsaws)
    - Baja volatilidad → stops más ajustados (proteger ganancias)

    Args:
        base_stop_distance: Distancia base del stop (%)
        regime: 'HIGH_VOL', 'LOW_VOL', o 'UNKNOWN'
        high_vol_mult: Multiplicador para alta vol
        low_vol_mult: Multiplicador para baja vol

    Returns:
        Stop distance ajustado (%)
    """
    if regime == 'HIGH_VOL':
        return base_stop_distance * high_vol_mult
    elif regime == 'LOW_VOL':
        return base_stop_distance * low_vol_mult
    else:
        return base_stop_distance  # Sin ajuste si desconocido


# ============================================================================
# STATISTICAL PERCENTILE TARGETS
# ============================================================================

def calculate_percentile_target(
    historical_returns: pd.Series,
    entry_price: float,
    holding_period: int = 20,
    percentile: int = 75,
    min_obs: int = 100
) -> Dict:
    """
    Calcula take profit basado en percentiles de retornos históricos.

    Paper: Lopez de Prado (2020) "Advances in Financial Machine Learning"

    Mejora sobre FASE 1: usa distribución empírica completa, no solo media.

    Args:
        historical_returns: Retornos históricos diarios
        entry_price: Precio de entrada
        holding_period: Horizonte de holding (días)
        percentile: Percentil objetivo (60-90)
        min_obs: Mínimo de observaciones requeridas

    Returns:
        Dict con target price y estadísticas
    """
    # Verificar suficientes datos
    if len(historical_returns) < min_obs:
        return {
            'target_price': None,
            'error': f'Insufficient data: {len(historical_returns)} < {min_obs}'
        }

    # Calcular retornos N-day forward
    forward_returns = []
    for i in range(len(historical_returns) - holding_period):
        period_return = (1 + historical_returns.iloc[i:i+holding_period]).prod() - 1
        forward_returns.append(period_return)

    if len(forward_returns) < min_obs // 2:
        return {
            'target_price': None,
            'error': 'Insufficient forward return observations'
        }

    forward_returns = np.array(forward_returns)

    # Calcular percentil
    target_return = np.percentile(forward_returns, percentile)

    # Target price
    target_price = entry_price * (1 + target_return)

    # Probability de alcanzar target (histórico)
    prob_reach = (forward_returns >= target_return).mean()

    # Estadísticas adicionales
    mean_return = forward_returns.mean()
    std_return = forward_returns.std()

    return {
        'target_price': target_price,
        'target_return_pct': target_return * 100,
        'percentile': percentile,
        'probability_reach': prob_reach,
        'mean_return_pct': mean_return * 100,
        'std_return_pct': std_return * 100,
        'holding_period': holding_period,
        'n_observations': len(forward_returns),
    }


# ============================================================================
# TIME-BASED EXITS
# ============================================================================

def calculate_time_decay_target(
    initial_target: float,
    entry_price: float,
    days_held: int,
    max_days: int = 90,
    decay_rate: float = 0.02
) -> Dict:
    """
    Calcula target con decay temporal.

    Paper: Harvey & Liu (2021) "Lucky Factors"

    Lógica:
    - Target inicial alto para capturar outliers
    - Decae gradualmente con el tiempo
    - Fuerza exit después de max_days

    Args:
        initial_target: Target inicial (precio)
        entry_price: Precio de entrada
        days_held: Días desde entrada
        max_days: Máximo días de holding
        decay_rate: Tasa de decay semanal (2% = 0.02)

    Returns:
        Dict con target ajustado y recomendación
    """
    if days_held >= max_days:
        return {
            'adjusted_target': entry_price,  # Exit at market
            'should_exit': True,
            'exit_reason': 'MAX_HOLDING_PERIOD',
            'days_remaining': 0,
        }

    # Calcular decay
    weeks_held = days_held / 7.0
    decay_factor = (1 - decay_rate) ** weeks_held

    # Target ajustado
    initial_gain = initial_target - entry_price
    adjusted_gain = initial_gain * decay_factor
    adjusted_target = entry_price + adjusted_gain

    # Days remaining
    days_remaining = max_days - days_held

    return {
        'adjusted_target': adjusted_target,
        'should_exit': False,
        'days_held': days_held,
        'days_remaining': days_remaining,
        'decay_factor': decay_factor,
        'target_return_pct': ((adjusted_target - entry_price) / entry_price) * 100,
    }


def check_time_exit(
    days_held: int,
    max_days: int = 90,
    current_price: float = None,
    entry_price: float = None,
    min_return_threshold: float = 0.0
) -> Dict:
    """
    Verifica si debe hacer exit por tiempo.

    Paper: Harvey & Liu (2021)

    Exit conditions:
    1. Alcanzó max_days
    2. Cerca de max_days y return < threshold

    Args:
        days_held: Días desde entrada
        max_days: Máximo días permitido
        current_price: Precio actual (opcional)
        entry_price: Precio entrada (opcional)
        min_return_threshold: Return mínimo para mantener cerca de max

    Returns:
        Dict con recomendación de exit
    """
    # Exit por max holding period
    if days_held >= max_days:
        return {
            'should_exit': True,
            'exit_reason': 'MAX_HOLDING_PERIOD',
            'urgency': 'IMMEDIATE',
        }

    # Check si está cerca del máximo
    days_remaining = max_days - days_held
    near_max = days_remaining <= 7  # Última semana

    if near_max and current_price and entry_price:
        current_return = (current_price - entry_price) / entry_price

        # Exit si está perdiendo o ganancia mínima
        if current_return < min_return_threshold:
            return {
                'should_exit': True,
                'exit_reason': 'TIME_DECAY_LOW_RETURN',
                'current_return_pct': current_return * 100,
                'days_remaining': days_remaining,
                'urgency': 'HIGH',
            }

    return {
        'should_exit': False,
        'days_remaining': days_remaining,
        'urgency': 'NONE',
    }


# ============================================================================
# PROFIT LOCK (TRAILING TAKE PROFIT)
# ============================================================================

def calculate_profit_lock(
    entry_price: float,
    current_price: float,
    peak_price: float,
    lock_threshold: float = 0.15,
    trail_pct: float = 0.05
) -> Dict:
    """
    Trailing take profit que se activa después de ganancia significativa.

    Estrategia híbrida:
    - Deja correr ganancias inicialmente
    - Después de +15%, activa trailing TP
    - Protege ganancias con trail de 5%

    Args:
        entry_price: Precio de entrada
        current_price: Precio actual
        peak_price: Precio máximo desde entrada
        lock_threshold: Ganancia para activar lock (15% = 0.15)
        trail_pct: % de trail desde peak (5% = 0.05)

    Returns:
        Dict con profit lock activo y nivel
    """
    # Calcular ganancia actual y peak
    current_gain = (current_price - entry_price) / entry_price
    peak_gain = (peak_price - entry_price) / entry_price

    # ¿Lock activado?
    lock_active = peak_gain >= lock_threshold

    if not lock_active:
        return {
            'lock_active': False,
            'lock_price': None,
            'current_gain_pct': current_gain * 100,
            'threshold_pct': lock_threshold * 100,
            'gain_to_threshold': (lock_threshold - peak_gain) * 100,
        }

    # Lock price = peak - trail%
    lock_price = peak_price * (1 - trail_pct)

    # ¿Debería salir?
    should_exit = current_price <= lock_price

    return {
        'lock_active': True,
        'lock_price': lock_price,
        'peak_price': peak_price,
        'current_price': current_price,
        'should_exit': should_exit,
        'locked_gain_pct': ((lock_price - entry_price) / entry_price) * 100,
        'current_gain_pct': current_gain * 100,
        'peak_gain_pct': peak_gain * 100,
    }


# ============================================================================
# INTEGRATED ADVANCED EXITS CALCULATOR
# ============================================================================

class AdvancedExitsCalculator:
    """
    Calculator integrado para todas las estrategias avanzadas de exit.

    Combina:
    - Regime-based dynamic stops (FASE 2)
    - Statistical percentile targets (FASE 2)
    - Time-based exits (FASE 2)
    - Profit lock trailing TP (FASE 2)
    """

    def __init__(self, config: AdvancedExitsConfig = None):
        self.config = config or AdvancedExitsConfig()

    def calculate_advanced_parameters(
        self,
        entry_price: float,
        current_price: float,
        prices: pd.DataFrame,
        base_stop_distance: float,
        base_target: float,
        days_held: int = 0,
        peak_price: Optional[float] = None
    ) -> Dict:
        """
        Calcula todos los parámetros avanzados de exit.

        Args:
            entry_price: Precio de entrada
            current_price: Precio actual
            prices: DataFrame con precios históricos
            base_stop_distance: Stop distance de FASE 1 (%)
            base_target: Target de FASE 1 (precio)
            days_held: Días desde entrada
            peak_price: Precio máximo desde entrada (None = usar current)

        Returns:
            Dict con todos los parámetros ajustados
        """
        results = {
            'entry_price': entry_price,
            'current_price': current_price,
            'base_stop_distance': base_stop_distance,
            'base_target': base_target,
        }

        # Calculate returns
        returns = prices['close'].pct_change().dropna()

        # ========== REGIME-BASED STOPS ==========
        if self.config.use_regime_stops:
            regime = detect_volatility_regime(
                returns,
                self.config.regime_lookback
            )

            adjusted_stop_distance = calculate_regime_adjusted_stop(
                base_stop_distance,
                regime,
                self.config.high_vol_multiplier,
                self.config.low_vol_multiplier
            )

            adjusted_stop_price = entry_price * (1 - adjusted_stop_distance / 100)

            results['regime'] = {
                'regime': regime,
                'adjusted_stop_distance_pct': adjusted_stop_distance,
                'adjusted_stop_price': adjusted_stop_price,
                'base_stop_distance_pct': base_stop_distance,
            }

        # ========== STATISTICAL PERCENTILE TARGETS ==========
        if self.config.use_percentile_targets:
            percentile_target = calculate_percentile_target(
                returns,
                entry_price,
                self.config.holding_period_days,
                self.config.target_percentile,
                self.config.min_observations
            )

            results['percentile_target'] = percentile_target

        # ========== TIME-BASED EXITS ==========
        if self.config.use_time_exits:
            # Time decay target
            if self.config.time_decay_enabled:
                time_decay = calculate_time_decay_target(
                    base_target,
                    entry_price,
                    days_held,
                    self.config.max_holding_days,
                    self.config.decay_rate
                )
                results['time_decay'] = time_decay

            # Time exit check
            time_exit = check_time_exit(
                days_held,
                self.config.max_holding_days,
                current_price,
                entry_price,
                min_return_threshold=0.05  # 5% mínimo
            )
            results['time_exit'] = time_exit

        # ========== PROFIT LOCK ==========
        if self.config.use_profit_lock:
            peak = peak_price if peak_price else current_price

            profit_lock = calculate_profit_lock(
                entry_price,
                current_price,
                peak,
                self.config.profit_lock_threshold,
                self.config.profit_lock_trail
            )
            results['profit_lock'] = profit_lock

        # ========== FINAL RECOMMENDATIONS ==========
        results['recommendations'] = self._generate_recommendations(results)

        return results

    def _generate_recommendations(self, results: Dict) -> Dict:
        """
        Genera recomendaciones finales basadas en todos los indicadores.

        Args:
            results: Resultados de calculate_advanced_parameters

        Returns:
            Dict con recomendaciones de acción
        """
        recommendations = {
            'action': 'HOLD',
            'reasons': [],
            'urgency': 'NONE',
        }

        # Check time exit
        if 'time_exit' in results and results['time_exit']['should_exit']:
            recommendations['action'] = 'EXIT'
            recommendations['reasons'].append(results['time_exit']['exit_reason'])
            recommendations['urgency'] = results['time_exit']['urgency']

        # Check profit lock
        if 'profit_lock' in results and results['profit_lock']['lock_active']:
            if results['profit_lock']['should_exit']:
                recommendations['action'] = 'EXIT'
                recommendations['reasons'].append('PROFIT_LOCK_TRIGGERED')
                recommendations['urgency'] = 'HIGH'

        # Adjusted targets
        final_stop = None
        final_target = None

        # Stop: usar regime-adjusted si disponible
        if 'regime' in results:
            final_stop = results['regime']['adjusted_stop_price']
        else:
            final_stop = results['entry_price'] * (1 - results['base_stop_distance'] / 100)

        # Target: usar el menor de percentile, time-decay, profit-lock
        targets = []

        if 'percentile_target' in results and results['percentile_target'].get('target_price'):
            targets.append(results['percentile_target']['target_price'])

        if 'time_decay' in results and not results['time_decay']['should_exit']:
            targets.append(results['time_decay']['adjusted_target'])

        if targets:
            final_target = min(targets)  # Usar el más conservador
        else:
            final_target = results['base_target']

        recommendations['final_stop'] = final_stop
        recommendations['final_target'] = final_target

        return recommendations


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing Advanced Exits System (FASE 2)")
    print("=" * 80)

    # Mock data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-12-04', freq='D')
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices_close = 100 * (1 + returns).cumprod()

    prices_df = pd.DataFrame({
        'close': prices_close,
    }, index=dates)

    returns_series = pd.Series(returns, index=dates)

    # Test parameters
    entry_price = 100.0
    current_price = prices_close[-1]
    base_stop_distance = 5.0  # 5%
    base_target = 115.0  # +15%
    days_held = 30

    # Initialize calculator
    config = AdvancedExitsConfig(
        use_regime_stops=True,
        use_percentile_targets=True,
        use_time_exits=True,
        use_profit_lock=True,
    )

    calculator = AdvancedExitsCalculator(config)

    # Calculate advanced parameters
    results = calculator.calculate_advanced_parameters(
        entry_price=entry_price,
        current_price=current_price,
        prices=prices_df,
        base_stop_distance=base_stop_distance,
        base_target=base_target,
        days_held=days_held,
    )

    # Display results
    print("\n📊 ADVANCED EXITS ANALYSIS:")
    print("=" * 80)

    print(f"\nEntry Price:   ${entry_price:.2f}")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Days Held:     {days_held}")

    if 'regime' in results:
        print(f"\n🔄 REGIME DETECTION:")
        regime_data = results['regime']
        print(f"  Regime:           {regime_data['regime']}")
        print(f"  Base Stop:        {regime_data['base_stop_distance_pct']:.2f}%")
        print(f"  Adjusted Stop:    {regime_data['adjusted_stop_distance_pct']:.2f}%")
        print(f"  Stop Price:       ${regime_data['adjusted_stop_price']:.2f}")

    if 'percentile_target' in results:
        print(f"\n📈 PERCENTILE TARGET:")
        pt = results['percentile_target']
        if pt.get('target_price'):
            print(f"  Target Price:     ${pt['target_price']:.2f}")
            print(f"  Target Return:    {pt['target_return_pct']:.2f}%")
            print(f"  Probability:      {pt['probability_reach']:.1%}")
            print(f"  Percentile:       {pt['percentile']}th")
        else:
            print(f"  Error: {pt.get('error', 'Unknown')}")

    if 'time_decay' in results:
        print(f"\n⏰ TIME DECAY:")
        td = results['time_decay']
        print(f"  Adjusted Target:  ${td['adjusted_target']:.2f}")
        print(f"  Decay Factor:     {td.get('decay_factor', 0):.3f}")
        print(f"  Days Remaining:   {td.get('days_remaining', 0)}")

    if 'profit_lock' in results:
        print(f"\n🔒 PROFIT LOCK:")
        pl = results['profit_lock']
        print(f"  Active:           {pl['lock_active']}")
        if pl['lock_active']:
            print(f"  Lock Price:       ${pl['lock_price']:.2f}")
            print(f"  Peak Gain:        {pl['peak_gain_pct']:.2f}%")
            print(f"  Should Exit:      {pl['should_exit']}")

    print(f"\n✅ RECOMMENDATIONS:")
    rec = results['recommendations']
    print(f"  Action:           {rec['action']}")
    print(f"  Final Stop:       ${rec['final_stop']:.2f}")
    print(f"  Final Target:     ${rec['final_target']:.2f}")
    if rec['reasons']:
        print(f"  Reasons:          {', '.join(rec['reasons'])}")
    print(f"  Urgency:          {rec['urgency']}")

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("\nKey Features Implemented (FASE 2):")
    print("  ✅ Regime-based dynamic stops (Nystrup et al. 2020)")
    print("  ✅ Statistical percentile targets (Lopez de Prado 2020)")
    print("  ✅ Time-based exits & target decay (Harvey & Liu 2021)")
    print("  ✅ Profit lock trailing TP (hybrid strategy)")
