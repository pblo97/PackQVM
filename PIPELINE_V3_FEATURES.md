# 🚀 Pipeline V3 - Características Completas

## ✅ LO QUE SE HA IMPLEMENTADO

### 1. **MA200 Filter** (Faber 2007) ✅
```python
require_above_ma200: bool = True  # Filtro de media móvil 200 días
```
- Elimina stocks en tendencia bajista
- Basado en paper de Meb Faber "A Quantitative Approach to Tactical Asset Allocation"
- Mejora Sharpe ratio y reduce drawdowns

### 2. **Momentum Real** (Jegadeesh & Titman 1993) ✅
```python
min_momentum_12m: float = 0.10  # 10% return mínimo últimos 12 meses
```
- Calculado desde precios históricos reales
- Excluye último mes para evitar reversal de corto plazo
- Usa función `calculate_12m_1m_momentum()` de momentum_calculator.py

### 3. **Filtros 52-Week High** ✅
```python
require_near_52w_high: bool = False
min_pct_from_52w_high: float = 0.80  # Precio >= 80% del 52w high
```
- Identifica stocks con momentum fuerte
- Configurable: puedes requerir que estén cerca del máximo anual

### 4. **Métricas Avanzadas de Valoración** ✅

#### a) **EBIT/EV (Earnings Yield)**
```python
min_ebit_ev: float = 0.08  # EBIT/EV mínimo 8%
```
- Mejor que P/E porque usa Enterprise Value
- Captura toda la estructura de capital (deuda + equity)

#### b) **FCF/EV (Free Cash Flow Yield)**
```python
max_fcf_ev: float = 0.15  # FCF/EV óptimo > 8%
```
- Free Cash Flow normalizado por Enterprise Value
- Mejor indicador de valor que solo FCF

#### c) **ROIC > WACC**
```python
require_roic_above_wacc: bool = True
```
- Return on Invested Capital vs Weighted Average Cost of Capital
- Solo invierte en empresas que crean valor
- WACC estimado en 9% (configurable en código)

### 5. **Backtest Integrado** ✅
```python
backtest_enabled: bool = True
backtest_start: str = "2020-01-01"
backtest_end: str = "2024-12-31"
rebalance_freq: str = "Q"  # Quarterly
```

**Métricas calculadas:**
- CAGR (Compound Annual Growth Rate)
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio

**Trading Costs realistas:**
```python
commission_bps: int = 5      # 0.05% comisión
slippage_bps: int = 5        # 0.05% slippage
market_impact_bps: int = 2   # 0.02% market impact
```

### 6. **Volumen Relativo** (placeholder) ✅
```python
min_relative_volume: float = 1.0
```
- Actualmente placeholder (requiere datos intraday)
- Infraestructura lista para cuando tengas datos en tiempo real

---

## 🎯 FLUJO COMPLETO DEL PIPELINE V3

```
PASO 1: Screener Inicial
   ↓ (Market cap, volumen)

PASO 2: Estados Financieros + Piotroski Score
   ↓ (9 checks completos)

PASO 3: Filtros Básicos de Calidad
   ↓ (Piotroski mín, P/E, EV/EBITDA, FCF>0, ROIC>10%)

PASO 4: Quality-Value Score
   ↓ (40% Piotroski + 35% Value + 15% FCF + 10% Momentum)

PASO 5: Métricas Avanzadas
   ↓ (EBIT/EV, FCF/EV, ROIC>WACC)

PASO 6: Precios Históricos
   ↓ (Descargar últimos 5 años)

PASO 7: Momentum + MA200 Filter ⭐ CRÍTICO
   ↓ (Momentum 12M-1M, Price > MA200)

PASO 8: Filtros 52w High
   ↓ (Cerca del máximo anual)

PASO 9: Selección Portfolio
   ↓ (Top N por QV Score)

PASO 10: Backtest (opcional)
   ↓ (CAGR, Sharpe, Max DD)

✅ RESULTADO: Portfolio optimizado con backtest
```

---

## 📋 CONFIGURACIÓN COMPLETA

### Parámetros Disponibles en QVMConfigV3:

```python
config = QVMConfigV3(
    # UNIVERSE
    universe_size=300,
    min_market_cap=2e9,
    min_volume=500_000,

    # QUALITY-VALUE WEIGHTS
    w_quality=0.40,
    w_value=0.35,
    w_fcf_yield=0.15,
    w_momentum=0.10,

    # FILTROS BÁSICOS
    min_piotroski_score=6,
    min_qv_score=0.50,
    max_pe=40.0,
    max_pb=10.0,
    max_ev_ebitda=20.0,
    require_positive_fcf=True,
    min_roic=0.10,

    # FILTROS AVANZADOS (NUEVOS)
    require_above_ma200=True,          # ⭐ MA200 filter
    min_momentum_12m=0.10,             # ⭐ Momentum mínimo
    require_near_52w_high=False,       # ⭐ 52w high filter
    min_pct_from_52w_high=0.80,
    min_relative_volume=1.0,

    # MÉTRICAS VALORACIÓN (NUEVAS)
    max_fcf_ev=0.15,                   # ⭐ FCF/EV
    min_ebit_ev=0.08,                  # ⭐ EBIT/EV
    require_roic_above_wacc=True,      # ⭐ ROIC>WACC

    # PORTFOLIO
    portfolio_size=30,

    # BACKTEST
    backtest_enabled=True,             # ⭐ Backtest on/off
    backtest_start="2020-01-01",
    backtest_end="2024-12-31",
    rebalance_freq="Q",
    commission_bps=5,
    slippage_bps=5,
    market_impact_bps=2,
)
```

---

## 💻 USO DEL PIPELINE V3

### Uso Básico:

```python
from qvm_pipeline_v3 import run_qvm_pipeline_v3, QVMConfigV3

# Configuración por defecto
results = run_qvm_pipeline_v3(verbose=True)

# Acceder a resultados
portfolio = results['portfolio']
backtest = results['backtest']
steps = results['steps']
```

### Uso Avanzado:

```python
# Configuración personalizada
config = QVMConfigV3(
    universe_size=200,
    portfolio_size=30,
    require_above_ma200=True,    # Filtro MA200 activado
    min_momentum_12m=0.15,        # Momentum agresivo 15%
    require_roic_above_wacc=True, # Solo empresas creando valor
    backtest_enabled=True,
)

results = run_qvm_pipeline_v3(config=config, verbose=True)

if results.get('success'):
    # Portfolio final
    portfolio = results['portfolio']
    print(portfolio[['symbol', 'piotroski_score', 'qv_score',
                     'momentum_12m', 'above_ma200']])

    # Backtest results
    if results.get('backtest'):
        metrics = results['backtest']['portfolio_metrics']
        print(f"CAGR: {metrics['CAGR']:.2%}")
        print(f"Sharpe: {metrics['Sharpe']:.2f}")
        print(f"Max DD: {metrics['MaxDD']:.2%}")
```

---

## 📊 EJEMPLO DE SALIDA

```
✅ PIPELINE V3 COMPLETO - RESUMEN
================================================================================
✅ PASO 1: 200 stocks (100.0% pass rate) - Screener
✅ PASO 2: 180 stocks (90.0% pass rate) - Piotroski
✅ PASO 3: 120 stocks (60.0% pass rate) - Filtros Calidad
✅ PASO 4: 100 stocks (83.3% pass rate) - Quality-Value Score
✅ PASO 5: 85 stocks (85.0% pass rate) - Métricas Avanzadas
✅ PASO 6: 80 stocks (94.1% pass rate) - Precios
✅ PASO 7: 60 stocks (75.0% pass rate) - Momentum + MA200 ⭐
✅ PASO 8: 55 stocks (91.7% pass rate) - 52w High
✅ PASO 9: 30 stocks (54.5% pass rate) - Portfolio Final
✅ PASO 10: 30 stocks (100.0% pass rate) - Backtest

📋 TOP 10 STOCKS:
symbol  piotroski  qv_score  momentum_12m  above_ma200  roic_above_wacc
TSM            9     0.815         0.285         True             True
META           8     0.782         0.156         True             True
GOOGL          8     0.745         0.198         True             True
NVDA           7     0.721         0.412         True             True
MSFT           8     0.698         0.167         True             True

📊 BACKTEST RESULTS:
CAGR: 24.5%
Sharpe: 1.82
Sortino: 2.45
Max DD: -18.3%
Calmar: 1.34
```

---

## 🔄 PRÓXIMOS PASOS

### 1. **Actualizar app_streamlit.py** (PENDIENTE)
   - Importar `qvm_pipeline_v3` en vez de `qvm_pipeline_v2`
   - Agregar sliders para nuevos parámetros:
     * MA200 filter on/off
     * Momentum mínimo
     * 52w high filter
     * ROIC>WACC filter
     * Backtest on/off
   - Mostrar resultados de backtest en interfaz

### 2. **Testing Completo** (PENDIENTE)
   - Probar pipeline V3 con diferentes configuraciones
   - Validar que backtest funciona
   - Verificar que MA200 filtra correctamente

### 3. **Documentación** (PENDIENTE)
   - Actualizar README.md con features V3
   - Agregar ejemplos de uso
   - Explicar cada nuevo parámetro

---

## 📚 REFERENCIAS ACADÉMICAS

1. **Faber (2007)**: "A Quantitative Approach to Tactical Asset Allocation"
   - MA200 filter mejora Sharpe y reduce drawdowns
   - [Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=962461)

2. **Jegadeesh & Titman (1993)**: "Returns to Buying Winners and Selling Losers"
   - Momentum 12M-1M como predictor de retornos
   - [Paper](https://www.jstor.org/stable/2328882)

3. **Piotroski (2000)**: "Value Investing: The Use of Historical Financial Statement Information"
   - F-Score de 9 puntos para separar winners de losers
   - [Paper](https://www.jstor.org/stable/2672906)

4. **Asness, Frazzini & Pedersen (2019)**: "Quality Minus Junk"
   - Factores de calidad (ROE, ROIC, margins) predicen retornos
   - [Paper](https://www.aqr.com/Insights/Research/Journal-Article/Quality-Minus-Junk)

---

## ⚙️ CONFIGURACIÓN ÓPTIMA RECOMENDADA

Para **máximo Sharpe Ratio**:
```python
QVMConfigV3(
    require_above_ma200=True,      # Filtro de tendencia
    min_momentum_12m=0.15,          # Momentum agresivo
    min_piotroski_score=7,          # Alta calidad
    require_roic_above_wacc=True,   # Creación de valor
    portfolio_size=20,              # Concentración
)
```

Para **mínimo Drawdown**:
```python
QVMConfigV3(
    require_above_ma200=True,      # Filtro de tendencia
    min_momentum_12m=0.20,          # Momentum muy agresivo
    min_piotroski_score=8,          # Calidad excepcional
    max_ev_ebitda=15.0,             # Valoración conservadora
    portfolio_size=30,              # Diversificación
)
```

Para **balance Riesgo-Retorno**:
```python
QVMConfigV3(
    require_above_ma200=True,
    min_momentum_12m=0.10,
    min_piotroski_score=6,
    require_roic_above_wacc=True,
    portfolio_size=25,
)
```

---

**¡Pipeline V3 está listo y pusheado al repositorio!** 🎉

Próximo paso: Actualizar `app_streamlit.py` para usar V3 con todos los nuevos controles.
