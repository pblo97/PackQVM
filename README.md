# 🎯 PackQVM - Quality-Value-Momentum Strategy

Sistema de screening de acciones usando **Piotroski Score Real** + **Quality-Value Score** sin multicolinealidad + **MA200 Filter** + **Backtest**.

---

## 🚀 Inicio Rápido

```bash
# 1. Configurar API key de Financial Modeling Prep
export FMP_API_KEY="tu_api_key_aqui"

# 2. Instalar dependencias
pip install pandas numpy requests streamlit plotly

# 3. Ejecutar la aplicación V3 (RECOMENDADO)
streamlit run app_streamlit_v3.py
```

**La aplicación se abrirá en tu navegador en http://localhost:8501**

### Versiones Disponibles:
- **`app_streamlit_v3.py`** ⭐ **RECOMENDADO** - Versión completa con MA200, Backtest, Momentum, ROIC>WACC
- `app_streamlit_v2.py` - Versión con Piotroski real y sliders ajustables
- `app_streamlit.py` - Versión básica original

---

## ✨ Características

### 🎯 Piotroski Score Real (9 Checks)
Calculado desde estados financieros completos:
- ✅ Profitability (4): ROA, OCF, ΔROA, Accruals
- ✅ Leverage/Liquidity (3): ΔLeverage, ΔCurrent Ratio, ΔShares
- ✅ Operating Efficiency (2): ΔGross Margin, ΔAsset Turnover

### 💎 Quality-Value Score SIN Multicolinealidad
```
QV Score = 40% Piotroski + 35% Value + 15% FCF Yield + 10% Momentum
```
- **Piotroski**: Captura calidad operacional completa
- **Value**: EV/EBITDA, P/B, P/E (independiente de Piotroski)
- **FCF Yield**: Free Cash Flow / Market Cap
- **Momentum**: Retornos históricos 12M-1M

### 🚀 NUEVAS CARACTERÍSTICAS V3

#### 📈 MA200 Filter (Faber 2007)
- Filtro de tendencia: solo stocks por encima de MA de 200 días
- Implementa "A Quantitative Approach to Tactical Asset Allocation"

#### 🎯 Momentum 12M-1M (Jegadeesh & Titman 1993)
- Momentum real calculado desde precios históricos
- Excluye último mes para evitar reversión de corto plazo

#### 💎 Métricas Avanzadas de Valoración
- **EBIT/EV**: Earnings Yield (mejor indicador que P/E)
- **FCF/EV**: Free Cash Flow Yield normalizado
- **ROIC > WACC**: Filtro de creación de valor

#### 📊 Backtest Integrado
- Performance histórica del portfolio
- Métricas: CAGR, Sharpe Ratio, Sortino Ratio, Max Drawdown
- Rebalanceo configurable (Mensual/Trimestral/Anual)
- Costos de trading incluidos (comisión, slippage, market impact)
- Visualización de equity curve

#### 🔍 Filtros Heurísticos
- 52-Week High filter: precio cerca del máximo anual
- ROIC > WACC: solo empresas que crean valor

### 🎛️ Parámetros Ajustables (Sliders)
- Pesos de Quality/Value/FCF Yield/Momentum
- Filtros de Piotroski mínimo (0-9)
- QV Score mínimo (0-1)
- Límites de valoración (P/E, EV/EBITDA, EBIT/EV, FCF/EV)
- Momentum mínimo 12M
- Tamaño de universo y portfolio
- Configuración de backtest y costos

### 📊 Visualizaciones Interactivas
- Funnel de selección por pasos (10 pasos en V3)
- Distribución de scores
- Análisis por sector
- Componentes de Piotroski
- Métricas de valoración avanzadas
- **NUEVO**: Análisis de Momentum y MA200
- **NUEVO**: Equity curve del backtest
- **NUEVO**: Performance metrics dashboard
- Exportación a CSV

---

## 📁 Estructura del Proyecto

### Archivos Principales V3 (RECOMENDADO):
```
app_streamlit_v3.py          # ⭐ Interfaz V3 completa (MA200, Backtest, Momentum)
qvm_pipeline_v3.py           # ⭐ Pipeline V3 con 10 pasos y backtest integrado
quality_value_score.py       # Score sin multicolinealidad
data_fetcher.py              # Descarga de datos + Piotroski Score real
```

### Archivos V2 (Funcionales):
```
app_streamlit_v2.py          # Interfaz V2 con sliders ajustables
qvm_pipeline_v2.py           # Pipeline V2 optimizado (6 pasos)
```

### Archivos de Soporte:
```
factor_calculator.py         # Cálculo de factores QVM
piotroski_fscore.py         # F-Score simplificado
screener_filters.py         # Filtros de calidad
backtest_engine.py          # Motor de backtesting con métricas
momentum_calculator.py      # Cálculo de momentum y MA200
```

### Documentación:
```
PIPELINE_V3_FEATURES.md     # ⭐ Documentación completa de V3
USAGE_V2.md                 # Guía detallada de V2
README.md                   # Este archivo
```

---

## 📊 Ejemplo de Resultados

```
Portfolio Final (5 stocks):

symbol  piotroski_score  qv_score
   TSM                9  0.811   ⭐⭐⭐ (Excelente)
  META                8  0.769   ⭐⭐⭐
  GOOG                8  0.733   ⭐⭐⭐
 GOOGL                8  0.733   ⭐⭐⭐
  AMZN                7  0.631   ⭐⭐

Piotroski Promedio: 8.0/9 (Excelente Calidad)
QV Score Promedio: 0.74 (Muy Atractivo)
```

---

## 🎓 Interpretación de Scores

### Piotroski Score (0-9)
- **8-9**: Excelente calidad → STRONG BUY ⭐⭐⭐
- **6-7**: Buena calidad → BUY ⭐⭐
- **4-5**: Calidad media → HOLD ⭐
- **0-3**: Baja calidad → AVOID ❌

### QV Score (0-1)
- **> 0.70**: Muy atractivo → STRONG BUY 🎯
- **0.50-0.70**: Atractivo → BUY ✅
- **0.30-0.50**: Neutral → HOLD ⚠️
- **< 0.30**: No atractivo → AVOID ❌

---

## 🔧 Uso Programático

### Pipeline V3 (RECOMENDADO):

```python
from qvm_pipeline_v3 import run_qvm_pipeline_v3, QVMConfigV3

# Configurar parámetros V3
config = QVMConfigV3(
    universe_size=200,
    portfolio_size=30,
    min_piotroski_score=6,
    min_qv_score=0.50,
    w_quality=0.40,
    w_value=0.35,
    w_fcf_yield=0.15,
    w_momentum=0.10,
    # Nuevos parámetros V3
    require_above_ma200=True,          # MA200 filter
    min_momentum_12m=0.10,             # 10% momentum mínimo
    require_roic_above_wacc=True,      # ROIC > WACC
    backtest_enabled=True,             # Ejecutar backtest
)

# Ejecutar pipeline V3
results = run_qvm_pipeline_v3(config=config, verbose=True)

if results.get('success'):
    portfolio = results['portfolio']
    print(portfolio[['symbol', 'piotroski_score', 'qv_score', 'momentum_12m', 'above_ma200']])

    # Resultados del backtest
    if results.get('backtest'):
        metrics = results['backtest']['portfolio_metrics']
        print(f"\nBacktest Results:")
        print(f"CAGR: {metrics['CAGR']:.2%}")
        print(f"Sharpe: {metrics['Sharpe']:.2f}")
        print(f"Max DD: {metrics['MaxDD']:.2%}")
```

### Pipeline V2 (Alternativa):

```python
from qvm_pipeline_v2 import run_qvm_pipeline_v2, QVMConfig

config = QVMConfig(
    universe_size=200,
    portfolio_size=30,
    min_piotroski_score=6,
    min_qv_score=0.50,
)

results = run_qvm_pipeline_v2(config=config, verbose=True)
```

---

## 📚 Referencias Académicas

1. **Piotroski, J. D. (2000)**. "Value Investing: The Use of Historical Financial Statement Information to Separate Winners from Losers." *Journal of Accounting Research*, 38, 1-41.

2. **Asness, C. S., Frazzini, A., & Pedersen, L. H. (2019)**. "Quality Minus Junk." *Review of Accounting Studies*, 24(1), 34-112.

3. **Fama, E. F., & French, K. R. (1992)**. "The Cross-Section of Expected Stock Returns." *Journal of Finance*, 47(2), 427-465.

4. **Faber, M. T. (2007)**. "A Quantitative Approach to Tactical Asset Allocation." *The Journal of Wealth Management*, 9(4), 69-79. (MA200 Filter)

5. **Jegadeesh, N., & Titman, S. (1993)**. "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency." *Journal of Finance*, 48(1), 65-91. (Momentum Strategy)

---

## 🔑 API Key

Necesitas una API key de **Financial Modeling Prep**:
- **Gratis**: https://financialmodelingprep.com/developer/docs/
- **Configurar**: `export FMP_API_KEY="tu_api_key"`

---

## ⚠️ Troubleshooting

### Error: "FMP_API_KEY no configurada"
```bash
export FMP_API_KEY="tu_api_key_aqui"
```

### Error: "_to_float no está definida"
```bash
# Limpiar caché de Python
rm -rf __pycache__
python3 -c "import data_fetcher"  # Reimportar
```

### Streamlit no encuentra módulos
```bash
# Ejecutar desde el directorio del proyecto
cd PackQVM
streamlit run app_streamlit.py
```

### Datos desactualizados
```bash
# Limpiar caché de FMP
rm -rf .cache/fmp/
```

---

## 📧 Soporte

Para preguntas o issues: https://github.com/pblo97/PackQVM/issues

---

## 📝 Notas

- **Caché**: Los datos se cachean en `.cache/fmp/` (TTL: 24h)
- **Rate Limiting**: Respeta límites de 6-7 requests/segundo
- **Multicolinealidad**: Evitada - Piotroski NO se mezcla con ROA, ROIC, ROE crudos

---

**Desarrollado con ❤️ usando metodología académica robusta**
