# 🎯 PackQVM - Quality-Value-Momentum Strategy

Sistema de screening de acciones usando **Piotroski Score Real** + **Quality-Value Score** sin multicolinealidad.

---

## 🚀 Inicio Rápido

```bash
# 1. Configurar API key de Financial Modeling Prep
export FMP_API_KEY="tu_api_key_aqui"

# 2. Instalar dependencias
pip install pandas numpy requests streamlit plotly

# 3. Ejecutar la aplicación
streamlit run app_streamlit.py
```

**La aplicación se abrirá en tu navegador en http://localhost:8501**

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
- **Momentum**: Retornos históricos

### 🎛️ Parámetros Ajustables (Sliders)
- Pesos de Quality/Value/FCF Yield/Momentum
- Filtros de Piotroski mínimo (0-9)
- QV Score mínimo (0-1)
- Límites de valoración (P/E, EV/EBITDA)
- Tamaño de universo y portfolio

### 📊 Visualizaciones Interactivas
- Funnel de selección por pasos
- Distribución de scores
- Análisis por sector
- Componentes de Piotroski
- Métricas de valoración
- Exportación a CSV

---

## 📁 Estructura del Proyecto

### Archivos Principales:
```
app_streamlit.py              # ⭐ Interfaz principal (Streamlit)
qvm_pipeline_v2.py           # Pipeline optimizado con análisis por pasos
quality_value_score.py       # Score sin multicolinealidad
data_fetcher.py              # Descarga de datos + Piotroski Score
```

### Archivos de Soporte:
```
factor_calculator.py         # Cálculo de factores QVM
piotroski_fscore.py         # F-Score simplificado
screener_filters.py         # Filtros de calidad
backtest_engine.py          # Motor de backtesting
momentum_calculator.py      # Cálculo de momentum
```

### Documentación:
```
USAGE_V2.md                 # Guía detallada de uso
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

```python
from qvm_pipeline_v2 import run_qvm_pipeline_v2, QVMConfig

# Configurar parámetros
config = QVMConfig(
    universe_size=200,
    portfolio_size=30,
    min_piotroski_score=6,
    min_qv_score=0.50,
    w_quality=0.40,
    w_value=0.35,
    w_fcf_yield=0.15,
    w_momentum=0.10,
)

# Ejecutar pipeline
results = run_qvm_pipeline_v2(config=config, verbose=True)

if results.get('success'):
    portfolio = results['portfolio']
    print(portfolio[['symbol', 'piotroski_score', 'qv_score', 'sector']])
```

---

## 📚 Referencias Académicas

1. **Piotroski, J. D. (2000)**. "Value Investing: The Use of Historical Financial Statement Information to Separate Winners from Losers." *Journal of Accounting Research*, 38, 1-41.

2. **Asness, C. S., Frazzini, A., & Pedersen, L. H. (2019)**. "Quality Minus Junk." *Review of Accounting Studies*, 24(1), 34-112.

3. **Fama, E. F., & French, K. R. (1992)**. "The Cross-Section of Expected Stock Returns." *Journal of Finance*, 47(2), 427-465.

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
