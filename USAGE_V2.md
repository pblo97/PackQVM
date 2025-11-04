# QVM Strategy V2 - Guía de Uso

## 🎯 Nuevas Características

Esta versión incluye mejoras fundamentales que eliminan la multicolinealidad y calculan el Piotroski Score real:

### 1. **Piotroski Score Real (9 Checks Completos)**
   - **Profitability (4 checks)**:
     - ROA > 0
     - Operating Cash Flow > 0
     - ΔROA > 0 (mejora YoY)
     - Accruals < 0 (calidad de ganancias)

   - **Leverage/Liquidity (3 checks)**:
     - Δ(Long-term Debt / Assets) < 0
     - Δ(Current Ratio) > 0
     - No emisión de acciones (ΔShares ≤ 0)

   - **Operating Efficiency (2 checks)**:
     - Δ(Gross Margin) > 0
     - Δ(Asset Turnover) > 0

### 2. **Quality-Value Score SIN Multicolinealidad**
   ```
   QV Score = 40% Quality + 35% Value + 15% FCF Yield + 10% Momentum
   ```

   Donde:
   - **Quality**: Piotroski Score normalizado (0-9 → 0-1)
   - **Value**: Múltiplos de valoración (EV/EBITDA, P/B, P/E) - INDEPENDIENTE
   - **FCF Yield**: Free Cash Flow / Market Cap - INDEPENDIENTE
   - **Momentum**: Retornos históricos (actualmente placeholder)

   ⚠️ **IMPORTANTE**: NO mezclamos ROA, ROIC, ROE con Piotroski porque ya están incluidos en los 9 checks.

### 3. **Métricas Calculadas**
   - ROIC (Return on Invested Capital)
   - FCF Yield (Free Cash Flow Yield)
   - ROA (Return on Assets)
   - ROE (Return on Equity)
   - Gross Margin, Operating Margin, Net Margin

---

## 🚀 Cómo Usar

### Opción 1: Interfaz Streamlit (Recomendado)

```bash
# 1. Configurar API key
export FMP_API_KEY="tu_api_key_aqui"

# 2. Ejecutar app
streamlit run app_streamlit_v2.py
```

En la interfaz podrás:
- ✅ Ajustar parámetros con sliders
- ✅ Ver análisis paso por paso
- ✅ Visualizar distribuciones de scores
- ✅ Exportar resultados a CSV
- ✅ Analizar por sector
- ✅ Ver componentes de Piotroski

### Opción 2: Script Python

```python
from qvm_pipeline_v2 import run_qvm_pipeline_v2, QVMConfig

# Configurar parámetros
config = QVMConfig(
    universe_size=200,
    portfolio_size=30,
    min_piotroski_score=6,      # Mínimo 6/9 (calidad media-alta)
    min_qv_score=0.50,           # Mínimo 0.50 (atractivo)
    w_quality=0.40,              # 40% peso Piotroski
    w_value=0.35,                # 35% peso Value
    w_fcf_yield=0.15,            # 15% peso FCF Yield
    w_momentum=0.10,             # 10% peso Momentum
)

# Ejecutar pipeline
results = run_qvm_pipeline_v2(config=config, verbose=True)

if results.get('success'):
    portfolio = results['portfolio']
    print(portfolio[['symbol', 'piotroski_score', 'qv_score', 'sector']])
```

---

## 📊 Interpretación de Scores

### Piotroski Score (0-9)
- **8-9**: Excelente calidad → STRONG BUY
- **6-7**: Buena calidad → BUY
- **4-5**: Calidad media → HOLD
- **0-3**: Baja calidad → AVOID

### QV Score (0-1)
- **> 0.70**: Muy atractivo → STRONG BUY
- **0.50-0.70**: Atractivo → BUY
- **0.30-0.50**: Neutral → HOLD
- **< 0.30**: No atractivo → AVOID

---

## 🎛️ Parámetros Ajustables

### Universo
- `universe_size`: Tamaño inicial (50-500)
- `min_market_cap`: Market cap mínimo en $ (ej. 2e9 = $2B)
- `min_volume`: Volumen diario mínimo

### Pesos del Score (deben sumar 1.0)
- `w_quality`: Peso de Piotroski (recomendado: 0.35-0.45)
- `w_value`: Peso de Value (recomendado: 0.30-0.40)
- `w_fcf_yield`: Peso de FCF Yield (recomendado: 0.10-0.20)
- `w_momentum`: Peso de Momentum (recomendado: 0.05-0.15)

### Filtros
- `min_piotroski_score`: Mínimo Piotroski (recomendado: 5-7)
- `min_qv_score`: Mínimo QV Score (recomendado: 0.40-0.60)
- `max_pe`: P/E máximo (recomendado: 30-50)
- `max_ev_ebitda`: EV/EBITDA máximo (recomendado: 15-25)
- `require_positive_fcf`: Requerir FCF > 0 (recomendado: True)

### Portfolio
- `portfolio_size`: Número de stocks (recomendado: 20-40)

---

## 📁 Archivos Principales

- **`data_fetcher.py`**: Descarga de datos de FMP API
  - Estados financieros completos
  - Cálculo de Piotroski Score
  - Métricas avanzadas (ROIC, FCF Yield, etc.)

- **`quality_value_score.py`**: Score compuesto sin multicolinealidad
  - Quality: basado en Piotroski
  - Value: múltiplos de valoración
  - FCF Yield: rentabilidad de flujo de caja
  - Momentum: retornos históricos

- **`qvm_pipeline_v2.py`**: Pipeline completo con análisis por pasos
  - 6 pasos con checks y validaciones
  - Funnel analysis
  - Configuración flexible

- **`app_streamlit_v2.py`**: Interfaz interactiva
  - Sliders para ajustar parámetros
  - Visualizaciones con Plotly
  - Exportación de resultados

---

## 🔬 Ejemplo de Resultados

```
Portfolio de 10 stocks:

symbol  piotroski_score  qv_score             sector
   TSM                9  0.772282         Technology
  BABA                7  0.764498  Consumer Cyclical
 BRK-B                6  0.741667 Financial Services
  META                8  0.724956         Technology
 GOOGL                8  0.720801         Technology

Promedio Piotroski: 7.1/9 (Excelente)
Promedio QV Score: 0.70 (Muy atractivo)
Sectores únicos: 5
```

---

## 📚 Referencias Académicas

1. **Piotroski, J. D. (2000)**. "Value Investing: The Use of Historical Financial Statement Information to Separate Winners from Losers." *Journal of Accounting Research*, 38, 1-41.

2. **Asness, C. S., Frazzini, A., & Pedersen, L. H. (2019)**. "Quality Minus Junk." *Review of Accounting Studies*, 24(1), 34-112.

3. **Fama, E. F., & French, K. R. (1992)**. "The Cross-Section of Expected Stock Returns." *Journal of Finance*, 47(2), 427-465.

4. **Fama, E. F., & French, K. R. (2015)**. "A Five-Factor Asset Pricing Model." *Journal of Financial Economics*, 116(1), 1-22.

---

## ⚠️ Notas Importantes

1. **API Key**: Necesitas una API key de Financial Modeling Prep
   - Gratis: https://financialmodelingprep.com/developer/docs/
   - Configura: `export FMP_API_KEY="tu_api_key"`

2. **Rate Limiting**: El código respeta límites de 6-7 requests/segundo

3. **Caché**: Los datos se cachean en `.cache/fmp/` para reducir llamadas

4. **Multicolinealidad**: Esta versión evita combinar Piotroski con métricas crudas

---

## 🐛 Troubleshooting

### Error: "FMP_API_KEY no configurada"
```bash
export FMP_API_KEY="tu_api_key_aqui"
```

### Error: "No symbols in initial universe"
- Aumenta `universe_size`
- Reduce `min_market_cap` o `min_volume`

### Error: "No stocks passed quality filters"
- Reduce `min_piotroski_score`
- Reduce `min_qv_score`
- Aumenta `max_pe` o `max_ev_ebitda`

### Lento o timeout
- Reduce `universe_size`
- Usa caché (archivos en `.cache/`)

---

## 📧 Contacto

Para preguntas o issues: https://github.com/pblo97/PackQVM/issues
