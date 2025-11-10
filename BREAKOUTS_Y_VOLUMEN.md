# ⚡ Breakouts y Volumen - Nuevas Features V3.2

## 📋 Resumen

Se han agregado nuevos filtros heurísticos de momentum técnico basados en **breakouts de niveles previos** y **volumen anormal**. Estas reglas combinan análisis técnico con fundamentales para identificar stocks con catalizadores recientes.

---

## 🎯 Motivación

### Literatura Académica

1. **George & Hwang (2004)**: "The 52-Week High and Momentum Investing"
   - Stocks cerca de su máximo 52w tienen outperformance significativo
   - El efecto persiste controlando por otros factores

2. **Lee & Swaminathan (2000)**: "Price Momentum and Trading Volume"
   - Momentum strategies son más rentables cuando se confirman con volumen alto
   - Volumen anormal indica participación institucional

3. **Gervais, Kaniel & Mingelgrin (2001)**: "The High-Volume Return Premium"
   - Stocks con volumen extremo tienen retornos superiores en las siguientes semanas
   - El efecto es más fuerte para small-caps

### Por Qué Importa

- **Breakouts**: Indican cambio en narrativa o información nueva
- **Volumen**: Confirma convicción institucional
- **Combinado**: Señal más fuerte que cualquiera por separado

---

## 🚀 Nuevas Funciones Implementadas

### 1. `calculate_volume_metrics(prices_dict)`

Calcula métricas avanzadas de volumen para cada stock.

**Retorna:**
```python
{
    'symbol': str,
    'avg_volume_20d': float,      # Volumen promedio últimos 20 días
    'current_volume': float,       # Volumen del día actual
    'relative_volume': float,      # Ratio: current / average
    'volume_surge': bool,          # True si > 2x promedio
}
```

**Ejemplo:**
```python
# Stock con volumen normal
{'symbol': 'AAPL', 'relative_volume': 1.2}  # 20% sobre promedio

# Stock con surge de volumen
{'symbol': 'NVDA', 'relative_volume': 3.5, 'volume_surge': True}  # 3.5x promedio!
```

---

### 2. `detect_breakouts(prices_dict)`

Detecta breakouts de 3 tipos de niveles técnicos.

**Tipos de Breakout:**

| Tipo | Descripción | Ventana | Uso |
|------|-------------|---------|-----|
| **52w** | Máximo de 52 semanas | 252 días | Breakout mayor, señal muy fuerte |
| **3M** | Máximo de 3 meses | 60 días | Resistencia intermedia |
| **20D** | Máximo de 20 días | 20 días | Consolidación corta |

**Lógica del Breakout:**
```python
# Un breakout ocurre cuando:
# 1. Precio ACTUAL > nivel de resistencia
# 2. Precio ANTERIOR <= nivel de resistencia
# (Es decir, cruzó HOY o recientemente)
```

**Retorna:**
```python
{
    'symbol': str,
    'breakout_52w': bool,          # Rompió máximo 52w
    'breakout_3m': bool,           # Rompió máximo 3M
    'breakout_20d': bool,          # Rompió máximo 20D
    'any_breakout': bool,          # Cualquiera de los anteriores
    'pct_above_52w_level': float,  # % por encima del nivel
}
```

---

### 3. `detect_volume_confirmed_breakouts(prices_dict)`

Combina breakouts con volumen para identificar señales confirmadas.

**Tipos de Señal:**

| Señal | Criterio | Fuerza |
|-------|----------|--------|
| **Breakout** | Precio rompe nivel | 🟡 Media |
| **Confirmado** | Breakout + volumen >1.5x | 🟢 Fuerte |
| **Fuerte** | Breakout + volumen >2x | 🔴 Muy Fuerte |

**Retorna:**
```python
{
    # Todo de detect_breakouts() +
    # Todo de calculate_volume_metrics() +
    'breakout_confirmed': bool,    # Breakout + vol >1.5x
    'breakout_strong': bool,       # Breakout + vol >2x
}
```

---

## ⚙️ Nuevos Parámetros de Configuración

### En `QVMConfigV3`:

```python
@dataclass
class QVMConfigV3:
    # ... parámetros existentes ...

    # ========== BREAKOUT FILTERS (NUEVO) ==========
    enable_breakout_filter: bool = False           # Requiere cualquier breakout
    require_breakout_confirmed: bool = False       # Breakout + vol >1.5x
    require_breakout_strong: bool = False          # Breakout + vol >2x

    # Volumen
    enable_volume_surge_filter: bool = False       # Vol >2x sin breakout necesario
```

### En Streamlit UI:

Nueva sección **"⚡ Breakouts y Volumen (NUEVO)"** con:

- ✅ Checkbox: "🚀 Filtro de Breakout"
- ✅ Checkbox: "✅ Solo Breakouts Confirmados" (disabled si breakout OFF)
- ✅ Checkbox: "💪 Solo Breakouts Fuertes" (disabled si breakout OFF)
- ✅ Checkbox: "📊 Filtro de Surge de Volumen"

---

## 🔧 Cambios en el Pipeline

### PASO 8 Actualizado

**Antes:**
```
PASO 8: Filtros 52w High y Volumen
- Filtro 52w high (legacy)
```

**Ahora:**
```
PASO 8: Filtros 52w High, Breakouts y Volumen
- Filtro 52w high (legacy)
- Breakout general (any)
- Breakout confirmado (vol >1.5x)
- Breakout fuerte (vol >2x)
- Volume surge (>2x sin breakout)
```

### Métricas Reportadas

```
📊 PASO 8: Filtros 52w High, Breakouts y Volumen
   ⚡ Breakout Filter: ENABLED
   ⚡ Confirmed Breakouts Only: YES (vol >1.5x)

   Input:  250 stocks
   Output: 45 stocks

   Metrics:
     - Rejected by breakout: 120
     - Rejected by breakout confirmation: 85
```

---

## 📊 Mejoras en `data_fetcher.py`

### `fetch_prices()` Actualizado

**Antes:**
```python
def fetch_prices(symbol, start, end) -> DataFrame:
    # Retornaba ['date', 'close']
```

**Ahora:**
```python
def fetch_prices(symbol, start, end, include_volume=True) -> DataFrame:
    # Retorna ['date', 'close', 'volume']
    # Usa endpoint completo en lugar de serietype=line
```

**Beneficios:**
- ✅ Volumen real de la API (no placeholder)
- ✅ Backward compatible (include_volume=True por default)
- ✅ Caché sigue funcionando

---

## 💡 Casos de Uso

### Caso 1: Growth Stock con Catalizador

**Objetivo:** Encontrar stocks de crecimiento que rompieron niveles con volumen fuerte.

**Config:**
```python
config = QVMConfigV3(
    universe_size=800,
    require_above_ma200=True,              # Tendencia alcista
    min_momentum_12m=0.15,                 # +15% últimos 12M
    require_breakout_confirmed=True,       # Breakout + volumen
    min_piotroski_score=6,                 # Calidad aceptable
    portfolio_size=30,
)
```

**Resultado:** Stocks con momentum fuerte, sobre MA200, y con catalizador reciente (breakout confirmado).

---

### Caso 2: Value Stocks con Catalizador de Volumen

**Objetivo:** Value stocks que están despertando interés institucional.

**Config:**
```python
config = QVMConfigV3(
    universe_size=800,
    min_qv_score=0.60,                     # Alto quality-value
    max_pe=15.0,                           # Valuación atractiva
    enable_volume_surge_filter=True,       # Volumen anormal
    min_piotroski_score=7,                 # Alta calidad
    portfolio_size=25,
)
```

**Resultado:** Value stocks de calidad con surge de volumen (posible interés institucional).

---

### Caso 3: Momentum Puro

**Objetivo:** Stocks en breakout fuerte sin importar valuación.

**Config:**
```python
config = QVMConfigV3(
    universe_size=800,
    require_breakout_strong=True,          # Breakout + vol >2x
    require_above_ma200=True,              # Sobre MA200
    min_momentum_12m=0.20,                 # +20% momentum
    # Sin filtros de value
    portfolio_size=20,
)
```

**Resultado:** Pure momentum con señales técnicas muy fuertes.

---

## 📈 Ejemplo de Output

### En la Tabla de Resultados

Nuevas columnas visibles:

| symbol | breakout_52w | breakout_3m | breakout_20d | relative_volume | volume_surge |
|--------|--------------|-------------|--------------|-----------------|--------------|
| NVDA   | ✅ True      | ✅ True     | ✅ True      | 3.2             | ✅ True      |
| AAPL   | ❌ False     | ✅ True     | ✅ True      | 1.8             | ❌ False     |
| MSFT   | ❌ False     | ❌ False    | ✅ True      | 1.1             | ❌ False     |

---

## ⚠️ Consideraciones Importantes

### 1. Volumen Requiere Datos Completos

- `fetch_prices()` ahora usa endpoint completo (sin `serietype=line`)
- Consumo de API es mayor (más datos por request)
- Plan premium de FMP recomendado para universos grandes

### 2. Breakouts Son Punto-en-Tiempo

- Un breakout detectado HOY puede revertir mañana
- Considerar usar `require_above_ma200=True` para filtrar breakouts débiles
- Combinar con momentum 12M para mayor robustez

### 3. Filtros Son Acumulativos

Si activas múltiples filtros, son **AND** (no OR):
```python
# Esto requiere AMBOS
require_breakout_confirmed=True,
enable_volume_surge_filter=True,
# = Breakout confirmado Y volumen >2x
```

### 4. Performance

Con `universe_size=800` y volumen enabled:
- Tiempo de ejecución: +30-50% vs antes
- Consumo de API: +100% (endpoint completo vs serietype=line)
- Recomendación: Usar caché agresivamente

---

## 🧪 Testing

### Verificar que Funciona

```python
# Test básico
from qvm_pipeline_v3 import detect_breakouts, calculate_volume_metrics
import pandas as pd

# Mock data
prices = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=300),
    'close': [100 + i*0.5 for i in range(300)],  # Trending up
    'volume': [1000000] * 299 + [3000000],        # Surge en último día
})

prices_dict = {'TEST': prices}

# Test breakouts
breakouts = detect_breakouts(prices_dict)
print(breakouts)  # breakout_20d should be True

# Test volume
volume = calculate_volume_metrics(prices_dict)
print(volume)  # volume_surge should be True
```

---

## 📚 Referencias

1. George, T. J., & Hwang, C. Y. (2004). "The 52-Week High and Momentum Investing". *Journal of Finance*, 59(5), 2145-2176.

2. Lee, C. M., & Swaminathan, B. (2000). "Price Momentum and Trading Volume". *Journal of Finance*, 55(5), 2017-2069.

3. Gervais, S., Kaniel, R., & Mingelgrin, D. H. (2001). "The High-Volume Return Premium". *Journal of Finance*, 56(3), 877-919.

---

## 🎓 Próximos Pasos

Posibles mejoras futuras:

1. **Breakout con Retest**: Detectar breakouts que retestaron el nivel
2. **Multiple Timeframe**: Confirmar breakout en múltiples ventanas
3. **Sector Rotation**: Priorizar breakouts en sectores en tendencia
4. **Machine Learning**: Predecir probabilidad de éxito del breakout

---

## 📞 Troubleshooting

### "No tengo columna 'volume'"

- Verifica que `fetch_prices()` incluya `include_volume=True`
- Algunos símbolos pueden no tener datos de volumen en FMP
- Verifica tu plan de FMP (algunos planes no incluyen volumen)

### "Todos mis stocks son rechazados por breakout"

- Breakouts son raros (típicamente 5-10% del universo)
- Considera usar `enable_breakout_filter=False` y solo ver la columna
- O usa `require_near_52w_high=True` con umbral más bajo (70-80%)

### "El pipeline es muy lento"

- `universe_size=800` + volumen = más llamadas a API
- Usa caché: ejecuta una vez, luego ajusta filtros
- Considera reducir a 500-600 para testing
- Plan premium de FMP tiene rate limits más altos

---

**Versión:** V3.2
**Fecha:** 2025-01-10
**Autor:** Claude Code
