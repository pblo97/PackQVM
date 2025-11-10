# 🔍 DEBUG: Filtro MA200 - Guía de Resolución

## ❓ Problema Reportado
Varios stocks aparecen en los resultados pero están bajo su MA200:
- PYPL, CPRT, DECK, IP, y otros

## ✅ Código Verificado
He revisado el código completo y **TODO es correcto**:
1. ✅ La función `is_above_ma200()` calcula correctamente
2. ✅ El filtro se aplica en el pipeline (línea 704-708 de qvm_pipeline_v3.py)
3. ✅ La configuración se pasa correctamente desde Streamlit

## 🎯 Causas Más Probables

### 1. 🔴 Filtro MA200 Desactivado en Tu Ejecución
**Probabilidad: ALTA**

**Cómo ocurre:**
- En la interfaz Streamlit, el checkbox "✅ Filtro MA200" estaba DESMARCADO
- Aunque el default es `True`, si lo desmarcaste manualmente, el filtro NO se aplica

**Cómo verificarlo:**
```
Busca en la salida del screener la sección:
"🚀 PASO 7: Momentum + MA200 Filter"

Debe decir:
   MA200 Filter: ENABLED (Faber 2007)
   Min Momentum 12M: 10%

Si NO dice "ENABLED", el filtro estaba desactivado.
```

**Solución:**
1. Abre la app Streamlit
2. Verifica que el checkbox "✅ Filtro MA200 (Faber 2007)" esté MARCADO
3. Ejecuta el screener nuevamente

---

### 2. 📊 Datos Cacheados/Desactualizados
**Probabilidad: MEDIA**

**Cómo ocurre:**
- La API cachea precios por 1 hora
- Un stock calculado como "above MA200" hace 1 hora puede estar "below" ahora
- Los precios cambian constantemente durante el mercado

**Cómo verificarlo:**
```bash
# Verifica precios actuales vs cacheados
python3 validate_ma200.py PYPL CPRT DECK IP
```

**Solución:**
1. En la app Streamlit, haz clic en "🗑️ Limpiar Caché"
2. Espera unos segundos
3. Ejecuta el screener nuevamente con datos frescos

---

### 3. 🗂️ Resultados de Sesión Anterior
**Probabilidad: BAJA**

**Cómo ocurre:**
- Estás viendo resultados guardados/exportados de una ejecución anterior
- Esa ejecución fue con filtro desactivado o con precios diferentes

**Solución:**
- Ejecuta el screener AHORA y verifica resultados en tiempo real

---

## 📋 Checklist de Verificación

Sigue estos pasos EN ORDEN:

### ✅ PASO 1: Verificar Estado Actual del Filtro
```
1. Abre app_streamlit_v3.py en el navegador
2. Ve a la sección "🚀 Filtros Avanzados (NUEVO)"
3. Confirma que "✅ Filtro MA200 (Faber 2007)" está MARCADO
4. Si NO está marcado, márcalo
```

### ✅ PASO 2: Limpiar Caché
```
1. Haz clic en el botón "🗑️ Limpiar Caché"
2. Verás el mensaje "Caché limpiado!"
3. Espera 3-5 segundos
```

### ✅ PASO 3: Ejecutar Screener con Verbose
```
1. Asegúrate de tener filtro MA200 marcado
2. Haz clic en "🚀 Ejecutar Screening V3"
3. BUSCA en la salida esta sección:

   🚀 PASO 7: Momentum + MA200 Filter
      MA200 Filter: ENABLED (Faber 2007)    <-- DEBE DECIR ESTO
      Min Momentum 12M: 10%

4. Si NO dice "ENABLED", algo está mal
```

### ✅ PASO 4: Verificar Resultados
```
1. Los resultados finales DEBEN mostrar columna "above_ma200"
2. TODOS los valores deben ser True (✅)
3. Si hay False (❌), el filtro NO se aplicó
```

---

## 🔧 Si el Problema Persiste

Si después de seguir TODOS los pasos anteriores sigues viendo stocks bajo MA200:

### Opción A: Debugging Manual
```python
# Crea este archivo: test_filter.py
from qvm_pipeline_v3 import run_qvm_pipeline_v3, QVMConfigV3

config = QVMConfigV3(
    require_above_ma200=True,  # FORZAR activación
    min_momentum_12m=0.10,
    portfolio_size=30,
    universe_size=500
)

results, stats = run_qvm_pipeline_v3(config=config, verbose=True)

# Verifica que PASO 7 muestre:
# "MA200 Filter: ENABLED"
# "Rejected by MA200: X" (donde X > 0)

# Verifica resultados
print(results[['symbol', 'above_ma200']].head(20))
# TODOS deben ser True
```

### Opción B: Validación Manual de Stocks
```bash
# Instala dependencias
pip install pandas requests

# Verifica stocks específicos
python3 validate_ma200.py PYPL CPRT DECK IP DFS DAL KR MO
```

---

## 📊 Interpretación de Resultados

### ✅ Resultado Esperado (Filtro Funcionando)
```
🚀 PASO 7: Momentum + MA200 Filter
   MA200 Filter: ENABLED (Faber 2007)
   Min Momentum 12M: 10%

   Input:  350 stocks
   Output: 180 stocks
   Metrics:
     - Rejected by MA200: 120    <-- Stocks filtrados
     - Rejected by Momentum: 50
     - Avg Momentum 12M: 23.4%
```

### ❌ Resultado Incorrecto (Filtro Desactivado)
```
🚀 PASO 7: Momentum + MA200 Filter
   Min Momentum 12M: 10%
   (NO menciona "MA200 Filter: ENABLED")

   Input:  350 stocks
   Output: 280 stocks
   Metrics:
     - Rejected by Momentum: 70
     - Avg Momentum 12M: 18.2%
   (NO menciona "Rejected by MA200")
```

---

## 💡 Recomendaciones

1. **SIEMPRE verifica** que el checkbox esté marcado antes de ejecutar
2. **LIMPIA el caché** antes de cada ejecución importante
3. **REVISA la salida** de PASO 7 para confirmar "MA200 Filter: ENABLED"
4. **USA** el script `validate_ma200.py` para verificar stocks individuales

---

## 📞 ¿Todavía no funciona?

Si después de todo esto el problema persiste, necesito que me proporciones:

1. Screenshot o copia de la configuración de filtros en Streamlit
2. Salida completa del PASO 7 cuando ejecutas el screener
3. Primeros 10 resultados con sus valores de `above_ma200`

Con esa información podré diagnosticar el problema exacto.
