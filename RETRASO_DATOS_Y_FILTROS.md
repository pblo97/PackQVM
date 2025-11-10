# 📊 Retraso de Datos y Priorización de Filtros

## ❓ Preguntas del Usuario

1. **¿Con cuánto retraso están los datos?**
2. **¿Se prioriza el momentum sobre el filtro MA200?**
3. **Royal Caribbean cruzó abajo el 3 de noviembre, ¿por qué aparece en resultados?**

---

## 🕐 1. RETRASO DE DATOS

### Tipo de Datos: EOD (End of Day)

La API de FMP (Financial Modeling Prep) utiliza datos **EOD (End of Day)**:

```
Cierre del Mercado (4:00 PM ET)
         ⬇
    Procesamiento
         ⬇
Datos Disponibles (5-6 PM ET mismo día)
         ⬇
    Tu Ejecución
```

### Retrasos Típicos

| Momento de Ejecución | Retraso Esperado |
|---------------------|------------------|
| Durante mercado abierto (9:30 AM - 4:00 PM ET) | Datos del día anterior |
| Después del cierre (5:00 PM+ ET) | Datos del mismo día |
| Fines de semana | Datos del viernes |

### Cache Local: 1 Hora

El código cachea los datos por **3600 segundos (1 hora)**:

```python
# data_fetcher.py línea 660
ttl = 3600 if use_cache else None
```

**Ejemplo:**
- Ejecutas screener a las 10:00 AM → Usa datos de ayer
- Ejecutas de nuevo a las 10:30 AM → Usa datos cacheados de la ejecución anterior
- Haces clic en "Limpiar Caché" → Próxima ejecución obtiene datos frescos

### Caso Real: Royal Caribbean (RCL)

Si RCL cruzó hacia abajo el **3 de noviembre** y hoy es **10 de noviembre**:

```
3 Nov: RCL cruza abajo MA200
       ⬇ (7 días de diferencia)
10 Nov: Tu ejecución del screener
```

**Si RCL aparece en resultados:**
- Los datos usados son de **ANTES del 3 de noviembre**, O
- El filtro MA200 no estaba activado

---

## ⚖️ 2. PRIORIZACIÓN DE FILTROS

### ❌ NO HAY PRIORIZACIÓN

**El momentum NO se prioriza sobre MA200.** Ambos son filtros **eliminatorios** que se aplican en secuencia.

### Flujo de Filtrado (PASO 7)

```python
# qvm_pipeline_v3.py líneas 703-714

# 1️⃣ FILTRO MA200 (si está activado)
if config.require_above_ma200:
    df_merged = df_merged[df_merged['above_ma200'] == True]
    # ❌ Stocks bajo MA200 son ELIMINADOS

# 2️⃣ FILTRO MOMENTUM (siempre se aplica)
df_merged = df_merged[df_merged['momentum_12m'] >= config.min_momentum_12m]
    # ❌ Stocks con bajo momentum son ELIMINADOS
```

### Ejemplo Práctico

Supongamos 3 stocks:

| Stock | Above MA200 | Momentum 12M | Pasa MA200? | Pasa Momentum? | En Resultados? |
|-------|-------------|--------------|-------------|----------------|----------------|
| AAPL  | ✅ True     | 25%          | ✅ SÍ       | ✅ SÍ (≥10%)   | ✅ SÍ          |
| RCL   | ❌ False    | 30%          | ❌ NO       | -              | ❌ NO          |
| TSLA  | ✅ True     | 5%           | ✅ SÍ       | ❌ NO (<10%)   | ❌ NO          |

**Observaciones:**
- RCL tiene excelente momentum (30%) pero está bajo MA200 → **RECHAZADO**
- TSLA está sobre MA200 pero tiene bajo momentum → **RECHAZADO**
- AAPL pasa ambos filtros → **ACEPTADO**

**Conclusión:** El momentum NO compensa estar bajo MA200. Ambos filtros deben pasar.

---

## 🚢 3. CASO RCL: ¿Por qué aparece en resultados?

### Análisis del Problema

**Fecha del cruce:** 3 de noviembre
**Fecha de hoy:** 10 de noviembre (7 días después)

Si RCL aparece en tus resultados, hay **3 escenarios posibles:**

### Escenario 1: Filtro MA200 Desactivado ❌
**Probabilidad: 🔴 ALTA**

```
Checkbox "Filtro MA200" → DESMARCADO
              ⬇
    Filtro NO se aplica
              ⬇
    RCL pasa sin verificar MA200
              ⬇
    Solo se verifica momentum
              ⬇
    RCL aparece en resultados
```

**Cómo verificar:**
Busca en la salida del PASO 7:
```
🚀 PASO 7: Momentum + MA200 Filter
   MA200 Filter: ENABLED (Faber 2007)    <-- ¿Dice esto?
   ...
   Rejected by MA200: X                   <-- ¿Aparece esto?
```

Si NO dice "ENABLED" o no muestra "Rejected by MA200", **el filtro estaba OFF**.

---

### Escenario 2: Datos Desactualizados/Cacheados 📊
**Probabilidad: 🟡 MEDIA**

```
Tu ejecución: 10 Nov, 10:00 AM
              ⬇
    Datos EOD más recientes: 9 Nov cierre
              ⬇
    Cache local: Datos de ejecución anterior
              ⬇
    Datos usados: Podrían ser del 1-2 Nov
              ⬇
    En esos datos: RCL todavía sobre MA200
```

**Cómo verificar:**
1. Haz clic en "🗑️ Limpiar Caché"
2. Re-ejecuta el screener
3. Si RCL desaparece → Eran datos cacheados

---

### Escenario 3: Timing del Mercado 🕐
**Probabilidad: 🟢 BAJA**

El "cruce" del 3 de noviembre fue:
- **Intraday:** Cayó temporalmente pero cerró sobre MA200
- **EOD:** El precio de cierre siguió sobre MA200

Entonces técnicamente RCL no cruzó hasta días después.

---

## ✅ SOLUCIÓN DEFINITIVA

### Pasos a Seguir (EN ORDEN):

#### 1️⃣ Verificar Configuración Actual
```
□ Abre app Streamlit
□ Busca sección "🚀 Filtros Avanzados"
□ Verifica que "✅ Filtro MA200 (Faber 2007)" esté MARCADO
```

#### 2️⃣ Limpiar Caché
```
□ Haz clic en "🗑️ Limpiar Caché"
□ Espera mensaje "Caché limpiado!"
□ Espera 3-5 segundos
```

#### 3️⃣ Re-Ejecutar Screener
```
□ Haz clic en "🚀 Ejecutar Screening V3"
□ OBSERVA la salida del PASO 7
□ CONFIRMA que diga "MA200 Filter: ENABLED"
□ CONFIRMA que muestre "Rejected by MA200: X" (X > 0)
```

#### 4️⃣ Verificar Resultados
```
□ RCL NO debe aparecer en la tabla final
□ Si aparece, el filtro sigue desactivado
□ O los datos de FMP tienen >7 días de retraso (raro)
```

---

## 📈 VERIFICACIÓN MANUAL DE RCL

Si quieres verificar el estado actual de RCL:

```bash
# Ejecuta este script
python3 check_rcl_ma200.py
```

Este script te mostrará:
- ✅ Precio actual de RCL
- ✅ Valor de MA200
- ✅ Si está sobre o bajo MA200
- ✅ Cuántos días de retraso tienen los datos
- ✅ Últimos 10 días de precios vs MA200

---

## 🎯 RESUMEN EJECUTIVO

### Retraso de Datos
- **EOD típico:** 0-1 día de retraso
- **Cache local:** 1 hora
- **Solución:** "Limpiar Caché" antes de ejecutar

### Priorización de Filtros
- **NO hay priorización**
- **Ambos son eliminatorios:**
  1. Filtro MA200 (si activado)
  2. Filtro Momentum (siempre)
- **No importa el momentum:** Si está bajo MA200 → RECHAZADO

### Caso RCL
- **Más probable:** Filtro MA200 desactivado
- **También posible:** Datos cacheados de antes del 3 Nov
- **Solución:** Activar filtro + Limpiar caché + Re-ejecutar

---

## 🔧 DEBUGGING ADICIONAL

Si después de todo esto RCL SIGUE apareciendo:

### Opción A: Validar Configuración en Código

```python
# Crea test_config.py
from qvm_pipeline_v3 import QVMConfigV3

config = QVMConfigV3()
print(f"require_above_ma200 = {config.require_above_ma200}")
# Debe imprimir: require_above_ma200 = True
```

### Opción B: Ver Logs Detallados

En la salida del screener, busca:
```
🚀 PASO 7: Momentum + MA200 Filter
   MA200 Filter: ENABLED (Faber 2007)
   Min Momentum 12M: 10%

   Input:  450 stocks
   Output: 180 stocks

   Metrics:
     - Rejected by MA200: 150    <-- ¿Cuántos rechazó?
     - Rejected by Momentum: 120
```

Si "Rejected by MA200" = 0 → El filtro NO se aplicó.

---

## 📞 ¿Necesitas Más Ayuda?

Si después de seguir TODOS estos pasos:
1. El filtro MA200 está marcado ✓
2. Limpiaste el caché
3. La salida dice "MA200 Filter: ENABLED"
4. "Rejected by MA200" > 0

**Y RCL SIGUE apareciendo:**

Entonces proporciona:
- Screenshot de la configuración de filtros
- Salida completa del PASO 7
- Primeros 20 resultados con columnas: symbol, above_ma200, momentum_12m

Con eso podré diagnosticar un posible bug.
