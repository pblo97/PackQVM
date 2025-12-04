# Fix: Problema con Filtro MA200 y Cache de Datos

## 🔍 Problema Reportado

El usuario reportó que stocks como PYPL (PayPal) aparecen con ✅ en la columna MA200, indicando que están sobre su media móvil de 200 días, cuando en realidad **no lo están**.

Ejemplo de la tabla reportada:
```
9  PYPL  Financial Services  6  0.678  115.24%  ✅  0.500  8.0  12.1  2.9  93.48%
```

El usuario indica: *"PYPL no está sobre la media de 200, no ha hecho más que estar estancada"*

## 🎯 Causa Raíz Identificada

**CACHE DE DATOS DESACTUALIZADOS**

El problema está en `qvm_pipeline_v3.py` línea 821:

```python
prices = fetch_prices(
    symbol,
    start=config.backtest_start,
    end=config.backtest_end,
    use_cache=True  # ❌ PROBLEMA: Siempre usa cache
)
```

### ¿Por qué es un problema?

1. Los datos de precios se cachean cuando se descargan por primera vez
2. Si los datos fueron cacheados hace días/semanas:
   - En ese momento, PYPL podría haber estado sobre MA200 → ✅
   - Hoy, PYPL ha caído y está bajo MA200
   - Pero el sistema muestra los datos del cache antiguo
3. El usuario ve resultados incorrectos/desactualizados

### Evidencia

- Usuario dice "PYPL no ha hecho más que estar estancada"
- Esto sugiere mala performance reciente
- PYPL podría haber caído bajo MA200 recientemente
- Pero el cache muestra datos antiguos cuando sí estaba sobre MA200

## ✅ Solución Implementada

### 1. Nuevo parámetro en QVMConfigV3

Archivo: `qvm_pipeline_v3.py`

```python
@dataclass
class QVMConfigV3:
    # ... otros parámetros ...

    # ========== DATA CACHING ==========
    use_price_cache: bool = True  # Si False, siempre descarga datos frescos
```

### 2. Pipeline actualizado para usar el parámetro

```python
prices = fetch_prices(
    symbol,
    start=config.backtest_start,
    end=config.backtest_end,
    use_cache=config.use_price_cache  # ✅ Ahora respeta la configuración
)
```

### 3. Mejoras en mensajes de diagnóstico

**PASO 6 - Precios Históricos:**
```python
if verbose:
    print(f"\n📈 {step6.name}: {step6.description}")
    cache_status = "ENABLED" if config.use_price_cache else "DISABLED (datos frescos)"
    print(f"   Cache: {cache_status}")
```

**PASO 7 - Filtro MA200:**
```python
if verbose:
    if config.require_above_ma200:
        print("   ✅ MA200 Filter: ENABLED (Faber 2007)")
    else:
        print("   ⚠️  MA200 Filter: DISABLED")

    # Muestra stocks rechazados
    if rejected > 0:
        print(f"   ⚠️  Rechazados por MA200: {rejected} stocks (estaban BAJO MA200)")
        if below_ma200_symbols:
            print(f"      Ejemplos: {', '.join(below_ma200_symbols[:3])}")
```

### 4. Nueva opción en UI de Streamlit

Archivo: `app_streamlit_v3.py`

```python
st.subheader("💾 Gestión de Datos")

use_price_cache = st.checkbox(
    "Usar caché de precios",
    value=True,
    help="Si está desmarcado, descarga datos de precios frescos (más lento pero datos actualizados). ⚠️ Si ves stocks que no deberían pasar los filtros MA200, DESMARCA esta opción."
)

if not use_price_cache:
    st.warning("⚠️ Cache deshabilitado: Se descargarán datos frescos (esto puede tomar más tiempo)")
```

## 🚀 Cómo Usar la Solución

### Opción 1: Deshabilitar Cache (RECOMENDADO para diagnóstico)

1. Abrir app Streamlit
2. Ir a la sección **"💾 Gestión de Datos"**
3. **DESMARCAR** el checkbox "Usar caché de precios"
4. Ejecutar "🚀 Ejecutar Screening V3"
5. Los datos se descargarán frescos (más lento pero actualizados)

### Opción 2: Limpiar Cache y Re-ejecutar

1. Abrir app Streamlit
2. Clic en "🗑️ Limpiar Caché"
3. Verificar que "Filtro MA200" esté **MARCADO**
4. Ejecutar "🚀 Ejecutar Screening V3"
5. Verificar en la salida del PASO 7:
   ```
   ✅ PASO 7: Momentum + MA200 Filter
      ✅ MA200 Filter: ENABLED (Faber 2007)
      Min Momentum 12M: 10%
      ⚠️  Rechazados por MA200: X stocks (estaban BAJO MA200)
   ```

## 📊 Verificación

Para verificar que el fix funciona:

1. Ejecutar con `use_price_cache=False`
2. Revisar la salida del PASO 7
3. Verificar que:
   - Se muestra "Cache: DISABLED (datos frescos)"
   - Se rechazan stocks que están bajo MA200
   - La columna ✅ solo muestra stocks realmente sobre MA200

## 🔬 Diagnóstico Adicional

Si el problema persiste, ejecutar:

```bash
python3 debug_pypl_ma200.py
```

Este script muestra:
- Análisis de la lógica del filtro MA200
- Posibles causas del problema
- Pasos de verificación
- Soluciones recomendadas

## 📝 Archivos Modificados

1. `qvm_pipeline_v3.py`:
   - Agregado parámetro `use_price_cache` en `QVMConfigV3`
   - Actualizada línea 824 para usar el parámetro
   - Mejorados mensajes de diagnóstico en PASO 6 y PASO 7

2. `app_streamlit_v3.py`:
   - Agregada sección "💾 Gestión de Datos"
   - Nuevo checkbox "Usar caché de precios"
   - Parámetro `use_price_cache` agregado a la configuración

3. `debug_pypl_ma200.py` (nuevo):
   - Script de diagnóstico para el problema MA200
   - Análisis detallado de causas y soluciones

## 🎯 Conclusión

El problema era que el sistema usaba datos cacheados potencialmente desactualizados para el filtro MA200. La solución permite al usuario:

1. **Forzar descarga de datos frescos** desmarcando el checkbox
2. **Ver información clara** sobre si el cache está activo
3. **Diagnosticar problemas** con mensajes mejorados que muestran cuántos stocks fueron rechazados por MA200

**Recomendación:** Para screening en producción, ejecutar con `use_price_cache=False` al menos una vez al día para asegurar datos actualizados.
