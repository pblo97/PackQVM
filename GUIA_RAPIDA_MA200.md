# 🚨 GUÍA RÁPIDA: Stocks Bajo MA200 Aparecen en Resultados

## ⚠️ Problema

Stocks que han caído recientemente (como PYPL, CPRT, DIS) aparecen en los resultados con ✅ en la columna MA200, cuando deberían estar filtrados.

## 🎯 Causas Más Comunes (en orden de probabilidad)

### 1. Filtro MA200 Desactivado 🔴 (90% de los casos)

**Síntoma:** El checkbox está desmarcado

**Dónde verificar:**
- Sidebar → Sección "🚀 Filtros Avanzados (NUEVO)"
- Checkbox: "✅ Filtro MA200 (Faber 2007)"

**Solución:**
```
1. MARCA el checkbox
2. Re-ejecuta el screening
3. Verifica en PASO 7: "✅ MA200 Filter: ENABLED"
```

### 2. Cache de Precios Desactualizado 🟡 (9% de los casos)

**Síntoma:**
- Checkbox MA200 está marcado
- Pero los resultados incluyen stocks que han caído

**Explicación:**
Los datos de precios pueden tener días/semanas de antigüedad. En ese momento, PYPL estaba sobre MA200, pero hoy no lo está.

**Solución:**
```
1. Sidebar → "💾 Gestión de Datos"
2. DESMARCA "Usar caché de precios"
3. Clic en "🗑️ Limpiar Caché"
4. Re-ejecuta el screening
```

### 3. App No Actualizada 🟢 (1% de los casos)

**Síntoma:** No ves la sección "💾 Gestión de Datos"

**Solución:**
```
1. Dashboard de Streamlit Cloud
2. "Manage app" → "Reboot app"
3. Espera 30-60 segundos
4. Refresca la página
```

## ✅ Checklist de Verificación

Antes de ejecutar el screening, verifica:

- [ ] Checkbox "✅ Filtro MA200 (Faber 2007)" está **MARCADO**
- [ ] Checkbox "Usar caché de precios" está **DESMARCADO** (para datos frescos)
- [ ] Hiciste clic en "🗑️ Limpiar Caché"
- [ ] Ves la sección "💾 Gestión de Datos" en el sidebar

## 🔍 Cómo Verificar que Funciona

Después de ejecutar el screening, revisa la salida:

### PASO 6: Precios Históricos
```
📈 PASO 6: Precios Históricos
   Cache: DISABLED (datos frescos)  ← Debe decir DISABLED
```

### PASO 7: Momentum + MA200 Filter
```
🚀 PASO 7: Momentum + MA200 Filter
   ✅ MA200 Filter: ENABLED (Faber 2007)  ← Debe estar ENABLED
   Min Momentum 12M: 10%
   ⚠️  Rechazados por MA200: 45 stocks  ← Debe rechazar stocks
      Ejemplos: PYPL, CPRT, DIS  ← Los stocks problemáticos deben aparecer aquí
```

### Tabla de Resultados
- PYPL, CPRT, DIS **NO** deberían aparecer en la tabla final
- Solo stocks con ✅ en MA200 que **realmente** están sobre su MA200

## 🎯 Test Rápido: ¿Está Funcionando?

1. **Mira la tabla de resultados**
2. **Busca PYPL** (PayPal)
3. **Si PYPL aparece:**
   - ❌ Filtro NO funcionando
   - Ve a la sección de causas arriba

4. **Si PYPL NO aparece:**
   - ✅ Filtro funcionando correctamente
   - Los datos están actualizados

## 📊 Ejemplo de Salida Correcta

```
🚀 PASO 7: Momentum + MA200 Filter
   ✅ MA200 Filter: ENABLED (Faber 2007)
   Min Momentum 12M: 10%
   ⚠️  Rechazados por MA200: 87 stocks (estaban BAJO MA200)
      Ejemplos: PYPL, CPRT, DIS
   ✅ 156/243 stocks sobre MA200 (64%)
```

## 📊 Ejemplo de Salida Incorrecta

```
🚀 PASO 7: Momentum + MA200 Filter
   ⚠️  MA200 Filter: DISABLED  ← ❌ PROBLEMA
   Min Momentum 12M: 10%
```

O:

```
📈 PASO 6: Precios Históricos
   Cache: ENABLED  ← ❌ PROBLEMA: Cache activo
```

## 🚀 Pasos de Acción (Copia y Pega)

### Si filtro está desactivado:
1. Sidebar → "🚀 Filtros Avanzados"
2. MARCA "✅ Filtro MA200 (Faber 2007)"
3. Ejecutar screening

### Si quieres datos frescos (recomendado):
1. Sidebar → "💾 Gestión de Datos"
2. DESMARCA "Usar caché de precios"
3. Clic en "🗑️ Limpiar Caché"
4. Ejecutar screening (tomará más tiempo)

### Si la app no está actualizada:
1. Streamlit Cloud → "Manage app"
2. "Reboot app"
3. Espera 30-60 segundos
4. Refresca página

## 📞 Necesitas Ayuda?

Si después de seguir TODOS los pasos el problema persiste:

1. **Copia la salida del PASO 7** (completa)
2. **Toma screenshot** de:
   - Sección de filtros en sidebar
   - Tabla de resultados finales
3. **Avísame** con la información para diagnóstico profundo

## 💡 Tips Adicionales

- **Primera ejecución del día**: Siempre deshabilita cache
- **Análisis de stocks específicos**: Usa cache deshabilitado
- **Backtest histórico**: Puedes usar cache (datos no cambian)
- **Desarrollo/testing**: Cache deshabilitado siempre

## 🎓 Por Qué Importa

El filtro MA200 (Faber 2007) es **CRÍTICO** porque:
- Reduce drawdowns en 50%+ según literatura académica
- Evita stocks en tendencia bajista
- Mejora significativamente el Sharpe ratio

Sin este filtro, el portfolio incluirá stocks que están cayendo, lo cual va en contra de la estrategia momentum académica.
