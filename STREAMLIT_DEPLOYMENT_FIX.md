# Fix: Error de Streamlit - TypeError en QVMConfigV3

## 🔴 Error Reportado

```
TypeError: This app has encountered an error. The original error message is redacted to prevent data leaks.
Traceback:
File "/mount/src/packqvm/app_streamlit_v3.py", line 470, in <module>
    config = QVMConfigV3(...)
```

## 🎯 Causa

El error ocurre porque **Streamlit Cloud está usando una versión desactualizada del código** que no incluye el nuevo parámetro `use_price_cache` en la clase `QVMConfigV3`.

### ¿Qué pasó?

1. Se agregó el parámetro `use_price_cache` a `QVMConfigV3` (línea 179 de `qvm_pipeline_v3.py`)
2. Se agregó el checkbox en la UI de Streamlit (línea 446 de `app_streamlit_v3.py`)
3. El código se pusheó al repositorio
4. **PERO** Streamlit Cloud no se actualizó automáticamente o está usando código cacheado

## ✅ Soluciones

### Opción 1: Reiniciar la App (MÁS RÁPIDO)

1. Ve a tu dashboard de Streamlit Cloud
2. Haz clic en **"Manage app"** (esquina inferior derecha de la app)
3. Haz clic en **"Reboot app"** o **"Restart"**
4. Espera a que la app se reinicie (30-60 segundos)
5. La app debería cargar con el código actualizado

### Opción 2: Redeploy Manual

1. Ve a tu dashboard de Streamlit Cloud
2. Haz clic en **"Manage app"**
3. Haz clic en el menú de opciones (⋮)
4. Selecciona **"Redeploy"** o **"Force redeploy"**
5. Confirma la acción
6. Espera a que el deployment complete

### Opción 3: Verificar Branch (SI LAS ANTERIORES FALLAN)

1. Ve a **"Manage app"** → **"Settings"**
2. Verifica que el **Branch** sea: `claude/stock-portfolio-dashboard-01WnZvpeSLmgLPyD1g7agMME`
3. Si no es el correcto, cámbialo y guarda
4. La app se redeployará automáticamente

### Opción 4: Pull y Actualizar Localmente (ALTERNATIVA)

Si estás corriendo la app localmente:

```bash
# Actualizar código
git pull origin claude/stock-portfolio-dashboard-01WnZvpeSLmgLPyD1g7agMME

# Ejecutar test de configuración
python3 test_config_v3.py

# Si el test pasa, ejecutar la app
streamlit run app_streamlit_v3.py
```

## 🧪 Verificación

Para verificar que el problema está resuelto:

### Localmente:
```bash
python3 test_config_v3.py
```

Deberías ver:
```
✅ TODOS LOS TESTS PASARON
```

### En Streamlit Cloud:
1. Abre la app
2. Ve al sidebar
3. Busca la sección **"💾 Gestión de Datos"**
4. Deberías ver el checkbox **"Usar caché de precios"**
5. Si lo ves, el problema está resuelto ✅

## 📝 Commits Relacionados

- **f471518**: `fix: Solucionar problema de cache desactualizado en filtro MA200`
  - Agrega parámetro `use_price_cache` a `QVMConfigV3`
  - Agrega checkbox en UI de Streamlit
  - Mejora mensajes de diagnóstico

## 🔍 Diagnóstico Técnico

El error específico es:
```python
TypeError: __init__() got an unexpected keyword argument 'use_price_cache'
```

Esto ocurre cuando:
- `app_streamlit_v3.py` (actualizado) intenta pasar `use_price_cache` al constructor
- Pero `qvm_pipeline_v3.py` (versión vieja en Streamlit Cloud) no tiene ese parámetro

Solución: Asegurar que Streamlit Cloud use la versión más reciente del código.

## ⚠️ Prevención Futura

Para evitar este problema en el futuro:

1. **Siempre redeploy después de cambios en dataclasses**
   - Los cambios en `@dataclass` requieren reinicio de la app

2. **Verificar deployment automático**
   - A veces Streamlit Cloud no detecta cambios inmediatamente
   - Espera 1-2 minutos o fuerza el redeploy

3. **Usar versioning**
   - Considera agregar un `__version__` en el código para tracking

## 📞 Si el Problema Persiste

Si después de intentar todas las opciones el error persiste:

1. Verifica que los cambios estén en GitHub:
   ```bash
   git log --oneline -5
   ```
   Deberías ver el commit `f471518`

2. Verifica el branch en Streamlit Cloud settings

3. Intenta un **"Force redeploy"** desde el dashboard

4. Como último recurso, crea una nueva app en Streamlit Cloud apuntando al mismo repositorio

## ✅ Estado Actual

- ✅ Código corregido localmente
- ✅ Tests pasan localmente
- ✅ Commit pusheado al repositorio
- ⏳ Pendiente: Redeploy en Streamlit Cloud

Una vez que hagas el redeploy, la app debería funcionar correctamente con la nueva funcionalidad de gestión de cache.
