# 🚀 Deployment en Streamlit Cloud

Guía paso a paso para desplegar PackQVM en Streamlit Cloud.

---

## 📋 Pre-requisitos

1. **Cuenta en Streamlit Cloud**: https://streamlit.io/cloud
2. **API Key de Financial Modeling Prep**: https://financialmodelingprep.com/developer/docs/
3. **Repositorio en GitHub** con el código

---

## 🔧 Pasos de Configuración

### 1. Configurar el Repositorio

Asegúrate de que estos archivos estén en tu repositorio:

```
✅ requirements.txt          # Dependencias
✅ app_streamlit_v2.py       # Aplicación principal
✅ data_fetcher.py           # Datos + Piotroski
✅ quality_value_score.py    # Scores sin multicolinealidad
✅ qvm_pipeline_v2.py        # Pipeline
✅ .streamlit/config.toml    # Configuración de tema
```

### 2. Conectar a Streamlit Cloud

1. Ve a https://share.streamlit.io/
2. Click en **"New app"**
3. Conecta tu repositorio de GitHub
4. Selecciona el branch: `claude/fix-program-functionality-011CUmvuTmZGQbG8YQEEkFgP` (o `main`)

### 3. Configurar la App

En el formulario de configuración:

- **Repository**: `pblo97/PackQVM`
- **Branch**: Tu branch actual
- **Main file path**: `app_streamlit_v2.py`  ⭐ IMPORTANTE
- **App URL**: Elige tu URL personalizada

### 4. Configurar Secrets (API Key)

⚠️ **PASO CRÍTICO** - La app no funcionará sin esto:

1. En Streamlit Cloud, ve a tu app
2. Click en **Settings** (⚙️)
3. Click en **Secrets**
4. Pega este contenido (reemplaza con tu API key real):

```toml
FMP_API_KEY = "tu_api_key_aqui"
```

5. Click **Save**

### 5. Deploy

Click en **"Deploy!"** y espera a que se construya (1-3 minutos).

---

## 🔍 Verificación

Una vez desplegada, verifica que:

- ✅ La app carga sin errores
- ✅ Los sliders funcionan
- ✅ Puedes ejecutar el screening
- ✅ Los datos se descargan correctamente

---

## ⚠️ Troubleshooting

### Error: "No module named 'plotly'"

**Causa**: Falta `requirements.txt` o no está bien configurado

**Solución**:
```bash
# Verifica que requirements.txt existe
cat requirements.txt

# Debe contener:
plotly>=5.18.0
streamlit>=1.28.0
pandas>=2.0.0
```

### Error: "FMP_API_KEY no configurada"

**Causa**: Secrets no configurados correctamente

**Solución**:
1. Ve a Settings > Secrets en Streamlit Cloud
2. Asegúrate de que `FMP_API_KEY` esté configurado
3. Reinicia la app (click en "Reboot")

### Error: "This app has exceeded its resource limits"

**Causa**: Plan gratuito tiene límites

**Soluciones**:
- Reduce `universe_size` (usa 100-150 en vez de 300)
- Reduce `portfolio_size` (usa 20-30 en vez de 50)
- Upgrade a plan de pago de Streamlit Cloud

### Error de Caché / Timeout

**Solución**:
```python
# En app_streamlit_v2.py, reduce los TTL de caché:
@st.cache_data(ttl=1800)  # Cambiar de 3600 a 1800
```

---

## 📊 Optimización para Producción

### 1. Reducir Límites por Defecto

En `qvm_pipeline_v2.py`, ajusta valores predeterminados:

```python
@dataclass
class QVMConfig:
    universe_size: int = 150      # Reducir de 300 a 150
    portfolio_size: int = 25       # Reducir de 30 a 25
```

### 2. Configurar Caché Agresivo

En `app_streamlit_v2.py`:

```python
@st.cache_data(ttl=3600)  # 1 hora de caché
def cached_pipeline(config_dict):
    # ... pipeline code
```

### 3. Limitar Concurrencia

En `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 200
enableCORS = false
```

---

## 🔄 Actualizar la App

Para actualizar después de cambios en el código:

1. **Push cambios** a tu repositorio:
   ```bash
   git add .
   git commit -m "Update: descripción"
   git push
   ```

2. **Streamlit Cloud** detectará los cambios automáticamente
3. La app se **redesplegará** sola (1-2 minutos)

O manualmente:
- Settings > Reboot (reinicio rápido)
- Settings > Clear cache (limpia caché)

---

## 📱 Compartir tu App

Una vez desplegada, tu app tendrá una URL como:

```
https://packqvm.streamlit.app
```

Puedes compartir esta URL directamente. Los usuarios pueden:
- ✅ Ajustar parámetros con sliders
- ✅ Ejecutar screening
- ✅ Descargar resultados en CSV
- ❌ NO necesitan configurar API key (está en secrets)

---

## 💡 Tips Adicionales

### Monitoreo

Streamlit Cloud provee:
- **Logs**: Ver errores en tiempo real
- **Analytics**: Número de visitantes
- **Performance**: Tiempo de carga

### Seguridad

- ✅ API key está en secrets (no en código)
- ✅ `.gitignore` excluye caché y secrets
- ⚠️ No commites `secrets.toml` al repositorio

### Costos

- **Plan Gratuito**: 1 app pública
- **Plan Starter**: $20/mes - 3 apps privadas
- **Plan Teams**: $250/mes - Ilimitadas

---

## 📚 Documentación Oficial

- Streamlit Cloud: https://docs.streamlit.io/streamlit-community-cloud
- Secrets Management: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management
- Resource Limits: https://docs.streamlit.io/streamlit-community-cloud/get-started/limitations-and-known-issues

---

## 🆘 Soporte

Si tienes problemas:

1. Verifica logs en Streamlit Cloud
2. Revisa este troubleshooting
3. Abre issue en GitHub: https://github.com/pblo97/PackQVM/issues

---

**¡Tu app está lista para producción!** 🎉
