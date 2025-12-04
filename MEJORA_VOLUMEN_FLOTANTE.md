# 📊 Mejora: Volumen y Flotante - Análisis Académico

## 🔍 Problema Actual

### Lo que tenemos ahora:
```python
min_volume: int = 500_000  # Volumen diario mínimo (básico)
```

### Lo que nos falta:
1. ❌ **Dollar Volume** (precio × volumen)
2. ❌ **Float** (shares disponibles para trading)
3. ❌ **Float Rotation** (volumen / float)
4. ❌ **Volumen Relativo** (vs promedio)
5. ❌ **Institutional Ownership**
6. ❌ **Short Interest**
7. ❌ **Liquidity Score** compuesto

---

## 📚 Literatura Académica Relevante

### **1. Volume & Liquidity**
- **Amihud (2002)**: "Illiquidity and Stock Returns"
  - Stocks ilíquidos tienen mayor expected return (compensation for risk)
  - Pero también mayor execution cost y slippage

- **Brennan & Subrahmanyam (1996)**: "Market Microstructure and Asset Pricing"
  - Trading costs afectan significativamente returns netos

- **Chordia et al. (2001)**: "Market Liquidity and Trading Activity"
  - Volumen anormal predice returns de corto plazo

### **2. Float & Share Structure**
- **D'Avolio (2002)**: "The Market for Borrowing Stock"
  - Low float stocks más caros de shortear
  - Short squeezes más probables

- **Asquith et al. (2005)**: "Short Interest and Stock Returns"
  - High short interest + low float = alta volatilidad

### **3. Institutional Ownership**
- **Gompers & Metrick (2001)**: "Institutional Investors and Equity Prices"
  - Alta institutional ownership → mejor price discovery
  - Pero también crowded trades

---

## ✅ Mejoras Propuestas

### **1. Dollar Volume (Crítico)**

**¿Por qué importa?**
- Un stock a $1 con 10M volumen = $10M dollar volume
- Un stock a $100 con 100K volumen = $10M dollar volume
- **Mismo dollar volume, muy diferente liquidez real**

```python
def calculate_dollar_volume(df_stocks):
    """
    Dollar Volume = Average Daily Dollar Value Traded

    Paper: Amihud (2002) - Illiquidity measure

    Benefits:
    - Mejor proxy de liquidez real que share volume
    - Institucionales usan dollar volume para capacity constraints
    - Evita low-price/high-volume penny stocks
    """

    # Calcular para últimos 20 días
    for symbol in df_stocks['symbol']:
        prices = fetch_prices(symbol)

        # Dollar volume diario
        dollar_volume = prices['close'] * prices['volume']

        # Promedio 20 días
        avg_dollar_volume_20d = dollar_volume.tail(20).mean()

        # Promedio 60 días (más estable)
        avg_dollar_volume_60d = dollar_volume.tail(60).mean()

        df_stocks.loc[symbol, 'dollar_volume_20d'] = avg_dollar_volume_20d
        df_stocks.loc[symbol, 'dollar_volume_60d'] = avg_dollar_volume_60d

    return df_stocks

# Filtro sugerido
MIN_DOLLAR_VOLUME = 10_000_000  # $10M/día mínimo
# Para portfolios grandes: $50M+
```

**Impacto:**
- ✅ Evita penny stocks con volumen inflado
- ✅ Asegura liquidez real para entrar/salir
- ✅ Reduce slippage en ejecución

---

### **2. Float & Float Rotation**

**¿Qué es Float?**
```
Float = Shares Outstanding - Insider Holdings - Institutional Lock-up
      = Shares realmente disponibles para trading
```

**¿Por qué importa?**
- **Low float (< 50M shares)**: Muy volátil, difícil de ejecutar
- **High float (> 500M shares)**: Más estable, mejor liquidez
- **Float rotation**: ¿Cuánto del float se tradea diariamente?

```python
def calculate_float_metrics(df_stocks):
    """
    Float Metrics usando FMP API

    Papers:
    - D'Avolio (2002): Low float stocks más caros de shortear
    - Asquith et al. (2005): Float vs short interest

    Metrics:
    1. Float (shares)
    2. Float Rotation = Daily Volume / Float
    3. Days to Cover = Short Interest / Avg Daily Volume
    """

    for symbol in df_stocks['symbol']:
        # Obtener key stats de FMP
        key_stats = fetch_key_stats(symbol)  # FMP API call

        shares_outstanding = key_stats.get('sharesOutstanding', 0)
        float_shares = key_stats.get('floatShares', shares_outstanding * 0.8)  # Estimate if not available
        short_interest = key_stats.get('shortInterest', 0)

        # Volumen promedio
        avg_volume = df_stocks.loc[symbol, 'avg_volume_20d']

        # Float rotation (¿qué % del float se tradea diariamente?)
        float_rotation = avg_volume / float_shares if float_shares > 0 else 0

        # Days to Cover (para shorts)
        days_to_cover = short_interest / avg_volume if avg_volume > 0 else 0

        df_stocks.loc[symbol, 'float_shares'] = float_shares
        df_stocks.loc[symbol, 'float_rotation_pct'] = float_rotation * 100
        df_stocks.loc[symbol, 'short_interest'] = short_interest
        df_stocks.loc[symbol, 'days_to_cover'] = days_to_cover

        # Clasificación de float
        if float_shares < 50_000_000:
            float_category = 'LOW'  # High volatility
        elif float_shares < 200_000_000:
            float_category = 'MEDIUM'  # Balanced
        else:
            float_category = 'HIGH'  # Institutional friendly

        df_stocks.loc[symbol, 'float_category'] = float_category

    return df_stocks

# Filtros sugeridos
MIN_FLOAT_SHARES = 20_000_000  # Mínimo 20M shares
MAX_FLOAT_SHARES = 2_000_000_000  # Máximo 2B shares (evita mega-caps demasiado lentos)

# Float rotation ideal: 0.5% - 5%
# < 0.5%: Demasiado ilíquido
# > 5%: Demasiado especulativo
MIN_FLOAT_ROTATION = 0.005  # 0.5%
MAX_FLOAT_ROTATION = 0.05   # 5%
```

---

### **3. Amihud Illiquidity Ratio**

**Paper:** Amihud (2002) - Medida académica estándar de iliquidez

```python
def calculate_amihud_illiquidity(prices):
    """
    Amihud Illiquidity Ratio (2002)

    ILLIQ = Average( |Return| / Dollar Volume )

    Interpretación:
    - Alto ILLIQ = Ilíquido (price impact alto por cada $1 tradeado)
    - Bajo ILLIQ = Líquido (price impact bajo)

    Paper: Amihud (2002) "Illiquidity and Stock Returns"
    """

    returns = prices['close'].pct_change().abs()
    dollar_volume = prices['close'] * prices['volume']

    # Amihud ratio diario
    illiq_daily = returns / (dollar_volume / 1_000_000)  # Per $1M

    # Promedio últimos 20 días
    illiq_ratio = illiq_daily.tail(20).mean()

    return {
        'amihud_illiquidity': illiq_ratio,
        'liquidity_score': 1 / (1 + illiq_ratio),  # Normalize to 0-1
    }

# Filtro: Rechazar stocks muy ilíquidos
MAX_AMIHUD_ILLIQ = 0.001  # Ajustar según necesidad
```

**Impacto:**
- ✅ Medida académicamente validada
- ✅ Captura price impact real
- ✅ Usado por institucionales

---

### **4. Volume Surge Detection (Momentum Técnico)**

**Papers:**
- Lee & Swaminathan (2000): "Price Momentum and Trading Volume"
- Chordia et al. (2001): "Trading Activity and Expected Stock Returns"

```python
def detect_volume_surge(prices):
    """
    Detecta anomalías de volumen (smart money entry?)

    Papers:
    - Lee & Swaminathan (2000): Volume predicts momentum strength
    - Chordia et al. (2001): Volume shocks & returns

    Signals:
    1. Volume > 2× promedio = Surge (institutional activity?)
    2. Price + Volume = Bullish (accumulation)
    3. Price - Volume = Bearish (distribution)
    """

    # Volumen promedio (20 días)
    avg_volume = prices['volume'].rolling(20).mean()

    # Volumen actual
    current_volume = prices['volume'].iloc[-1]

    # Ratio
    volume_ratio = current_volume / avg_volume.iloc[-1]

    # Clasificación
    if volume_ratio > 2.0:
        volume_signal = 'SURGE'  # Institucionales entrando?
    elif volume_ratio > 1.5:
        volume_signal = 'ELEVATED'  # Interés creciente
    elif volume_ratio > 0.75:
        volume_signal = 'NORMAL'
    else:
        volume_signal = 'LOW'  # Red flag: pérdida de interés

    # Combine con precio para detección de accumulation/distribution
    price_change = prices['close'].pct_change().iloc[-1]

    if volume_signal in ['SURGE', 'ELEVATED']:
        if price_change > 0.02:  # +2%
            pattern = 'ACCUMULATION'  # Bullish
        elif price_change < -0.02:
            pattern = 'DISTRIBUTION'  # Bearish
        else:
            pattern = 'NEUTRAL'
    else:
        pattern = 'LOW_INTEREST'

    return {
        'volume_ratio': volume_ratio,
        'volume_signal': volume_signal,
        'volume_pattern': pattern,
    }
```

**Impacto:**
- ✅ Detecta smart money entry
- ✅ Confirma breakouts con volumen
- ✅ Evita low-interest stocks

---

### **5. Institutional Ownership**

**Papers:**
- Gompers & Metrick (2001): "Institutional Investors and Equity Prices"
- Sias (1996): "Institutional Ownership and Stock Returns"

```python
def get_institutional_ownership(symbol):
    """
    Institutional Ownership Metrics

    Papers:
    - Gompers & Metrick (2001): Institutions prefer liquid stocks
    - Sias (1996): Institutional herding

    Sweet spot:
    - 30-70% institutional ownership
    - < 30%: Under-researched, poor liquidity
    - > 70%: Crowded, prone to herding
    """

    # FMP API call
    ownership_data = fetch_institutional_ownership(symbol)

    inst_ownership_pct = ownership_data.get('institutionalOwnership', 0)
    num_institutions = ownership_data.get('numberOfInstitutions', 0)

    # Clasificación
    if inst_ownership_pct < 30:
        inst_category = 'LOW'  # Under-owned, potential opportunity
    elif inst_ownership_pct < 70:
        inst_category = 'OPTIMAL'  # Sweet spot
    else:
        inst_category = 'HIGH'  # Crowded, high correlation risk

    return {
        'institutional_ownership_pct': inst_ownership_pct,
        'num_institutions': num_institutions,
        'institutional_category': inst_category,
    }

# Filtro sugerido (opcional)
MIN_INSTITUTIONAL_OWNERSHIP = 20.0  # Mínimo 20%
MAX_INSTITUTIONAL_OWNERSHIP = 80.0  # Máximo 80%
```

---

### **6. Composite Liquidity Score**

**Combina todos los factores en un score único**

```python
def calculate_composite_liquidity_score(stock_data):
    """
    Liquidity Score compuesto (0-100)

    Components:
    1. Dollar Volume (40%)
    2. Amihud Illiquidity (30%)
    3. Float Rotation (20%)
    4. Volume Consistency (10%)
    """

    # 1. Dollar Volume Score (0-100)
    # Normalize: $10M = 0, $1B = 100
    dv_score = min(100, (stock_data['dollar_volume_20d'] / 1_000_000_000) * 100)

    # 2. Amihud Score (invert: lower is better)
    # Normalize: 0.001 = 100, 0.01 = 0
    amihud = stock_data['amihud_illiquidity']
    amihud_score = max(0, 100 - (amihud / 0.0001) * 100)

    # 3. Float Rotation Score
    # Optimal: 1-3%
    float_rot = stock_data['float_rotation_pct']
    if 1.0 <= float_rot <= 3.0:
        float_score = 100
    elif 0.5 <= float_rot < 1.0 or 3.0 < float_rot <= 5.0:
        float_score = 60
    else:
        float_score = 20

    # 4. Volume Consistency Score
    # CV (Coefficient of Variation) of volume last 20 days
    vol_std = stock_data['volume_std_20d']
    vol_mean = stock_data['volume_mean_20d']
    cv = vol_std / vol_mean if vol_mean > 0 else 999
    consistency_score = max(0, 100 - cv * 100)

    # Composite
    liquidity_score = (
        0.40 * dv_score +
        0.30 * amihud_score +
        0.20 * float_score +
        0.10 * consistency_score
    )

    return {
        'liquidity_score': liquidity_score,
        'liquidity_grade': 'A' if liquidity_score >= 80 else
                          'B' if liquidity_score >= 60 else
                          'C' if liquidity_score >= 40 else 'F',
    }

# Filtro final
MIN_LIQUIDITY_SCORE = 50  # Mínimo score B
```

---

## 🎯 Implementación en el Pipeline

### **Nuevos parámetros en QVMConfigV3**

```python
@dataclass
class QVMConfigV3:
    # ... existing parameters ...

    # ========== VOLUMEN & LIQUIDEZ (NUEVO) ==========
    # Dollar Volume
    min_dollar_volume: float = 10_000_000  # $10M/día

    # Float
    min_float_shares: int = 20_000_000      # 20M shares
    max_float_shares: int = 2_000_000_000   # 2B shares
    min_float_rotation: float = 0.005       # 0.5%
    max_float_rotation: float = 0.05        # 5%

    # Amihud Illiquidity
    max_amihud_illiquidity: float = 0.001

    # Volume Surge
    require_volume_surge: bool = False       # Opcional
    min_volume_ratio: float = 1.5           # 1.5× promedio

    # Institutional Ownership (opcional)
    min_institutional_ownership: float = 20.0  # %
    max_institutional_ownership: float = 80.0  # %

    # Composite Liquidity Score
    use_liquidity_score: bool = True
    min_liquidity_score: float = 50.0       # Score B
```

### **Nuevo PASO 6.5 en Pipeline**

```python
# PASO 6.5: ANÁLISIS DE VOLUMEN Y LIQUIDEZ
step6_5 = PipelineStep("PASO 6.5", "Volumen, Float y Liquidez")

for symbol in df_with_prices['symbol']:
    prices = prices_dict[symbol]

    # 1. Dollar Volume
    dollar_volume = calculate_dollar_volume(prices)

    # 2. Float & Rotation
    float_metrics = calculate_float_metrics(symbol)

    # 3. Amihud Illiquidity
    illiquidity = calculate_amihud_illiquidity(prices)

    # 4. Volume Surge
    volume_surge = detect_volume_surge(prices)

    # 5. Institutional Ownership
    inst_ownership = get_institutional_ownership(symbol)

    # 6. Composite Score
    liquidity_score = calculate_composite_liquidity_score({
        **dollar_volume,
        **float_metrics,
        **illiquidity,
        **volume_surge,
        **inst_ownership,
    })

    # Agregar al DataFrame
    df_with_prices.loc[symbol, 'dollar_volume'] = dollar_volume['avg_dollar_volume_20d']
    df_with_prices.loc[symbol, 'float_shares'] = float_metrics['float_shares']
    df_with_prices.loc[symbol, 'liquidity_score'] = liquidity_score['liquidity_score']
    # etc.

# Filtrar por liquidity score
df_filtered = df_with_prices[
    df_with_prices['liquidity_score'] >= config.min_liquidity_score
]
```

---

## 📊 UI en Streamlit

```python
st.subheader("💧 Volumen y Liquidez (NUEVO)")

st.info("Basado en Amihud (2002), Lee & Swaminathan (2000), D'Avolio (2002)")

min_dollar_volume = st.number_input(
    "Dollar Volume Mínimo ($M/día)",
    min_value=1,
    max_value=500,
    value=10,
    help="Amihud (2002): Dollar volume mejor proxy que share volume"
) * 1_000_000

col1, col2 = st.columns(2)

with col1:
    min_float_shares = st.number_input(
        "Float Mínimo (M shares)",
        min_value=10,
        max_value=1000,
        value=20,
        help="D'Avolio (2002): Low float stocks más volátiles"
    ) * 1_000_000

with col2:
    max_float_shares = st.number_input(
        "Float Máximo (M shares)",
        min_value=100,
        max_value=5000,
        value=2000,
        help="Evita mega-caps muy lentos"
    ) * 1_000_000

use_liquidity_score = st.checkbox(
    "✅ Usar Liquidity Score Compuesto",
    value=True,
    help="Combina Dollar Volume, Amihud Illiquidity, Float Rotation"
)

min_liquidity_score = st.slider(
    "Liquidity Score Mínimo",
    min_value=0,
    max_value=100,
    value=50,
    step=10,
    help="A: 80+, B: 60-79, C: 40-59, F: <40"
)
```

---

## 📈 Beneficios Esperados

### **1. Mejor Ejecución**
- ✅ Menor slippage (stocks más líquidos)
- ✅ Menor market impact al entrar/salir
- ✅ Mejor fill prices

### **2. Menos Crowding**
- ✅ Evita micro-caps demasiado ilíquidos
- ✅ Evita mega-caps demasiado lentos
- ✅ Sweet spot de liquidez

### **3. Detección de Smart Money**
- ✅ Volume surges indican institutional interest
- ✅ Accumulation patterns (price + volume)
- ✅ Evita distribution patterns

### **4. Risk Management**
- ✅ Liquidity score → mejor gestión de riesgo
- ✅ Float rotation → volatility prediction
- ✅ Institutional ownership → crowding risk

---

## 🎯 Prioridad de Implementación

### **ALTA** (Implementar YA)
1. ✅ **Dollar Volume** - Crítico para liquidez real
2. ✅ **Amihud Illiquidity** - Estándar académico
3. ✅ **Composite Liquidity Score** - Métrica unificada

### **MEDIA** (Implementar después)
4. ✅ **Float Metrics** - Útil pero requiere API adicional
5. ✅ **Volume Surge Detection** - Nice to have

### **BAJA** (Opcional)
6. ⚪ **Institutional Ownership** - Interesante pero no crítico

---

## 📚 Referencias Completas

1. **Amihud, Y. (2002)** - "Illiquidity and Stock Returns: Cross-Section and Time-Series Effects"
2. **Lee, C. M. C., & Swaminathan, B. (2000)** - "Price Momentum and Trading Volume"
3. **Chordia, T., Subrahmanyam, A., & Anshuman, V. R. (2001)** - "Trading Activity and Expected Stock Returns"
4. **D'Avolio, G. (2002)** - "The Market for Borrowing Stock"
5. **Asquith, P., Pathak, P. A., & Ritter, J. R. (2005)** - "Short Interest, Institutional Ownership, and Stock Returns"
6. **Gompers, P. A., & Metrick, A. (2001)** - "Institutional Investors and Equity Prices"
7. **Brennan, M. J., & Subrahmanyam, A. (1996)** - "Market Microstructure and Asset Pricing"
8. **Sias, R. W. (1996)** - "Volatility and the Institutional Investor"

---

## ✅ Resumen Ejecutivo

**Problema:** Sistema actual solo usa volumen simple (shares), ignora liquidez real

**Solución:** Implementar métricas académicas de liquidez:
- Dollar Volume (Amihud 2002)
- Illiquidity Ratio (Amihud 2002)
- Float & Rotation (D'Avolio 2002)
- Volume Patterns (Lee & Swaminathan 2000)

**Impacto:**
- ✅ Mejor ejecución (menos slippage)
- ✅ Stocks más "tradeables"
- ✅ Detección de smart money
- ✅ Risk management mejorado

**Tiempo:** 2-3 días de implementación

**Prioridad:** ALTA - Fundamental para ejecución real
