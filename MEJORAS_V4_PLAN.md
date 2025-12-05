# 🚀 Plan de Mejoras: QVM Strategy V4

## 📊 Problemas Identificados

1. **❌ Demasiado restrictivo con momentum**
   - Filtro MA200 binario (sí/no) puede rechazar stocks en recuperación temprana
   - Momentum 12M-1M puede perder cambios de tendencia recientes
   - No considera momentum relativo (vs mercado/sector)

2. **❌ No detecta rotación de capital sectorial**
   - No identifica qué sectores están recibiendo flujos
   - No ajusta exposición según ciclo económico
   - Pierde oportunidades en sectores emergentes

3. **❌ Bibliografía técnica desactualizada**
   - Falta incorporar papers 2016-2024
   - Filtros técnicos muy básicos (solo MA200)
   - No usa técnicas modernas de detección de régimen

---

## ✅ Mejoras Propuestas (Priorizado)

### 🔥 **PRIORIDAD ALTA** (Implementar primero)

#### 1. **Momentum Multi-Dimensional Mejorado**

**Problema actual:** Solo usa momentum 12M-1M, muy rígido

**Mejora:**
```python
# Momentum compuesto con múltiples horizontes
# Paper: Novy-Marx (2012) "Intermediate Horizon Returns"
# Paper: Gupta & Kelly (2019) "Factor Momentum Everywhere"

def calculate_multi_horizon_momentum(prices):
    """
    Combina múltiples horizontes temporales con pesos adaptativos

    Papers:
    - Novy-Marx (2012): 6M momentum > 12M momentum
    - Gupta & Kelly (2019): Factor momentum
    - Moskowitz et al. (2012): Time series momentum
    """

    # 1M momentum (muy reciente, detecta cambios de tendencia)
    ret_1m = (prices[-1] / prices[-21]) - 1  # Weight: 0.15

    # 3M momentum (tendencia de corto plazo)
    ret_3m = (prices[-1] / prices[-63]) - 1  # Weight: 0.25

    # 6M momentum (mejor Sharpe según Novy-Marx)
    ret_6m = (prices[-1] / prices[-126]) - 1  # Weight: 0.35

    # 12M momentum (clásico Jegadeesh & Titman)
    ret_12m = (prices[-21] / prices[-252]) - 1  # Weight: 0.25

    # Composite con pesos adaptativos
    composite = 0.15*ret_1m + 0.25*ret_3m + 0.35*ret_6m + 0.25*ret_12m

    return {
        'mom_1m': ret_1m,
        'mom_3m': ret_3m,
        'mom_6m': ret_6m,
        'mom_12m': ret_12m,
        'mom_composite': composite,
    }
```

**Paper clave:**
- Novy-Marx (2012): "The Other Side of Value: The Gross Profitability Premium"
- Gupta & Kelly (2019): "Factor Momentum Everywhere"

#### 2. **Detección de Rotación Sectorial**

**Problema actual:** No detecta qué sectores están en tendencia alcista

**Mejora:**
```python
# Sector Rotation Detection
# Paper: Moskowitz & Grinblatt (1999) "Do Industries Explain Momentum?"
# Paper: Arnott et al. (2019) "Alice in Factorland"

def calculate_sector_momentum(sector_returns, lookback=60):
    """
    Detecta sectores con momentum positivo y rotación de capital

    Returns:
        sector_scores: Dict con score de momentum por sector
        rotation_signal: Sectores ganando/perdiendo momentum
    """

    sector_scores = {}

    for sector, returns in sector_returns.items():
        # Momentum absoluto del sector
        abs_momentum = returns.iloc[-lookback:].mean()

        # Momentum relativo vs mercado
        market_momentum = all_returns.iloc[-lookback:].mean()
        rel_momentum = abs_momentum - market_momentum

        # Aceleración de momentum (2nd derivative)
        recent_mom = returns.iloc[-20:].mean()
        past_mom = returns.iloc[-60:-20].mean()
        acceleration = recent_mom - past_mom

        sector_scores[sector] = {
            'absolute': abs_momentum,
            'relative': rel_momentum,
            'acceleration': acceleration,
            'score': 0.4*abs_momentum + 0.4*rel_momentum + 0.2*acceleration
        }

    return sector_scores


def filter_by_sector_rotation(stocks_df, sector_scores, top_n_sectors=5):
    """
    Filtra stocks solo de sectores con momentum positivo

    Esto permite 'surfear' la rotación de capital
    """
    # Top N sectores con mejor momentum
    top_sectors = sorted(
        sector_scores.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )[:top_n_sectors]

    top_sector_names = [s[0] for s in top_sectors]

    # Filtrar solo stocks de esos sectores
    filtered = stocks_df[stocks_df['sector'].isin(top_sector_names)]

    return filtered, top_sector_names
```

**Papers:**
- Moskowitz & Grinblatt (1999): "Do Industries Explain Momentum?"
- Lewellen (2002): "Momentum and Autocorrelation in Stock Returns"

#### 3. **Filtro MA200 Adaptativo (No Binario)**

**Problema actual:** MA200 es todo-o-nada (muy restrictivo)

**Mejora:**
```python
# Adaptive MA200 Filter con gradiente
# Paper: Zakamulin (2017) "Market Timing with Moving Averages"

def calculate_ma200_score(prices):
    """
    Score continuo en lugar de binario

    Paper: Zakamulin (2017) - Moving average timing strategies
    """
    current_price = prices['close'].iloc[-1]
    ma200 = prices['close'].rolling(200).mean().iloc[-1]

    # % sobre MA200
    pct_above = (current_price - ma200) / ma200

    # Scoring:
    # > +10% sobre MA200: score = 1.0 (fuerte uptrend)
    # 0% to +10%: score = 0.5 to 1.0 (uptrend débil)
    # -5% to 0%: score = 0.0 to 0.5 (near MA200, acceptable)
    # < -5%: score = 0.0 (downtrend, reject)

    if pct_above >= 0.10:
        score = 1.0
    elif pct_above >= 0.0:
        score = 0.5 + (pct_above / 0.10) * 0.5  # Linear 0.5 to 1.0
    elif pct_above >= -0.05:
        score = 0.5 + (pct_above / 0.05) * 0.5  # Linear 0.0 to 0.5
    else:
        score = 0.0  # Reject

    return {
        'ma200_score': score,
        'pct_above_ma200': pct_above,
        'accept': score >= 0.0,  # Menos restrictivo
    }
```

**Paper:** Zakamulin (2017): "Market Timing with Moving Averages"

---

### 🔥 **PRIORIDAD MEDIA**

#### 4. **Detección de Régimen de Mercado**

**Nueva funcionalidad:** Identificar bull/bear markets

```python
# Market Regime Detection
# Paper: Nystrup et al. (2018) "Multi-Period Portfolio Selection"
# Paper: Kritzman et al. (2012) "Regime Shifts"

def detect_market_regime(market_index_prices):
    """
    Detecta régimen de mercado (bull/bear/sideways)

    Papers:
    - Nystrup et al. (2018): Regime switching models
    - Kritzman et al. (2012): Turbulence index
    """

    returns = market_index_prices.pct_change()

    # Volatility Regime
    vol_20d = returns.rolling(20).std()
    vol_60d = returns.rolling(60).std()
    high_vol = vol_20d.iloc[-1] > vol_60d.iloc[-1] * 1.5

    # Trend Regime
    ma50 = market_index_prices.rolling(50).mean()
    ma200 = market_index_prices.rolling(200).mean()
    bullish_trend = ma50.iloc[-1] > ma200.iloc[-1]

    # Momentum Regime
    ret_3m = (market_index_prices.iloc[-1] / market_index_prices.iloc[-63]) - 1
    positive_momentum = ret_3m > 0.05

    # Combine signals
    if bullish_trend and positive_momentum and not high_vol:
        regime = 'BULL'
        risk_appetite = 1.2  # Increase exposure
    elif bullish_trend and positive_momentum and high_vol:
        regime = 'BULL_VOLATILE'
        risk_appetite = 1.0  # Normal exposure
    elif not bullish_trend and not positive_momentum:
        regime = 'BEAR'
        risk_appetite = 0.5  # Reduce exposure
    else:
        regime = 'SIDEWAYS'
        risk_appetite = 0.8

    return {
        'regime': regime,
        'risk_appetite': risk_appetite,
        'high_vol': high_vol,
        'bullish_trend': bullish_trend,
        'positive_momentum': positive_momentum,
    }
```

**Papers:**
- Nystrup et al. (2018): "Multi-Period Portfolio Selection and Bayesian Adaptive Multivariate Regression"
- Kritzman et al. (2012): "Regime Shifts: Implications for Dynamic Strategies"

#### 5. **Trend Strength (ADX-based)**

**Problema:** MA200 no mide FUERZA de tendencia

```python
# Average Directional Index (ADX)
# Classic: Wilder (1978), Modern: Chande (2013)

def calculate_adx(high, low, close, period=14):
    """
    ADX mide fuerza de tendencia (no dirección)

    ADX > 25: Tendencia fuerte
    ADX < 20: Tendencia débil / lateral
    """

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift()
    down_move = low.shift() - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smooth with Wilder's smoothing
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()

    return {
        'adx': adx.iloc[-1],
        'plus_di': plus_di.iloc[-1],
        'minus_di': minus_di.iloc[-1],
        'trend_strength': 'STRONG' if adx.iloc[-1] > 25 else 'WEAK',
        'trend_direction': 'UP' if plus_di.iloc[-1] > minus_di.iloc[-1] else 'DOWN',
    }
```

#### 6. **Risk-Adjusted Momentum (Volatility Scaling)**

**Paper:** Barroso & Santa-Clara (2015) "Momentum Has Its Moments"

```python
def calculate_volatility_scaled_momentum(prices, returns):
    """
    Escala momentum por volatilidad

    Paper: Barroso & Santa-Clara (2015)
    Key insight: Momentum crashes ocurren en alta volatilidad
    """

    # Momentum crudo
    raw_momentum = (prices.iloc[-1] / prices.iloc[-252]) - 1

    # Volatilidad realizada
    realized_vol = returns.iloc[-60:].std() * np.sqrt(252)

    # Target volatility (15% anualizada)
    target_vol = 0.15

    # Scaled momentum
    scaled_momentum = raw_momentum * (target_vol / realized_vol)

    # Momentum risk-adjusted score
    sharpe_style_score = raw_momentum / realized_vol

    return {
        'raw_momentum': raw_momentum,
        'realized_vol': realized_vol,
        'scaled_momentum': scaled_momentum,
        'risk_adjusted_score': sharpe_style_score,
    }
```

---

### 🔥 **PRIORIDAD BAJA** (Opcionales)

#### 7. **Machine Learning para Ranking**

```python
# Gradient Boosting para mejorar ranking
# Paper: Gu et al. (2020) "Empirical Asset Pricing via Machine Learning"

from sklearn.ensemble import GradientBoostingRegressor

def train_ranking_model(historical_data):
    """
    Entrena modelo ML para predecir futura performance

    Features:
    - Momentum multi-horizon
    - Value metrics
    - Quality metrics
    - Sector momentum
    - Technical indicators

    Target: Forward 3-month return
    """

    # Features
    X = historical_data[[
        'mom_1m', 'mom_3m', 'mom_6m', 'mom_12m',
        'piotroski_score', 'roic', 'roe',
        'pe_inv', 'pb_inv', 'fcf_yield',
        'sector_momentum', 'adx', 'rsi',
        'vol_20d', 'vol_60d',
    ]]

    # Target: forward 3M return
    y = historical_data['forward_3m_return']

    # Train model
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
    )
    model.fit(X, y)

    return model
```

**Paper:** Gu et al. (2020): "Empirical Asset Pricing via Machine Learning"

#### 8. **Sentiment Analysis**

```python
# News Sentiment Scoring
# Paper: Tetlock (2007) "Giving Content to Investor Sentiment"

def get_news_sentiment(symbol):
    """
    Analiza sentiment de noticias

    Usa APIs como:
    - NewsAPI
    - Finnhub
    - AlphaVantage News Sentiment
    """
    # Placeholder - requires API integration
    pass
```

---

## 📚 **Bibliografía Actualizada (2016-2024)**

### **Momentum & Technical**
1. **Gupta & Kelly (2019)** - "Factor Momentum Everywhere"
2. **Daniel & Moskowitz (2016)** - "Momentum Crashes"
3. **Barroso & Santa-Clara (2015)** - "Momentum Has Its Moments"
4. **Zakamulin (2017)** - "Market Timing with Moving Averages"
5. **Hurst et al. (2017)** - "Demystifying Managed Futures"

### **Sector Rotation & Multi-Asset**
6. **Moskowitz & Grinblatt (1999)** - "Do Industries Explain Momentum?"
7. **Arnott et al. (2019)** - "Alice in Factorland"
8. **Asness et al. (2013)** - "Value and Momentum Everywhere"

### **Regime Detection**
9. **Nystrup et al. (2018)** - "Multi-Period Portfolio Selection"
10. **Kritzman et al. (2012)** - "Regime Shifts"

### **Machine Learning**
11. **Gu et al. (2020)** - "Empirical Asset Pricing via Machine Learning"
12. **Chen et al. (2019)** - "Deep Learning in Asset Pricing"

### **Risk Management**
13. **Moreira & Muir (2017)** - "Volatility-Managed Portfolios"
14. **Harvey et al. (2016)** - "...and the Cross-Section of Expected Returns"

---

## 🎯 **Plan de Implementación Sugerido**

### **Fase 1 (2-3 días)** - Mejoras de Momentum
- [ ] Implementar multi-horizon momentum
- [ ] Agregar MA200 adaptativo (no binario)
- [ ] Risk-adjusted momentum (volatility scaling)

### **Fase 2 (3-4 días)** - Rotación Sectorial
- [ ] Calcular momentum por sector
- [ ] Implementar filtro de top N sectores
- [ ] Visualizaciones de rotación sectorial

### **Fase 3 (2-3 días)** - Indicadores Técnicos
- [ ] Implementar ADX
- [ ] Implementar RSI
- [ ] Trend strength scoring

### **Fase 4 (3-4 días)** - Detección de Régimen
- [ ] Market regime detection
- [ ] Ajuste dinámico de parámetros por régimen
- [ ] Backtesting por régimen

### **Fase 5 (Opcional)** - ML & Advanced
- [ ] Feature engineering
- [ ] Train/test gradient boosting model
- [ ] Integrate predictions into ranking

---

## 💾 **Archivos a Crear/Modificar**

```
momentum_advanced.py          # Multi-horizon momentum
sector_rotation.py            # Sector momentum & rotation
technical_indicators.py       # ADX, RSI, etc.
regime_detection.py          # Market regime classifier
risk_management.py           # Volatility scaling, position sizing
ml_ranking.py (opcional)     # ML models
```

---

## 📊 **Ejemplo de Nuevos Filtros en UI**

```python
# En app_streamlit_v3.py

st.subheader("🎯 Momentum Avanzado")

momentum_mode = st.selectbox(
    "Tipo de Momentum",
    ["Single (12M-1M)", "Multi-Horizon (1M+3M+6M+12M)", "Risk-Adjusted"],
    help="Multi-Horizon usa Novy-Marx (2012)"
)

ma200_mode = st.selectbox(
    "Filtro MA200",
    ["Binario (ON/OFF)", "Adaptativo (Score)", "Desactivado"],
    help="Adaptativo permite stocks cerca de MA200"
)

st.subheader("🔄 Rotación Sectorial")

enable_sector_rotation = st.checkbox(
    "Detectar Rotación de Capital",
    value=True,
    help="Moskowitz & Grinblatt (1999)"
)

top_n_sectors = st.slider(
    "Top N Sectores con Momentum",
    min_value=3,
    max_value=11,
    value=6,
    help="Solo invertir en sectores con momentum positivo"
)

st.subheader("📊 Detección de Régimen")

enable_regime_detection = st.checkbox(
    "Ajustar por Régimen de Mercado",
    value=True,
    help="Reduce exposición en mercados bajistas"
)
```

---

## 🚀 **Próximos Pasos**

1. **¿Qué mejora quieres implementar primero?**
   - Momentum multi-horizon (menos restrictivo)
   - Rotación sectorial (detecta capital flows)
   - MA200 adaptativo (menos binario)

2. **¿Cuál es tu prioridad?**
   - Capturar más oportunidades (menos filtros)
   - Mejor timing sectorial
   - Reducir drawdowns

3. **¿Nivel técnico deseado?**
   - Básico (solo mejorar momentum)
   - Intermedio (momentum + sector rotation)
   - Avanzado (todo lo anterior + regime detection)

**Dime qué quieres que implemente primero y empezamos!** 🎯
