#!/usr/bin/env python3
"""
Test de la lógica de breakout para entender por qué rechaza todo
"""

import pandas as pd
from datetime import datetime, timedelta

# Simular datos de un stock
dates = pd.date_range('2024-01-01', periods=300, freq='D')
prices_trending_up = [100 + i*0.3 for i in range(300)]  # Trending up

df = pd.DataFrame({
    'date': dates,
    'close': prices_trending_up,
    'volume': [1000000] * 300
})

# Simular la lógica actual
current_price = df['close'].iloc[-1]
prev_price = df['close'].iloc[-2]

# 52w high (excluyendo día actual)
high_52w_prev = df['close'].iloc[-252:-1].max()

print("=" * 80)
print("🔍 TEST DE LÓGICA DE BREAKOUT")
print("=" * 80)
print()
print(f"Precio actual (día -1):     ${current_price:.2f}")
print(f"Precio anterior (día -2):   ${prev_price:.2f}")
print(f"Máximo 52w (excluyendo hoy): ${high_52w_prev:.2f}")
print()

# Lógica ACTUAL (muy estricta)
breakout_strict = current_price > high_52w_prev and prev_price <= high_52w_prev
print("LÓGICA ACTUAL (estricta):")
print(f"  current_price ({current_price:.2f}) > high_52w ({high_52w_prev:.2f})? {current_price > high_52w_prev}")
print(f"  prev_price ({prev_price:.2f}) <= high_52w ({high_52w_prev:.2f})? {prev_price <= high_52w_prev}")
print(f"  Breakout detectado? {breakout_strict}")
print()

if not breakout_strict:
    print("❌ NO SE DETECTA BREAKOUT")
    print()
    print("🔍 PROBLEMA:")
    print("   La lógica requiere que el breakout ocurra EXACTAMENTE HOY")
    print("   Si el stock rompió hace 2-3 días, NO se detecta")
    print()
    print("   En este caso:")
    if current_price <= high_52w_prev:
        print(f"   - El precio actual NO supera el máximo 52w")
    else:
        print(f"   - El precio actual SÍ supera el máximo 52w")
        print(f"   - PERO el precio de ayer TAMBIÉN superaba el máximo")
        print(f"   - Entonces el breakout fue ANTES de ayer, no hoy")
        print(f"   - La lógica actual NO lo detecta")

print()
print("=" * 80)
print("💡 SOLUCIÓN PROPUESTA")
print("=" * 80)
print()

# Lógica RELAJADA (últimos N días)
lookback_days = 5  # Detectar breakouts de los últimos 5 días

# Verificar si en los últimos N días hubo un breakout
recent_prices = df['close'].iloc[-lookback_days:]
older_high = df['close'].iloc[-252:-lookback_days].max()

breakout_relaxed = current_price > older_high

print(f"LÓGICA RELAJADA (últimos {lookback_days} días):")
print(f"  Máximo ANTES de los últimos {lookback_days} días: ${older_high:.2f}")
print(f"  Precio actual: ${current_price:.2f}")
print(f"  ¿Precio actual > máximo anterior? {breakout_relaxed}")
print()

if breakout_relaxed:
    # Encontrar cuándo fue el breakout
    for i in range(len(recent_prices)):
        if recent_prices.iloc[i] > older_high:
            days_ago = len(recent_prices) - i - 1
            print(f"   ✅ Breakout detectado hace {days_ago} día(s)")
            print(f"      Precio del breakout: ${recent_prices.iloc[i]:.2f}")
            break

print()
print("=" * 80)
print("📊 COMPARACIÓN")
print("=" * 80)
print()
print(f"Lógica ACTUAL (estricta):  {breakout_strict} ❌")
print(f"Lógica RELAJADA (5 días):  {breakout_relaxed} ✅")
print()
print("La lógica relajada detecta breakouts recientes (últimos 5 días)")
print("en lugar de solo el día exacto, lo cual es más útil.")
