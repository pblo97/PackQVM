#!/usr/bin/env python3
"""
Diagnóstico específico: ¿Por qué PYPL aparece si está bajo MA200?
"""

import sys
sys.path.insert(0, '/home/user/PackQVM')

from datetime import datetime, timedelta
from data_fetcher import fetch_prices
from momentum_calculator import is_above_ma200, calculate_ma200

print("=" * 80)
print("🔍 DIAGNÓSTICO: PYPL y MA200")
print("=" * 80)
print()

# Obtener datos SIN cache
symbol = "PYPL"
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')

print(f"📅 Periodo: {start_date} a {end_date}")
print(f"📊 Symbol: {symbol}")
print()

# Test 1: Con cache
print("=" * 80)
print("TEST 1: CON CACHE (como lo hace el pipeline)")
print("=" * 80)
prices_cached = fetch_prices(symbol, start_date, end_date, use_cache=True)
if prices_cached is not None and len(prices_cached) >= 200:
    current_price_cached = prices_cached['close'].iloc[-1]
    ma200_cached = calculate_ma200(prices_cached)
    above_cached = is_above_ma200(prices_cached)
    last_date_cached = prices_cached['date'].iloc[-1] if 'date' in prices_cached.columns else 'N/A'

    print(f"   Última fecha en datos: {last_date_cached}")
    print(f"   Precio actual:         ${current_price_cached:.2f}")
    print(f"   MA200:                 ${ma200_cached:.2f}")
    print(f"   Diferencia:            ${current_price_cached - ma200_cached:.2f} ({(current_price_cached/ma200_cached - 1)*100:.1f}%)")
    print(f"   ¿Sobre MA200?:         {above_cached}")
else:
    print("   ❌ No hay suficientes datos con cache")
print()

# Test 2: Sin cache
print("=" * 80)
print("TEST 2: SIN CACHE (datos frescos)")
print("=" * 80)
prices_fresh = fetch_prices(symbol, start_date, end_date, use_cache=False)
if prices_fresh is not None and len(prices_fresh) >= 200:
    current_price_fresh = prices_fresh['close'].iloc[-1]
    ma200_fresh = calculate_ma200(prices_fresh)
    above_fresh = is_above_ma200(prices_fresh)
    last_date_fresh = prices_fresh['date'].iloc[-1] if 'date' in prices_fresh.columns else 'N/A'

    print(f"   Última fecha en datos: {last_date_fresh}")
    print(f"   Precio actual:         ${current_price_fresh:.2f}")
    print(f"   MA200:                 ${ma200_fresh:.2f}")
    print(f"   Diferencia:            ${current_price_fresh - ma200_fresh:.2f} ({(current_price_fresh/ma200_fresh - 1)*100:.1f}%)")
    print(f"   ¿Sobre MA200?:         {above_fresh}")
else:
    print("   ❌ No hay suficientes datos sin cache")
print()

# Test 3: Mostrar últimos 10 días
print("=" * 80)
print("TEST 3: ÚLTIMOS 10 DÍAS DE PRECIOS")
print("=" * 80)
if prices_fresh is not None and len(prices_fresh) >= 10:
    last_10 = prices_fresh.tail(10)
    ma200_values = prices_fresh['close'].rolling(200).mean().tail(10)

    print(f"{'Fecha':<12} {'Close':>10} {'MA200':>10} {'vs MA200':>10} {'Status':>10}")
    print("-" * 55)

    for i, (idx, row) in enumerate(last_10.iterrows()):
        date_str = str(row['date'])[:10] if 'date' in row else f"Day {i}"
        close = row['close']
        ma200_val = ma200_values.iloc[i] if not pd.isna(ma200_values.iloc[i]) else 0
        diff = close - ma200_val
        status = "✅ ABOVE" if close > ma200_val else "❌ BELOW"
        print(f"{date_str:<12} ${close:>8.2f} ${ma200_val:>8.2f} {diff:>+9.2f} {status:>10}")
print()

# Análisis
print("=" * 80)
print("📋 ANÁLISIS")
print("=" * 80)
print()

if prices_fresh is not None and len(prices_fresh) >= 200:
    if above_fresh:
        print("✅ PYPL ESTÁ SOBRE MA200")
        print("   El filtro NO debería eliminarlo")
        print("   Si aparece en resultados, es CORRECTO")
    else:
        print("❌ PYPL ESTÁ BAJO MA200")
        print("   El filtro DEBERÍA eliminarlo")
        print()
        print("   Si aparece en resultados, hay un PROBLEMA:")
        print()
        print("   POSIBLES CAUSAS:")
        print("   1. config.require_above_ma200 = False")
        print("      → Verifica que el checkbox esté MARCADO en Streamlit")
        print()
        print("   2. Cache desactualizado")
        print("      → Haz clic en 'Limpiar Caché' y re-ejecuta")
        print()
        print("   3. Bug en el merge de dataframes")
        print("      → El df_merged podría estar incluyendo datos incorrectos")
print()

# Verificar si hay diferencia entre cache y fresh
if prices_cached is not None and prices_fresh is not None:
    if len(prices_cached) >= 200 and len(prices_fresh) >= 200:
        print("=" * 80)
        print("📊 COMPARACIÓN CACHE vs FRESH")
        print("=" * 80)

        if above_cached != above_fresh:
            print("⚠️  INCONSISTENCIA DETECTADA:")
            print(f"   Cache dice: {'ABOVE' if above_cached else 'BELOW'} MA200")
            print(f"   Fresh dice: {'ABOVE' if above_fresh else 'BELOW'} MA200")
            print()
            print("   → SOLUCIÓN: Limpiar caché")
        else:
            print(f"✅ Consistente: Ambos dicen {'ABOVE' if above_fresh else 'BELOW'} MA200")

import pandas as pd
