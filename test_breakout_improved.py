#!/usr/bin/env python3
"""
Test de la lógica mejorada de breakout
"""

import sys
sys.path.insert(0, '/home/user/PackQVM')

import pandas as pd
from datetime import datetime, timedelta
from qvm_pipeline_v3 import detect_breakouts, detect_volume_confirmed_breakouts

# Crear datos de prueba
print("=" * 80)
print("🧪 TEST DE LÓGICA MEJORADA DE BREAKOUTS")
print("=" * 80)
print()

# Escenario 1: Stock que rompió hace 3 días
dates = pd.date_range('2024-01-01', periods=300, freq='D')

# Precio estable hasta día 295, luego breakout
prices1 = [100.0] * 295 + [105.0, 106.0, 107.0, 108.0, 109.0]

df1 = pd.DataFrame({
    'date': dates,
    'close': prices1,
    'volume': [1000000] * 295 + [2500000, 2000000, 1500000, 1200000, 1100000]
})

# Escenario 2: Stock en tendencia pero sin breakout
prices2 = [100 + i*0.1 for i in range(300)]

df2 = pd.DataFrame({
    'date': dates,
    'close': prices2,
    'volume': [1000000] * 300
})

# Escenario 3: Stock que rompió hace 10 días (fuera de ventana)
prices3 = [100.0] * 285 + [105.0] * 15

df3 = pd.DataFrame({
    'date': dates,
    'close': prices3,
    'volume': [1000000] * 285 + [3000000] + [1000000] * 14
})

prices_dict = {
    'STOCK_A': df1,
    'STOCK_B': df2,
    'STOCK_C': df3,
}

# Test con lookback de 5 días (default)
print("📊 TEST 1: Lookback = 5 días (default)")
print("-" * 80)
breakouts_5d = detect_breakouts(prices_dict, lookback_days=5)
print(breakouts_5d[['symbol', 'breakout_52w', 'breakout_3m', 'breakout_20d', 'any_breakout']])
print()

# Test con lookback de 15 días (más permisivo)
print("📊 TEST 2: Lookback = 15 días (más permisivo)")
print("-" * 80)
breakouts_15d = detect_breakouts(prices_dict, lookback_days=15)
print(breakouts_15d[['symbol', 'breakout_52w', 'breakout_3m', 'breakout_20d', 'any_breakout']])
print()

# Test con volumen confirmado
print("📊 TEST 3: Breakouts confirmados con volumen")
print("-" * 80)
confirmed = detect_volume_confirmed_breakouts(prices_dict, lookback_days=5)
print(confirmed[['symbol', 'any_breakout', 'breakout_confirmed', 'breakout_strong', 'relative_volume']])
print()

print("=" * 80)
print("✅ CONCLUSIONES")
print("=" * 80)
print()
print("1. STOCK_A (rompió hace 3 días con volumen):")
print("   - Con lookback=5:  Debería detectar breakout ✓")
print("   - Debería ser 'confirmado' por volumen ✓")
print()
print("2. STOCK_B (tendencia gradual, sin breakout):")
print("   - No debería detectar breakout")
print()
print("3. STOCK_C (rompió hace 10 días):")
print("   - Con lookback=5:  NO debería detectar")
print("   - Con lookback=15: SÍ debería detectar ✓")
print()

# Verificar resultados
print("=" * 80)
print("🔍 VERIFICACIÓN")
print("=" * 80)
print()

stock_a_5d = breakouts_5d[breakouts_5d['symbol'] == 'STOCK_A']['any_breakout'].iloc[0]
stock_c_5d = breakouts_5d[breakouts_5d['symbol'] == 'STOCK_C']['any_breakout'].iloc[0]
stock_c_15d = breakouts_15d[breakouts_15d['symbol'] == 'STOCK_C']['any_breakout'].iloc[0]

if stock_a_5d:
    print("✅ STOCK_A detectado con lookback=5")
else:
    print("❌ STOCK_A NO detectado con lookback=5 (PROBLEMA)")

if not stock_c_5d:
    print("✅ STOCK_C NO detectado con lookback=5 (correcto, fuera de ventana)")
else:
    print("⚠️  STOCK_C detectado con lookback=5 (breakout antiguo)")

if stock_c_15d:
    print("✅ STOCK_C detectado con lookback=15 (correcto, dentro de ventana)")
else:
    print("❌ STOCK_C NO detectado con lookback=15 (PROBLEMA)")
