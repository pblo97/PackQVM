#!/usr/bin/env python3
"""
Verificación específica de PYPL, CPRT, DIS y MA200
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

print("="*80)
print("🔍 VERIFICACIÓN: PYPL, CPRT, DIS vs MA200")
print("="*80)
print()

# Stocks reportados por el usuario
stocks_to_check = {
    'PYPL': 'PayPal',
    'CPRT': 'Copart',
    'DIS': 'Disney',
}

print(f"📅 Fecha de verificación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

results = []

for symbol, name in stocks_to_check.items():
    print(f"📊 Analizando {symbol} ({name})...")

    try:
        # Descargar datos usando yfinance (no requiere API key)
        ticker = yf.Ticker(symbol)

        # Obtener datos de 2 años para calcular MA200
        end_date = datetime.now()
        start_date = end_date - timedelta(days=500)

        hist = ticker.history(start=start_date, end=end_date)

        if hist.empty or len(hist) < 200:
            print(f"   ❌ Datos insuficientes ({len(hist)} días)\n")
            continue

        # Calcular MA200
        hist['MA200'] = hist['Close'].rolling(window=200).mean()

        # Datos más recientes
        current_price = hist['Close'].iloc[-1]
        ma200 = hist['MA200'].iloc[-1]

        # Verificar si está sobre o bajo MA200
        above_ma200 = current_price > ma200
        pct_diff = ((current_price - ma200) / ma200) * 100

        # Información adicional
        high_52w = hist['Close'].tail(252).max()
        low_52w = hist['Close'].tail(252).min()
        pct_from_high = ((current_price - high_52w) / high_52w) * 100

        # Tendencia reciente (últimos 20 días)
        ret_20d = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-20]) - 1) * 100
        ret_60d = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-60]) - 1) * 100

        results.append({
            'symbol': symbol,
            'name': name,
            'current_price': current_price,
            'ma200': ma200,
            'above_ma200': above_ma200,
            'pct_diff': pct_diff,
            'high_52w': high_52w,
            'pct_from_high': pct_from_high,
            'ret_20d': ret_20d,
            'ret_60d': ret_60d,
        })

        # Mostrar resultado
        status = "✅ SOBRE" if above_ma200 else "❌ BAJO"
        print(f"   {status} MA200")
        print(f"   Precio actual: ${current_price:.2f}")
        print(f"   MA200: ${ma200:.2f}")
        print(f"   Diferencia: {pct_diff:+.2f}%")
        print(f"   52w High: ${high_52w:.2f} (actual es {pct_from_high:+.1f}%)")
        print(f"   Retorno 20D: {ret_20d:+.2f}%")
        print(f"   Retorno 60D: {ret_60d:+.2f}%")
        print()

    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        continue

# Resumen
print("="*80)
print("📊 RESUMEN")
print("="*80)
print()

if not results:
    print("❌ No se pudieron obtener datos para ningún stock")
    sys.exit(1)

below_ma200 = [r for r in results if not r['above_ma200']]
above_ma200 = [r for r in results if r['above_ma200']]

print(f"✅ Sobre MA200: {len(above_ma200)}/{len(results)}")
if above_ma200:
    for r in above_ma200:
        print(f"   - {r['symbol']}: ${r['current_price']:.2f} vs MA200 ${r['ma200']:.2f} ({r['pct_diff']:+.1f}%)")

print()
print(f"❌ Bajo MA200: {len(below_ma200)}/{len(results)}")
if below_ma200:
    for r in below_ma200:
        print(f"   - {r['symbol']}: ${r['current_price']:.2f} vs MA200 ${r['ma200']:.2f} ({r['pct_diff']:+.1f}%)")

print()
print("="*80)
print("🎯 CONCLUSIÓN")
print("="*80)
print()

if below_ma200:
    print(f"⚠️  CONFIRMADO: {len(below_ma200)} stock(s) están BAJO su MA200")
    print()
    print("Esto significa que el filtro MA200 NO se aplicó correctamente.")
    print()
    print("POSIBLES CAUSAS:")
    print()
    print("1️⃣  FILTRO MA200 DESACTIVADO")
    print("   ✓ En Streamlit, verifica que el checkbox 'Filtro MA200' esté MARCADO")
    print()
    print("2️⃣  CACHE DE PRECIOS ACTIVO")
    print("   ✓ En Streamlit, ve a '💾 Gestión de Datos'")
    print("   ✓ DESMARCA el checkbox 'Usar caché de precios'")
    print("   ✓ Re-ejecuta el screening")
    print()
    print("3️⃣  STREAMLIT CLOUD NO ACTUALIZADO")
    print("   ✓ Ve a 'Manage app' → 'Reboot app'")
    print("   ✓ Espera 30-60 segundos")
    print("   ✓ Re-ejecuta el screening")
    print()
    print("ACCIÓN INMEDIATA:")
    print("1. Abre la app de Streamlit")
    print("2. En el sidebar, busca '💾 Gestión de Datos'")
    print("3. DESMARCA 'Usar caché de precios'")
    print("4. Verifica que 'Filtro MA200' esté MARCADO")
    print("5. Haz clic en '🗑️ Limpiar Caché'")
    print("6. Ejecuta '🚀 Ejecutar Screening V3'")
else:
    print("✅ Todos los stocks verificados están SOBRE su MA200")
    print()
    print("Esto sugiere que:")
    print("- Los precios han cambiado desde tu última ejecución")
    print("- O estás viendo resultados de una sesión anterior")
    print()
    print("RECOMENDACIÓN:")
    print("- Re-ejecuta el screener con cache deshabilitado")
    print("- Verifica que los datos sean de HOY")

print()
print("="*80)
