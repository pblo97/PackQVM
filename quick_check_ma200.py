#!/usr/bin/env python3
"""
Quick Check: Verifica si el filtro MA200 está realmente funcionando
==================================================================

Este script hace una verificación rápida de un subset de stocks
para determinar si están realmente sobre su MA200.

Uso: python3 quick_check_ma200.py
"""

import sys
import os

# Agregar el directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pandas as pd
    from data_fetcher import fetch_prices
    from momentum_calculator import is_above_ma200, calculate_ma200
    from datetime import datetime, timedelta
except ImportError as e:
    print(f"❌ Error: Falta instalar dependencias")
    print(f"   {e}")
    print("\n💡 Solución:")
    print("   pip install pandas requests")
    sys.exit(1)


def quick_check():
    """
    Verificación rápida de los stocks mencionados por el usuario.
    """
    print("=" * 80)
    print("🔍 QUICK CHECK: Verificación MA200")
    print("=" * 80)
    print()

    # Stocks mencionados por el usuario como problemáticos
    stocks_to_check = [
        'PYPL',   # Usuario dice que está bajo MA200
        'CPRT',   # Usuario dice que está bajo MA200
        'DECK',   # Usuario dice que está bajo MA200
        'IP',     # Usuario dice que está bajo MA200
    ]

    # También algunos que deberían estar sobre MA200
    stocks_control = [
        'DFS',    # En resultados del usuario
        'DAL',    # En resultados del usuario
        'KR',     # En resultados del usuario
        'MO',     # En resultados del usuario
    ]

    all_stocks = stocks_to_check + stocks_control

    print(f"📊 Verificando {len(all_stocks)} stocks...")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Verificar si tenemos API key
    if not os.environ.get('FMP_API_KEY'):
        print("⚠️  WARNING: FMP_API_KEY no está configurada")
        print("   No puedo obtener datos de precios reales\n")
        print("📌 SOLUCIÓN:")
        print("   1. Crea un archivo .env en el directorio del proyecto")
        print("   2. Agrega: FMP_API_KEY=tu_api_key_aqui")
        print("   3. O exporta la variable: export FMP_API_KEY=tu_api_key")
        print("\n🎯 PERO PUEDO DECIRTE CUÁL ES EL PROBLEMA:")
        print()
        print("=" * 80)
        print("🔍 ANÁLISIS SIN NECESIDAD DE API")
        print("=" * 80)
        print()
        print("Basándome en tu reporte de que PYPL, CPRT, DECK, IP están")
        print("BAJO su MA200 pero aparecen en tus resultados, hay 3 opciones:\n")
        print("1️⃣  EL FILTRO MA200 NO ESTÁ ACTIVADO")
        print("   ▸ Probabilidad: 🔴 ALTA")
        print("   ▸ En Streamlit, el checkbox 'Filtro MA200' estaba DESMARCADO")
        print("   ▸ Verifica: Busca en la salida 'MA200 Filter: ENABLED'")
        print()
        print("2️⃣  DATOS CACHEADOS/DESACTUALIZADOS")
        print("   ▸ Probabilidad: 🟡 MEDIA")
        print("   ▸ Los precios cambiaron desde que se ejecutó el screener")
        print("   ▸ Solución: Clic en '🗑️ Limpiar Caché' y re-ejecutar")
        print()
        print("3️⃣  RESULTADOS DE SESIÓN ANTERIOR")
        print("   ▸ Probabilidad: 🟢 BAJA")
        print("   ▸ Estás viendo resultados guardados de una ejecución anterior")
        print("   ▸ Solución: Ejecutar screener AHORA y verificar")
        print()
        print("=" * 80)
        print("📋 ACCIÓN INMEDIATA:")
        print("=" * 80)
        print()
        print("1. Abre la app Streamlit")
        print("2. Verifica que 'Filtro MA200 (Faber 2007)' esté ✅ MARCADO")
        print("3. Haz clic en '🗑️ Limpiar Caché'")
        print("4. Ejecuta '🚀 Ejecutar Screening V3'")
        print("5. En la salida, BUSCA:")
        print()
        print("   🚀 PASO 7: Momentum + MA200 Filter")
        print("      MA200 Filter: ENABLED (Faber 2007)    <-- DEBE DECIR ESTO")
        print("      ...")
        print("      Rejected by MA200: X                   <-- Debe rechazar stocks")
        print()
        print("6. Si NO dice 'ENABLED', el filtro NO se aplicó")
        print()
        print("=" * 80)
        print()
        print("📖 Para más detalles, lee: DEBUG_MA200_FILTER.md")
        return

    # Si tenemos API key, hacer verificación real
    print("✅ API Key configurada, obteniendo datos...\n")

    results_problematic = []
    results_control = []

    for symbol in stocks_to_check:
        result = check_stock(symbol)
        results_problematic.append(result)

    for symbol in stocks_control:
        result = check_stock(symbol)
        results_control.append(result)

    # Mostrar resultados
    print("\n" + "=" * 80)
    print("📊 RESULTADOS: Stocks 'Problemáticos' (usuario dice que están bajo MA200)")
    print("=" * 80)
    print()

    below_count = 0
    for r in results_problematic:
        if r.get('error'):
            print(f"❓ {r['symbol']:6s} | ERROR: {r['error']}")
        elif r['above_ma200']:
            print(f"✅ {r['symbol']:6s} | SOBRE MA200 ({r['pct_above']:+.2f}%) | Precio: ${r['price']:.2f} | MA200: ${r['ma200']:.2f}")
        else:
            below_count += 1
            print(f"❌ {r['symbol']:6s} | BAJO MA200  ({r['pct_above']:+.2f}%) | Precio: ${r['price']:.2f} | MA200: ${r['ma200']:.2f}")

    print("\n" + "=" * 80)
    print("📊 RESULTADOS: Stocks de Control")
    print("=" * 80)
    print()

    for r in results_control:
        if r.get('error'):
            print(f"❓ {r['symbol']:6s} | ERROR: {r['error']}")
        elif r['above_ma200']:
            print(f"✅ {r['symbol']:6s} | SOBRE MA200 ({r['pct_above']:+.2f}%) | Precio: ${r['price']:.2f} | MA200: ${r['ma200']:.2f}")
        else:
            below_count += 1
            print(f"❌ {r['symbol']:6s} | BAJO MA200  ({r['pct_above']:+.2f}%) | Precio: ${r['price']:.2f} | MA200: ${r['ma200']:.2f}")

    # Análisis
    print("\n" + "=" * 80)
    print("🎯 ANÁLISIS")
    print("=" * 80)
    print()

    if below_count > 0:
        print(f"⚠️  CONFIRMADO: {below_count} stocks están BAJO su MA200")
        print()
        print("🔍 DIAGNÓSTICO:")
        print("   El filtro MA200 NO se aplicó en tu ejecución")
        print()
        print("📌 CAUSAS POSIBLES:")
        print("   1. El checkbox 'Filtro MA200' estaba DESMARCADO")
        print("   2. El parámetro require_above_ma200=False en la config")
        print()
        print("✅ SOLUCIÓN:")
        print("   1. Abre app Streamlit")
        print("   2. MARCA el checkbox 'Filtro MA200 (Faber 2007)'")
        print("   3. Haz clic en 'Limpiar Caché'")
        print("   4. Ejecuta 'Ejecutar Screening V3'")
        print("   5. Verifica que la salida diga 'MA200 Filter: ENABLED'")
    else:
        print("✅ Todos los stocks verificados están SOBRE su MA200")
        print()
        print("🤔 POSIBLES EXPLICACIONES:")
        print("   1. Los precios cambiaron desde que ejecutaste el screener")
        print("   2. Estás viendo resultados de una sesión anterior")
        print("   3. Necesitas re-ejecutar el screener con datos actuales")

    print()


def check_stock(symbol: str) -> dict:
    """Verifica un stock específico."""
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')

        prices = fetch_prices(symbol, start=start_date, end=end_date, use_cache=False)

        if prices is None or prices.empty or len(prices) < 200:
            return {
                'symbol': symbol,
                'error': f'Datos insuficientes ({len(prices) if prices is not None else 0} días)',
                'above_ma200': None
            }

        current_price = prices['close'].iloc[-1]
        ma200 = calculate_ma200(prices)
        above_ma200 = is_above_ma200(prices)
        pct_above = ((current_price - ma200) / ma200) * 100 if ma200 else None

        return {
            'symbol': symbol,
            'above_ma200': above_ma200,
            'price': current_price,
            'ma200': ma200,
            'pct_above': pct_above,
        }

    except Exception as e:
        return {
            'symbol': symbol,
            'error': str(e),
            'above_ma200': None
        }


if __name__ == "__main__":
    quick_check()
