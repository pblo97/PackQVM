#!/usr/bin/env python3
"""
Verificar específicamente Royal Caribbean (RCL) y su estado MA200
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pandas as pd
    from data_fetcher import fetch_prices
    from momentum_calculator import is_above_ma200, calculate_ma200
    from datetime import datetime, timedelta
except ImportError as e:
    print(f"❌ Error: {e}")
    print("Ejecuta: pip install pandas requests")
    sys.exit(1)


def check_rcl_detailed():
    """
    Verifica RCL con datos recientes para ver cuándo cruzó MA200
    """
    print("=" * 80)
    print("🚢 ROYAL CARIBBEAN (RCL) - Análisis MA200")
    print("=" * 80)
    print()

    # Verificar API key
    if not os.environ.get('FMP_API_KEY'):
        print("⚠️  FMP_API_KEY no configurada")
        print()
        print("📊 ANÁLISIS DEL RETRASO DE DATOS:")
        print()
        print("1️⃣  TIPO DE DATOS: FMP usa datos EOD (End of Day)")
        print("   ▸ Los precios se actualizan DESPUÉS del cierre del mercado")
        print("   ▸ Típicamente disponibles: 5-6 PM ET del mismo día")
        print("   ▸ Pueden tener hasta 12-24 horas de retraso")
        print()
        print("2️⃣  CACHÉ LOCAL: 1 hora (3600 segundos)")
        print("   ▸ Si ejecutaste el screener hace 30 minutos")
        print("   ▸ Los datos están cacheados de esa ejecución")
        print("   ▸ Solución: 'Limpiar Caché' y re-ejecutar")
        print()
        print("3️⃣  PRIORIZACIÓN DE FILTROS:")
        print("   ▸ NO hay priorización de momentum sobre MA200")
        print("   ▸ Ambos son filtros ELIMINATORIOS secuenciales:")
        print()
        print("      PASO 1: Filtrar por MA200 (si está activado)")
        print("      PASO 2: Filtrar por Momentum mínimo")
        print()
        print("   ▸ Si RCL aparece en resultados:")
        print("      a) Filtro MA200 NO estaba activado, O")
        print("      b) Datos son de cuando RCL estaba sobre MA200")
        print()
        print("=" * 80)
        print("🔍 CASO ESPECÍFICO: RCL cruzó abajo el 3 de noviembre")
        print("=" * 80)
        print()
        print("📅 Hoy: 10 de noviembre (7 días después)")
        print()
        print("Si RCL aparece en tus resultados, hay 3 escenarios:")
        print()
        print("❌ ESCENARIO 1: Filtro MA200 desactivado")
        print("   ▸ El checkbox estaba desmarcado")
        print("   ▸ RCL pasó por alto el filtro MA200")
        print("   ▸ Solo se aplicó filtro de momentum")
        print()
        print("📊 ESCENARIO 2: Datos desactualizados/cacheados")
        print("   ▸ El screener usó datos de hace varios días")
        print("   ▸ En esos datos, RCL todavía estaba sobre MA200")
        print("   ▸ La API puede tener 1-2 días de retraso")
        print()
        print("🕐 ESCENARIO 3: Timing del mercado")
        print("   ▸ RCL cruzó hacia abajo DESPUÉS de tu ejecución")
        print("   ▸ O los datos EOD no reflejaban el cruce aún")
        print()
        print("=" * 80)
        print("✅ SOLUCIÓN GARANTIZADA:")
        print("=" * 80)
        print()
        print("1. Haz clic en '🗑️ Limpiar Caché'")
        print("2. Verifica que 'Filtro MA200' esté MARCADO ✓")
        print("3. Ejecuta el screener AHORA")
        print("4. RCL NO debe aparecer en los resultados")
        print()
        print("Si RCL SIGUE apareciendo:")
        print("▸ Los datos de FMP tienen retraso (>7 días)")
        print("▸ O hay un bug en la API de FMP")
        print()
        print("=" * 80)
        print("📖 SOBRE EL RETRASO DE DATOS DE FMP:")
        print("=" * 80)
        print()
        print("FMP (Financial Modeling Prep) API:")
        print("▸ Plan gratuito: Datos EOD con posible retraso de 1 día")
        print("▸ Plan premium: Datos EOD más actualizados")
        print("▸ Datos intraday: Solo en planes superiores")
        print()
        print("Para datos en tiempo real necesitarías:")
        print("▸ Plan premium de FMP")
        print("▸ O usar otra fuente (Yahoo Finance, Alpha Vantage)")
        print()
        return

    # Si tenemos API key, hacer verificación real
    print("✅ API Key configurada\n")

    symbol = 'RCL'

    # Obtener últimos 400 días
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')

    print(f"📥 Obteniendo datos de {symbol}...")
    print(f"   Período: {start_date} a {end_date}")
    print()

    try:
        # Fetch sin caché para obtener datos más frescos
        df = fetch_prices(symbol, start=start_date, end=end_date, use_cache=False)

        if df.empty or len(df) < 200:
            print(f"❌ Datos insuficientes para {symbol}: {len(df)} días")
            return

        # Calcular MA200
        prices = df[['close']].copy()
        current_price = prices['close'].iloc[-1]
        ma200 = calculate_ma200(prices)
        above_ma200 = is_above_ma200(prices)
        pct_diff = ((current_price - ma200) / ma200) * 100 if ma200 else None

        last_date = df['date'].iloc[-1]

        print(f"📊 ESTADO ACTUAL DE {symbol}:")
        print(f"   Último dato disponible: {last_date.strftime('%Y-%m-%d')}")
        print(f"   Precio actual: ${current_price:.2f}")
        print(f"   MA200: ${ma200:.2f}")
        print(f"   Diferencia: {pct_diff:+.2f}%")
        print()

        if above_ma200:
            print(f"✅ {symbol} está SOBRE su MA200")
            print("   ▸ Pasaría el filtro MA200")
        else:
            print(f"❌ {symbol} está BAJO su MA200")
            print("   ▸ Sería RECHAZADO por el filtro MA200")
            print("   ▸ NO debe aparecer en resultados (si filtro activo)")

        print()
        print("=" * 80)
        print("📅 ANÁLISIS TEMPORAL:")
        print("=" * 80)
        print()

        # Calcular cuántos días de retraso tenemos
        today = datetime.now().date()
        last_data_date = last_date.date()
        days_lag = (today - last_data_date).days

        if days_lag == 0:
            print("✅ Datos de HOY - Sin retraso")
        elif days_lag == 1:
            print("⚠️  Datos de AYER - Retraso de 1 día (normal para EOD)")
        else:
            print(f"❌ Datos de hace {days_lag} días - Retraso significativo")

        print()
        print("📌 CONCLUSIÓN:")
        print()

        if above_ma200:
            print(f"   Según los datos más recientes ({last_data_date}):")
            print(f"   {symbol} ESTÁ sobre MA200")
            print()
            print("   Si cruzó hacia abajo el 3 de noviembre:")
            print(f"   ▸ Los datos de FMP están desactualizados (>{days_lag} días)")
            print("   ▸ O el cruce fue intraday y EOD cerró sobre MA200")
        else:
            print(f"   Según los datos más recientes ({last_data_date}):")
            print(f"   {symbol} está BAJO MA200")
            print()
            print("   Si aparece en tus resultados:")
            print("   ▸ El filtro MA200 NO estaba activado, O")
            print("   ▸ Usaste datos cacheados de días anteriores")

        # Mostrar últimos 10 días de precios vs MA200
        print()
        print("=" * 80)
        print("📈 ÚLTIMOS 10 DÍAS (Precio vs MA200):")
        print("=" * 80)
        print()

        # Calcular MA200 rolling para últimos 10 días
        recent = df.tail(10).copy()
        for i in range(len(recent)):
            date = recent.iloc[i]['date']
            price = recent.iloc[i]['close']
            # Calcular MA200 hasta ese punto
            prices_until = df[df['date'] <= date][['close']].copy()
            if len(prices_until) >= 200:
                ma = calculate_ma200(prices_until)
                status = "✅" if price > ma else "❌"
                diff = ((price - ma) / ma) * 100
                print(f"   {date.strftime('%Y-%m-%d')} | ${price:7.2f} | MA200: ${ma:7.2f} | {diff:+6.2f}% {status}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    check_rcl_detailed()
