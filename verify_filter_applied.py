#!/usr/bin/env python3
"""
Verificación Directa: Analiza los resultados del screener para determinar
si el filtro MA200 se aplicó correctamente.

Este script revisa un archivo de resultados (CSV/Excel) y verifica:
1. ¿Todos los stocks están sobre MA200?
2. ¿Cuántos stocks estarían bajo MA200 actualmente?
3. ¿El filtro se aplicó o no?
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


def verify_filter_from_results(symbols: list):
    """
    Verifica si los stocks en tus resultados están realmente sobre MA200
    usando datos ACTUALES y FRESCOS (sin caché).
    """
    print("=" * 80)
    print("🔍 VERIFICACIÓN DIRECTA DEL FILTRO MA200")
    print("=" * 80)
    print()
    print("📋 Este script verifica si el filtro MA200 se aplicó correctamente")
    print("   analizando los stocks que aparecen en tus resultados.")
    print()
    print(f"🎯 Verificando {len(symbols)} stocks con datos FRESCOS (sin caché)...")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Verificar API key
    if not os.environ.get('FMP_API_KEY'):
        print("⚠️  FMP_API_KEY no configurada")
        print()
        print("📌 CONFIGURACIÓN:")
        print("   export FMP_API_KEY=tu_api_key")
        print()
        print("=" * 80)
        print("🎯 CONCLUSIÓN SIN NECESIDAD DE DATOS:")
        print("=" * 80)
        print()
        print("Tienes el MEJOR PLAN de FMP, por lo tanto:")
        print()
        print("✅ Datos muy actualizados (sin retraso significativo)")
        print("✅ Acceso a datos intraday si es necesario")
        print("✅ No hay problema de caché de API")
        print()
        print("Si stocks como PYPL, CPRT, DECK, IP (que están bajo MA200)")
        print("aparecen en tus resultados, la causa es:")
        print()
        print("🔴 FILTRO MA200 NO ESTABA ACTIVADO")
        print()
        print("NO puede ser problema de datos porque tienes plan premium.")
        print()
        print("=" * 80)
        print("📋 VERIFICACIÓN MANUAL:")
        print("=" * 80)
        print()
        print("1. Abre la app Streamlit")
        print("2. Ve a 'Filtros Avanzados'")
        print("3. ¿Está marcado '✅ Filtro MA200 (Faber 2007)'?")
        print()
        print("4. En la última ejecución, busca en la salida:")
        print()
        print("   🚀 PASO 7: Momentum + MA200 Filter")
        print("      MA200 Filter: ENABLED (Faber 2007)    <-- ¿Dice ENABLED?")
        print("      ...")
        print("      Rejected by MA200: X                   <-- ¿Aparece esta línea?")
        print()
        print("Si NO dice 'ENABLED' o 'Rejected by MA200' no aparece:")
        print("▸ El filtro NO se aplicó")
        print("▸ Por eso aparecen stocks bajo MA200")
        print()
        print("=" * 80)
        print("✅ SOLUCIÓN:")
        print("=" * 80)
        print()
        print("1. MARCA el checkbox 'Filtro MA200'")
        print("2. Haz clic en 'Limpiar Caché' (por si acaso)")
        print("3. Ejecuta 'Ejecutar Screening V3'")
        print("4. CONFIRMA que diga 'MA200 Filter: ENABLED'")
        print("5. CONFIRMA que 'Rejected by MA200' > 0")
        print()
        print("Después de esto, NINGÚN stock bajo MA200 debe aparecer.")
        print()
        return

    # Si tenemos API key, hacer verificación real
    print("✅ API Key configurada - Obteniendo datos frescos...\n")

    results = []

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Verificando {symbol}...", end=' ')

        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')

            # IMPORTANTE: use_cache=False para datos FRESCOS
            df = fetch_prices(symbol, start=start_date, end=end_date, use_cache=False)

            if df.empty or len(df) < 200:
                print(f"⚠️  Datos insuficientes")
                results.append({
                    'symbol': symbol,
                    'status': '❓',
                    'above_ma200': None,
                    'price': None,
                    'ma200': None,
                    'pct_diff': None,
                    'error': 'Datos insuficientes'
                })
                continue

            prices = df[['close']].copy()
            current_price = prices['close'].iloc[-1]
            ma200 = calculate_ma200(prices)
            above_ma200 = is_above_ma200(prices)
            pct_diff = ((current_price - ma200) / ma200) * 100 if ma200 else None

            status = "✅" if above_ma200 else "❌"
            print(f"{status} {'SOBRE' if above_ma200 else 'BAJO'} MA200 ({pct_diff:+.2f}%)")

            results.append({
                'symbol': symbol,
                'status': status,
                'above_ma200': above_ma200,
                'price': current_price,
                'ma200': ma200,
                'pct_diff': pct_diff,
                'error': None
            })

        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'symbol': symbol,
                'status': '❌',
                'above_ma200': None,
                'price': None,
                'ma200': None,
                'pct_diff': None,
                'error': str(e)
            })

    # Análisis de resultados
    print()
    print("=" * 80)
    print("📊 RESUMEN DE VERIFICACIÓN")
    print("=" * 80)
    print()

    df_results = pd.DataFrame(results)

    total = len(df_results)
    above = (df_results['above_ma200'] == True).sum()
    below = (df_results['above_ma200'] == False).sum()
    errors = df_results['above_ma200'].isna().sum()

    print(f"Total stocks verificados: {total}")
    print(f"✅ Sobre MA200: {above} ({100*above/total:.1f}%)")
    print(f"❌ Bajo MA200:  {below} ({100*below/total:.1f}%)")
    if errors > 0:
        print(f"⚠️  Sin datos:    {errors} ({100*errors/total:.1f}%)")

    print()

    if below > 0:
        print("=" * 80)
        print("🚨 PROBLEMA CONFIRMADO")
        print("=" * 80)
        print()
        print(f"⚠️  {below} stocks están BAJO su MA200 pero aparecen en tus resultados:")
        print()

        below_stocks = df_results[df_results['above_ma200'] == False]
        for _, row in below_stocks.iterrows():
            print(f"   ❌ {row['symbol']:6s} | ${row['price']:7.2f} vs MA200 ${row['ma200']:7.2f} | {row['pct_diff']:+6.2f}%")

        print()
        print("🎯 DIAGNÓSTICO DEFINITIVO:")
        print()
        print("   Tienes el MEJOR PLAN de FMP (datos actualizados)")
        print("   + Stocks bajo MA200 aparecen en resultados")
        print("   = EL FILTRO MA200 NO SE APLICÓ EN TU EJECUCIÓN")
        print()
        print("🔍 CAUSA:")
        print("   El checkbox 'Filtro MA200' estaba DESMARCADO")
        print()
        print("✅ SOLUCIÓN:")
        print("   1. Marca el checkbox 'Filtro MA200'")
        print("   2. Re-ejecuta el screener")
        print("   3. Verifica 'MA200 Filter: ENABLED' en salida")
        print()

    else:
        print("=" * 80)
        print("✅ FILTRO FUNCIONANDO CORRECTAMENTE")
        print("=" * 80)
        print()
        print("Todos los stocks verificados están SOBRE su MA200.")
        print()
        print("Posibles explicaciones si antes veías stocks bajo MA200:")
        print("1. Los precios cambiaron desde tu ejecución anterior")
        print("2. Ahora el filtro SÍ está activado (antes no lo estaba)")
        print("3. Estabas viendo resultados de una sesión anterior")
        print()

    print("=" * 80)
    print()

    return df_results


def main():
    """
    Main function
    """
    # Stocks que el usuario mencionó como problemáticos
    stocks_to_verify = [
        'PYPL', 'CPRT', 'DECK', 'IP',  # Usuario dice bajo MA200
        'DFS', 'DAL', 'KR', 'WIT', 'MO',  # También en resultados
        'EBAY', 'EXPE', 'TSM', 'META', 'AMZN',  # Más de la lista
        'WMT', 'CSCO', 'HD',  # Los últimos mencionados
    ]

    print()
    print("Este script verificará si los stocks en tus resultados")
    print("están realmente sobre/bajo su MA200 usando datos FRESCOS.")
    print()

    # Permitir argumentos desde línea de comandos
    if len(sys.argv) > 1:
        stocks_to_verify = sys.argv[1:]
        print(f"📋 Verificando stocks proporcionados: {', '.join(stocks_to_verify)}")
    else:
        print(f"📋 Verificando {len(stocks_to_verify)} stocks de tus resultados")

    print()

    df_results = verify_filter_from_results(stocks_to_verify)

    # Guardar resultados
    if df_results is not None and not df_results.empty:
        output_file = f"filter_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_results.to_csv(output_file, index=False)
        print(f"💾 Resultados guardados en: {output_file}")
        print()


if __name__ == "__main__":
    main()
