"""
Validador MA200 - Verifica en tiempo real si stocks están above MA200
========================================================================

Uso:
    python validate_ma200.py MO RSG TRV

Para validar el portfolio completo del screener.
"""

import sys
import pandas as pd
from data_fetcher import fetch_prices
from momentum_calculator import is_above_ma200, calculate_ma200
from datetime import datetime, timedelta


def validate_ma200_status(symbols: list) -> pd.DataFrame:
    """
    Valida el status MA200 actual de una lista de símbolos.

    Args:
        symbols: Lista de símbolos a validar

    Returns:
        DataFrame con resultados de validación
    """
    results = []

    print(f"\n🔍 Validando MA200 para {len(symbols)} símbolos...")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for symbol in symbols:
        try:
            # Fetch precios actuales (últimos 300 días para tener suficiente historia)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
            prices = fetch_prices(symbol, start=start_date, end=end_date)

            if prices is None or prices.empty or len(prices) < 200:
                results.append({
                    'symbol': symbol,
                    'status': '❓',
                    'reason': 'Datos insuficientes (<200 días)',
                    'current_price': None,
                    'ma200': None,
                    'distance': None,
                })
                continue

            # Verificar MA200
            current_price = prices['close'].iloc[-1]
            ma200 = calculate_ma200(prices)
            above = is_above_ma200(prices)

            # Calcular distancia % de la MA200
            distance_pct = ((current_price - ma200) / ma200) * 100 if ma200 else None

            results.append({
                'symbol': symbol,
                'status': '✅' if above else '❌',
                'reason': 'Above MA200' if above else 'BELOW MA200',
                'current_price': current_price,
                'ma200': ma200,
                'distance': distance_pct,
            })

            # Print individual
            status_icon = '✅' if above else '⚠️'
            print(f"{status_icon} {symbol:6s} | Price: ${current_price:>7.2f} | MA200: ${ma200:>7.2f} | Distance: {distance_pct:>+6.2f}%")

        except Exception as e:
            results.append({
                'symbol': symbol,
                'status': '❌',
                'reason': f'Error: {str(e)}',
                'current_price': None,
                'ma200': None,
                'distance': None,
            })
            print(f"❌ {symbol:6s} | Error: {str(e)}")

    df = pd.DataFrame(results)

    # Summary
    print("\n" + "="*70)
    print("📊 RESUMEN:")
    above_count = (df['status'] == '✅').sum()
    below_count = (df['status'] == '❌').sum()
    error_count = (df['status'] == '❓').sum()

    print(f"   ✅ Above MA200: {above_count}/{len(symbols)} ({100*above_count/len(symbols):.1f}%)")
    print(f"   ❌ Below MA200: {below_count}/{len(symbols)} ({100*below_count/len(symbols):.1f}%)")
    if error_count > 0:
        print(f"   ❓ Sin datos:    {error_count}/{len(symbols)}")

    # Alertas
    if below_count > 0:
        print("\n⚠️  ALERTA: Las siguientes acciones están DEBAJO de MA200:")
        below_stocks = df[df['status'] == '❌']['symbol'].tolist()
        for s in below_stocks:
            print(f"      - {s}")
        print("\n   Esto puede indicar:")
        print("   1. Los precios cambiaron desde que se ejecutó el screener")
        print("   2. Hay un lag en los datos de la API")
        print("   3. Necesitas re-ejecutar el screener con datos frescos")

    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Error: Debes proporcionar al menos un símbolo")
        print("\nUso:")
        print("  python validate_ma200.py MO RSG TRV")
        print("  python validate_ma200.py AAPL MSFT GOOGL")
        sys.exit(1)

    symbols = sys.argv[1:]

    print("="*70)
    print("🔍 VALIDADOR MA200 - Verificación en Tiempo Real")
    print("="*70)

    df_results = validate_ma200_status(symbols)

    # Opcional: guardar resultados
    output_file = f"ma200_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n💾 Resultados guardados en: {output_file}")
