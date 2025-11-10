#!/usr/bin/env python3
"""
Script para validar MA200 de stocks específicos mencionados por el usuario.
Verifica si PYPL, CPRT, DECK, IP realmente están sobre/bajo su MA200.
"""

import pandas as pd
from datetime import datetime, timedelta
from momentum_calculator import is_above_ma200, calculate_ma200
from data_fetcher import fetch_prices


def check_stock_ma200(symbol: str) -> dict:
    """
    Verifica el estado MA200 de un stock específico.

    Returns:
        Dict con información detallada del stock
    """
    try:
        # Obtener 300 días para asegurar 200 días hábiles
        end_date = datetime.now()
        start_date = end_date - timedelta(days=400)

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        # Usar data_fetcher en lugar de yfinance
        df = fetch_prices(symbol, start_str, end_str, use_cache=False)

        if df.empty or len(df) < 200:
            return {
                'symbol': symbol,
                'error': f'Datos insuficientes: {len(df)} días',
                'above_ma200': None,
                'current_price': None,
                'ma200': None,
                'pct_above_ma200': None
            }

        # Preparar DataFrame con formato esperado (solo columna 'close')
        prices = df[['close']].copy()

        # Calcular MA200 y verificar posición
        current_price = prices['close'].iloc[-1]
        ma200 = calculate_ma200(prices)
        above_ma200 = is_above_ma200(prices)

        if ma200 is not None:
            pct_above = ((current_price - ma200) / ma200) * 100
        else:
            pct_above = None

        return {
            'symbol': symbol,
            'above_ma200': above_ma200,
            'current_price': round(current_price, 2),
            'ma200': round(ma200, 2) if ma200 else None,
            'pct_above_ma200': round(pct_above, 2) if pct_above else None,
            'data_points': len(prices),
            'last_date': df['date'].iloc[-1].strftime('%Y-%m-%d') if not df.empty else 'N/A'
        }

    except Exception as e:
        return {
            'symbol': symbol,
            'error': str(e),
            'above_ma200': None,
            'current_price': None,
            'ma200': None,
            'pct_above_ma200': None
        }


def main():
    """
    Valida los stocks mencionados por el usuario.
    """
    print("=" * 80)
    print("🔍 VALIDACIÓN MA200 EN TIEMPO REAL")
    print("=" * 80)
    print()

    # Stocks mencionados por el usuario como problemáticos
    problematic_stocks = ['PYPL', 'CPRT', 'DECK', 'IP']

    # También verificar algunos otros de la lista
    additional_stocks = ['DFS', 'DAL', 'KR', 'WIT', 'MO']

    all_stocks = problematic_stocks + additional_stocks

    results = []
    for symbol in all_stocks:
        print(f"📊 Verificando {symbol}...", end=' ')
        result = check_stock_ma200(symbol)
        results.append(result)

        if result.get('error'):
            print(f"❌ ERROR: {result['error']}")
        elif result['above_ma200']:
            print(f"✅ SOBRE MA200 ({result['pct_above_ma200']:+.2f}%)")
        else:
            print(f"❌ BAJO MA200 ({result['pct_above_ma200']:+.2f}%)")

    print()
    print("=" * 80)
    print("📋 RESUMEN DETALLADO")
    print("=" * 80)
    print()

    # Tabla detallada
    df_results = pd.DataFrame(results)

    # Separar por categoría
    above = df_results[df_results['above_ma200'] == True]
    below = df_results[df_results['above_ma200'] == False]
    errors = df_results[df_results['above_ma200'].isna()]

    print(f"✅ SOBRE MA200: {len(above)} stocks")
    if not above.empty:
        for _, row in above.iterrows():
            print(f"   {row['symbol']:6s} | Precio: ${row['current_price']:7.2f} | MA200: ${row['ma200']:7.2f} | Diferencia: {row['pct_above_ma200']:+6.2f}%")

    print()
    print(f"❌ BAJO MA200: {len(below)} stocks")
    if not below.empty:
        for _, row in below.iterrows():
            print(f"   {row['symbol']:6s} | Precio: ${row['current_price']:7.2f} | MA200: ${row['ma200']:7.2f} | Diferencia: {row['pct_above_ma200']:+6.2f}%")

    if not errors.empty:
        print()
        print(f"⚠️  ERRORES: {len(errors)} stocks")
        for _, row in errors.iterrows():
            print(f"   {row['symbol']:6s} | {row.get('error', 'Unknown error')}")

    print()
    print("=" * 80)
    print("🎯 ANÁLISIS")
    print("=" * 80)
    print()

    if not below.empty:
        print("⚠️  PROBLEMA DETECTADO:")
        print(f"   Los siguientes stocks están BAJO su MA200 pero aparecen en los resultados:")
        for _, row in below.iterrows():
            print(f"   - {row['symbol']} ({row['pct_above_ma200']:+.2f}% de su MA200)")
        print()
        print("📌 Posibles causas:")
        print("   1. El filtro MA200 NO está activado (require_above_ma200=False)")
        print("   2. Los datos de precios están desactualizados/cacheados")
        print("   3. Bug en la función is_above_ma200()")
        print("   4. Los resultados son de una ejecución anterior con filtro desactivado")
    else:
        print("✅ Todos los stocks verificados están SOBRE su MA200")
        print("   El filtro parece estar funcionando correctamente")

    print()


if __name__ == '__main__':
    main()
