#!/usr/bin/env python3
"""
Test de QVMConfigV3 para verificar que todos los parámetros funcionan correctamente
"""
import sys

print("="*80)
print("🧪 TEST: QVMConfigV3 Configuration")
print("="*80)
print()

# Test 1: Imports
print("1️⃣  Verificando imports...")
try:
    from qvm_pipeline_v3 import QVMConfigV3
    print("   ✅ QVMConfigV3 importado correctamente")
except ImportError as e:
    print(f"   ❌ Error al importar: {e}")
    sys.exit(1)

# Test 2: Crear configuración con defaults
print("\n2️⃣  Creando configuración con valores por defecto...")
try:
    config = QVMConfigV3()
    print("   ✅ Configuración creada con defaults")
    print(f"   - use_price_cache: {config.use_price_cache}")
except Exception as e:
    print(f"   ❌ Error al crear configuración: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Crear configuración con use_price_cache explícito
print("\n3️⃣  Creando configuración con use_price_cache=True...")
try:
    config_true = QVMConfigV3(use_price_cache=True)
    print("   ✅ Configuración creada con use_price_cache=True")
    print(f"   - use_price_cache: {config_true.use_price_cache}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Crear configuración con use_price_cache=False
print("\n4️⃣  Creando configuración con use_price_cache=False...")
try:
    config_false = QVMConfigV3(use_price_cache=False)
    print("   ✅ Configuración creada con use_price_cache=False")
    print(f"   - use_price_cache: {config_false.use_price_cache}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Verificar que el parámetro existe
print("\n5️⃣  Verificando que el parámetro existe en la clase...")
try:
    from dataclasses import fields
    config_fields = {f.name for f in fields(QVMConfigV3)}
    if 'use_price_cache' in config_fields:
        print("   ✅ Parámetro 'use_price_cache' existe en la clase")
    else:
        print("   ❌ Parámetro 'use_price_cache' NO existe en la clase")
        print(f"   Parámetros disponibles: {sorted(config_fields)}")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Crear configuración completa como en app_streamlit_v3.py
print("\n6️⃣  Creando configuración completa (simulando app_streamlit_v3.py)...")
try:
    config_full = QVMConfigV3(
        universe_size=800,
        min_market_cap=2e9,
        min_volume=500000,
        w_quality=0.35,
        w_value=0.40,
        w_fcf_yield=0.10,
        w_momentum=0.15,
        min_piotroski_score=6,
        min_qv_score=0.50,
        max_pe=40.0,
        max_ev_ebitda=20.0,
        require_positive_fcf=True,
        min_roic=0.10,
        require_above_ma200=True,
        min_momentum_12m=0.10,
        require_near_52w_high=False,
        min_pct_from_52w_high=0.80,
        require_roic_above_wacc=True,
        enable_breakout_filter=False,
        require_breakout_confirmed=False,
        require_breakout_strong=False,
        enable_volume_surge_filter=False,
        min_ebit_ev=0.08,
        max_fcf_ev=0.15,
        portfolio_size=30,
        backtest_enabled=True,
        backtest_start="2020-01-01",
        backtest_end="2024-12-31",
        rebalance_freq="Q",
        commission_bps=5,
        slippage_bps=5,
        market_impact_bps=2,
        use_price_cache=True,  # ← ESTE ES EL PARÁMETRO QUE AGREGAMOS
        enable_earnings_quality=True,
        enable_red_flags=True,
        enable_reversal_filter=True,
        use_enhanced_value_score=True,
        enable_fundamental_momentum=False,
        enable_sector_relative=False,
    )
    print("   ✅ Configuración completa creada exitosamente")
    print(f"   - use_price_cache: {config_full.use_price_cache}")
    print(f"   - require_above_ma200: {config_full.require_above_ma200}")
    print(f"   - portfolio_size: {config_full.portfolio_size}")
except TypeError as e:
    print(f"   ❌ TypeError: {e}")
    print("\n   🔍 DIAGNÓSTICO:")
    print("   Este error sugiere que algún parámetro no existe en la clase QVMConfigV3")
    print("   o que hay un mismatch entre el código de qvm_pipeline_v3.py y app_streamlit_v3.py")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✅ TODOS LOS TESTS PASARON")
print("="*80)
print()
print("💡 Si la app de Streamlit sigue fallando, intenta:")
print("   1. Reiniciar la app de Streamlit")
print("   2. Hacer pull del código más reciente")
print("   3. Verificar que estés usando el branch correcto")
print("   4. Limpiar el cache de Streamlit (botón '🗑️ Limpiar Caché')")
print()
