"""
Test Risk Management Integration
==================================

Quick test to verify that risk management is properly integrated
with the QVM pipeline V3.
"""

import sys
from qvm_pipeline_v3 import QVMConfigV3, run_qvm_pipeline_v3

print("=" * 80)
print("🧪 Testing Risk Management Integration")
print("=" * 80)

# Create config with risk management enabled
config = QVMConfigV3(
    universe_size=20,  # Small universe for fast test
    portfolio_size=5,   # Small portfolio
    min_piotroski_score=5,
    min_qv_score=0.40,
    require_above_ma200=True,
    backtest_enabled=False,  # Disable backtest for faster test
    # Enable Risk Management
    enable_risk_management=True,
    use_volatility_stop=True,
    volatility_stop_confidence=2.0,
    use_trailing_stop=True,
    trailing_stop_method='ATR',
    trailing_atr_multiplier=2.5,
    use_take_profit=True,
    risk_reward_ratio=2.5,
    use_volatility_sizing=True,
    target_volatility=0.15,
    max_position_size=0.20,
)

print("\n📊 Running pipeline with Risk Management enabled...")
print(f"   Universe: {config.universe_size} stocks")
print(f"   Portfolio: {config.portfolio_size} stocks")
print(f"   Risk Management: {config.enable_risk_management}")

try:
    # Run pipeline
    results = run_qvm_pipeline_v3(config=config, verbose=True)

    if results.get('success'):
        portfolio = results['portfolio']

        print("\n" + "=" * 80)
        print("✅ PIPELINE EXITOSO")
        print("=" * 80)

        # Check if risk management columns exist
        risk_cols = ['entry_price', 'stop_loss', 'take_profit', 'position_size_pct', 'risk_pct', 'reward_pct', 'rr_ratio']

        print("\n📊 Verificando columnas de Risk Management:")
        for col in risk_cols:
            if col in portfolio.columns:
                print(f"   ✅ {col}: {portfolio[col].notna().sum()}/{len(portfolio)} valores")
            else:
                print(f"   ❌ {col}: NO ENCONTRADO")

        # Display sample risk data
        if all(col in portfolio.columns for col in risk_cols):
            print("\n" + "=" * 80)
            print("📋 MUESTRA DE RISK MANAGEMENT (primeras 5 posiciones)")
            print("=" * 80)

            sample = portfolio[['symbol'] + risk_cols].head(5)

            for _, row in sample.iterrows():
                print(f"\n{row['symbol']}:")
                print(f"  Entry:    ${row['entry_price']:.2f}")
                print(f"  Stop:     ${row['stop_loss']:.2f} (-{row['risk_pct']:.2f}%)")
                print(f"  Target:   ${row['take_profit']:.2f} (+{row['reward_pct']:.2f}%)")
                print(f"  R:R:      {row['rr_ratio']:.2f}:1")
                print(f"  Position: {row['position_size_pct']:.1f}%")

            # Summary stats
            print("\n" + "=" * 80)
            print("📊 ESTADÍSTICAS SUMMARY")
            print("=" * 80)
            print(f"Position Size promedio: {portfolio['position_size_pct'].mean():.1f}%")
            print(f"R:R Ratio promedio:     {portfolio['rr_ratio'].mean():.2f}:1")
            print(f"Risk promedio:          {portfolio['risk_pct'].mean():.2f}%")
            print(f"Reward promedio:        {portfolio['reward_pct'].mean():.2f}%")

            # Calculate total portfolio risk
            total_risk = (portfolio['position_size_pct'] / 100 * portfolio['risk_pct'] / 100 * 100).sum()
            print(f"\n💎 Total Portfolio Risk: {total_risk:.2f}%")

            print("\n" + "=" * 80)
            print("✅ TEST COMPLETO - Risk Management funcionando correctamente!")
            print("=" * 80)

        else:
            print("\n" + "=" * 80)
            print("⚠️  WARNING: Algunas columnas de risk management no se encontraron")
            print("=" * 80)
            sys.exit(1)

    else:
        print(f"\n❌ Pipeline failed: {results.get('error')}")
        sys.exit(1)

except Exception as e:
    print(f"\n❌ Error durante test: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
