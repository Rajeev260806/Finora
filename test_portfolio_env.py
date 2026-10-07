"""
Root Test Runner for PortfolioEnv (Phase 3 Person B)
"""
from scripts.test_portfolio_env import (
    test_basic_functionality,
    test_embedding_mode_placeholder,
    test_real_embedding_files_loading,
    test_economic_logic,
    test_phase2_reward_integration,
    test_gym_compliance,
)

if __name__ == "__main__":
    print("=========================================================")
    print("  FINORA PHASE 3: PORTFOLIO ENVIRONMENT TEST SUITE  ")
    print("=========================================================\n")
    test_basic_functionality()
    test_embedding_mode_placeholder()
    test_real_embedding_files_loading()
    test_economic_logic()
    test_phase2_reward_integration()
    test_gym_compliance()
    print("\n[SUCCESS] ALL PHASE 3 TESTS PASSED SUCCESSFULLY!")
