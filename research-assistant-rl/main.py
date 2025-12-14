import sys
import os

print("\n" + "="*70)
print("RESEARCHMIND - COMPLETE PIPELINE")
print("="*70)

print("\n" + "="*60)
print("STEP 1: Running Experiments")
print("="*60)
os.system(f"{sys.executable} experiments\\run_experiments.py")

print("\n" + "="*60)
print("STEP 2: Analyzing Results")
print("="*60)
os.system(f"{sys.executable} experiments\\analyze_results.py")

print("\n" + "="*60)
print("STEP 3: Statistical Validation")
print("="*60)
os.system(f"{sys.executable} experiments\\validation.py")

print("\n" + "="*60)
print("STEP 4: Theoretical Analysis")
print("="*60)
os.system(f"{sys.executable} experiments\\theoretical_analysis.py")

print("\n" + "="*70)
print("✓ COMPLETE - ALL ANALYSES FINISHED")
print("="*70)
print("\nGenerated files in results/ directory:")
print("  1. experiment_data.json")
print("  2. learning_curves.png")
print("  3. source_preferences.png")
print("  4. strategy_usage.png")
print("  5. summary_report.txt")
print("  6. comprehensive_validation.txt")
print("  7. theoretical_analysis.txt")
print("\n" + "="*70)
print("To use RAG + LLM web interface:")
print("  streamlit run app.py")
print("="*70)