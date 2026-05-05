"""Force-regenerate the final report. Stand-alone wrapper to avoid module-import quirks."""
import sys, traceback
sys.path.insert(0, '.')
try:
    from src.final_report import build_final_report
    p = build_final_report()
    print('OK -> wrote', p)
except Exception as e:
    print('FAILED:', e)
    traceback.print_exc()
    sys.exit(1)
