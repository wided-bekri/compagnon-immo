import sys
import os

# Rend inference_service importable depuis les tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "services", "inference"))
