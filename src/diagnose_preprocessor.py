"""
diagnose_preprocessor.py

Run this from your project root (where models/ and src/ live) to find out
exactly which column(s) the preprocessor is treating as numeric that are
actually sending strings like "Yes"/"No".

Usage:
    python diagnose_preprocessor.py
"""

from pathlib import Path
import joblib

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"

preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")

print("=" * 70)
print("ColumnTransformer breakdown")
print("=" * 70)

# preprocessor.transformers_ is populated after fitting
for name, transformer, columns in preprocessor.transformers_:
    print(f"\nTransformer name : {name}")
    print(f"Transformer type : {transformer}")
    print(f"Columns assigned : {list(columns)}")

print("\n" + "=" * 70)
print("Now compare the 'numeric' transformer's column list above against")
print("your input_data dict in app.py. Any column in that list whose value")
print("you're sending as a string (e.g. 'Yes'/'No') instead of a number")
print("(e.g. 1/0) is the cause of the ValueError.")
print("=" * 70)