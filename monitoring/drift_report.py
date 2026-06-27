"""
Phase 4 - Drift Monitoring avec Evidently 0.7+
Référence : données 2022
Courant    : données 2024 ou 2025
"""
import os
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# ── Colonnes utilisées ────────────────────────────────────────────────────────
NUM_FEATURES = [
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "surface_terrain",
    "longitude",
    "latitude",
    "mois",
]
CAT_FEATURES = [
    "type_local",
    "nature_mutation",
    "code_departement",
]
TARGET = "prix_m2"
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")


def load_and_prepare(path: str, sample: int = 50000) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    cols = ALL_FEATURES + [TARGET]
    df = df[cols].dropna(subset=[TARGET])
    df["code_departement"] = df["code_departement"].astype(str)
    if len(df) > sample:
        df = df.sample(sample, random_state=42)
    return df


def run_data_drift_report(reference: pd.DataFrame, current: pd.DataFrame, label: str):
    os.makedirs(REPORTS_DIR, exist_ok=True)

    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)

    out = os.path.join(REPORTS_DIR, f"data_drift_{label}.html")
    result.save_html(out)
    print(f"[drift] Rapport sauvegardé : {out}")

    # Extraire le statut drift
    result_dict = result.dict()
    try:
        drift_info = result_dict["metrics"][0]["value"]
        drift_detected = drift_info.get("dataset_drift", drift_info.get("share_of_drifted_columns", 0) > 0.5)
        print(f"[drift] 2022 → {label.split('_')[-1]} : drift={drift_detected}")
    except Exception:
        print(f"[drift] Rapport {label} généré.")

    return result


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(__file__))

    print("=== Chargement des données ===")
    ref = load_and_prepare(os.path.join(base, "dvf_2022_clean.csv"))
    cur_2024 = load_and_prepare(os.path.join(base, "dvf_2024_clean.csv"))
    cur_2025 = load_and_prepare(os.path.join(base, "dvf_2025_clean.csv"))

    print(f"Référence 2022 : {len(ref)} lignes")
    print(f"Courant  2024  : {len(cur_2024)} lignes")
    print(f"Courant  2025  : {len(cur_2025)} lignes")

    print("\n=== Rapport drift 2022 -> 2024 ===")
    run_data_drift_report(ref, cur_2024, "2022_vs_2024")

    print("\n=== Rapport drift 2022 -> 2025 ===")
    run_data_drift_report(ref, cur_2025, "2022_vs_2025")

    print("\nPhase 4 terminee. Rapports dans monitoring/reports/")
