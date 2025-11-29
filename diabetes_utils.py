import pandas as pd
import numpy as np

def clean_diabetes_data(df):
    """
    Clean and prepare the Diabetes 130-US dataset for modelling.

    Steps:
    - Replace "?" with proper missing values (NaN).
    - Drop columns with extremely high missingness:
      * weight (~97% missing)
      * medical_specialty (~49% missing, many categories)
      * payer_code (~40% missing)
    - Keep race, but fill missing values with an explicit "Unknown" category so we
      do not drop those rows and the model can treat missing race as its own level.
    - Group diagnosis codes (diag_1, diag_2, diag_3) into broad disease categories
      instead of using every raw code:
        Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal,
        Genitourinary, Neoplasms, Other, Unknown.
    - Create a binary 30-day readmission label readmit_30d:
        1 if readmitted == "<30"
        0 if readmitted is "NO" or ">30"
    """
    df_clean = df.copy()

    # Treat '?' as missing everywhere
    df_clean = df_clean.replace("?", np.nan)

    # 1. Drop columns with huge missingness
    high_missing_cols = ["weight", "medical_specialty", "payer_code"]
    df_clean = df_clean.drop(
        columns=[c for c in high_missing_cols if c in df_clean.columns]
    )

    # 2. Handle race: keep it, but make missing explicit
    if "race" in df_clean.columns:
        df_clean["race"] = df_clean["race"].fillna("Unknown")

    # 3. Group diagnosis codes into broad disease categories
    diag_cols = ["diag_1", "diag_2", "diag_3"]

    def map_diag_to_group(code):
        """Map a diagnosis code into a broad group."""
        if pd.isna(code):
            return "Unknown"

        try:
            num = float(code)
        except ValueError:
            return "Other"

        if (390 <= num <= 459) or (num == 785):
            return "Circulatory"
        elif (460 <= num <= 519) or (num == 786):
            return "Respiratory"
        elif (520 <= num <= 579) or (num == 787):
            return "Digestive"
        elif num == 250:
            return "Diabetes"
        elif 800 <= num <= 999:
            return "Injury"
        elif 710 <= num <= 739:
            return "Musculoskeletal"
        elif 580 <= num <= 629:
            return "Genitourinary"
        elif 140 <= num <= 239:
            return "Neoplasms"
        else:
            return "Other"

    for col in diag_cols:
        if col in df_clean.columns:
            df_clean[col + "_group"] = df_clean[col].apply(map_diag_to_group)

    # 4. Create binary 30-day readmission label
    if "readmitted" in df_clean.columns:
        df_clean["readmit_30d"] = (df_clean["readmitted"] == "<30").astype(int)

    return df_clean
