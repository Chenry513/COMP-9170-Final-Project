import sys
!{sys.executable} -m pip install ucimlrepo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

# Fetch dataset from UCI ML Repository (ID 296)
diabetes_data = fetch_ucirepo(id=296)
X = diabetes_data.data.features
y = diabetes_data.data.targets

# Make sure target column has a nice name
if "readmitted" not in y.columns:
    y.columns = ["readmitted"]

# Combine into a single DataFrame
df = pd.concat([X, y], axis=1)

print("Shape:", df.shape)
print(df.head())
print(df.columns)
print(y.head())


# Cleaning pipeline function
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
      This reduces the number of categories while keeping the main clinical signal.
    - Create a binary 30-day readmission label readmit_30d:
        1 if readmitted == "<30"
        0 if readmitted is "NO" or ">30"

    Returns
    -------
    df_clean : pandas.DataFrame
        Cleaned dataframe with high-missing columns removed, race handled,
        diagnosis groups added, and readmit_30d defined.
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
        """
        Map a diagnosis code (like '250', '414') into a broad group
        such as 'Circulatory', 'Respiratory', 'Diabetes', etc.
        Missing values become 'Unknown'.
        """
        if pd.isna(code):
            return "Unknown"

        # codes can be like '250', '414', 'V45', 'E870', etc.
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

    # apply mapping to each diag column and create *_group versions
    for col in diag_cols:
        if col in df_clean.columns:
            df_clean[col + "_group"] = df_clean[col].apply(map_diag_to_group)

    # (optional) drop the raw diag codes if you only want the grouped ones
    # df_clean = df_clean.drop(columns=[c for c in diag_cols if c in df_clean.columns])

    # 4. Create binary 30-day readmission label
    if "readmitted" in df_clean.columns:
        # 1 = readmitted within 30 days, 0 = NO or >30
        df_clean["readmit_30d"] = (df_clean["readmitted"] == "<30").astype(int)

    return df_clean

    df_clean = clean_diabetes_data(df)

    df_clean.head()