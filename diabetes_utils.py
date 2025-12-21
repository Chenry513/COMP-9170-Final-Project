import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import os

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

def plot_and_save_metrics(model_name, y_test, y_prob, threshold=0.5):
    """
    Make ROC, PR, and confusion matrix plots for one model
    and save them under figures/.
    """
    os.makedirs("figures", exist_ok=True)

    FONT_LABEL = 14
    FONT_TITLE = 16
    FONT_TICKS = 13
    FONT_LEGEND = 13
    LINE_WIDTH_MAIN = 4
    LINE_WIDTH_BASE = 3

    # Binary predictions at chosen threshold
    y_pred = (y_prob >= threshold).astype(int)

    #ROC curve 
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", lw=LINE_WIDTH_MAIN)
    plt.plot([0, 1], [0, 1], "k--", lw=LINE_WIDTH_BASE)
    plt.xlabel("False Positive Rate", fontsize=FONT_LABEL)
    plt.ylabel("True Positive Rate", fontsize=FONT_LABEL)
    plt.title(f"{model_name} – ROC curve", fontsize=FONT_TITLE)
    plt.xticks(fontsize=FONT_TICKS)
    plt.yticks(fontsize=FONT_TICKS)
    plt.legend(loc="lower right", fontsize=FONT_LEGEND)
    plt.tight_layout()
    plt.savefig(f"figures/roc_{model_name}.png", dpi=300)
    plt.close()

    #Precision–Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.3f}", linewidth=LINE_WIDTH_MAIN)
    plt.xlabel("Recall", fontsize=FONT_LABEL)
    plt.ylabel("Precision", fontsize=FONT_LABEL)
    plt.title(f"{model_name} – Precision–Recall curve", fontsize=FONT_TITLE)
    plt.xticks(fontsize=FONT_TICKS)
    plt.yticks(fontsize=FONT_TICKS)
    plt.legend(loc="lower left", fontsize=FONT_LEGEND)
    plt.tight_layout()
    plt.savefig(f"figures/pr_{model_name}.png", dpi=300)
    plt.close()

    #Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No 30d readmit", "30d readmit"])
    disp.plot(values_format="d")
    plt.title(f"{model_name} – Confusion matrix (thr={threshold})", fontsize=FONT_TITLE)
    plt.xticks(fontsize=FONT_TICKS)
    plt.yticks(fontsize=FONT_TICKS)
    
    # Increase font size of the cell values
    for text in plt.gca().texts:
        text.set_fontsize(14)
        
    plt.tight_layout()
    plt.savefig(f"figures/cm_{model_name}.png", dpi=300)
    plt.close()
