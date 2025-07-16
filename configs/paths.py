import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "CN_JP_US_data")
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")

# === DATA NEED TO EXIST ===
US_WEB_DATA = os.path.join(DATA_DIR, "us_web_with_company.csv")
US_PATENT_DATA = os.path.join(DATA_DIR, "us_patent202506.csv")
US_PATENT_DATA_CLEANED = os.path.join(DATA_DIR, "cleaned_patents.csv")
FASTTEXT_VEC = os.path.join(DATA_DIR, "fasttext_web_patent.vec")
# === END ===

# === OUTPUT === 
DUAL_ATT_SAVE = os.path.join(MODEL_DIR, "dual_attn_model_state.pth")
DUAL_ATT_OUTPUT = os.path.join(OUTPUT_DIR, "save_dual_attn_output.csv")
EMBEDDINGS_OUTPUT = os.path.join(OUTPUT_DIR, "full_company_embeddings_multilabel.csv")
FIRM_PRODUCT_KEYWORDS = os.path.join(DATA_DIR, "DualAtt_Firm_Product_Keywords_Table_with_Vector.csv")
COMPANY_EMBEDDINGS = os.path.join(OUTPUT_DIR, "company_embeddings.pkl")
# === END ===



