import pandas as pd
import numpy as np
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier
import re

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── VAT Training Data ──────────────────────────────────────────────────────────
TRAINING_DATA = [
    # STANDARD RATE 20%
    ("AMAZON PRIME MEMBERSHIP", "standard_rate"),
    ("AMAZON PRIME", "standard_rate"),
    ("AMZN MKTP UK", "standard_rate"),
    ("AMAZON WEB SERVICES", "standard_rate"),
    ("AMAZON AWS", "standard_rate"),
    ("APPLE COM BILL", "standard_rate"),
    ("APPLE ICLOUD", "standard_rate"),
    ("APPLE MUSIC", "standard_rate"),
    ("GOOGLE GSUITE", "standard_rate"),
    ("GOOGLE WORKSPACE", "standard_rate"),
    ("GOOGLE STORAGE", "standard_rate"),
    ("MICROSOFT 365", "standard_rate"),
    ("MICROSOFT OFFICE", "standard_rate"),
    ("MICROSOFT AZURE", "standard_rate"),
    ("ADOBE CREATIVE", "standard_rate"),
    ("ADOBE SYSTEMS", "standard_rate"),
    ("DROPBOX", "standard_rate"),
    ("SLACK TECHNOLOGIES", "standard_rate"),
    ("ZOOM VIDEO", "standard_rate"),
    ("LINKEDIN PREMIUM", "standard_rate"),
    ("LINKEDIN LEARNING", "standard_rate"),
    ("SPOTIFY PREMIUM", "standard_rate"),
    ("NETFLIX", "standard_rate"),
    ("DELIVEROO", "standard_rate"),
    ("UBER EATS", "standard_rate"),
    ("JUST EAT", "standard_rate"),
    ("MCDONALD S", "standard_rate"),
    ("KFC UK", "standard_rate"),
    ("BURGER KING", "standard_rate"),
    ("PIZZA HUT", "standard_rate"),
    ("DOMINOS PIZZA", "standard_rate"),
    ("COSTA COFFEE", "standard_rate"),
    ("STARBUCKS", "standard_rate"),
    ("CAFFE NERO", "standard_rate"),
    ("PRET A MANGER", "standard_rate"),
    ("MARKS SPENCER CAFE", "standard_rate"),
    ("VODAFONE UK", "standard_rate"),
    ("EE LIMITED", "standard_rate"),
    ("O2 UK", "standard_rate"),
    ("THREE MOBILE", "standard_rate"),
    ("BT GROUP", "standard_rate"),
    ("SKY UK", "standard_rate"),
    ("VIRGIN MEDIA", "standard_rate"),
    ("CURRYS PC WORLD", "standard_rate"),
    ("ARGOS LIMITED", "standard_rate"),
    ("JOHN LEWIS", "standard_rate"),
    ("SELFRIDGES", "standard_rate"),
    ("HARRODS", "standard_rate"),
    ("ASOS", "standard_rate"),
    ("NEXT RETAIL", "standard_rate"),
    ("H AND M", "standard_rate"),
    ("ZARA UK", "standard_rate"),
    ("PRIMARK", "standard_rate"),
    ("EBAY UK", "standard_rate"),
    ("ETSY UK", "standard_rate"),
    ("SAGE ACCOUNTING", "standard_rate"),
    ("QUICKBOOKS", "standard_rate"),
    ("XERO LIMITED", "standard_rate"),
    ("HUBSPOT", "standard_rate"),
    ("MAILCHIMP", "standard_rate"),
    ("SHOPIFY", "standard_rate"),
    ("SQUARESPACE", "standard_rate"),
    ("WIXCOM", "standard_rate"),
    ("GODADDY", "standard_rate"),
    ("NAMECHEAP", "standard_rate"),
    ("CLOUDFLARE", "standard_rate"),
    ("GITHUB", "standard_rate"),
    ("DIGITALOCEAN", "standard_rate"),
    ("HEROKU", "standard_rate"),
    ("VERCEL", "standard_rate"),
    ("RAILWAY APP", "standard_rate"),
    ("RENDER SERVICES", "standard_rate"),
    ("UBER RIDE", "standard_rate"),
    ("ADDISON LEE", "standard_rate"),
    ("ENTERPRISE CAR HIRE", "standard_rate"),
    ("EUROPCAR", "standard_rate"),
    ("AVIS CAR RENTAL", "standard_rate"),
    ("PREMIER INN", "standard_rate"),
    ("TRAVELODGE", "standard_rate"),
    ("HOLIDAY INN", "standard_rate"),
    ("BOOKING COM", "standard_rate"),
    ("AIRBNB", "standard_rate"),
    ("EXPEDIA", "standard_rate"),
    ("RYANAIR", "standard_rate"),
    ("EASYJET", "standard_rate"),
    ("BRITISH AIRWAYS", "standard_rate"),
    ("VIRGIN ATLANTIC", "standard_rate"),
    ("STAPLES UK", "standard_rate"),
    ("RYMAN STATIONERY", "standard_rate"),
    ("VIKING DIRECT", "standard_rate"),
    ("AMAZON BUSINESS SUPPLIES", "standard_rate"),
    ("IKEA UK", "standard_rate"),
    ("SCREWFIX", "standard_rate"),
    ("TOOLSTATION", "standard_rate"),
    ("SPORTS DIRECT", "standard_rate"),
    ("JD SPORTS", "standard_rate"),
    ("HALFORDS", "standard_rate"),
    ("RAC BREAKDOWN", "standard_rate"),
    ("AA MEMBERSHIP", "standard_rate"),
    ("B AND Q", "standard_rate"),
    ("HOMEBASE", "standard_rate"),
    ("DUNELM", "standard_rate"),

    # REDUCED RATE 5%
    ("BRITISH GAS", "reduced_rate"),
    ("BRITISH GAS ENERGY", "reduced_rate"),
    ("BRITISH GAS SERVICES", "reduced_rate"),
    ("EON ENERGY", "reduced_rate"),
    ("EON NEXT", "reduced_rate"),
    ("EDF ENERGY", "reduced_rate"),
    ("NPOWER", "reduced_rate"),
    ("SCOTTISH POWER", "reduced_rate"),
    ("OCTOPUS ENERGY", "reduced_rate"),
    ("BULB ENERGY", "reduced_rate"),
    ("OVO ENERGY", "reduced_rate"),
    ("SSE ENERGY", "reduced_rate"),
    ("SEVERN TRENT WATER", "reduced_rate"),
    ("THAMES WATER", "reduced_rate"),
    ("ANGLIAN WATER", "reduced_rate"),
    ("YORKSHIRE WATER", "reduced_rate"),
    ("SOUTHERN WATER", "reduced_rate"),
    ("UNITED UTILITIES", "reduced_rate"),
    ("CALOR GAS", "reduced_rate"),
    ("FLOGAS", "reduced_rate"),
    ("NATIONAL GAS", "reduced_rate"),

    # ZERO RATED 0%
    ("TESCO STORES", "zero_rated"),
    ("TESCO EXTRA", "zero_rated"),
    ("TESCO EXPRESS", "zero_rated"),
    ("TESCO METRO", "zero_rated"),
    ("TESCO SUPERSTORE", "zero_rated"),
    ("SAINSBURYS", "zero_rated"),
    ("SAINSBURYS LOCAL", "zero_rated"),
    ("ASDA STORES", "zero_rated"),
    ("ASDA SUPERMARKET", "zero_rated"),
    ("MORRISONS", "zero_rated"),
    ("MORRISONS DAILY", "zero_rated"),
    ("WAITROSE", "zero_rated"),
    ("WAITROSE PARTNERS", "zero_rated"),
    ("LIDL UK", "zero_rated"),
    ("ALDI STORES", "zero_rated"),
    ("CO OP FOOD", "zero_rated"),
    ("MARKS SPENCER FOOD", "zero_rated"),
    ("MARKS AND SPENCER FOODHALL", "zero_rated"),
    ("ICELAND FOODS", "zero_rated"),
    ("FARMFOODS", "zero_rated"),
    ("AMAZON KINDLE", "zero_rated"),
    ("KINDLE BOOKS", "zero_rated"),
    ("AMAZON BOOKS", "zero_rated"),
    ("WATERSTONES", "zero_rated"),
    ("WHSmith BOOKS", "zero_rated"),
    ("THE BOOK PEOPLE", "zero_rated"),
    ("HIVE BOOKS", "zero_rated"),
    ("TRAINLINE", "zero_rated"),
    ("NATIONAL RAIL", "zero_rated"),
    ("AVANTI WEST COAST", "zero_rated"),
    ("LNER TRAINS", "zero_rated"),
    ("GREAT WESTERN RAILWAY", "zero_rated"),
    ("SOUTHEASTERN TRAINS", "zero_rated"),
    ("SOUTHERN RAIL", "zero_rated"),
    ("TFL TRAVEL", "zero_rated"),
    ("TFL OYSTER", "zero_rated"),
    ("TRANSPORT FOR LONDON", "zero_rated"),
    ("STAGECOACH BUS", "zero_rated"),
    ("FIRST BUS", "zero_rated"),
    ("ARRIVA BUS", "zero_rated"),
    ("MEGABUS", "zero_rated"),
    ("NATIONAL EXPRESS BUS", "zero_rated"),
    ("EUROSTAR", "zero_rated"),
    ("CHILDRENS PLACE", "zero_rated"),
    ("NEXT KIDS", "zero_rated"),
    ("MOTHERCARE", "zero_rated"),
    ("BOOTS PHARMACY", "zero_rated"),
    ("LLOYDS PHARMACY", "zero_rated"),
    ("SUPERDRUG PHARMACY", "zero_rated"),

    # EXEMPT
    ("BARCLAYS BANK CHARGE", "exempt"),
    ("BARCLAYS FEE", "exempt"),
    ("HSBC BANK CHARGE", "exempt"),
    ("HSBC SERVICE FEE", "exempt"),
    ("LLOYDS BANK CHARGE", "exempt"),
    ("LLOYDS MONTHLY FEE", "exempt"),
    ("NATWEST BANK CHARGE", "exempt"),
    ("NATWEST SERVICE CHARGE", "exempt"),
    ("SANTANDER BANK FEE", "exempt"),
    ("STARLING BANK FEE", "exempt"),
    ("MONZO BANK CHARGE", "exempt"),
    ("REVOLUT FEE", "exempt"),
    ("WISE TRANSFER FEE", "exempt"),
    ("PAYPAL FEE", "exempt"),
    ("STRIPE FEE", "exempt"),
    ("SQUARE FEE", "exempt"),
    ("GOCARDLESS FEE", "exempt"),
    ("ROYAL MAIL POSTAGE", "exempt"),
    ("ROYAL MAIL STAMPS", "exempt"),
    ("PARCELFORCE", "exempt"),
    ("DPD POSTAGE", "exempt"),
    ("HERMES POSTAGE", "exempt"),
    ("EVRI DELIVERY", "exempt"),
    ("AVIVA INSURANCE", "exempt"),
    ("AXA INSURANCE", "exempt"),
    ("ZURICH INSURANCE", "exempt"),
    ("DIRECT LINE INSURANCE", "exempt"),
    ("ADMIRAL INSURANCE", "exempt"),
    ("CHURCHILL INSURANCE", "exempt"),
    ("LEGAL AND GENERAL", "exempt"),
    ("BUPA HEALTH INSURANCE", "exempt"),
    ("VITALITY INSURANCE", "exempt"),
    ("NHS PRESCRIPTION", "exempt"),
    ("NHS DENTAL", "exempt"),
    ("GP SURGERY", "exempt"),
    ("PRIVATE MEDICAL CENTRE", "exempt"),
    ("INTEREST CHARGED", "exempt"),
    ("BANK INTEREST", "exempt"),
    ("OVERDRAFT FEE", "exempt"),
    ("LATE PAYMENT FEE", "exempt"),

    # OUTSIDE SCOPE
    ("HMRC SHIPLEY", "outside_scope"),
    ("HMRC CUMBERNAULD", "outside_scope"),
    ("HMRC PAYE", "outside_scope"),
    ("HMRC VAT PAYMENT", "outside_scope"),
    ("HMRC CORPORATION TAX", "outside_scope"),
    ("HMRC SELF ASSESSMENT", "outside_scope"),
    ("HMRC NIC", "outside_scope"),
    ("HMRC NATIONAL INSURANCE", "outside_scope"),
    ("COMPANIES HOUSE", "outside_scope"),
    ("COMPANIES HOUSE FEE", "outside_scope"),
    ("DVLA VEHICLE TAX", "outside_scope"),
    ("DVLA LICENCE", "outside_scope"),
    ("COUNCIL TAX", "outside_scope"),
    ("LOCAL AUTHORITY RATES", "outside_scope"),
    ("BUSINESS RATES", "outside_scope"),
    ("PAYROLL TRANSFER", "outside_scope"),
    ("SALARY PAYMENT", "outside_scope"),
    ("WAGES TRANSFER", "outside_scope"),
    ("DIVIDEND PAYMENT", "outside_scope"),
    ("DIRECTOR DRAWINGS", "outside_scope"),
    ("SHAREHOLDER DIVIDEND", "outside_scope"),
    ("INTER ACCOUNT TRANSFER", "outside_scope"),
    ("OWN ACCOUNT TRANSFER", "outside_scope"),
    ("SAVINGS TRANSFER", "outside_scope"),
    ("LOAN REPAYMENT", "outside_scope"),
    ("MORTGAGE PAYMENT", "outside_scope"),
    ("MORTGAGE DD", "outside_scope"),
    ("NATWEST LOAN", "outside_scope"),
    ("BARCLAYS LOAN REPAY", "outside_scope"),
    ("PENSION CONTRIBUTION", "outside_scope"),
    ("NEST PENSION", "outside_scope"),
    ("PEOPLES PENSION", "outside_scope"),
    ("CASH WITHDRAWAL", "outside_scope"),
    ("ATM WITHDRAWAL", "outside_scope"),
    ("CASH DEPOSIT", "outside_scope"),
]

# ── Augment Data ───────────────────────────────────────────────────────────────
def augment_description(desc, label, n=15):
    rows = [(desc, label)]
    suffixes = [
        "3421", "LONDON", "MANCHESTER", "001", "REF12345",
        "DD", "DIRECT DEBIT", "PAYMENT", "LTD", "PLC",
        "UK", "GB", "LIMITED", "ONLINE", "DIGITAL"
    ]
    for i in range(n):
        suffix = np.random.choice(suffixes)
        variation = f"{desc} {suffix}"
        rows.append((variation, label))
    return rows

augmented = []
for desc, label in TRAINING_DATA:
    augmented.extend(augment_description(desc, label, n=20))

df = pd.DataFrame(augmented, columns=["description", "vat_code"])
print(f"Total training samples: {len(df)}")
print(df["vat_code"].value_counts())

# ── Clean Text ─────────────────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

df["clean"] = df["description"].apply(clean_text)

# ── Encode Labels ──────────────────────────────────────────────────────────────
le = LabelEncoder()
df["label"] = le.fit_transform(df["vat_code"])
print(f"Classes: {le.classes_}")

# ── Train/Test Split ───────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"]
)

# ── TF-IDF Vectorization ───────────────────────────────────────────────────────
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    min_df=1
)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# ── Train XGBoost ──────────────────────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42
)
model.fit(X_train_vec, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_vec)
print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=le.classes_))
macro_f1 = f1_score(y_test, y_pred, average="macro")
print(f"Macro F1 Score: {macro_f1:.4f}")

# ── Save Artifacts ─────────────────────────────────────────────────────────────
joblib.dump(tfidf, os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.joblib"))
joblib.dump(model, os.path.join(ARTIFACTS_DIR, "xgboost_model.joblib"))
joblib.dump(le, os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"))

print(f"\n✅ Artifacts saved to: {ARTIFACTS_DIR}")
print("  - tfidf_vectorizer.joblib")
print("  - xgboost_model.joblib")
print("  - label_encoder.joblib")