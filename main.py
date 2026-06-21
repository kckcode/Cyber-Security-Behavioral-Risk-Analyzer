import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans

# ---------------- SETTINGS ----------------
plt.style.use('default')

plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12
})

def save_png(filename):
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data.csv")
print("Dataset Shape:", df.shape)

questions = [f"Q{i}" for i in range(1, 31)]
df_q = df[questions]

# ---------------- PREPROCESS ----------------
df_q = df_q.apply(pd.to_numeric, errors='coerce')
df_q = df_q.fillna(df_q.mean())
df_q = df_q.clip(1, 4)

# ---------------- FEATURE ENGINEERING ----------------
df["phishing"] = df_q.loc[:, "Q1":"Q5"].mean(axis=1)
df["password"] = df_q.loc[:, "Q6":"Q10"].mean(axis=1)
df["risk"] = df_q.loc[:, "Q11":"Q15"].mean(axis=1)
df["device"] = df_q.loc[:, "Q16":"Q20"].mean(axis=1)
df["attention"] = df_q.loc[:, "Q21":"Q25"].mean(axis=1)
df["security"] = df_q.loc[:, "Q26":"Q30"].mean(axis=1)

df["impulsiveness"] = df[["Q3", "Q11", "Q19"]].mean(axis=1)
df["trust"] = df[["Q1", "Q13", "Q24"]].mean(axis=1)
df["urgency"] = df[["Q3", "Q11"]].mean(axis=1)

df["impulse_risk"] = df["impulsiveness"] * df["urgency"]
df["trust_risk"] = df["phishing"] * df["trust"]
df["awareness_gap"] = df["security"] - df["attention"]

# ---------------- BVI SCORE ----------------
df["total_score"] = df_q.sum(axis=1)
df["BVI"] = ((df["total_score"] - 30) / 90) * 100

def categorize(bvi):
    if bvi <= 33:
        return "Low"
    elif bvi <= 66:
        return "Medium"
    else:
        return "High"

df["risk_category"] = df["BVI"].apply(categorize)

# ---------------- MODEL ----------------
features = [
    "phishing", "password", "risk", "device",
    "attention", "security",
    "impulsiveness", "trust", "urgency",
    "impulse_risk", "trust_risk", "awareness_gap"
]

label_map = {
    "phishing": "Phishing Awareness",
    "password": "Password Hygiene",
    "risk": "Risk Behavior",
    "device": "Device Security",
    "attention": "Attention Level",
    "security": "Security Awareness",
    "impulsiveness": "Impulsiveness",
    "trust": "Trust Level",
    "urgency": "Urgency Response",
    "impulse_risk": "Impulse Risk",
    "trust_risk": "Trust Risk",
    "awareness_gap": "Awareness Gap"
}

pretty_features = [label_map[f] for f in features]

X = df[features]
y = df["risk_category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
accuracy = model.score(X_test, y_test)
print("\nAccuracy:", accuracy)

y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------- CONFUSION MATRIX ----------------
from sklearn.metrics import ConfusionMatrixDisplay

plt.figure()
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
plt.title("Confusion Matrix", fontsize=10)
plt.tight_layout()
save_png("Figure_6.png")

# ---------------- FEATURE IMPORTANCE ----------------
importances = model.feature_importances_

plt.figure()

sorted_idx = np.argsort(importances)

plt.barh(np.array(pretty_features)[sorted_idx],
         importances[sorted_idx],
         color="black")   # 🔥 grayscale

plt.xlabel("Importance", fontsize=9)
plt.title("Feature Importance", fontsize=10)

plt.grid(axis='x', linestyle=':', linewidth=0.5)

plt.tight_layout()
save_png("Figure_1.png")

# ---------------- CORRELATION ----------------
plt.figure(figsize=(6,4))

sns.heatmap(
    df[features].corr(),
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    linewidths=0.3,
    cbar=True
)

plt.title("Feature Correlation", fontsize=10)
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(fontsize=7)

plt.tight_layout()
save_png("Figure_2.png")

# ---------------- SAMPLE PREDICTION ----------------
sample = X.iloc[[0]]
prediction = model.predict(sample)
print("\nSample Prediction:", prediction[0])

# ---------------- ANALYSIS ----------------
print("\n--- MEAN FEATURE VALUES BY RISK CATEGORY ---")
print(df.groupby("risk_category")[features].mean())

print("\n--- HIGH RISK USER PROFILE ---")
print(df[df["risk_category"] == "High"][features].mean())

# ---------------- CORRELATION WITH RISK ----------------
df["risk_numeric"] = df["risk_category"].map({
    "Low": 1, "Medium": 2, "High": 3
})

correlation = df[features + ["risk_numeric"]].corr()
print("\n--- CORRELATION WITH RISK ---")
print(correlation["risk_numeric"].sort_values(ascending=False))

# ---------------- BOXPLOTS ----------------
def clean_boxplot(x, y, title, filename):
    plt.figure()

    sns.boxplot(
        x=x,
        y=y,
        data=df,
        color="lightgray",
        width=0.5,
        fliersize=3,
        linewidth=1
    )

    sns.stripplot(
        x=x,
        y=y,
        data=df,
        color="black",
        size=2,
        alpha=0.3
    )

    plt.title(title, fontsize=10)
    plt.xlabel("Risk Category", fontsize=9)
    plt.ylabel(y.capitalize(), fontsize=9)

    save_png(filename)

clean_boxplot("risk_category", "impulsiveness",
              "Impulsiveness vs Risk", "Figure_3.png")

clean_boxplot("risk_category", "attention",
              "Attention vs Risk", "Figure_4.png")

clean_boxplot("risk_category", "password",
              "Password vs Risk", "Figure_5.png")

# ---------------- CLUSTERING ----------------
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

print("\n--- CLUSTER ANALYSIS ---")
print(df.groupby("cluster")[features].mean())

# ---------------- TOP FEATURES ----------------
print("\n--- TOP FEATURES ---")
important = sorted(zip(features, importances),
                   key=lambda x: x[1], reverse=True)

for f, v in important[:5]:
    print(f"{f} → {v:.3f}")

# ---------------- SAVE MODEL ----------------
joblib.dump(model, "model.pkl")