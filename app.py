import streamlit as st
import pandas as pd
import joblib
import random
import os
from datetime import datetime

st.set_page_config(page_title="Cyber Risk Analyzer", layout="wide", page_icon="🔐")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-meter {
        background: linear-gradient(90deg, #ff4444 0%, #ffaa00 50%, #44aa44 100%);
        border-radius: 10px;
        height: 20px;
        margin: 10px 0;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        text-align: center;
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        margin: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🔐 Cyber Security Behavioral Risk Analyzer</h1>', unsafe_allow_html=True)

# Load model
model = joblib.load("model.pkl")

# ---------------- FEATURE FUNCTION ----------------
def calculate_features(df):
    df["phishing"] = df.loc[:, "Q1":"Q5"].mean(axis=1)
    df["password"] = df.loc[:, "Q6":"Q10"].mean(axis=1)
    df["risk"] = df.loc[:, "Q11":"Q15"].mean(axis=1)
    df["device"] = df.loc[:, "Q16":"Q20"].mean(axis=1)
    df["attention"] = df.loc[:, "Q21":"Q25"].mean(axis=1)
    df["security"] = df.loc[:, "Q26":"Q30"].mean(axis=1)

    df["impulsiveness"] = df[["Q3", "Q11", "Q19"]].mean(axis=1)
    df["trust"] = df[["Q1", "Q13", "Q24"]].mean(axis=1)
    df["urgency"] = df[["Q3", "Q11"]].mean(axis=1)

    df["impulse_risk"] = df["impulsiveness"] * df["urgency"]
    df["trust_risk"] = df["phishing"] * df["trust"]
    df["awareness_gap"] = df["security"] - df["attention"]

    return df


def analyze_behavior(df):
    suggestions = []
    row = df.iloc[0]

    if row["phishing"] < 2.5:
        suggestions.append("⚠️ You are vulnerable to phishing attacks. Avoid clicking unknown links and verify sources.")

    if row["password"] < 2.5:
        suggestions.append("🔑 Improve password practices. Use strong, unique passwords and avoid reuse.")

    if row["risk"] < 2.5:
        suggestions.append("🚨 You tend to take risky actions. Always verify before responding to urgent messages.")

    if row["device"] < 2.5:
        suggestions.append("📱 Device security is weak. Avoid public Wi-Fi or use VPN.")

    if row["attention"] < 2.5:
        suggestions.append("👀 You may miss security warnings. Stay focused while browsing.")

    if row["security"] < 2.5:
        suggestions.append("🛡️ Enable security features like 2FA, updates, and backups.")

    return suggestions


def clean_question_text(q_text):
    return q_text.split(". ", 1)[-1]


def create_risk_gauge(risk_score):
    """Create a visual risk gauge using HTML"""
    percentage = min(100, max(0, risk_score))
    color = "#ff4444" if percentage > 66 else "#ffaa00" if percentage > 33 else "#44aa44"

    gauge_html = f"""
    <div style="width: 100%; background-color: #e0e0e0; border-radius: 10px; height: 25px; margin: 10px 0;">
        <div style="width: {percentage}%; background-color: {color}; height: 25px; border-radius: 10px; text-align: center; color: white; font-weight: bold; line-height: 25px;">
            {percentage:.1f}%
        </div>
    </div>
    """
    return gauge_html


features = [
    "phishing","password","risk","device",
    "attention","security",
    "impulsiveness","trust","urgency",
    "impulse_risk","trust_risk","awareness_gap"
]

feature_names = {
    "phishing": "Phishing Awareness",
    "password": "Password Security",
    "risk": "Risk Handling",
    "device": "Device Security",
    "attention": "Attention Level",
    "security": "Security Practices",
    "awareness_gap": "Awareness vs Action Gap",
    "impulse_risk": "Impulsive Behavior Risk",
    "trust_risk": "Overtrust Risk",
    "impulsiveness": "Impulsiveness",
    "trust": "Trust Level",
    "urgency": "Urgency Response"
}

feature_explanations = {
    "phishing": "How well you identify and avoid phishing attempts",
    "password": "Quality and management of your password practices",
    "risk": "Your response to potentially risky online situations",
    "device": "Security measures on your devices and networks",
    "attention": "Focus and awareness during online activities",
    "security": "Adoption of security best practices and tools",
    "awareness_gap": "Difference between security knowledge and actual application",
    "impulse_risk": "Risk from impulsive decisions under pressure",
    "trust_risk": "Risk from over-trusting sources or communications",
    "impulsiveness": "Tendency to act quickly without full consideration",
    "trust": "Level of trust placed in online communications",
    "urgency": "Response to urgent or time-pressured situations"
}

question_bank = {

"Q1": ("1. Unexpected Bank Email\nSuppose you receive an email saying your bank account needs urgent verification.",
[
"Click the link immediately and complete it",
"Open the link and quickly look through it",
"Visit the official website separately to check",
"Call or confirm with the bank before doing anything"
]),

"Q2": ("2. Slightly Strange Website Name\nYou notice a website name that looks correct but has a small spelling change.",
[
"Ignore the difference",
"Notice it but continue anyway",
"Pause and double-check",
"Avoid using that website"
]),

"Q3": ("3. Urgent Account Block Message\nIf you receive a message saying your account will be blocked soon:",
[
"Act immediately",
"Feel stressed and click quickly",
"Verify first, then act",
"Ignore and verify calmly"
]),

"Q4": ("4. OTP Entry on Website\nSuppose you are about to enter an OTP on a website.",
[
"Enter the OTP immediately without checking much",
"Quickly look around the page and then enter",
"Check the website address and security details first",
"Confirm the site is genuine before entering anything"
]),

"Q5": ("5. Hearing About a New Scam\nYou hear about a new online scam trending.",
[
"Ignore it since it probably won’t affect you",
"Just read the headline and move on",
"Read briefly to understand what happened",
"Try to understand how the scam works to avoid it"
]),

"Q6": ("6. Creating a New Account\nWhile creating a new account on a website, you usually:",
[
"Use the same password you already use",
"Slightly modify your old password",
"Create a completely new strong password",
"Use a password manager to generate one"
]),

"Q7": ("7. After Hearing About a Data Breach\nIf you hear that a website you use was hacked, you would:",
[
"Do nothing and wait",
"Assume it’s not serious",
"Change your password later",
"Change your password immediately"
]),

"Q8": ("8. A Friend Asks for Your Password\nIf a close friend asks for your password saying it’s urgent, you would:",
[
"Share it without hesitation",
"Share but plan to change later",
"Politely refuse",
"Suggest another safe way to help"
]),

"Q9": ("9. Saving Password in Browser\nWhen your browser asks 'Save this password?', you usually:",
[
"Always click save",
"Save most of the time",
"Save only on personal devices",
"Prefer using a secure password manager instead"
]),

"Q10": ("10. Managing Multiple Accounts\nWhen handling multiple accounts, you generally:",
[
"Use the same login credentials everywhere",
"Use slightly modified versions",
"Use different passwords for each",
"Use securely generated and stored credentials"
]),

"Q11": ("11. Prize or Offer Email\nIf you receive an email saying you won a prize, you would:",
[
"Click the link immediately",
"Feel curious and open it",
"Search online to confirm first",
"Ignore it completely"
]),

"Q12": ("12. Unknown Attachment\nIf you receive an attachment from someone unfamiliar, you would:",
[
"Open it directly",
"Preview it quickly",
"Verify the sender first",
"Delete it immediately"
]),

"Q13": ("13. Email from a Popular Brand\nIf an email looks like it’s from a popular brand, you would:",
[
"Trust it and follow the link",
"Click but scan quickly",
"Check the sender’s domain carefully",
"Visit the official app or website separately"
]),

"Q14": ("14. WhatsApp Offer Link\nIf someone forwards a 'Free subscription' link, you would:",
[
"Click it immediately",
"Ask if it’s real",
"Search online before opening",
"Ignore it"
]),

"Q15": ("15. After Clicking Something Suspicious\nIf you realize you clicked a suspicious link, you would:",
[
"Ignore it",
"Close the tab and forget",
"Change your password",
"Run a security scan and change credentials"
]),

"Q16": ("16. Public Wi-Fi Usage\nAt a café with free Wi-Fi, you usually:",
[
"Log into everything normally",
"Avoid only sensitive accounts",
"Use mobile data instead",
"Use VPN or additional security"
]),

"Q17": ("17. App Permissions\nWhen installing an app that asks for many permissions, you:",
[
"Allow all without reading",
"Quickly accept to finish installation",
"Review permissions carefully",
"Deny unnecessary permissions"
]),

"Q18": ("18. Security Warning in Browser\nIf your browser shows 'Not Secure', you:",
[
"Continue anyway",
"Ignore and proceed",
"Pause and reconsider",
"Exit the website immediately"
]),

"Q19": ("19. Security Steps Slowing You Down\nIf extra security steps take time, you:",
[
"Try to bypass them",
"Feel irritated but continue",
"Complete them patiently",
"Appreciate the added security"
]),

"Q20": ("20. Downloading Trending Apps\nIf you see a trending unofficial app, you:",
[
"Download instantly",
"Think briefly and download",
"Research before downloading",
"Avoid unofficial apps completely"
]),

"Q21": ("21. Pop-Up Notification\nWhen a pop-up says 'Allow Notifications', you:",
[
"Click Allow immediately",
"Close it without reading",
"Read and then decide",
"Block such notifications"
]),

"Q22": ("22. Using a Shared Device\nIf using a shared or public device, you:",
[
"Stay logged in after use",
"Log out later",
"Log out immediately",
"Avoid logging in if possible"
]),

"Q23": ("23. Multitasking Online\nWhile browsing and chatting together, you:",
[
"Often miss small security warnings",
"Sometimes overlook details",
"Usually stay aware",
"Prefer focusing on one task"
]),

"Q24": ("24. Sharing Personal Information\nIf a website asks for extra personal details, you:",
[
"Fill everything requested",
"Fill without thinking much",
"Provide only necessary information",
"Question why it’s needed"
]),

"Q25": ("25. Reusing Credentials\nWhen creating new accounts, you:",
[
"Use same login details",
"Slightly modify old ones",
"Create unique credentials",
"Use managed secure credentials"
]),

"Q26": ("26. Software Update Notification\nWhen your device shows an update, you:",
[
"Ignore it",
"Delay it for long",
"Update within some time",
"Update immediately"
]),

"Q27": ("27. Device Lock\nYour device security setup is:",
[
"No lock at all",
"Simple PIN",
"Strong password",
"Biometric with strong backup"
]),

"Q28": ("28. Two-Factor Authentication\nWhen 2FA is available, you:",
[
"Never enable it",
"Enable rarely",
"Enable for important accounts",
"Enable for most accounts"
]),

"Q29": ("29. Data Backup\nYour data backup habit is:",
[
"Never back up",
"Rarely back up",
"Occasionally back up",
"Regular automatic backup"
]),

"Q30": ("30. Antivirus Usage\nRegarding antivirus or built-in protection:",
[
"I don’t use any",
"Installed but not updated",
"Updated occasionally",
"Always active and updated"
])
}
# ---------------- SECTIONS ----------------
sections = {
    "phishing": ["Q1","Q2","Q3","Q4","Q5"],
    "password": ["Q6","Q7","Q8","Q9","Q10"],
    "risk": ["Q11","Q12","Q13","Q14","Q15"],
    "device": ["Q16","Q17","Q18","Q19","Q20"],
    "attention": ["Q21","Q22","Q23","Q24","Q25"],
    "security": ["Q26","Q27","Q28","Q29","Q30"]
}

# ---------------- MODE ----------------
mode = st.sidebar.selectbox("Mode", ["Quick Test (6Q)", "Full Survey (30Q)", "Analytics Dashboard"])

# ---------------- QUICK TEST ----------------
if mode == "Quick Test (6Q)":

    st.header("⚡ Quick Behavioral Test")

    if "selected_questions" not in st.session_state:
        selected_questions = []
        for sec in sections.values():
            selected_questions.append(random.choice(sec))
        st.session_state.selected_questions = selected_questions

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 New Test"):
            del st.session_state.selected_questions
            st.experimental_rerun()

    selected_questions = st.session_state.selected_questions
    answers = {}

    for i, q in enumerate(selected_questions, start=1):
        q_text, options = question_bank[q]
        clean_text = clean_question_text(q_text)
        answers[q] = st.radio(f"Q{i}. {clean_text}", options, key=f"quick_{q}")

    if st.button("Analyze"):

        data = [2]*30

        for q, ans in answers.items():
            options = question_bank[q][1]
            score = options.index(ans) + 1
            data[int(q[1:]) - 1] = score

        df = pd.DataFrame([data], columns=[f"Q{i}" for i in range(1,31)])
        df = calculate_features(df)

        risk_score = df[features].mean(axis=1).iloc[0] * 25

        pred = model.predict(df[features])[0]

        # Enhanced Results Layout
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader(f"Risk Level: {pred}")
            st.write(f"**Risk Score: {risk_score:.1f}/100**")
            st.markdown(create_risk_gauge(risk_score), unsafe_allow_html=True)

            if pred == "High":
                st.error("🚨 High Risk")
            elif pred == "Medium":
                st.warning("⚠️ Medium Risk")
            else:
                st.success("✅ Low Risk")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("### Key Behavioral Features")
            st.dataframe(df[features[:6]].T.style.format("{:.2f}"))
            st.markdown('</div>', unsafe_allow_html=True)

        # Personalized Suggestions
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### 🎯 Personalized Suggestions")
        suggestions = analyze_behavior(df)
        if suggestions:
            for s in suggestions:
                st.warning(s)
        else:
            st.success("✅ Great! Your behavior shows strong cybersecurity awareness.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Highlight worst feature
        worst = df[features].iloc[0].idxmin()
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(f"### 🔍 Most Critical Weakness: {feature_names.get(worst, worst)}")
        st.info(feature_explanations.get(worst, "Behavioral risk factor"))
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FULL SURVEY ----------------
elif mode == "Full Survey (30Q)":

    st.header("📋 Full Behavioral Survey")

    responses = []

    for q in question_bank:
        q_text, options = question_bank[q]

        ans = st.radio(
            q_text,
            options,
            key=f"full_{q}"
        )

        score = options.index(ans) + 1
        responses.append(score)

    if st.button("Submit"):

        df = pd.DataFrame([responses], columns=[f"Q{i}" for i in range(1,31)])
        df_features = calculate_features(df.copy())
        df_final = pd.concat([df, df_features[features]], axis=1)

        pred = model.predict(df_features[features])[0]
        df_final["risk_label"] = pred

        df_final["timestamp"] = datetime.now()

        file_path = "collected_data.csv"

        if not os.path.exists(file_path):
            df_final.to_csv(file_path, index=False)
        else:
            df_final.to_csv(file_path, mode='a', header=False, index=False)

        st.success("Response saved successfully!")

# ---------------- ANALYTICS DASHBOARD ----------------
elif mode == "Analytics Dashboard":

    st.header("📊 Analytics Dashboard")

    file_path = "collected_data.csv"

    if not os.path.exists(file_path):
        st.warning("No data collected yet. Complete some surveys to see analytics.")
    else:
        df = pd.read_csv(file_path)

        # Overview Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Responses", len(df))
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            if "risk_label" in df.columns:
                high_count = (df["risk_label"] == "High").sum()
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("High Risk Users", high_count)
                st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                recent = df[df["timestamp"] > datetime.now() - pd.Timedelta(days=7)].shape[0]
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Last 7 Days", recent)
                st.markdown('</div>', unsafe_allow_html=True)

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if "risk_label" in df.columns:
                risk_counts = df["risk_label"].value_counts()
                st.subheader("Risk Level Distribution")
                st.bar_chart(risk_counts)

                # Percentages
                total = len(df)
                high_pct = (risk_counts.get("High", 0) / total * 100)
                med_pct = (risk_counts.get("Medium", 0) / total * 100)
                low_pct = (risk_counts.get("Low", 0) / total * 100)
                st.write(f"High: {high_pct:.1f}% | Medium: {med_pct:.1f}% | Low: {low_pct:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            feature_cols = [col for col in features if col in df.columns]
            if feature_cols:
                st.subheader("Average Behavioral Scores")
                avg_scores = df[feature_cols].mean()
                st.bar_chart(avg_scores)
            st.markdown('</div>', unsafe_allow_html=True)

        # Trends
        if "timestamp" in df.columns:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["date"] = df["timestamp"].dt.date
            daily_counts = df.groupby("date").size()
            st.subheader("Response Trends Over Time")
            st.line_chart(daily_counts)
            st.markdown('</div>', unsafe_allow_html=True)