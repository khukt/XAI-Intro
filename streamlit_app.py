# streamlit_app.py
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="AI Decision Transparency Dashboard", page_icon="🔎", layout="wide")

# -----------------------------
# Scenarios & compliance template
# -----------------------------
SCENARIOS = {
    "Loan Approval": {
        "risk_category": "High Risk (Annex III: Creditworthiness)",
        "features": ["income_k", "debt_ratio", "credit_years", "savings_k", "age"],
        "feature_info": {
            "income_k": {"min": 10, "max": 200, "step": 1, "label": "Monthly Income (k SEK)"},
            "debt_ratio": {"min": 0.05, "max": 0.95, "step": 0.01, "label": "Debt-to-Income Ratio"},
            "credit_years": {"min": 0, "max": 25, "step": 1, "label": "Credit History (years)"},
            "savings_k": {"min": 0, "max": 1000, "step": 5, "label": "Savings (k SEK)"},
            "age": {"min": 18, "max": 80, "step": 1, "label": "Age (years)"},
        },
        "positive_label": "Approve",
        "negative_label": "Decline",
    },
    "Predictive Maintenance": {
        "risk_category": "High Risk (Critical Infrastructure)",
        "features": ["temp_c", "vibration", "usage_hours", "operator_flag", "humidity"],
        "feature_info": {
            "temp_c": {"min": 20, "max": 120, "step": 1, "label": "Temperature (°C)"},
            "vibration": {"min": 0.0, "max": 1.0, "step": 0.01, "label": "Vibration Index"},
            "usage_hours": {"min": 0, "max": 20000, "step": 100, "label": "Usage Hours"},
            "operator_flag": {"min": 0, "max": 1, "step": 1, "label": "Operator Fault Flag (0/1)"},
            "humidity": {"min": 0, "max": 100, "step": 1, "label": "Humidity (%)"},
        },
        "positive_label": "Trigger Alert",
        "negative_label": "No Alert",
    },
    "Recruitment Screening": {
        "risk_category": "High Risk (Employment)",
        "features": ["experience_yrs", "education_level", "skills_score", "test_score", "gap_months"],
        "feature_info": {
            "experience_yrs": {"min": 0, "max": 30, "step": 1, "label": "Experience (years)"},
            "education_level": {"min": 0, "max": 3, "step": 1, "label": "Education Level (0-3)"},
            "skills_score": {"min": 0, "max": 100, "step": 1, "label": "Skills Score (0-100)"},
            "test_score": {"min": 0, "max": 100, "step": 1, "label": "Assessment Score (0-100)"},
            "gap_months": {"min": 0, "max": 60, "step": 1, "label": "Career Gap (months)"},
        },
        "positive_label": "Shortlist",
        "negative_label": "Reject",
    },
}

COMPLIANCE_TEMPLATE = {
    "Article 9: Risk Management": True,
    "Article 10: Data & Governance": True,
    "Article 13: Transparency": True,
    "Article 14: Human Oversight": True,
    "Article 15: Accuracy/Robustness": True,
    "Impact Assessment (internal)": False,
}

# -----------------------------
# Data synthesis
# -----------------------------
def synthesize_dataset(scenario: str, n: int = 1400, seed: int = 42):
    rng = np.random.default_rng(seed)
    if scenario == "Loan Approval":
        income = rng.normal(55, 20, n).clip(10, 200)
        debt_ratio = rng.beta(2, 3, n) * 0.9 + 0.05
        credit_years = rng.integers(0, 25, n)
        savings = (rng.exponential(50, n)).clip(0, 1000)
        age = rng.integers(18, 80, n)
        X = pd.DataFrame({
            "income_k": income, "debt_ratio": debt_ratio, "credit_years": credit_years,
            "savings_k": savings, "age": age
        })
        score = (0.035*income - 2.6*debt_ratio + 0.08*credit_years + 0.003*savings + 0.01*np.maximum(age-21,0)
                 + rng.normal(0,0.5,n))
        y = (score > 0.4).astype(int)
    elif scenario == "Predictive Maintenance":
        temp = rng.normal(70, 15, n).clip(20, 120)
        vibration = rng.uniform(0, 1, n)
        usage = rng.integers(0, 20000, n)
        operator_flag = rng.integers(0, 2, n)
        humidity = rng.integers(0, 100, n)
        X = pd.DataFrame({
            "temp_c": temp, "vibration": vibration, "usage_hours": usage,
            "operator_flag": operator_flag, "humidity": humidity
        })
        score = (0.03*(temp-60) + 1.8*vibration + 0.00006*usage + 0.9*operator_flag + 0.01*(humidity-40)
                 + rng.normal(0,0.5,n))
        y = (score > 1.5).astype(int)
    else:
        exp = rng.integers(0, 30, n)
        edu = rng.integers(0, 4, n)
        skills = rng.normal(55, 15, n).clip(0, 100)
        test = rng.normal(60, 18, n).clip(0, 100)
        gap = rng.integers(0, 60, n)
        X = pd.DataFrame({
            "experience_yrs": exp, "education_level": edu, "skills_score": skills,
            "test_score": test, "gap_months": gap
        })
        score = (0.08*exp + 0.4*edu + 0.03*skills + 0.025*test - 0.03*gap + rng.normal(0,0.6,n))
        y = (score > 6.0).astype(int)
    return X, pd.Series(y, name="label")

# -----------------------------
# Modeling & XAI helpers
# -----------------------------
@dataclass
class ModelBundle:
    name: str
    pipeline: Pipeline
    features: List[str]
    is_white_box: bool

@st.cache_resource(show_spinner=False)
def build_models(scenario: str, seed: int = 7) -> Dict[str, ModelBundle]:
    X, y = synthesize_dataset(scenario, n=1800, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)

    lr_pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200, solver="lbfgs"))])
    lr_pipe.fit(X_train, y_train)
    lr_auc = roc_auc_score(y_test, lr_pipe.predict_proba(X_test)[:, 1])

    rf_pipe = Pipeline([("scaler", MinMaxScaler()), ("clf", RandomForestClassifier(n_estimators=160, max_depth=6, random_state=seed))])
    rf_pipe.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf_pipe.predict_proba(X_test)[:, 1])

    return {
        "White-box (Logistic Regression)": ModelBundle(name=f"White-box (AUC={lr_auc:.2f})", pipeline=lr_pipe, features=list(X.columns), is_white_box=True),
        "Black-box (Random Forest)": ModelBundle(name=f"Black-box (AUC={rf_auc:.2f})", pipeline=rf_pipe, features=list(X.columns), is_white_box=False),
    }

def predict_proba(bundle: ModelBundle, x_row: pd.DataFrame) -> float:
    return float(bundle.pipeline.predict_proba(x_row[bundle.features])[0, 1])

def global_importance(bundle: ModelBundle) -> pd.DataFrame:
    clf = bundle.pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        vals = np.abs(clf.coef_).flatten()
    else:
        vals = np.ones(len(bundle.features))
    vals = vals / (vals.sum() + 1e-9)
    return pd.DataFrame({"feature": bundle.features, "importance": vals}).sort_values("importance", ascending=False)

def local_contributions(bundle: ModelBundle, x_row: pd.DataFrame, background: pd.DataFrame) -> List[Tuple[str, float]]:
    clf = bundle.pipeline.named_steps["clf"]
    scaler = bundle.pipeline.named_steps.get("scaler", None)
    x_scaled = x_row[bundle.features].values.astype(float).reshape(1, -1)
    if scaler is not None:
        x_scaled = scaler.transform(x_scaled)
    if hasattr(clf, "coef_"):
        weights = clf.coef_.flatten()
        contribs = (x_scaled.flatten() * weights).tolist()
    elif hasattr(clf, "feature_importances_"):
        bg_mean = background[bundle.features].mean().values
        if scaler is not None:
            bg_mean = scaler.transform(bg_mean.reshape(1, -1)).flatten()
        contribs = (x_scaled.flatten() - bg_mean) * (clf.feature_importances_ / (clf.feature_importances_.sum() + 1e-9))
        contribs = contribs.tolist()
    else:
        contribs = [0.0] * len(bundle.features)
    pairs = list(zip(bundle.features, contribs))
    denom = sum(abs(v) for _, v in pairs) + 1e-9
    pairs = [(f, float(v / denom)) for f, v in pairs]
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    return pairs

def minimal_change_to_flip(bundle: ModelBundle, x_row: pd.DataFrame, background: pd.DataFrame, positive_target: bool) -> Optional[Tuple[str, float]]:
    p = predict_proba(bundle, x_row)
    if positive_target and p >= 0.5:
        return None
    if (not positive_target) and p < 0.5:
        return None
    best = (None, None, -1.0)
    for feat in bundle.features:
        fmin = float(background[feat].quantile(0.05)); fmax = float(background[feat].quantile(0.95))
        for val in np.linspace(fmin, fmax, 15):
            trial = x_row.copy(); trial.iloc[0][feat] = val
            p_trial = predict_proba(bundle, trial)
            if positive_target:
                delta = p_trial - p
                if p_trial >= 0.5 and delta > best[2]: best = (feat, float(val), delta)
            else:
                delta = p - p_trial
                if p_trial < 0.5 and delta > best[2]: best = (feat, float(val), delta)
    if best[0] is None: return None
    return best[0], best[1]

# -----------------------------
# UI helpers
# -----------------------------
def header():
    st.markdown("""
# 🔎 AI Decision Transparency Dashboard
- **Transparent vs. Complex models** (white-box vs black-box)  
- **Global, Local, Targeted explanations**  
- **Stakeholder views & EU AI Act compliance**
""")

def sidebar_controls():
    st.sidebar.markdown("## ⚙️ Controls")
    scenario = st.sidebar.selectbox("Scenario", list(SCENARIOS.keys()), index=0)
    model_choice = st.sidebar.radio("🔬 Model Transparency Level", ["White-Box (Transparent & Explainable)", "Black-Box (Complex & Opaque)"], index=0)
    if "White-Box" in model_choice:
        st.sidebar.caption("✅ Like a clear calculator — every step is visible.")
    else:
        st.sidebar.caption("🧠 Like a smart assistant — powerful but harder to explain (we add XAI).")
    stakeholder = st.sidebar.selectbox("Stakeholder view", ["End User", "Business Decision Maker", "CTO / System Owner", "AI Developer / Auditor", "Field Engineer", "Compliance Officer"], index=0)
    return scenario, model_choice, stakeholder

def feature_inputs(scenario: str, key_prefix: str = "") -> pd.DataFrame:
    cfg = SCENARIOS[scenario]
    cols = st.columns(3)
    values = {}
    for i, feat in enumerate(cfg["features"]):
        meta = cfg["feature_info"][feat]
        with cols[i % 3]:
            values[feat] = st.slider(
                label=meta["label"],
                min_value=float(meta["min"]),
                max_value=float(meta["max"]),
                value=float((meta["min"] + meta["max"]) / 2),
                step=float(meta["step"]),
                key=f"{key_prefix}-{feat}",
            )
    return pd.DataFrame([values])

def decision_card(prob: float, pos_label: str, neg_label: str):
    label = pos_label if prob >= 0.5 else neg_label
    conf = prob if prob >= 0.5 else (1 - prob)
    st.metric("Decision", label, f"Confidence: {conf*100:.0f}%")
    if st.session_state.get("bundle_key_name"):
        st.caption(st.session_state["bundle_key_name"])

def reasons_card(contribs: List[Tuple[str, float]], top_k: int = 3):
    st.markdown("### Top reasons for this decision")
    for feat, val in contribs[:top_k]:
        arrow = "⬆️ helps" if val >= 0 else "⬇️ hurts"
        st.write(f"- **{feat}**: {arrow} (influence {abs(val):.2f})")

def global_bar(df_imp: pd.DataFrame, title: str = "Global feature influence"):
    st.markdown(f"### {title}")
    m = df_imp["importance"].max() + 1e-9
    for _, row in df_imp.iterrows():
        bar = "█" * int(40 * row["importance"] / m)
        st.write(f"{row['feature']:>18} | {bar} {row['importance']:.2f}")

def transparency_note(is_white_box: bool):
    t = "White-box (transparent)" if is_white_box else "Black-box (complex)"
    st.info(f"**Model Transparency:** {t}. Even with a transparent model, people still need case-level reasons, audit logs, and fairness checks.")

def stakeholder_panel(role: str, prob: float, contribs, bundle, x_row, background, pos_label, neg_label):
    label = pos_label if prob >= 0.5 else neg_label
    if role == "End User":
        st.subheader("End User")
        st.write(f"**Outcome:** {label} (confidence {max(prob,1-prob):.0%})")
        reasons_card(contribs, top_k=3)
        suggestion = minimal_change_to_flip(bundle, x_row, background, positive_target=(label==neg_label))
        st.write("**How could this change?**")
        if suggestion:
            f, newv = suggestion
            st.success(f"If **{f}** became **{newv:.2f}**, the decision would likely change.")
        else:
            st.info("Already at target or needs a combination of changes.")
        st.write("**Appeal:** A human review is available.")
    elif role == "Business Decision Maker":
        st.subheader("Business Decision Maker")
        st.write(f"**Outcome:** {label} • **Confidence:** {max(prob,1-prob):.0%}")
        st.write("**Top drivers overall (policy levers):**")
        global_bar(global_importance(bundle))
        st.write("**Fairness & Stability (demo):** gap 2.1% • stability 97%")
    elif role == "CTO / System Owner":
        st.subheader("CTO / System Owner")
        st.write(f"**Pipeline:** {bundle.name}")
        st.write("**Monitoring:** input drift OK • version v1.2")
        global_bar(global_importance(bundle)); reasons_card(contribs, top_k=5)
    elif role == "AI Developer / Auditor":
        st.subheader("AI / Data Scientist")
        global_bar(global_importance(bundle)); reasons_card(contribs, top_k=8)
        st.caption("Store full attributions for audit in production.")
    elif role == "Field Engineer":
        st.subheader("Field Engineer / Operator")
        reasons_card(contribs, top_k=3)
        suggestion = minimal_change_to_flip(bundle, x_row, background, positive_target=(label==neg_label))
        if suggestion:
            f, newv = suggestion
            st.success(f"Adjust **{f}** to **{newv:.2f}** to change the outcome.")
        else:
            st.info("No single-parameter change flips the decision; escalate.")
    else:
        st.subheader("Compliance / Risk")
        st.write(f"- **Risk Category:** {st.session_state.get('risk_category','High Risk')}")
        checklist = dict(COMPLIANCE_TEMPLATE); checklist["Article 13: Transparency"] = True
        for k,v in checklist.items():
            st.write(("✅ " if v else "⚠️ ") + k)
        st.caption("Educational checklist (not legal advice).")

def compliance_overlay(risk_category: str):
    st.markdown("---")
    st.markdown("### 🧩 Compliance Lens (EU AI Act — educational)")
    st.write(f"**Risk Tier:** {risk_category}")
    for k,v in COMPLIANCE_TEMPLATE.items():
        st.write(("✅ " if v else "⚠️ ") + k)

# -----------------------------
# Interactive pipeline diagram
# -----------------------------
def render_pipeline_sankey(mode: str, key: str):
    labels = ["Training Data", "Machine Learning\\nProcess",
              "Black-Box Model" if mode=="Black-Box" else "Explainable Model",
              "Explainable Interface" if mode=="Explainable" else "—",
              "User / Decision\\n& Justification"]
    colors = ["#F59E0B", "#FB923C", "#9CA3AF" if mode=="Black-Box" else "#60A5FA",
              "#93C5FD" if mode=="Explainable" else "#E5E7EB", "#A7F3D0"]
    src, dst, val, link_color, hover = [], [], [], [], []
    src += [0,1]; dst += [1,2]; val += [1,1]
    link_color += ["#F59E0B","#FB923C"]; hover += ["Data flows into training","ML pipeline builds the model"]
    if mode=="Black-Box":
        src += [2]; dst += [4]; val += [1]; link_color += ["#9CA3AF"]; hover += ["Decision arrives without reasons"]
    else:
        src += [2,3]; dst += [3,4]; val += [1,1]; link_color += ["#60A5FA","#93C5FD"]; hover += ["Model outputs scores + attributions","User sees reasons & guidance"]
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(label=labels, pad=25, thickness=22, color=colors, line=dict(color="#374151", width=1), hovertemplate="%{label}"),
        link=dict(source=src, target=dst, value=val, color=link_color, hovertemplate="%{customdata}", customdata=hover)
    )])
    title = "Black-Box Pipeline" if mode=="Black-Box" else "Explainable Pipeline"
    fig.update_layout(title=title, font=dict(size=14))
    st.plotly_chart(fig, use_container_width=True, key=key)

# -----------------------------
# Main App
# -----------------------------
def main():
    header()
    import uuid
    if "session_uid" not in st.session_state:
        st.session_state["session_uid"] = str(uuid.uuid4())[:8]

    scenario, model_choice, stakeholder = sidebar_controls()
    st.session_state["risk_category"] = SCENARIOS[scenario]["risk_category"]

    X_bg, _ = synthesize_dataset(scenario, n=1200, seed=13)
    bundles = build_models(scenario, seed=11)
    bundle = bundles["White-box (Logistic Regression)"] if "White-Box" in model_choice else bundles["Black-box (Random Forest)"]

    tab1, tab0vis, tab2, tab3, tab4, tab5 = st.tabs([
        "1) Black Box Decision",
        "2.0) Visual Explanation",
        "2) Explanations (Global • Local • Targeted)",
        "3) Stakeholder Views",
        "4) Compliance Lens",
        "5) Summary & Export",
    ])

    # Tab 1
    with tab1:
        st.markdown("### 1) Black Box Decision")
        x_row = feature_inputs(scenario, key_prefix=f"tab1-{scenario}-{model_choice}-{st.session_state['session_uid']}")
        prob = predict_proba(bundle, x_row)
        decision_card(prob, SCENARIOS[scenario]["positive_label"], SCENARIOS[scenario]["negative_label"])
        st.session_state["x_row"] = x_row
        st.session_state["prob"] = prob
        st.session_state["bundle_key_name"] = "🟩 Transparent (White-Box)" if bundle.is_white_box else "🟥 Opaque (Black-Box)"
        transparency_note(bundle.is_white_box)
        st.info("Notice how little you can infer without reasons. Next tab: the pipeline picture.")

    # Tab 2.0: Visual Explanation
    with tab0vis:
        st.markdown("### 2.0) Visual Explanation — from black box to explainable system")
        colA, colB = st.columns(2)
        with colA:
            st.caption("Black-Box (opaque): decision without reasons")
            render_pipeline_sankey("Black-Box", key=f"sankey-bb-{st.session_state['session_uid']}")
            st.warning("Users receive a decision but **no why**. Hard to contest or audit.")
        with colB:
            st.caption("Explainable (transparent to the user): decision **with reasons**")
            render_pipeline_sankey("Explainable", key=f"sankey-xai-{st.session_state['session_uid']}")
            st.success("Users receive a decision **and the top reasons**, with clear next steps.")
        st.divider()
        st.markdown("**Tips**")
        st.markdown("- Opaque ≠ illegal, but **high-risk** systems must provide explanations and human oversight.")
        st.markdown("- Even with a transparent model, users still need **case-level reasons** they can act on.")
        try:
            st.image("/mnt/data/8799ecf1-b603-44d3-9bde-1f36bf9760bf.png", caption="Black-box vs Explainable model flow (static reference)", use_column_width=True)
        except Exception:
            pass

    # Tab 2
    with tab2:
        st.markdown("### 2) Explanations (Global • Local • Targeted)")
        x_row = st.session_state.get("x_row")
        if x_row is None:
            x_row = feature_inputs(scenario, key_prefix=f"tab2-{scenario}-{model_choice}-{st.session_state['session_uid']}")
        prob = st.session_state.get("prob", predict_proba(bundle, x_row))

        st.subheader("Global Explanation — which factors usually matter?")
        global_bar(global_importance(bundle))

        st.subheader("Local Explanation — why this specific decision?")
        contribs = local_contributions(bundle, x_row, X_bg)
        reasons_card(contribs, top_k=5)

        st.subheader("Targeted Explanation — what would change?")
        if st.button("Suggest minimal change", key=f"cf-btn-{st.session_state['session_uid']}"):
            suggestion = minimal_change_to_flip(bundle, x_row, X_bg, positive_target=(prob < 0.5))
            if suggestion:
                f, newv = suggestion
                tgt = SCENARIOS[scenario]['positive_label'] if prob < 0.5 else SCENARIOS[scenario]['negative_label']
                st.success(f"Change **{f}** to **{newv:.2f}** to reach **{tgt}**.")
            else:
                st.info("No single-parameter change found; multiple changes may be needed.")

    # Tab 3
    with tab3:
        st.markdown("### 3) Stakeholder Views — role-aware explanations")
        x_row = st.session_state.get("x_row")
        if x_row is None:
            x_row = feature_inputs(scenario, key_prefix=f"tab3-{scenario}-{model_choice}-{st.session_state['session_uid']}")
        prob = st.session_state.get("prob", predict_proba(bundle, x_row))
        contribs = local_contributions(bundle, x_row, X_bg)
        stakeholder_panel(stakeholder, prob, contribs, bundle, x_row, X_bg, SCENARIOS[scenario]["positive_label"], SCENARIOS[scenario]["negative_label"])

    # Tab 4
    with tab4:
        st.markdown("### 4) Compliance Lens — EU AI Act (educational demo)")
        compliance_overlay(SCENARIOS[scenario]["risk_category"])

    # Tab 5
    with tab5:
        st.markdown("### 5) Summary & Export")
        x_row = st.session_state.get("x_row")
        if x_row is None:
            x_row = feature_inputs(scenario, key_prefix=f"tab5-{scenario}-{model_choice}-{st.session_state['session_uid']}")
        prob = st.session_state.get("prob", predict_proba(bundle, x_row))
        label = SCENARIOS[scenario]['positive_label'] if prob >= 0.5 else SCENARIOS[scenario]['negative_label']
        contribs = local_contributions(bundle, x_row, X_bg)[:5]
        summary = {
            "scenario": scenario,
            "risk_category": SCENARIOS[scenario]["risk_category"],
            "model": bundle.name,
            "decision": label,
            "confidence": float(max(prob, 1 - prob)),
            "top_reasons": [{"feature": f, "influence": float(v)} for f, v in contribs],
            "human_oversight": True,
            "checklist": COMPLIANCE_TEMPLATE,
        }
        st.json(summary, expanded=False)
        st.download_button("Download AI Accountability Card (JSON)", data=json.dumps(summary, indent=2), file_name="ai_accountability_card.json", mime="application/json")

if __name__ == "__main__":
    main()
