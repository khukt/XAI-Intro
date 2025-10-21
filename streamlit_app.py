# app.py
import math
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import streamlit as st

# Optional libraries (handled gracefully if missing)
try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# -----------------------------
# Configuration & Data
# -----------------------------

st.set_page_config(
    page_title="AI Decision Transparency Dashboard",
    page_icon="🔎",
    layout="wide"
)

SCENARIOS = {
    "Loan Approval": {
        "risk_category": "High Risk (Annex III: Creditworthiness)",
        "features": ["income_k", "debt_ratio", "credit_years", "savings_k", "age"],
        "feature_info": {
            "income_k": {"min": 10, "max": 200, "step": 1, "label": "Monthly Income (k SEK)"},
            "debt_ratio": {"min": 0.05, "max": 0.95, "step": 0.01, "label": "Debt-to-Income Ratio"},
            "credit_years": {"min": 0, "max": 25, "step": 1, "label": "Credit History (years)"},
            "savings_k": {"min": 0, "max": 1000, "step": 5, "label": "Savings (k SEK)"},
            "age": {"min": 18, "max": 80, "step": 1, "label": "Age (years)"}
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
            "humidity": {"min": 0, "max": 100, "step": 1, "label": "Humidity (%)"}
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
            "gap_months": {"min": 0, "max": 60, "step": 1, "label": "Career Gap (months)"}
        },
        "positive_label": "Shortlist",
        "negative_label": "Reject",
    },
}

# Simple compliance checklist (educational)
COMPLIANCE_TEMPLATE = {
    "Article 9: Risk Management": True,
    "Article 10: Data & Governance": True,
    "Article 13: Transparency": True,
    "Article 14: Human Oversight": True,
    "Article 15: Accuracy/Robustness": True,
    "Impact Assessment (internal)": False,  # left as TODO to illustrate gap
}

# -----------------------------
# Synthetic Data Generators
# -----------------------------

def synthesize_dataset(scenario: str, n: int = 1200, seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    info = SCENARIOS[scenario]
    feats = info["features"]

    if scenario == "Loan Approval":
        income = rng.normal(55, 20, n).clip(10, 200)
        debt_ratio = rng.beta(2, 3, n) * 0.9 + 0.05
        credit_years = rng.integers(0, 25, n)
        savings = (rng.exponential(50, n)).clip(0, 1000)
        age = rng.integers(18, 80, n)
        X = pd.DataFrame({
            "income_k": income,
            "debt_ratio": debt_ratio,
            "credit_years": credit_years,
            "savings_k": savings,
            "age": age,
        })
        # Weighted score with noise to form label
        score = (
            0.035 * income
            - 2.6 * debt_ratio
            + 0.08 * credit_years
            + 0.003 * savings
            + 0.01 * np.maximum(age - 21, 0)
            + rng.normal(0, 0.5, n)
        )
        y = (score > 0.4).astype(int)

    elif scenario == "Predictive Maintenance":
        temp = rng.normal(70, 15, n).clip(20, 120)
        vibration = rng.uniform(0, 1, n)
        usage = rng.integers(0, 20000, n)
        operator_flag = rng.integers(0, 2, n)
        humidity = rng.integers(0, 100, n)
        X = pd.DataFrame({
            "temp_c": temp,
            "vibration": vibration,
            "usage_hours": usage,
            "operator_flag": operator_flag,
            "humidity": humidity,
        })
        score = (
            0.03 * (temp - 60)
            + 1.8 * vibration
            + 0.00006 * usage
            + 0.9 * operator_flag
            + 0.01 * (humidity - 40)
            + rng.normal(0, 0.5, n)
        )
        y = (score > 1.5).astype(int)

    else:  # Recruitment Screening
        exp = rng.integers(0, 30, n)
        edu = rng.integers(0, 4, n)  # 0-3
        skills = rng.normal(55, 15, n).clip(0, 100)
        test = rng.normal(60, 18, n).clip(0, 100)
        gap = rng.integers(0, 60, n)
        X = pd.DataFrame({
            "experience_yrs": exp,
            "education_level": edu,
            "skills_score": skills,
            "test_score": test,
            "gap_months": gap,
        })
        score = (
            0.08 * exp
            + 0.4 * edu
            + 0.03 * skills
            + 0.025 * test
            - 0.03 * gap
            + rng.normal(0, 0.6, n)
        )
        y = (score > 6.0).astype(int)

    return X, pd.Series(y, name="label")


# -----------------------------
# Modeling Utilities
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

    # White-box: Logistic Regression
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, solver="lbfgs"))
    ])
    lr_pipe.fit(X_train, y_train)
    lr_auc = roc_auc_score(y_test, lr_pipe.predict_proba(X_test)[:, 1])

    # Black-box: Random Forest
    rf_pipe = Pipeline([
        ("scaler", MinMaxScaler()),  # keep input range normalized for widgets
        ("clf", RandomForestClassifier(n_estimators=160, max_depth=6, random_state=seed))
    ])
    rf_pipe.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf_pipe.predict_proba(X_test)[:, 1])

    bundles = {
        "White-box (Logistic Regression)": ModelBundle(
            name=f"White-box (AUC={lr_auc:.2f})", pipeline=lr_pipe, features=list(X.columns), is_white_box=True
        ),
        "Black-box (Random Forest)": ModelBundle(
            name=f"Black-box (AUC={rf_auc:.2f})", pipeline=rf_pipe, features=list(X.columns), is_white_box=False
        ),
    }
    return bundles

def global_importance(bundle: ModelBundle) -> pd.DataFrame:
    # Return a DF of feature -> importance (normalized)
    clf = bundle.pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        vals = np.abs(clf.coef_).flatten()
    else:
        # fallback to permutation-like from coefficients/weights
        vals = np.ones(len(bundle.features))
    vals = vals / (np.sum(vals) + 1e-9)
    return pd.DataFrame({"feature": bundle.features, "importance": vals}).sort_values("importance", ascending=False)

def predict_proba(bundle: ModelBundle, x_row: pd.DataFrame) -> float:
    p = float(bundle.pipeline.predict_proba(x_row)[0, 1])
    return p

def local_contributions(bundle: ModelBundle, x_row: pd.DataFrame, background: pd.DataFrame) -> List[Tuple[str, float]]:
    """
    Return list of (feature, contribution) where positive pushes toward positive class.
    Uses simple approximations; if SHAP is present, uses TreeExplainer/KernelExplainer.
    """
    try:
        if HAS_SHAP:
            clf = bundle.pipeline.named_steps["clf"]
            scaler = bundle.pipeline.named_steps.get("scaler", None)
            if scaler is not None:
                x_bg = scaler.transform(background[bundle.features])
                x_val = scaler.transform(x_row[bundle.features])
            else:
                x_bg = background[bundle.features].values
                x_val = x_row[bundle.features].values

            if hasattr(clf, "predict_proba"):
                model_fn = lambda a: clf.predict_proba(a)[:, 1]
            else:
                model_fn = lambda a: clf.decision_function(a)

            if hasattr(clf, "apply"):
                explainer = shap.TreeExplainer(clf, x_bg, model_output="probability")
                shap_vals = explainer.shap_values(x_val)[0]
            else:
                explainer = shap.KernelExplainer(model_fn, x_bg[:200])
                shap_vals = explainer.shap_values(x_val, nsamples=100)
            contribs = shap_vals.flatten().tolist()
        else:
            # Approximate: use gradient-like contribs
            clf = bundle.pipeline.named_steps["clf"]
            scaler = bundle.pipeline.named_steps.get("scaler", None)
            x_scaled = x_row[bundle.features].values.astype(float).reshape(1, -1)
            if scaler is not None:
                x_scaled = scaler.transform(x_scaled)

            if hasattr(clf, "coef_"):
                weights = clf.coef_.flatten()
                contribs = (x_scaled.flatten() * weights).tolist()
            elif hasattr(clf, "feature_importances_"):
                # use centered feature * normalized importance as proxy
                bg_mean = background[bundle.features].mean().values
                if scaler is not None:
                    bg_mean = scaler.transform(bg_mean.reshape(1, -1)).flatten()
                contribs = (x_scaled.flatten() - bg_mean) * (clf.feature_importances_ / (clf.feature_importances_.sum() + 1e-9))
                contribs = contribs.tolist()
            else:
                contribs = [0.0] * len(bundle.features)
    except Exception:
        contribs = [0.0] * len(bundle.features)

    pairs = list(zip(bundle.features, contribs))
    # Normalize to sum of magnitudes 1 for display
    denom = sum(abs(v) for _, v in pairs) + 1e-9
    pairs = [(f, float(v / denom)) for f, v in pairs]
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    return pairs

def minimal_change_to_flip(bundle: ModelBundle, x_row: pd.DataFrame, background: pd.DataFrame, positive_target: bool) -> Optional[Tuple[str, float]]:
    """
    Greedy one-feature change to cross 0.5 probability threshold.
    Returns (feature, new_value) or None.
    """
    p = predict_proba(bundle, x_row)
    target = 1.0 if positive_target else 0.0
    if positive_target and p >= 0.5:
        return None
    if (not positive_target) and p < 0.5:
        return None

    # Try nudging each feature within plausible bounds
    features = bundle.features
    best_feat, best_val, best_delta = None, None, 0.0

    for feat in features:
        # Create a small grid between min and max observed in background
        fmin = float(background[feat].quantile(0.05))
        fmax = float(background[feat].quantile(0.95))
        grid = np.linspace(fmin, fmax, 15)

        for val in grid:
            trial = x_row.copy()
            trial.iloc[0][feat] = val
            p_trial = predict_proba(bundle, trial)
            if positive_target:
                delta = p_trial - p
                if p_trial >= 0.5 and delta > best_delta:
                    best_feat, best_val, best_delta = feat, val, delta
            else:
                delta = p - p_trial
                if p_trial < 0.5 and delta > best_delta:
                    best_feat, best_val, best_delta = feat, val, delta

    if best_feat is None:
        return None
    return best_feat, float(best_val)

# -----------------------------
# UI Helpers
# -----------------------------

def header():
    st.markdown("""
# 🔎 AI Decision Transparency Dashboard
**Showcasing Explainable AI for mixed audiences, aligned with EU AI Act themes.**  
- White-box vs Black-box  
- Global, Local, and Targeted explanations  
- Stakeholder-specific views  
- Compliance checklist overlay
    """.strip())

def sidebar_controls() -> Tuple[str, str, str]:
    st.sidebar.markdown("## ⚙️ Controls")
    scenario = st.sidebar.selectbox("Scenario", list(SCENARIOS.keys()), index=0)
    model_choice = st.sidebar.selectbox("Model", ["White-box (Logistic Regression)", "Black-box (Random Forest)"], index=0)
    stakeholder = st.sidebar.selectbox(
        "Stakeholder view",
        ["End User", "Business Decision Maker", "CTO / System Owner", "AI Developer / Auditor", "Field Engineer", "Compliance Officer"],
        index=0
    )
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

def reasons_card(contribs: List[Tuple[str, float]], top_k: int = 3):
    st.markdown("### Top reasons for this decision")
    top = contribs[:top_k]
    # Positive/negative arrow indicators
    for feat, val in top:
        arrow = "⬆️" if val >= 0 else "⬇️"
        strength = abs(val)
        st.write(f"- {feat}: {arrow} (influence: {strength:.2f})")

def global_bar(df_imp: pd.DataFrame, title: str = "Global feature influence"):
    st.markdown(f"### {title}")
    # Simple text bars to avoid extra plotting packages
    max_len = df_imp["importance"].max() + 1e-9
    for _, row in df_imp.iterrows():
        bar = "█" * int(40 * row["importance"] / max_len)
        st.write(f"{row['feature']:>18} | {bar} {row['importance']:.2f}")

def transparency_note(is_white_box: bool):
    t = "White-box" if is_white_box else "Black-box"
    st.info(
        f"**Model Transparency:** {t}. "
        "Even with a transparent (white-box) model, we still need explanations for case-level reasoning, audits, fairness, and user trust."
    )

def stakeholder_panel(role: str, prob: float, contribs: List[Tuple[str, float]], bundle: ModelBundle, x_row: pd.DataFrame, background: pd.DataFrame, pos_label: str, neg_label: str):
    label = pos_label if prob >= 0.5 else neg_label
    gap_demo = 0.021  # demo fairness gap
    stability_demo = 0.97  # demo stability

    if role == "End User":
        st.subheader("End User View")
        st.write(f"**Outcome:** {label} (confidence {max(prob,1-prob):.0%})")
        st.write("**Why:**")
        reasons_card(contribs, top_k=3)
        target = pos_label if label == neg_label else neg_label
        st.write("**How could this change?**")
        suggestion = minimal_change_to_flip(bundle, x_row, background, positive_target=(label == neg_label))
        if suggestion:
            f, newv = suggestion
            st.success(f"If **{f}** became **{newv:.2f}**, the decision would likely change to **{target}**.")
        else:
            st.info("This case already satisfies the target outcome or needs a combination of changes.")
        st.write("**Appeal:** A human review is available if you disagree with this decision.")

    elif role == "Business Decision Maker":
        st.subheader("Business Decision Maker")
        st.write(f"**Outcome:** {label} • **Confidence:** {max(prob,1-prob):.0%}")
        st.write(f"**Transparency score:** 4.2/5  •  **Fairness gap:** {gap_demo*100:.1f}%  •  **Stability:** {stability_demo*100:.0f}%")
        st.write("**Top drivers overall (policy levers):**")
        global_bar(global_importance(bundle))

    elif role == "CTO / System Owner":
        st.subheader("CTO / AI System Owner")
        st.write(f"**Pipeline:** {bundle.name}")
        st.write("**Reproducibility:** model_version `v1.2`, hash `RF_v1.2_2025-03-18`")
        st.write("**Monitoring:** input drift: OK • error budget: OK")
        st.write("**Global vs Local:**")
        global_bar(global_importance(bundle))
        reasons_card(contribs, top_k=5)

    elif role == "AI Developer / Auditor":
        st.subheader("AI / Data Scientist")
        st.write("**Global importance:**")
        global_bar(global_importance(bundle))
        st.write("**Local contributions:**")
        reasons_card(contribs, top_k=8)
        st.caption("Tip: In production you would store full attribution arrays and link to an audit notebook.")

    elif role == "Field Engineer":
        st.subheader("Field Engineer / Operator")
        st.write(f"**Actionable insight for current case:**")
        reasons_card(contribs, top_k=3)
        suggestion = minimal_change_to_flip(bundle, x_row, background, positive_target=(label == neg_label))
        if suggestion:
            f, newv = suggestion
            st.success(f"Adjust **{f}** to **{newv:.2f}** to change the outcome.")
        else:
            st.info("No single-parameter change flips the outcome; escalate to manual review.")

    else:  # Compliance Officer
        st.subheader("Compliance / Risk")
        st.write("**EU AI Act alignment (educational demo):**")
        risk = st.session_state.get("risk_category", "High Risk")
        st.write(f"- **Risk Category:** {risk}")
        checklist = dict(COMPLIANCE_TEMPLATE)
        # Derive transparency from availability of local/global explanations
        checklist["Article 13: Transparency"] = True
        checklist["Impact Assessment (internal)"] = False  # show a gap
        for k, v in checklist.items():
            mark = "✅" if v else "⚠️"
            st.write(f"{mark} {k}")
        st.caption("Note: This simplified checklist is for lecture purposes only, not legal advice.")

def compliance_overlay(risk_category: str):
    st.markdown("---")
    st.markdown("### 🧩 Compliance Lens (EU AI Act — educational)")
    st.write(f"**Risk Tier:** {risk_category}")
    checks = dict(COMPLIANCE_TEMPLATE)
    checks["Article 13: Transparency"] = True
    for k, v in checks.items():
        mark = "✅" if v else "⚠️"
        st.write(f"{mark} {k}")
    st.caption("This overlay illustrates how explanations map to governance tasks.")

# -----------------------------
# Main App
# -----------------------------

def main():
    header()
    # Ensure unique key namespace per session
    import uuid
    if 'session_uid' not in st.session_state:
        st.session_state['session_uid'] = str(uuid.uuid4())[:8]
    scenario, model_choice, stakeholder = sidebar_controls()
    st.session_state["risk_category"] = SCENARIOS[scenario]["risk_category"]

    # Data and models
    X_bg, y_bg = synthesize_dataset(scenario, n=1200, seed=13)
    bundles = build_models(scenario, seed=11)
    bundle = bundles[model_choice]

    # --- Tabs for the live lecture flow ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1) Black Box Decision",
        "2) Explanations (Global • Local • Targeted)",
        "3) Stakeholder Views",
        "4) Compliance Lens",
        "5) Summary & Export",
    ])

    # ========== Tab 1: Black Box Decision ==========
    with tab1:
        st.markdown("### 1) Black Box Decision")
        st.write("Start with an opaque decision to highlight the need for explanations.")
        x_row = feature_inputs(scenario, key_prefix=f"tab1-{scenario}-{model_choice}-{st.session_state['session_uid']}")
        prob = predict_proba(bundle, x_row)
        decision_card(prob, SCENARIOS[scenario]['positive_label'], SCENARIOS[scenario]['negative_label'])
        transparency_note(bundle.is_white_box)
        st.info("Notice how little you can infer without reasons. Move to the next tab to reveal explanations.")

        # Store for other tabs
        st.session_state["x_row"] = x_row
        st.session_state["prob"] = prob
        st.session_state["bundle_key"] = model_choice

    # ========== Tab 2: Explanations ==========
    with tab2:
        st.markdown("### 2) Explanations (Global • Local • Targeted)")
        x_row = st.session_state.get("x_row")
        if x_row is None:
            x_row = feature_inputs(scenario, key_prefix=f"tab2-{scenario}-{model_choice}-{st.session_state['session_uid']}")
        prob = st.session_state.get("prob", predict_proba(bundle, x_row))

        st.subheader("Global Explanation — which factors usually matter?")
        imp_df = global_importance(bundle)
        global_bar(imp_df)

        st.subheader("Local Explanation — why this specific decision?")
        contribs = local_contributions(bundle, x_row, X_bg)
        reasons_card(contribs, top_k=5)

        st.subheader("Targeted Explanation — answer a specific question")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("Why not the other outcome? (contrastive)"):
                # Show strongest opposing reasons
                opposed = [(f, v) for f, v in contribs if (v < 0 and prob >= 0.5) or (v > 0 and prob < 0.5)]
                opposed = opposed[:3] if opposed else contribs[-3:]
                st.write("**Main opposing reasons:**")
                for f, v in opposed:
                    st.write(f"- {f}: {'⬇️' if v<0 else '⬆️'} (influence {abs(v):.2f})")
        with col_b:
            if st.button("What would change the decision? (counterfactual)"):
                suggestion = minimal_change_to_flip(bundle, x_row, X_bg, positive_target=(prob < 0.5))
                if suggestion:
                    f, newv = suggestion
                    tgt = SCENARIOS[scenario]['positive_label'] if prob < 0.5 else SCENARIOS[scenario]['negative_label']
                    st.success(f"Change **{f}** to **{newv:.2f}** to reach **{tgt}**.")
                else:
                    st.info("No single-parameter change found; multiple changes may be needed.")
        with col_c:
            if st.button("What should each role see? (role-aware)"):
                st.write("Use the next tab to switch stakeholder views and see tailored explanations.")

    # ========== Tab 3: Stakeholder Views ==========
    with tab3:
        st.markdown("### 3) Stakeholder Views — role-aware explanations")
        x_row = st.session_state.get("x_row")
        if x_row is None:
            x_row = feature_inputs(scenario, key_prefix=f"tab2-{scenario}-{model_choice}-{st.session_state['session_uid']}")
        prob = st.session_state.get("prob", predict_proba(bundle, x_row))
        contribs = local_contributions(bundle, x_row, X_bg)
        stakeholder_panel(stakeholder, prob, contribs, bundle, x_row, X_bg, SCENARIOS[scenario]['positive_label'], SCENARIOS[scenario]['negative_label'])

    # ========== Tab 4: Compliance Lens ==========
    with tab4:
        st.markdown("### 4) Compliance Lens — EU AI Act (educational demo)")
        compliance_overlay(SCENARIOS[scenario]["risk_category"])

    # ========== Tab 5: Summary & Export ==========
    with tab5:
        st.markdown("### 5) Summary & Export")
        x_row = st.session_state.get("x_row")
        if x_row is None:
            x_row = feature_inputs(scenario, key_prefix=f"tab2-{scenario}-{model_choice}-{st.session_state['session_uid']}")
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
        st.write("**Accountability Summary**")
        st.json(summary, expanded=False)

        st.download_button(
            "Download AI Accountability Card (JSON)",
            data=json.dumps(summary, indent=2),
            file_name="ai_accountability_card.json",
            mime="application/json"
        )

if __name__ == '__main__':
    main()
