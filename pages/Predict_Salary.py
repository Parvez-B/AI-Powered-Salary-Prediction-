import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
import time
from io import BytesIO

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="💼 Salary Predictor – Major Project",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ THEMING ------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    theme_choice = st.radio("Theme", ["Light", "Dark"], index=0)

LIGHT_CSS = """
<style>
:root { --bg: #f6f7fb; --text: #0f172a; }
</style>
"""
DARK_CSS = """
<style>
:root { --bg: #0b1020; --text: #e5e7eb; }
</style>
"""
BASE_CSS = """
<style>
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
}
h1,h2,h3,h4,h5,h6,p,div,span { color: var(--text) }
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem; border-radius: 15px;
    margin-bottom: 1.5rem; text-align: center;
    color: white; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}
.prediction-card {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    padding: 1.5rem; border-radius: 15px; text-align:center;
}
.prediction-amount { font-size:2.4rem; font-weight:700; color:white; }
.model-card {
    position: fixed; top: 16px; right: 16px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    padding: 12px 18px; border-radius: 14px; color: white;
    box-shadow: 0 5px 15px rgba(0,0,0,0.3); z-index: 1000;
}
.model-card h4 { margin: 0; font-size: 16px; font-weight: 700; }
.model-card p { margin: 2px 0 0 0; font-size: 13px; }
.small-muted { font-size: 12px; opacity: 0.8; }
</style>
"""
st.markdown(LIGHT_CSS if theme_choice=="Light" else DARK_CSS, unsafe_allow_html=True)
st.markdown(BASE_CSS, unsafe_allow_html=True)

# ------------------ HELPERS ------------------
def extract_algorithm_name(model_obj):
    """
    Best-effort extraction of the underlying estimator/algorithm name.
    Works for raw estimators, Pipelines, GridSearchCV, etc.
    """
    try:
        # GridSearch/RandomizedSearch
        if hasattr(model_obj, "best_estimator_") and model_obj.best_estimator_ is not None:
            return extract_algorithm_name(model_obj.best_estimator_)
        # Pipeline-like
        if hasattr(model_obj, "named_steps"):
            steps = model_obj.named_steps
            if "model" in steps:
                return extract_algorithm_name(steps["model"])
            # last step fallback
            if len(steps) > 0:
                last_key = list(steps.keys())[-1]
                return extract_algorithm_name(steps[last_key])
        # Composite
        if hasattr(model_obj, "estimator"):
            return extract_algorithm_name(model_obj.estimator)
        return type(model_obj).__name__
    except Exception:
        return "Unknown"

def get_pipeline_final_estimator(model_obj):
    """Return (preprocessor_or_None, final_estimator_or_model)."""
    try:
        if hasattr(model_obj, "best_estimator_") and model_obj.best_estimator_ is not None:
            return get_pipeline_final_estimator(model_obj.best_estimator_)
        if hasattr(model_obj, "named_steps"):
            steps = model_obj.named_steps
            pre = steps.get("preprocess") or steps.get("preprocessor") or None
            final = steps.get("model") or steps.get(list(steps.keys())[-1])
            return pre, final
        if hasattr(model_obj, "estimator"):
            return None, model_obj.estimator
        return None, model_obj
    except Exception:
        return None, model_obj

# ------------------ MODEL LOADING ------------------
@st.cache_resource
def load_model(uploaded=None, show_success=False):
    try:
        if uploaded:
            model = joblib.load(uploaded)
            if show_success:
                st.success("✅ Model uploaded successfully!")
            return model
        model = joblib.load("salary_prediction_pipeline.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Built-in model not found. Ensure 'salary_prediction_pipeline.pkl' exists.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        st.stop()

# ------------------ VALIDATION & INSIGHTS ------------------
def validate_age(age):
    if not (18 <= age <= 100):
        return False, "Age must be 18-100"
    return True, ""

def validate_experience(exp, age):
    if exp < 0:
        return False, "Experience cannot be negative"
    if exp > (age - 16):
        return False, "Experience cannot exceed Age - 16"
    return True, ""

def get_salary_insights(age, exp, edu, job):
    insights=[]
    if exp<2: insights.append("📈 Entry-level – focus on skills and portfolio.")
    elif exp<5: insights.append("🔄 Mid-junior level – strong growth potential.")
    elif exp<10: insights.append("⭐ Experienced – competitive market position.")
    else: insights.append("🏆 Senior expert – premium salaries likely.")
    if edu=="PhD": insights.append("🎓 PhD adds significant value for R&D/lead roles.")
    if edu=="Master": insights.append("📚 Master's provides an edge in specialized roles.")
    if (age > 18) and (exp/(age-18) > 0.8): insights.append("🚀 Excellent experience-to-age ratio.")
    return insights

def career_recommendation(age, exp, edu, job, pred):
    recs=[]
    if pred < 400000 and edu in ["High School","Bachelor"]:
        recs.append("🎓 Consider a Master's degree to access higher-paying roles.")
    if "Developer" in job and exp >= 5:
        recs.append("🚀 Transition towards Senior/Lead Developer responsibilities.")
    if "Data Scientist" in job and exp < 3:
        recs.append("📊 Build ML portfolio (Kaggle, end-to-end projects) to accelerate growth.")
    if exp > 10 and pred < 800000:
        recs.append("💡 Explore management tracks or niche certifications (Cloud/ML/Architecture).")
    if not recs:
        recs.append("✅ You're on a good trajectory. Keep sharpening in-demand skills.")
    return recs

# ------------------ PDF EXPORT ------------------
# def generate_pdf_report(age, gender, edu, job, exp, prediction, insights, algo_name, model_type):
#     from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
#     from reportlab.lib.styles import getSampleStyleSheet
#     from reportlab.lib.pagesizes import A4
#     from reportlab.lib import colors

#     buf = BytesIO()
#     doc = SimpleDocTemplate(buf, pagesize=A4)
#     styles = getSampleStyleSheet()
#     story = []

#     story.append(Paragraph("Salary Prediction Report", styles["Title"]))
#     story.append(Spacer(1, 8))
#     story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
#     story.append(Paragraph(f"<b>Model:</b> {model_type} — {algo_name}", styles["Normal"]))
#     story.append(Spacer(1, 12))

#     data = [
#         ["Age", age],
#         ["Gender", gender],
#         ["Education", edu],
#         ["Job Title", job],
#         ["Experience (years)", exp],
#         ["Predicted Monthly Salary", f"₹ {prediction:,.0f}"],
#     ]
#     tbl = Table(data, hAlign="LEFT")
#     tbl.setStyle(TableStyle([
#         ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
#         ('GRID',(0,0),(-1,-1),0.5,colors.grey),
#         ('FONTNAME',(0,0),(-1,-1),'Helvetica'),
#         ('PADDING',(0,0),(-1,-1),6),
#     ]))
#     story.append(tbl)
#     story.append(Spacer(1, 12))

#     story.append(Paragraph("<b>Career Insights</b>", styles["Heading2"]))
#     for i in insights:
#         story.append(Paragraph(f"- {i}", styles["Normal"]))

#     doc.build(story)
#     buf.seek(0)
#     return buf

def generate_pdf_report(age, gender, edu, job, exp, prediction, insights, model_type):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Salary Prediction Report", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Paragraph(f"<b>Model:</b> {model_type}", styles["Normal"]))   # 🚀 only model_type now
    story.append(Spacer(1, 12))

    data = [
        ["Age", age],
        ["Gender", gender],
        ["Education", edu],
        ["Job Title", job],
        ["Experience (years)", exp],
        ["Predicted Monthly Salary", f"₹ {prediction:,.0f}"],
    ]
    tbl = Table(data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey),
        ('FONTNAME',(0,0),(-1,-1),'Helvetica'),
        ('PADDING',(0,0),(-1,-1),6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Career Insights</b>", styles["Heading2"]))
    for i in insights:
        story.append(Paragraph(f"- {i}", styles["Normal"]))

    doc.build(story)
    buf.seek(0)
    return buf



# ------------------ SHAP UTILS ------------------
def compute_shap_for_input(model, X_df):
    """
    Return (fig, message). Tries to compute SHAP waterfall for a single-row X_df.
    Handles pipelines by extracting final estimator and transforming inputs if preprocessor exists.
    """
    pre, final_est = get_pipeline_final_estimator(model)

    # Transform X if we have a preprocessor
    X_trans = None
    feature_names = None
    try:
        if pre is not None:
            X_trans = pre.transform(X_df)
            # Attempt to get output feature names (OneHotEncoder etc.)
            if hasattr(pre, "get_feature_names_out"):
                feature_names = pre.get_feature_names_out()
        else:
            X_trans = X_df.values
            feature_names = list(X_df.columns)
    except Exception:
        # Fallback: try model.predict on raw if transform fails
        X_trans = X_df.values
        feature_names = list(X_df.columns)

    # Try SHAP with different explainers
    try:
        # Tree-based
        explainer = shap.TreeExplainer(final_est)
        sv = explainer(X_trans)
    except Exception:
        try:
            # Model-agnostic
            explainer = shap.Explainer(final_est, X_trans)
            sv = explainer(X_trans)
        except Exception as e:
            return None, f"SHAP not supported for this model: {e}"

    try:
        shap_values_row = sv[0]
        # Plot waterfall
        fig = plt.figure(figsize=(8, 6))
        shap.plots.waterfall(shap_values_row, show=False)
        return fig, "ok"
    except Exception as e:
        # Fallback to bar summary
        try:
            fig = plt.figure(figsize=(8, 6))
            shap.plots.bar(sv, max_display=12, show=False)
            return fig, "ok"
        except Exception as e2:
            return None, f"Failed to render SHAP plot: {e2}"

# ------------------ MAIN APP ------------------
def main():
    # HEADER
    st.markdown(
        '<div class="main-header"><h1>💼 AI-Powered Career and Salary Prediction System</h1>'
        '<p class="small-muted">Predict • Analyze • Explain • Recommend • Report</p></div>',
        unsafe_allow_html=True
    )

    # SIDEBAR: Model + Template
    with st.sidebar:
        st.markdown("### 📦 Model")
        use_demo = st.checkbox("Use built-in model", True)
        uploaded_model_file = None
        if not use_demo:
            uploaded_model_file = st.file_uploader("Upload Model (.pkl/.joblib)", type=["pkl", "joblib"])

        # Load model
        if not use_demo and uploaded_model_file is not None:
            model = load_model(uploaded_model_file, show_success=True)
        else:
            model = load_model()

        # Template CSV
        st.markdown("### 📄 Template")
        template = pd.DataFrame({
            "Age": [30],
            "Gender": ["Male"],
            "Education Level": ["Bachelor"],
            "Job Title": ["Developer"],
            "Years of Experience": [5],
        })
        st.download_button(
            "Download CSV Template",
            template.to_csv(index=False),
            file_name="input_template.csv",
            mime="text/csv"
        )

    # Floating model info card
    model_type = "Built-in Model" if (use_demo or uploaded_model_file is None) else "Uploaded Model"
    algo_name = extract_algorithm_name(model)
    # st.markdown(
        # f"""
        # <div class="model-card">
        #     <h4>📦 {model_type}</h4>
        #     <p>🔍 Algorithm: <b>{algo_name}</b></p>
        # </div>
        # """,
        # unsafe_allow_html=True
    # )

    # ------------------ TABS ------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Predict", " ", "🧪 What-If", " ", "🎯 Recommendations"
    ])

    # ---------- TAB 1: SINGLE PREDICTION ----------
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            edu = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"], index=1)
            job = st.selectbox("Job Title", ["Developer", "Data Scientist", "Manager", "Analyst", "Engineer"])
            exp = st.slider("Years of Experience", min_value=0, max_value=50, value=5)

        with col2:
            valid_age, msg1 = validate_age(age)
            valid_exp, msg2 = validate_experience(exp, age)
            if not valid_age: st.error(msg1)
            if not valid_exp: st.error(msg2)
            if valid_age and valid_exp:
                st.markdown("### 🔍 Career Insights")
                for tip in get_salary_insights(age, exp, edu, job):
                    st.info(tip)

        if st.button("🔮 Predict Salary", use_container_width=True):
            if not (valid_age and valid_exp):
                st.error("❌ Fix input errors before predicting.")
            else:
                with st.spinner("🤖 Analyzing..."):
                    time.sleep(1)
                    inp = pd.DataFrame({
                        "Age": [age],
                        "Gender": [gender],
                        "Education Level": [edu],
                        "Job Title": [job],
                        "Years of Experience": [exp],
                    })
                    try:
                        pred = float(model.predict(inp)[0])
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        st.stop()

                    st.markdown(
                        f"<div class='prediction-card'><p>Predicted Monthly Salary</p>"
                        f"<div class='prediction-amount'>₹ {pred:,.0f}</div></div>",
                        unsafe_allow_html=True
                    )

                    c1, c2 = st.columns(2)
                    # with c1:
                    #     hourly = pred / (52 * 40)  # approx 40h/week
                    #     st.metric("Estimated Hourly", f"₹ {hourly:,.2f}")
                    # with c2:
                    #     monthly = pred / 12
                    #     st.metric("Estimated Monthly", f"₹ {monthly:,.0f}")

                    # CSV download
                    result_df = pd.DataFrame([{
                        "Prediction_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Age": age,
                        "Gender": gender,
                        "Education": edu,
                        "Job_Title": job,
                        "Experience": exp,
                        "Predicted_Annual_Salary_INR": int(pred),
                        "Model_Type": model_type,
                        
                    }])
                    st.download_button(
                        "📥 Download Prediction CSV",
                        result_df.to_csv(index=False),
                        file_name=f"salary_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                    # PDF report
                    insight_list = get_salary_insights(age, exp, edu, job)
                    pdf_buf = generate_pdf_report(age, gender, edu, job, exp, pred, insight_list, model_type)
                    st.download_button(
                        "🧾 Download PDF Report",
                        data=pdf_buf,
                        file_name="salary_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

    # ---------- TAB 2: BATCH PREDICTIONS ----------
    with tab2:
        st.markdown("### 📂 Upload CSV for Batch Prediction")
        st.caption("Required columns: Age, Gender, Education Level, Job Title, Years of Experience")
        batch_file = st.file_uploader("Upload CSV", type="csv", key="batch_csv")
        if batch_file is not None:
            try:
                df_in = pd.read_csv(batch_file)
                preds = model.predict(df_in)
                df_out = df_in.copy()
                df_out["Predicted_Annual_Salary_INR"] = preds.astype(int)
                st.dataframe(df_out, use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 Download Predictions",
                    df_out.to_csv(index=False),
                    file_name="salary_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.success("✅ Batch prediction completed.")
            except Exception as e:
                st.error(f"❌ Failed to run batch prediction: {e}")

    # ---------- TAB 3: WHAT-IF ANALYSIS ----------
    with tab3:
        st.markdown("### 🧪 Experiment with Inputs to See Impact")
        c1, c2, c3 = st.columns(3)
        with c1:
            base_age = st.number_input("Base Age", 18, 100, 28, key="wa_base_age")
            base_exp = st.slider("Base Experience", 0, 40, 3, key="wa_base_exp")
        with c2:
            base_edu = st.selectbox("Base Education", ["High School","Bachelor","Master","PhD"], index=1, key="wa_base_edu")
            base_job = st.selectbox("Base Job", ["Developer","Data Scientist","Manager","Analyst","Engineer"], key="wa_base_job")
        with c3:
            delta_exp = st.slider("Add Experience (+ years)", 0, 10, 2, key="wa_delta_exp")
            upgraded_edu = st.selectbox("Upgrade Education", ["No Change","Master","PhD"], key="wa_upg_edu")

        base_inp = pd.DataFrame({
            "Age":[base_age], "Gender":["Male"],
            "Education Level":[base_edu], "Job Title":[base_job],
            "Years of Experience":[base_exp]
        })
        try:
            base_pred = float(model.predict(base_inp)[0])
        except Exception as e:
            st.error(f"Base scenario prediction failed: {e}")
            base_pred = np.nan

        # Scenario:
        new_edu = base_edu if upgraded_edu=="No Change" else upgraded_edu
        scen_inp = pd.DataFrame({
            "Age":[base_age + (1 if delta_exp>0 else 0)],  # assume time passes with experience
            "Gender":["Male"], "Education Level":[new_edu],
            "Job Title":[base_job], "Years of Experience":[base_exp + delta_exp]
        })
        try:
            scen_pred = float(model.predict(scen_inp)[0])
        except Exception as e:
            st.error(f"Scenario prediction failed: {e}")
            scen_pred = np.nan

        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Base Salary", f"₹ {base_pred:,.0f}" if not np.isnan(base_pred) else "—")
        with colB:
            st.metric("What-If Salary", f"₹ {scen_pred:,.0f}" if not np.isnan(scen_pred) else "—")
        with colC:
            if not (np.isnan(base_pred) or np.isnan(scen_pred)):
                st.metric("Change", f"₹ {(scen_pred - base_pred):,.0f}")

        # Simple bar compare
        if not (np.isnan(base_pred) or np.isnan(scen_pred)):
            df_plot = pd.DataFrame({
                "Scenario":["Base","What-If"],
                "Predicted Monthly Salary":[base_pred, scen_pred]
            })
            fig = px.bar(df_plot, x="Scenario", y="Predicted Monthly Salary", text="Predicted Monthly Salary",
                         title="What-If Comparison")
            st.plotly_chart(fig, use_container_width=True)

    # ---------- TAB 4: EXPLAINABILITY (SHAP) ----------
    with tab4:
        st.markdown("### 📊 Model Explainability (SHAP)")
        st.caption("Explains contribution of features to a single prediction (best effort; works best with tree/linear models).")
        ex_age = st.number_input("Age (Explain)", 18, 100, 30)
        ex_gender = st.selectbox("Gender (Explain)", ["Male","Female"])
        ex_edu = st.selectbox("Education (Explain)", ["High School","Bachelor","Master","PhD"], index=1)
        ex_job = st.selectbox("Job (Explain)", ["Developer","Data Scientist","Manager","Analyst","Engineer"])
        ex_exp = st.slider("Experience (Explain)", 0, 50, 5)

        X_explain = pd.DataFrame({
            "Age":[ex_age], "Gender":[ex_gender],
            "Education Level":[ex_edu], "Job Title":[ex_job],
            "Years of Experience":[ex_exp]
        })

        with st.spinner("Computing SHAP..."):
            fig, msg = compute_shap_for_input(model, X_explain)
        if fig is not None:
            st.pyplot(fig, clear_figure=True)
        else:
            st.warning(msg)

    # ---------- TAB 5: CAREER RECOMMENDATIONS ----------
    with tab5:
        st.markdown("### 🎯 Personalized Career Suggestions")
        r_age = st.number_input("Age ", 18, 100, 28, key="rec_age")
        r_exp = st.slider("Experience ", 0, 40, 3, key="rec_exp")
        r_edu = st.selectbox("Education ", ["High School","Bachelor","Master","PhD"], key="rec_edu")
        r_job = st.selectbox("Job ", ["Developer","Data Scientist","Manager","Analyst","Engineer"], key="rec_job")
        X_rec = pd.DataFrame({
            "Age":[r_age], "Gender":["Male"], "Education Level":[r_edu],
            "Job Title":[r_job], "Years of Experience":[r_exp]
        })
        try:
            rec_pred = float(model.predict(X_rec)[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            rec_pred = np.nan

        st.metric("Predicted Monthly Salary", f"₹ {rec_pred:,.0f}" if not np.isnan(rec_pred) else "—")

        if not np.isnan(rec_pred):
            recs = career_recommendation(r_age, r_exp, r_edu, r_job, rec_pred)
            for r in recs:
                st.success(r)
        else:
            st.info("Provide valid inputs to see recommendations.")

# ------------------ RUN ------------------
if __name__=="__main__":
    main()
