# ==============================================================================
# PRODUCTION DEPLOYMENT: CLINICAL RISK ASSESSMENT & COMPREHENSIVE EXPLANATION PORTAL
# ==============================================================================
import os
import io
import warnings
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import shap

# Structural PDF Layout Engines
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

warnings.filterwarnings("ignore")

# Set publication-grade frontend web configs
st.set_page_config(page_title="T2D Risk Assessment Portal", layout="wide")

# ==============================================================================
# 1. INGEST PRODUCTION PIPELINE ASSETS
# ==============================================================================
@st.cache_resource
def load_production_assets():
    bundle_path = "final_t2d_production_pipeline.pkl"
    try:
        return joblib.load(bundle_path)
    except FileNotFoundError:
        st.error(f"Critical Deployment Error: Execution bundle asset missing at {bundle_path}")
        return None

assets = load_production_assets()

if assets:
    FEATURES = assets["features_blueprint"]
    calibrated_pipeline = assets["production_pipeline"]
    THRESHOLD_RAW = assets["optimal_classification_threshold"]
    CALIBRATION_USED = assets["calibration_method_applied"]
    
    # Extract structural components for raw feature conversion and SHAP attributions
    raw_xgb_pipeline = calibrated_pipeline.calibrated_classifiers_[0].estimator
    fitted_preprocessor = raw_xgb_pipeline.named_steps["preprocessor"]
    underlying_xgb_model = raw_xgb_pipeline.named_steps["model"]
    TRANSFORMED_FEATURE_NAMES = fitted_preprocessor.get_feature_names_out()

    # Manual Clinical Label Mappings
    label_map = {
        "age": "Age", "education_years": "Years of Education", "bmi": "Body Mass Index (BMI)",
        "waist_circumference": "Waist Circumference", "waist_hip_ratio": "Waist-to-Hip Ratio",
        "mean_systolic_bp": "Systolic Blood Pressure", "mean_diastolic_bp": "Diastolic Blood Pressure",
        "total_cholesterol": "Total Cholesterol", "triglycerides": "Triglycerides",
        "average_daily_fv_intake": "Daily Fruit & Vegetable Intake", "sedentary_time_minutes_day": "Sedentary Minutes/Day",
        "sex": "Sex", "current_smoking": "Current Smoking Status", "adequate_fv_intake": "Adequate Fruit/Vegetable Intake",
        "vigorous_work_activity": "Vigorous Work Activity", "moderate_work_activity": "Moderate Work Activity",
        "active_transport": "Active Transport", "vigorous_leisure_activity": "Vigorous Leisure Activity",
        "moderate_leisure_activity": "Moderate Leisure Activity", "education_level": "Education Level"
    }

    # Clean the preprocessor output keys to match native clinical variables
    clean_feature_names = []
    for name in TRANSFORMED_FEATURE_NAMES:
        raw_key = name.split("__")[-1]
        if raw_key in label_map:
            clean_feature_names.append(label_map[raw_key])
        elif "marital_status" in raw_key:
            group_id = raw_key.split("_")[-1]
            clean_feature_names.append(f"Marital Status (Group {group_id})")
        else:
            clean_feature_names.append(raw_key.replace("_", " ").title())

    # ==============================================================================
    # 🔍 STREAMLIT SIDEBAR: RESEARCH & METADATA OVERVIEW PANEL
    # ==============================================================================
    with st.sidebar:
        st.markdown("## 📊 Model Metadata & Performance")
        st.markdown("This panel acts as a direct reference link to the methodology and validation metrics established in your thesis.")
        
        st.markdown("### 🧬 Architecture Specs")
        st.markdown(f"**Core Estimator:** Gradient Boosted Trees (XGBoost)")
        st.markdown(f"**Post-Hoc Calibration:** Isotonic Regression")
        st.markdown(f"**Screening Threshold:** {THRESHOLD_RAW * 100:.1f}%")
        st.markdown(f"**Validation Framework:** 5-Fold Cross-Validation Matrix")
        
        st.markdown("### 📉 Validation Metrics")
        # Direct mirroring of the out-of-fold benchmark metrics generated during pipeline training
        metrics_data = {
            "Metric": ["AUROC", "AUPRC Score", "Brier Score Loss", "Sensitivity (Recall)", "Specificity"],
            "Value": ["0.783", "0.199", "0.059", "61.8%", "78.8%"]
        }
        st.table(pd.DataFrame(metrics_data))
        st.markdown("---")
        st.caption("Deployment Node Build: v1.0.0 (July 2026)")

    # ==============================================================================
    # 2. APPLICATION INTERFACE HEADER & DIAGNOSTIC DISCLAIMER
    # ==============================================================================
    st.title("🔬 Type 2 Diabetes Clinical Risk Assessment Portal")
    
    st.info(
        "**Clinical Guidance Notice:** This tool provides an estimate of Type 2 Diabetes risk based on "
        "routinely collected clinical information. It is intended to support—not replace—clinical "
        "judgement and should be interpreted alongside patient history, examination findings, and confirmatory diagnostic testing."
    )
    st.markdown("---")

    # ==============================================================================
    # 3. PATIENT DATA INGESTION LAYOUT (THREE-COLUMN FRAMEWORK)
    # ==============================================================================
    st.subheader("📋 Patient Clinical Metrics Intake")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🧬 Biomarker Profiles")
        age = st.number_input("Age (Years)", min_value=1, max_value=120, value=48)
        bmi = st.number_input("Body Mass Index (BMI, kg/m²)", min_value=10.0, max_value=60.0, value=29.7, step=0.1)
        waist_circumference = st.number_input("Waist Circumference (cm)", min_value=40.0, max_value=180.0, value=96.0, step=0.1)
        waist_hip_ratio = st.number_input("Waist-to-Hip Ratio (WHR)", min_value=0.5, max_value=1.5, value=0.92, step=0.01)
        mean_systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=70, max_value=250, value=140)
        mean_diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=150, value=88)
        total_cholesterol = st.number_input("Total Cholesterol (mmol/L)", min_value=1.0, max_value=15.0, value=5.2, step=0.1)
        triglycerides = st.number_input("Triglycerides (mmol/L)", min_value=0.1, max_value=20.0, value=1.8, step=0.1)

    with col2:
        st.markdown("### 🏃‍♂️ Behavioral Vectors")
        sex = st.selectbox("Biological Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        current_smoking = st.selectbox("Current Tobacco Smoking Status", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        sedentary_time_minutes_day = st.number_input("Sedentary Time (Minutes/Day)", min_value=0, max_value=1440, value=420)
        average_daily_fv_intake = st.number_input("Fruit/Vegetable Intake (Daily Servings)", min_value=0, max_value=20, value=2)
        adequate_fv_intake = st.selectbox("Meets Adequate Fruit/Vegetable Intake Guidelines", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        vigorous_work_activity = st.selectbox("Engages in Vigorous Work Activity", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        moderate_work_activity = st.selectbox("Engages in Moderate Work Activity", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        active_transport = st.selectbox("Utilizes Active Transport (Walking/Cycling)", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        vigorous_leisure_activity = st.selectbox("Engages in Vigorous Leisure Activity", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        moderate_leisure_activity = st.selectbox("Engages in Moderate Leisure Activity", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    with col3:
        st.markdown("### 📊 Socio-Demographics")
        education_years = st.number_input("Education Background (Total Years)", min_value=0, max_value=30, value=10)
        education_level = st.selectbox("Highest Education Level Categorical", options=[1, 2, 3, 4], 
                                      format_func=lambda x: {1:"No Education", 2:"Primary", 3:"Secondary", 4:"Tertiary"}[x])
        marital_status = st.selectbox("Marital Status Categorical", options=[1, 2, 3, 4], 
                                     format_func=lambda x: {1:"Single", 2:"Married/Cohabiting", 3:"Separated/Divorced/Widowed", 4:"Refused"}[x])

    # ==============================================================================
    # 4. RISK ESTIMATION ENGINE & EXPLANATION PIPELINE
    # ==============================================================================
    st.markdown("---")
    if st.button("🚀 Estimate Type 2 Diabetes Risk", use_container_width=True):
        
        patient_record = {
            "age": age, "education_years": education_years, "bmi": bmi, 
            "waist_circumference": waist_circumference, "waist_hip_ratio": waist_hip_ratio,
            "mean_systolic_bp": mean_systolic_bp, "mean_diastolic_bp": mean_diastolic_bp, 
            "total_cholesterol": total_cholesterol, "triglycerides": triglycerides, 
            "average_daily_fv_intake": average_daily_fv_intake, "sedentary_time_minutes_day": sedentary_time_minutes_day,
            "sex": sex, "current_smoking": current_smoking, "adequate_fv_intake": adequate_fv_intake, 
            "vigorous_work_activity": vigorous_work_activity, "moderate_work_activity": moderate_work_activity, 
            "active_transport": active_transport, "vigorous_leisure_activity": vigorous_leisure_activity, 
            "moderate_leisure_activity": moderate_leisure_activity, "education_level": education_level, 
            "marital_status": marital_status
        }
        
        input_frame = pd.DataFrame([patient_record])[FEATURES]
        
        # Run inference via the pipeline's calibrated probabilities mapping
        calibrated_probability = calibrated_pipeline.predict_proba(input_frame)[0, 1]
        is_elevated_risk = 1 if calibrated_probability >= THRESHOLD_RAW else 0
        
        # Process raw metrics through the fitted preprocessor to support SHAP calculations
        processed_array = fitted_preprocessor.transform(input_frame)
        if hasattr(processed_array, "toarray"):
            processed_array = processed_array.toarray()
            
        # Extract local patient log-odds contribution arrays via tree explainer 
        explainer = shap.TreeExplainer(underlying_xgb_model)
        local_shap_values = explainer.shap_values(processed_array)
        local_base_value = explainer.expected_value
        
        # Map dynamic display string units to match local clinical audit standards
        display_values = processed_array[0].astype(str)
        display_map = {
            "Age": f"{age} years", "Body Mass Index (BMI)": f"{bmi} kg/m²",
            "Waist Circumference": f"{waist_circumference} cm", "Waist-to-Hip Ratio": f"{waist_hip_ratio:.2f}",
            "Systolic Blood Pressure": f"{mean_systolic_bp} mmHg", "Diastolic Blood Pressure": f"{mean_diastolic_bp} mmHg",
            "Total Cholesterol": f"{total_cholesterol:.2f} mmol/L", "Triglycerides": f"{triglycerides:.2f} mmol/L",
            "Years of Education": f"{education_years} years", "Sedentary Minutes/Day": f"{sedentary_time_minutes_day} min/day"
        }
        
        final_display_vector = []
        for i, f_name in enumerate(clean_feature_names):
            if f_name in display_map:
                final_display_vector.append(display_map[f_name])
            elif "Sex" in f_name:
                final_display_vector.append("Male" if sex == 1 else "Female")
            elif "Smoking" in f_name:
                final_display_vector.append("Yes" if current_smoking == 1 else "No")
            else:
                final_display_vector.append("Yes" if processed_array[0][i] == 1 else "No")

        # Compile localized explanation object structure for plotting
        explanation_obj_patient = shap.Explanation(
            values=local_shap_values[0],
            base_values=local_base_value,
            data=np.array(final_display_vector),
            feature_names=clean_feature_names
        )

        # ==============================================================================
        # 5. DISPLAY CLINICAL RESULTS (FIXED 1, 2) - TRAFFIC LIGHT DISPLAY
        # ==============================================================================
        st.markdown("---")
        st.subheader("📊 Predicted Risk Output Summary")
        
        if is_elevated_risk == 1:
            st.markdown(
                f"<div style='background-color:#FDEDEC; border-left: 6px solid #E74C3C; padding: 15px; border-radius: 4px;'>\n"
                f"<span style='font-size: 14px; color:#78281F; text-transform: uppercase; font-weight: bold;'>Estimated Risk of Type 2 Diabetes</span><br>\n"
                f"<span style='font-size: 42px; color:#C0392B; font-weight: bold;'>{calibrated_probability * 100:.1f}%</span>\n"
                f"<span style='font-size: 24px; color:#C0392B; margin-left: 20px; font-weight: bold;'>[ Above Referral Threshold ]</span>\n"
                f"</div>", unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='background-color:#EAF2F8; border-left: 6px solid #2980B9; padding: 15px; border-radius: 4px;'>\n"
                f"<span style='font-size: 14px; color:#1B4F72; text-transform: uppercase; font-weight: bold;'>Estimated Risk of Type 2 Diabetes</span><br>\n"
                f"<span style='font-size: 42px; color:#2980B9; font-weight: bold;'>{calibrated_probability * 100:.1f}%</span>\n"
                f"<span style='font-size: 24px; color:#27AE60; margin-left: 20px; font-weight: bold;'>[ Below Referral Threshold ]</span>\n"
                f"</div>", unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(float(calibrated_probability))
        # FIXED (1): Softened text to reference clean "Referral Threshold"
        st.markdown(
            f"<div style='display: flex; justify-content: space-between; font-size: 12px; color: #566573; margin-top: -10px;'>\n"
            f"<span>0% Baseline Risk</span>\n"
            f"<span style='font-weight: bold; color: #C0392B;'>Referral Threshold = {THRESHOLD_RAW * 100:.1f}%</span>\n"
            f"<span>100% Maximum Risk</span>\n"
            f"</div>", unsafe_allow_html=True
        )
        st.markdown("---")

        # ==============================================================================
        # 6. DYNAMIC CLINICAL INTERPRETATION COGNITIVE BOX
        # ==============================================================================
        shap_summary_df = pd.DataFrame({
            "Feature Name": clean_feature_names,
            "Observed Entry": final_display_vector,
            "Raw Score Weight": local_shap_values[0]
        })
        
        increasing_factors = shap_summary_df[shap_summary_df["Raw Score Weight"] > 0].sort_values(by="Raw Score Weight", ascending=False).head(3)
        lowering_factors = shap_summary_df[shap_summary_df["Raw Score Weight"] < 0].sort_values(by="Raw Score Weight", ascending=True).head(2)

        st.subheader("💡 Patient-Specific Clinical Rationale")
        # FIXED (1): Softened summary copy reference to match clinical naming paths
        risk_status_string = "exceeds the referral threshold boundary." if is_elevated_risk == 1 else "remains safely below the referral threshold boundary."
        st.markdown(f"The patient's estimated risk score of **{calibrated_probability * 100:.1f}%** {risk_status_string}")
        
        c_box_1, c_box_2 = st.columns(2)
        with c_box_1:
            st.markdown("#### 🔺 Top Factors Increasing Risk")
            for _, r in increasing_factors.iterrows():
                st.markdown(f"• **{r['Feature Name']}** ({r['Observed Entry']})")
        with c_box_2:
            st.markdown("#### 🔹 Top Factors Lowering Estimated Risk")
            for _, r in lowering_factors.iterrows():
                st.markdown(f"• **{r['Feature Name']}** ({r['Observed Entry']})")
                
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Core Pathophysiological Impact Summary Table:**")
        combined_factors_df = pd.concat([increasing_factors, lowering_factors])
        combined_factors_df["Clinical Effect Direction"] = np.where(combined_factors_df["Raw Score Weight"] > 0, "▲ Increased risk profile", "▼ Lower estimated risk contribution")
        
        display_table_view = combined_factors_df[["Feature Name", "Observed Entry", "Clinical Effect Direction"]].rename(
            columns={"Feature Name": "Risk Factor Variable", "Observed Entry": "Patient Recorded Value"}
        ).reset_index(drop=True)
        st.dataframe(display_table_view, use_container_width=True, hide_index=True)
        st.markdown("---")

        # ==============================================================================
        # 7. WATERFALL PLOT CORE (FIXED 2)
        # ==============================================================================
        st.subheader("📋 Factors influencing this patient's estimated risk")
        # FIXED (2): Completely natural, jargon-free visual text label mapping
        st.caption("This figure shows how each clinical factor influenced the estimated risk prediction.")
        
        fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
        shap.plots.waterfall(explanation_obj_patient, max_display=10, show=False)
        plt.title("Patient-Specific Metabolic Risk Contributions", fontsize=11, fontweight="bold", pad=15, loc="left")
        plt.xlabel("Influence on Estimated Risk Score", fontsize=9, labelpad=8)
        plt.tight_layout()
        st.pyplot(fig)
        
        # ==============================================================================
        # 8. AUTOMATED PDF REPORT EXPORT ENGINE (FIXED 3)
        # ==============================================================================
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        report_elements = []
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=18, leading=22, textColor=colors.HexColor('#2C3E50'), spaceAfter=15)
        h2_style = ParagraphStyle('SectionHeader', parent=styles['Heading2'], fontSize=12, leading=16, textColor=colors.HexColor('#2C3E50'), spaceBefore=12, spaceAfter=8, keepWithNext=True)
        body_style = ParagraphStyle('ReportBody', parent=styles['Normal'], fontSize=10, leading=14, textColor=colors.HexColor('#34495E'))
        alert_style = ParagraphStyle('AlertText', parent=body_style, fontSize=11, leading=15, fontName='Helvetica-Bold', textColor=colors.HexColor('#C0392B'))
        safe_style = ParagraphStyle('SafeText', parent=body_style, fontSize=11, leading=15, fontName='Helvetica-Bold', textColor=colors.HexColor('#2980B9'))
        
        report_elements.append(Paragraph("Type 2 Diabetes Risk Assessment Report", title_style))
        # FIXED (3): Replaced system jargon with clear academic build timestamp reference
        report_elements.append(Paragraph(f"<b>Generated On:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | <b>Prediction Model Version:</b> 1.0 (July 2026)", body_style))
        report_elements.append(Spacer(1, 15))
        
        report_elements.append(Paragraph("1. Screening Decision Metrics", h2_style))
        risk_category_text = "Above Referral Threshold Boundary" if is_elevated_risk == 1 else "Below Action/Referral Threshold Baseline"
        
        summary_table_data = [
            [Paragraph("<b>Parameter Label</b>", body_style), Paragraph("<b>Observed Operational Assessment</b>", body_style)],
            [Paragraph("Estimated Risk of Type 2 Diabetes", body_style), Paragraph(f"<b>{calibrated_probability * 100:.2f}%</b>", body_style)],
            [Paragraph("Clinical Risk Category Allocation", body_style), Paragraph(risk_category_text, alert_style if is_elevated_risk == 1 else safe_style)]
        ]
        
        t_summary = Table(summary_table_data, colWidths=[200, 300])
        t_summary.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (1,0), colors.HexColor('#ECF0F1')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 6),
            ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#BDC3C7')),
        ]))
        report_elements.append(t_summary)
        report_elements.append(Spacer(1, 15))
        
        report_elements.append(Paragraph("2. Core Risk Driver & Protective Attribution Summary", h2_style))
        patient_table_data = [[Paragraph("<b>Risk Factor Variable</b>", body_style), Paragraph("<b>Patient Recorded Value</b>", body_style), Paragraph("<b>Clinical Effect Direction</b>", body_style)]]
        for _, r in display_table_view.iterrows():
            patient_table_data.append([
                Paragraph(str(r['Risk Factor Variable']), body_style),
                Paragraph(str(r['Patient Recorded Value']), body_style),
                Paragraph(str(r['Clinical Effect Direction']), body_style)
            ])
            
        t_patient = Table(patient_table_data, colWidths=[200, 130, 170])
        t_patient.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (2,0), colors.HexColor('#ECF0F1')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 5),
            ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#E5E7E9')),
        ]))
        report_elements.append(t_patient)
        report_elements.append(Spacer(1, 15))
        
        # Dynamic care recommendations inside the PDF document layout structure
        report_elements.append(Paragraph("3. Recommended Follow-up Care Path", h2_style))
        if is_elevated_risk == 1:
            report_elements.append(Paragraph(
                "This patient is above the referral threshold. Consider confirmatory testing "
                "(HbA1c or fasting plasma glucose) according to local clinical guidelines.", body_style
            ))
        else:
            report_elements.append(Paragraph(
                "The estimated risk is below the referral threshold. Continue routine preventive care "
                "and reassess according to local clinical practice.", body_style
            ))
        report_elements.append(Spacer(1, 15))
        
        report_elements.append(Paragraph(
            "<b>Clinical Disclaimer Notice:</b> This report provides an estimate of Type 2 Diabetes risk "
            "based on routinely collected clinical information. It is intended to support—not replace—clinical "
            "judgement and should be interpreted alongside patient history, examination findings, and confirmatory diagnostic testing.", 
            body_style
        ))
        
        doc.build(report_elements)
        pdf_bytes = pdf_buffer.getvalue()
        pdf_buffer.close()
        
        # ==============================================================================
        # 9. INTEGRATED CLINICAL GUIDELINES BLOCK (FIXED 4)
        # ==============================================================================
        st.markdown("---")
        st.subheader("📋 Recommended Clinical Action Guidance")
        
        # FIXED (4): Conditional logic block altering UI message based on risk tier
        if is_elevated_risk == 1:
            st.error(
                "**Care Path Recommendation:** This patient is above the referral threshold. "
                "Consider confirmatory testing (HbA1c or fasting plasma glucose) according to local clinical guidelines."
            )
        else:
            st.success(
                "**Care Path Recommendation:** The estimated risk is below the referral threshold. "
                "Continue routine preventive care and reassess according to local clinical practice."
            )
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.download_button(
            label="📥 Download Comprehensive Patient Risk PDF Report",
            data=pdf_bytes,
            file_name=f"t2d_risk_assessment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )