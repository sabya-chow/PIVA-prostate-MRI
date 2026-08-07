"""PIVA quantitative prostate MRI patient-report demonstration."""

# Import Path so model and asset paths remain stable regardless of launch folder.
from pathlib import Path

# Import NumPy and pandas for chart and table preparation.
import numpy as np
import pandas as pd

# Import Plotly for polished, presentation-friendly interactive charts.
import plotly.graph_objects as go

# Import Streamlit for the local dashboard and downloadable report.
import streamlit as st

# Import the real physics-informed model, prepared cases, inference, and report builder.
from piva_core import B_GRID, COMPARTMENTS, PARAMETER_NAMES, PARAMETER_SHORT_NAMES, TE_GRID, analyze_patient, build_demo_patients, build_html_report, build_reference_profile, format_percentile, load_model


# Configure a wide presentation canvas before creating any Streamlit elements.
st.set_page_config(page_title="PIVA Patient Report", page_icon="◈", layout="wide")

# Add a restrained clinical-product visual system without external fonts or network assets.
st.markdown(
    """
    <style>
    .stApp {background: #f6f8fb; color: #132238;}
    .block-container {max-width: 1240px; padding-top: 1.5rem; padding-bottom: 3rem;}
    .hero {background: linear-gradient(120deg,#0d2744,#145a78); padding: 28px 32px; border-radius: 18px; color: white; margin-bottom: 20px;}
    .hero h1 {margin: 0; font-size: 2.15rem;} .hero p {opacity: .86; margin: 8px 0 0; font-size: 1.02rem;}
    .eyebrow {font-size: .78rem; letter-spacing: .12em; text-transform: uppercase; color: #75d5c6; font-weight: 700;}
    .patient-card {background:white;border:1px solid #dce5ee;border-radius:14px;padding:18px 20px;margin-bottom:12px;}
    .section-title {font-size:1.25rem;font-weight:750;color:#102a43;margin-top:24px;margin-bottom:8px;}
    .explain {background:#edf4f8;border-left:4px solid #167d8d;padding:13px 15px;border-radius:4px;color:#294a5f;}
    div[data-testid="stMetric"] {background:white;border:1px solid #dce5ee;padding:14px 16px;border-radius:14px;box-shadow:0 3px 12px rgba(18,46,70,.04);}
    .disclaimer {background:#fff6df;border:1px solid #efd59a;border-radius:12px;padding:13px 16px;color:#6f5315;margin-top:20px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Define stable local paths for the one-time exported model checkpoint.
APP_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = APP_DIR / "checkpoints" / "piva_full_ilr_demo.pt"

# Cache model loading so Streamlit reruns do not repeatedly deserialize weights.
@st.cache_resource
def cached_model():
    return load_model(CHECKPOINT_PATH)


# Cache the model-matched reference population so every patient comparison uses one stable baseline.
@st.cache_resource
def cached_reference():
    model, _ = cached_model()
    return build_reference_profile(model, n_voxels=5000)


# Render the product header used during the capstone pitch.
st.markdown(
    """<div class="hero"><div class="eyebrow">MSADS Capstone · Product Demonstration</div>
    <h1>PIVA Quantitative Patient Report</h1>
    <p>From noisy prostate MRI voxel signals to physical tissue estimates, posterior probabilities, and an understandable report.</p></div>""",
    unsafe_allow_html=True,
)

# Stop with a direct recovery instruction if the one-time checkpoint has not been exported.
if not CHECKPOINT_PATH.exists():
    st.error("The demonstration checkpoint is missing. Run: `.venv/bin/python demo_app/train_demo_model.py`")
    st.stop()

# Load prepared cases and give the presenter one simple, dependable selection control.
patients = build_demo_patients()
with st.sidebar:
    st.header("Demonstration case")
    selected_label = st.selectbox("Select a prepared patient", list(patients), index=2)
    patient = patients[selected_label]
    st.caption("All cases are synthetic and generated from the capstone notebook's physiological parameter ranges.")
    n_mc = st.select_slider("Posterior samples", options=[100, 200, 300, 500], value=300)
    run_analysis = st.button("Run PIVA analysis", type="primary", width="stretch")
    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("1. Validate 16 measurements\n2. Encode latent distribution\n3. Sample full covariance\n4. Decode tissue factors\n5. Build patient report")

# Show case metadata before analysis so the audience understands what enters the pipeline.
st.markdown('<div class="section-title">Selected patient</div>', unsafe_allow_html=True)
st.markdown(
    f"""<div class="patient-card"><b>{patient.patient_id}</b> · {patient.label}<br>
    Age {patient.age} &nbsp; | &nbsp; PSA {patient.psa:.1f} ng/mL &nbsp; | &nbsp; PSA density {patient.psa_density:.2f}<br>
    <span style="color:#5f7386">{patient.description}</span></div>""",
    unsafe_allow_html=True,
)

# Require one deliberate click during the pitch, while retaining results across harmless widget reruns.
if run_analysis:
    with st.status("Running the PIVA pipeline…", expanded=True) as status:
        st.write("✓ Validated 10 voxels × 16 quantitative measurements")
        model, checkpoint = cached_model()
        st.write("✓ Loaded Full-ILR Cholesky posterior")
        reference_profile = cached_reference()
        st.write(f"✓ Matched against {reference_profile['n']:,} synthetic reference voxels")
        result = analyze_patient(model, patient, n_mc=n_mc, reference_profile=reference_profile)
        st.write(f"✓ Drew {n_mc} posterior samples and reconstructed the MRI signal")
        st.write("✓ Generated quantitative patient interpretation")
        status.update(label="PIVA analysis complete", state="complete", expanded=False)
    st.session_state["result"] = result
    st.session_state["patient_label"] = selected_label

# Avoid displaying stale results when the presenter changes to a different case.
result = st.session_state.get("result") if st.session_state.get("patient_label") == selected_label else None
if result is None:
    st.info("Select a prepared case and click **Run PIVA analysis** to generate the dashboard and patient report.")
    st.stop()

# Present the four decision-level outputs before deeper technical charts.
st.markdown('<div class="section-title">Quantitative overview</div>', unsafe_allow_html=True)
metric_columns = st.columns(4)
metric_columns[0].metric("Concern score", f"{result['concern_score']:.0f}/100", result["concern_category"])
metric_columns[1].metric("Quantitative confidence", result["confidence_label"], f"{result['confidence_score']:.0f}/100")
metric_columns[2].metric("Signal reconstruction", f"{result['reconstruction_score']:.0f}%", "Physics consistency")
metric_columns[3].metric("Voxels analyzed", patient.signals.shape[0], "16 signals each")
st.markdown('<div class="explain"><b>Your position at a glance:</b> this score summarizes how strongly three tissue indicators differ from the model-matched synthetic reference cohort. The report below shows exactly where each measurement sits and how certain it is. It is not a cancer probability.</div>', unsafe_allow_html=True)

# Separate the dashboard around the questions a patient is most likely to ask.
patient_tab, reference_tab, evidence_tab, method_tab = st.tabs(["Your report", "Where you stand", "Inside the MRI signal", "How it was calculated"])

with patient_tab:
    # Lead with a single interpretation and a visual score dial instead of requiring table reading.
    left_column, right_column = st.columns([0.85, 1.35])
    with left_column:
        st.subheader("Quantitative position")
        gauge_color = "#277a5a" if result["concern_score"] < 35 else "#d19a2a" if result["concern_score"] < 65 else "#b94b45"
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["concern_score"],
            number={"suffix": "/100", "font": {"size": 34}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": gauge_color}, "steps": [
                {"range": [0, 35], "color": "#dcefe7"}, {"range": [35, 65], "color": "#faedc8"}, {"range": [65, 100], "color": "#f3d7d5"}
            ], "threshold": {"line": {"color": "#173f5f", "width": 3}, "value": result["concern_score"]}},
        ))
        gauge.update_layout(height=250, margin=dict(l=25, r=25, t=25, b=5))
        st.plotly_chart(gauge, width="stretch", config={"displayModeBar": False})
        st.caption("0–34 within range · 35–64 indeterminate · 65–100 elevated")

    with right_column:
        st.subheader("Patient-friendly interpretation")
        category_text = {
            "Within expected range": "The estimated tissue pattern is broadly consistent with the synthetic reference cohort.",
            "Indeterminate": "The analysis contains mixed quantitative findings, with no single indicator dominating the report.",
            "Elevated": "Several estimated tissue properties differ from the synthetic reference pattern, led by epithelial composition and diffusion.",
        }[result["concern_category"]]
        st.markdown(f"### {result['concern_category']}")
        st.write(category_text)
        st.write(f"The model explained the observed signal with a **{result['reconstruction_score']:.0f}% reconstruction score** and reported **{result['confidence_label'].lower()} quantitative confidence**.")
        st.caption("Interpretation is derived from synthetic-reference thresholds for this academic demonstration.")

    # Name the three strongest deviations so the patient immediately understands what drove the result.
    st.subheader("What stands out most")
    driver_columns = st.columns(3)
    for column, index in zip(driver_columns, result["driver_indices"]):
        percentile = result["percentiles"][index]
        direction = "higher" if percentile >= 50 else "lower"
        column.metric(PARAMETER_NAMES[index], format_percentile(percentile), f"{direction} than reference")
        column.caption(str(result["range_status"][index]))

    # Show estimated tissue composition as the easiest physical interpretation of the hidden factors.
    st.subheader("Estimated tissue composition")
    fractions = result["estimates"][6:9] * 100
    composition = go.Figure(go.Bar(
        x=fractions, y=list(COMPARTMENTS), orientation="h",
        marker_color=["#176b87", "#45a29e", "#d7a437"],
        text=[f"{value:.1f}%" for value in fractions], textposition="inside",
    ))
    composition.update_layout(height=260, margin=dict(l=5, r=10, t=10, b=20), xaxis=dict(range=[0, 100], title="Estimated fraction (%)"), showlegend=False)
    st.plotly_chart(composition, width="stretch", config={"displayModeBar": False})

    # Build a readable table of the nine hidden physical factors and their posterior intervals.
    st.subheader("Quantitative tissue factors")
    names = [f"D · {name}" for name in COMPARTMENTS] + [f"T2 · {name}" for name in COMPARTMENTS] + [f"v · {name}" for name in COMPARTMENTS]
    units = ["×10⁻³ mm²/s"] * 3 + ["ms"] * 3 + ["fraction"] * 3
    factor_table = pd.DataFrame({
        "Measurement": names,
        "Estimate": result["estimates"],
        "95% lower": result["lower"],
        "95% upper": result["upper"],
        "Reference low": result["reference_lower"],
        "Reference high": result["reference_upper"],
        "Percentile": result["percentiles"],
        "Position": result["range_status"],
        "Unit": units,
    })
    st.dataframe(factor_table.style.format({"Estimate": "{:.3f}", "95% lower": "{:.3f}", "95% upper": "{:.3f}", "Reference low": "{:.3f}", "Reference high": "{:.3f}", "Percentile": "{:.0f}th"}), hide_index=True, width="stretch")

    # Finish with a short action-oriented reading guide rather than a diagnostic recommendation.
    st.subheader("How to discuss this report")
    st.markdown("- Which measurements fall outside the synthetic reference band?\n- Which estimates have wide uncertainty intervals?\n- Does the reconstructed signal closely follow the observed measurements?\n- Would the same pattern remain after clinical-cohort validation?")

    # Generate a self-contained report and offer one-click download during the presentation.
    report_html = build_html_report(patient, result)
    st.download_button("Download detailed patient report", report_html, file_name=f"{patient.patient_id}_PIVA_report.html", mime="text/html", type="primary")

with reference_tab:
    # Plot every parameter on a percentile scale so unlike units can be compared in one visual.
    st.subheader("Your position within the synthetic reference cohort")
    st.caption(f"Compared with {result['reference_n']:,} model-matched synthetic voxels. The green band is the central 80% reference range.")
    colors = ["#b94b45" if value < 10 or value > 90 else "#d19a2a" if value < 20 or value > 80 else "#277a5a" for value in result["percentiles"]]
    percentile_chart = go.Figure()
    for row_index in range(9):
        percentile_chart.add_shape(type="rect", x0=10, x1=90, y0=row_index - 0.28, y1=row_index + 0.28, fillcolor="#dcefe7", line_width=0, layer="below")
    percentile_chart.add_trace(go.Scatter(
        x=result["percentiles"], y=list(PARAMETER_SHORT_NAMES), mode="markers+text",
        marker=dict(size=15, color=colors, line=dict(color="white", width=2)),
        text=[format_percentile(value).replace(" percentile", "") for value in result["percentiles"]], textposition="middle right", showlegend=False,
    ))
    percentile_chart.add_vline(x=50, line_dash="dot", line_color="#6c7f90")
    percentile_chart.update_layout(height=480, margin=dict(l=10, r=45, t=10, b=35), xaxis=dict(range=[0, 105], title="Reference percentile"), yaxis=dict(autorange="reversed"))
    st.plotly_chart(percentile_chart, width="stretch", config={"displayModeBar": False})
    st.markdown('<div class="explain"><b>Percentile does not mean probability of disease.</b> A 95th-percentile measurement means it is higher than 95% of this synthetic reference cohort.</div>', unsafe_allow_html=True)

    # Present raw reference ranges beside patient estimates for readers who want exact numbers.
    st.subheader("Exact reference comparison")
    reference_table = pd.DataFrame({
        "Measurement": PARAMETER_NAMES,
        "Patient estimate": result["estimates"],
        "Reference 10th": result["reference_lower"],
        "Reference median": result["reference_median"],
        "Reference 90th": result["reference_upper"],
        "Patient percentile": result["percentiles"],
        "Interpretation": result["range_status"],
    })
    st.dataframe(reference_table.style.format({"Patient estimate": "{:.3f}", "Reference 10th": "{:.3f}", "Reference median": "{:.3f}", "Reference 90th": "{:.3f}", "Patient percentile": "{:.0f}th"}), hide_index=True, width="stretch")

    # Show how the ten analyzed voxels vary across the region instead of hiding heterogeneity in one mean.
    st.subheader("Variation across the 10 analyzed voxels")
    reference_samples = cached_reference()["samples"]
    voxel_percentiles = np.empty_like(result["voxel_estimates"])
    for parameter_index in range(9):
        voxel_percentiles[:, parameter_index] = [100.0 * (reference_samples[:, parameter_index] <= value).mean() for value in result["voxel_estimates"][:, parameter_index]]
    heatmap = go.Figure(go.Heatmap(
        z=voxel_percentiles.T,
        x=[f"Voxel {index}" for index in range(1, 11)],
        y=list(PARAMETER_SHORT_NAMES),
        colorscale=[[0, "#2b6cb0"], [0.5, "#f6f8fb"], [1, "#b94b45"]],
        zmin=0, zmax=100, colorbar=dict(title="Percentile"),
        hovertemplate="%{y}<br>%{x}<br>%{z:.0f}th percentile<extra></extra>",
    ))
    heatmap.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=35))
    st.plotly_chart(heatmap, width="stretch", config={"displayModeBar": False})

with evidence_tab:
    # Display the three posterior probabilities that directly determine the transparent score.
    st.subheader("Probability dashboard")
    probability_labels = ["Epithelium above reference", "Lumen below reference", "Restricted epithelial diffusion"]
    probability_values = [result["probabilities"]["epithelium_high"], result["probabilities"]["lumen_low"], result["probabilities"]["diffusion_restricted"]]
    probability_chart = go.Figure(go.Bar(
        x=np.asarray(probability_values) * 100,
        y=probability_labels,
        orientation="h",
        marker_color=["#176b87", "#45a29e", "#d7a437"],
        text=[f"{value:.0%}" for value in probability_values],
        textposition="outside",
        cliponaxis=False,
    ))
    probability_chart.update_layout(height=300, margin=dict(l=5, r=55, t=10, b=30), xaxis=dict(range=[0, 100], title="Regularized posterior evidence (%)"), showlegend=False)
    st.plotly_chart(probability_chart, width="stretch", config={"displayModeBar": False})
    st.caption("Probabilities are conservatively regularized to avoid presenting finite Monte Carlo samples as absolute 0% or 100% certainty.")

    # Visualize interval width relative to the reference span so uncertainty becomes interpretable.
    st.subheader("How precise are the estimates?")
    interval_width = result["upper"] - result["lower"]
    reference_width = np.maximum(result["reference_upper"] - result["reference_lower"], 1e-6)
    relative_uncertainty = 100.0 * interval_width / reference_width
    uncertainty_chart = go.Figure(go.Bar(
        x=relative_uncertainty, y=list(PARAMETER_SHORT_NAMES), orientation="h",
        marker_color=["#277a5a" if value < 35 else "#d19a2a" if value < 70 else "#b94b45" for value in relative_uncertainty],
        text=[f"{value:.0f}% of reference span" for value in relative_uncertainty], textposition="outside", cliponaxis=False,
    ))
    uncertainty_chart.update_layout(height=430, margin=dict(l=10, r=35, t=10, b=35), xaxis=dict(title="95% posterior interval width relative to reference range", range=[0, max(110.0, float(relative_uncertainty.max()) * 1.22)]), yaxis=dict(autorange="reversed"), showlegend=False)
    st.plotly_chart(uncertainty_chart, width="stretch", config={"displayModeBar": False})
    st.markdown('<div class="explain"><b>Values above 100% are possible:</b> they mean the model’s 95% interval is wider than the reference cohort’s central 80% range. Those parameters are less precisely identified, not computationally invalid.</div>', unsafe_allow_html=True)

    # Plot measured and physics-reconstructed signals to make model behavior visually auditable.
    st.subheader("Observed signal versus PIVA reconstruction")
    labels = [f"b={int(b)}, TE={int(te)}" for b, te in zip(B_GRID, TE_GRID)]
    signal_chart = go.Figure()
    signal_chart.add_trace(go.Scatter(x=labels, y=result["observed_signal"], mode="markers+lines", name="Observed patient mean", line=dict(color="#176b87", width=3)))
    signal_chart.add_trace(go.Scatter(x=labels, y=result["reconstructed_signal"], mode="lines", name="PIVA reconstruction", line=dict(color="#d7a437", width=3, dash="dash")))
    signal_chart.update_layout(height=390, margin=dict(l=10, r=10, t=10, b=90), yaxis_title="MRI signal", xaxis_tickangle=-45, legend=dict(orientation="h", y=1.12))
    st.plotly_chart(signal_chart, width="stretch", config={"displayModeBar": False})

with method_tab:
    # Explain the complete score in language suitable for questions after the demo.
    st.subheader("Transparent demonstration score")
    st.latex(r"\mathrm{Concern\ Score}=100\left(0.40p_{v_{ep}}+0.30p_{v_{lum}}+0.30p_{D_{ep}}\right)")
    st.markdown(
        """
        - **40%:** posterior probability that epithelial volume fraction exceeds the reference 90th percentile.
        - **30%:** posterior probability that lumen volume fraction falls below the reference 10th percentile.
        - **30%:** posterior probability that epithelial diffusion falls below the reference 10th percentile.

        The thresholds come from the same 5,000-voxel synthetic reference cohort shown in the dashboard. The weights are transparent demonstration choices and were not fitted to biopsy outcomes.

        Displayed posterior evidence is regularized with a symmetric Beta prior so a finite set of Monte Carlo samples does not appear as absolute 0% or 100% certainty.
        """
    )
    st.subheader("What PIVA contributes")
    st.markdown("The network predicts a full Cholesky covariance in its latent space, draws correlated posterior samples, transforms them into constrained physical parameters, and reconstructs the measured signal through the fixed three-compartment MRI equation.")

# Keep the academic limitation visible without overwhelming the product demonstration.
st.markdown(
    '<div class="disclaimer"><b>Academic demonstration:</b> synthetic patient data and synthetic-reference thresholds only. Outputs are not PI-RADS scores, cancer probabilities, diagnoses, or clinical recommendations.</div>',
    unsafe_allow_html=True,
)
