import streamlit as st
from inference import predict_review


st.set_page_config(
    page_title="TruthReview",
    page_icon="🔎",
    layout="centered",
)

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #0b1020 0%, #111827 100%);
            color: #f9fafb;
        }

        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 2rem;
            max-width: 860px;
        }

        .hero-card {
            background: rgba(17, 24, 39, 0.72);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            padding: 28px 28px 22px 28px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.22);
            backdrop-filter: blur(10px);
            margin-bottom: 20px;
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.35rem;
            color: #ffffff;
            letter-spacing: -0.02em;
        }

        .hero-subtitle {
            font-size: 1rem;
            color: #cbd5e1;
            margin-bottom: 0;
        }

        .section-card {
            background: rgba(17, 24, 39, 0.72);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 22px;
            padding: 22px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
            backdrop-filter: blur(10px);
            margin-top: 14px;
        }

        .result-real {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.20), rgba(5, 150, 105, 0.18));
            border: 1px solid rgba(16, 185, 129, 0.35);
            border-radius: 18px;
            padding: 18px 20px;
            margin-top: 14px;
        }

        .result-fake {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.20), rgba(220, 38, 38, 0.18));
            border: 1px solid rgba(239, 68, 68, 0.35);
            border-radius: 18px;
            padding: 18px 20px;
            margin-top: 14px;
        }

        .result-label {
            font-size: 0.9rem;
            color: #cbd5e1;
            margin-bottom: 0.3rem;
        }

        .result-value {
            font-size: 1.35rem;
            font-weight: 800;
            color: #ffffff;
        }

        .small-note {
            font-size: 0.92rem;
            color: #94a3b8;
            margin-top: 0.4rem;
        }

        div[data-testid="stTextArea"] textarea {
            border-radius: 18px !important;
            background-color: rgba(15, 23, 42, 0.90) !important;
            color: #f8fafc !important;
            border: 1px solid rgba(255, 255, 255, 0.10) !important;
            min-height: 220px !important;
            font-size: 1rem !important;
            padding: 16px !important;
        }

        div[data-testid="stButton"] > button {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: white;
            border: none;
            border-radius: 999px;
            padding: 0.7rem 1.2rem;
            font-weight: 700;
            font-size: 0.98rem;
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.28);
        }

        div[data-testid="stButton"] > button:hover {
            background: linear-gradient(135deg, #1d4ed8, #1e40af);
            color: white;
        }

        .metric-title {
            color: #cbd5e1;
            font-size: 0.95rem;
            margin-bottom: 0.35rem;
            font-weight: 600;
        }

        .footer-note {
            text-align: center;
            color: #94a3b8;
            font-size: 0.88rem;
            margin-top: 18px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">TruthReview</div>
        <p class="hero-subtitle">
            A clean NLP-powered interface for detecting whether a hotel review is likely
            truthful or deceptive.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-card">', unsafe_allow_html=True)

review_text = st.text_area(
    "Review text",
    placeholder="Paste an English hotel review here...",
    label_visibility="collapsed",
)

analyze = st.button("Analyze Review")

st.markdown("</div>", unsafe_allow_html=True)

if analyze:
    try:
        result = predict_review(review_text)

        predicted_label = result["prediction_label"]
        truthful_prob = result["truthful_probability"]
        deceptive_prob = result["deceptive_probability"]

        is_real = result["prediction_numeric"] == 0
        result_class = "result-real" if is_real else "result-fake"

        st.markdown(
            f"""
            <div class="{result_class}">
                <div class="result-label">Prediction</div>
                <div class="result-value">{predicted_label}</div>
                <div class="small-note">
                    The model classified this review based on learned linguistic patterns from hotel review data.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if truthful_prob is not None and deceptive_prob is not None:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### Confidence Scores")

            st.markdown('<div class="metric-title">Truthful / Real</div>', unsafe_allow_html=True)
            st.progress(int(truthful_prob * 100))
            st.caption(f"{truthful_prob:.4f}")

            st.markdown('<div class="metric-title">Deceptive / Fake</div>', unsafe_allow_html=True)
            st.progress(int(deceptive_prob * 100))
            st.caption(f"{deceptive_prob:.4f}")

            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("View cleaned text"):
            st.write(result["clean_text"])

    except ValueError as error:
        st.warning(str(error))
    except Exception as error:
        st.error(f"An unexpected error occurred: {error}")

st.markdown(
    """
    <div class="footer-note">
        Built for review classification using a lightweight deployed NLP pipeline.
    </div>
    """,
    unsafe_allow_html=True,
)