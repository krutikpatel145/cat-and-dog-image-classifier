import io
import html as _html
import streamlit as st
from PIL import Image

from inference import POSSIBLE_MODEL_PATHS, load_model, predict_image, preprocess_image, resolve_model_path


MAX_FILE_BYTES = 10 * 1024 * 1024  # 10MB


def main() -> None:
    st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

    # Styling: closely match the Flask Tailwind look (colors + card layout).
    st.markdown(
        """
        <style>
          body {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
            background: radial-gradient(900px circle at 50% -10%, rgba(99, 102, 241, 0.18), transparent 55%),
                        radial-gradient(700px circle at 10% 20%, rgba(99, 102, 241, 0.08), transparent 55%);
          }

          .app-header {
            text-align: center;
            margin-top: 12px;
            margin-bottom: 22px;
          }
          .app-h1 {
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.02em;
          }
          .app-sub {
            margin-top: 10px;
            color: #64748b;
            line-height: 1.6;
          }

          .card {
            border-radius: 1rem;
            padding: 18px;
            border: 1px solid rgba(148, 163, 184, 0.35);
            background: rgba(255, 255, 255, 0.78);
            box-shadow: 0 10px 30px rgba(2, 6, 23, 0.08);
            backdrop-filter: blur(10px);
          }

          .card-dark {
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid rgba(51, 65, 85, 0.7);
          }

          .badge {
            border-radius: 9999px;
            padding: 10px 14px;
            font-weight: 700;
            display: inline-block;
            border: 1px solid transparent;
          }

          .badge-cat { background: rgba(237, 233, 254, 1); color: #5b21b6; border-color: rgba(199, 210, 254, 1); }
          .badge-dog { background: rgba(224, 242, 254, 1); color: #0369a1; border-color: rgba(186, 230, 253, 1); }

          .pill {
            border-radius: 1rem;
            padding: 12px 14px;
            border: 1px solid rgba(148, 163, 184, 0.35);
            background: rgba(248, 250, 252, 1);
          }

          .row { display: flex; gap: 14px; flex-wrap: wrap; }
          .row > div { flex: 1 1 240px; }

          .bar-shell {
            height: 10px;
            border-radius: 9999px;
            background: rgba(226, 232, 240, 1);
            overflow: hidden;
          }

          .bar {
            height: 100%;
            width: 0%;
            border-radius: 9999px;
            transition: width 600ms ease;
          }

          .bar-cat { background: rgba(124, 58, 237, 1); }
          .bar-dog { background: rgba(2, 132, 199, 1); }

          .small {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #64748b;
          }

          .value {
            margin-top: 6px;
            font-weight: 700;
          }

          .error-box {
            border-radius: 12px;
            border: 1px solid rgba(251, 113, 133, 0.35);
            background: rgba(254, 242, 242, 1);
            padding: 12px 14px;
            color: #9f1239;
          }
          .error-box pre { margin: 0; white-space: pre-wrap; }

          @media (prefers-color-scheme: dark) {
            .card { background: rgba(15, 23, 42, 0.50); border-color: rgba(51, 65, 85, 0.75); box-shadow: none; }
            .app-sub { color: rgba(148, 163, 184, 0.95); }
            .pill { background: rgba(2, 6, 23, 0.25); border-color: rgba(51, 65, 85, 0.8); }
            .small { color: rgba(148, 163, 184, 0.95); }
            .bar-shell { background: rgba(51, 65, 85, 0.7); }
            .error-box { border-color: rgba(251, 113, 133, 0.45); background: rgba(127, 29, 29, 0.15); color: rgba(225, 29, 72, 0.95); }
            .badge-cat { background: rgba(69, 20, 103, 0.35); color: rgba(216, 180, 254, 1); border-color: rgba(199, 210, 254, 0.45); }
            .badge-dog { background: rgba(7, 89, 133, 0.35); color: rgba(186, 230, 253, 1); border-color: rgba(186, 230, 253, 0.45); }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="app-header">
          <div class="app-h1">Cat vs Dog Classifier</div>
          <div class="app-sub">
            Upload an image and the model will predict whether it’s a <b>Cat</b> or a <b>Dog</b>.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    @st.cache_resource
    def get_model():
        model_path = resolve_model_path()
        model, err = load_model(model_path)
        return model, err, model_path

    model, model_error, model_path = get_model()

    if model is None:
        safe_model_error = _html.escape(model_error or "Unknown model load error.")
        safe_model_path = _html.escape(model_path or "none found")
        st.markdown(
            f"""
            <div class="card">
              <div style="font-weight:700;">Model not loaded.</div>
              <div class="app-sub" style="margin-top:10px;">
                Place a trained <code>.h5</code> model in the project. Checked: <code>{safe_model_path}</code>
              </div>
              <div style="margin-top:10px;" class="error-box">
                <pre>{safe_model_error}</pre>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "bmp"],
        )

        if uploaded_file is None:
            st.markdown(
                """
                <div class="app-sub" style="margin-top:10px;">
                  PNG, JPG, JPEG, BMP up to <b>10MB</b>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            if uploaded_file.size > MAX_FILE_BYTES:
                st.markdown(
                    '<div class="error-box">File too large. Please choose an image up to 10MB.</div>',
                    unsafe_allow_html=True,
                )
            else:
                try:
                    img = Image.open(io.BytesIO(uploaded_file.read()))
                except Exception:
                    st.markdown(
                        '<div class="error-box">Invalid image file. Please upload a valid PNG/JPG/JPEG/BMP.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.image(img, caption="Uploaded image", use_column_width=True)
                    if st.button("Predict", disabled=False):
                        with st.spinner("Running model inference…"):
                            img_array = preprocess_image(img)
                            label, confidence, raw_score, cat_prob, dog_prob = predict_image(model, img_array)

                        confidence_pct = float(confidence) * 100.0
                        cat_pct = float(cat_prob) * 100.0
                        dog_pct = float(dog_prob) * 100.0

                        badge_class = "badge-cat" if str(label).lower().find("cat") >= 0 else "badge-dog"
                        safe_label = _html.escape(str(label))

                        st.markdown(
                            f"""
                            <div style="margin-top:16px;">
                              <div style="display:flex; align-items:center; justify-content:space-between; gap:14px; flex-wrap:wrap;">
                                <div style="font-weight:700;">Prediction</div>
                                <div class="badge {badge_class}">{safe_label}</div>
                              </div>

                              <div class="row" style="margin-top:14px;">
                                <div class="pill">
                                  <div class="small">Confidence</div>
                                  <div class="value">{confidence_pct:.1f}%</div>
                                </div>
                                <div class="pill">
                                  <div class="small">Score</div>
                                  <div class="value">{float(raw_score):.4f}</div>
                                </div>
                              </div>

                              <div style="margin-top:14px;" class="pill">
                                <div style="display:flex; gap:16px; flex-wrap:wrap;">
                                  <div style="flex:1 1 240px;">
                                    <div class="small" style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
                                      <span style="display:inline-flex; align-items:center; gap:10px;">
                                        <span style="display:inline-block; width:10px; height:10px; border-radius:9999px; background: #7c3aed;"></span>
                                        Cat
                                      </span>
                                      <span>{cat_pct:.1f}%</span>
                                    </div>
                                    <div class="bar-shell" style="margin-top:10px;">
                                      <div class="bar bar-cat" style="width:{cat_pct:.0f}%"></div>
                                    </div>
                                  </div>

                                  <div style="flex:1 1 240px;">
                                    <div class="small" style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
                                      <span style="display:inline-flex; align-items:center; gap:10px;">
                                        <span style="display:inline-block; width:10px; height:10px; border-radius:9999px; background: #0284c7;"></span>
                                        Dog
                                      </span>
                                      <span>{dog_pct:.1f}%</span>
                                    </div>
                                    <div class="bar-shell" style="margin-top:10px;">
                                      <div class="bar bar-dog" style="width:{dog_pct:.0f}%"></div>
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
