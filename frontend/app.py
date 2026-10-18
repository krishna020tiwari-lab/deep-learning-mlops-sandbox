import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

st.set_page_config(
    page_title="Unified AI Classifier", page_icon="🤖", layout="centered"
)

st.title("🤖 Unified AI Inference Dashboard")
st.write("Perform image and text classification powered by ONNX models.")


def show_error(response):
  try:
    detail = response.json().get("detail")
  except ValueError:
    detail = response.text
  st.error(f"Error {response.status_code}: {detail}")


tab_image, tab_text = st.tabs(
    ["🖼️ Image Classification", "📝 Text Classification"]
)

# --- TAB 1: VISION (ResNet18) ---
with tab_image:
  st.header("ResNet18 Vision Model")
  uploaded_file = st.file_uploader(
      "Choose an image...", type=["jpg", "jpeg", "png"]
  )

  if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width="stretch")

    if st.button("Classify Image 🚀"):
      with st.spinner("Processing image..."):
        try:
          files = {
              "file": (
                  uploaded_file.name,
                  uploaded_file.getvalue(),
                  uploaded_file.type,
              )
          }
          response = requests.post(
              f"{BACKEND_URL}/predict/image", files=files, timeout=30
          )

          if response.status_code == 200:
            data = response.json()
            st.success(f"**Result:** {data['label_description']}")
            st.metric(
                label="Confidence", value=f"{data['confidence'] * 100:.2f}%"
            )
          else:
            show_error(response)
        except Exception as e:
          st.error(f"Could not connect to backend: {e}")

# --- TAB 2: TEXT (BERT) ---
with tab_text:
  st.header("BERT Text Model")
  user_text = st.text_area(
      "Enter text to analyze:", placeholder="Type a sentence here..."
  )

  if st.button("Classify Text 🚀"):
    if not user_text.strip():
      st.warning("Please enter some text before submitting.")
    else:
      with st.spinner("Processing text..."):
        try:
          payload = {"text": user_text}
          response = requests.post(
              f"{BACKEND_URL}/predict/text", json=payload, timeout=30
          )

          if response.status_code == 200:
            data = response.json()
            st.success(f"**Predicted Label:** {data['label']}")
            st.info(f"**Class Index:** {data['predicted_class']}")
            st.write("**Probabilities:**", data["probabilities"])
          else:
            show_error(response)
        except Exception as e:
          st.error(f"Could not connect to backend: {e}")