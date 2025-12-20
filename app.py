# app.py
import streamlit as st
from PIL import Image
import os
from pipeline import predict_image  # your pipeline function

# ===================== CONFIG =====================
OUTPUT_DIR = "streamlit_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== STREAMLIT APP =====================
st.set_page_config(
    page_title="Food & Fruit Classifier",
    layout="wide",
    page_icon="🍽️"
)

# ===================== SIDEBAR =====================
#st.sidebar.image("https://img.icons8.com/color/96/000000/apple.png", use_column_width=True)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Introduction", "🔍 Prediction"]
)

# Optional settings in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Settings")
show_masks = st.sidebar.checkbox("Show Segmentation Masks", value=True)

# ===================== INTRODUCTION PAGE =====================
if page == "🏠 Introduction":
    st.title("🍎 Food & Fruit Classifier with Calories & Segmentation")
    st.markdown("""
    Welcome to the **Food & Fruit Classifier**! 🥗🍌🍕  
    This app can help you:
    - 🍏 **Classify** an image as **Food** or **Fruit**
    - 🥑 Recognize the **specific type** of food or fruit
    - ⚖️ Compute **total calories** based on weight
    - 🖼️ Generate **segmentation masks** for fruits
    - 🎨 Visualize **multi-class segmentation** for fruits
    """)
   # st.image("https://img.icons8.com/external-flat-juicy-fish/344/external-fruits-health-food-flat-flat-juicy-fish.png", use_column_width=True)
    st.markdown("✨ **Enjoy exploring your food and fruits visually and learn their calorie content instantly!**")

# ===================== PREDICTION PAGE =====================
elif page == "🔍 Prediction":
    st.title("🖼️ Upload Image for Prediction")
    st.markdown("Upload a **food or fruit image** and click **Predict** to see results 🍴")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Save temporarily to pass path to pipeline
        temp_path = os.path.join(OUTPUT_DIR, uploaded_file.name)
        img.save(temp_path)

        if st.button("Predict"):
            with st.spinner("🔍 Running prediction..."):
                results = predict_image(temp_path, OUTPUT_DIR)

            st.success("✅ Prediction complete!")
            st.markdown(f"**Main Class:** {results['main_class']}")
            st.markdown(f"**Sub-class:** {results['sub_class']}")
            st.markdown(f"**Weight:** {results['grams']} g")
            st.markdown(f"**Total Calories:** {results['total_calories']:.2f} kcal")

            # Show masks if available and user wants to
            if show_masks:
                if results.get('binary_mask'):
                    st.subheader("Binary Segmentation Mask")
                    st.image(results['binary_mask'], use_column_width=True)
                if results.get('multi_mask'):
                    st.subheader("Multi-class Segmentation Mask")
                    st.image(results['multi_mask'], use_column_width=True)
