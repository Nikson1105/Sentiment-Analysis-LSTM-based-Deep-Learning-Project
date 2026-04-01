import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Page Config (MUST be first Streamlit command)
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="💬", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    body {
        background-color: #0f172a;
    }
    .main {
        background-color: #111827;
        padding: 2rem;
        border-radius: 15px;
    }
    h1 {
        text-align: center;
        color: #f9fafb;
    }
    .stTextArea textarea {
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #6366f1;
        color: white;
        font-size: 18px;
        height: 3em;
    }
    .footer {
        text-align: center;
        color: gray;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load model
model = tf.keras.models.load_model("sentiment_model.h5")

# UI Header
st.markdown("<h1>💬 AI Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.write("### Analyze movie reviews with Deep Learning 🤖")

# Input Box
review = st.text_area("✍️ Enter your movie review:")

# Button
if st.button("🔍 Analyze Sentiment"):
    if review:
        sequence = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(sequence, maxlen=200)

        prediction = model.predict(padded)[0][0]

        if prediction > 0.5:
            st.success("🎉 Positive Review 😍")
            st.balloons()
        else:
            st.error("😔 Negative Review")

        # Confidence score
        st.write(f"**Confidence Score:** {round(float(prediction)*100, 2)}%")

    else:
        st.warning("⚠️ Please enter a review")

# Footer
st.markdown("""
    <div class="footer">
        Made with ❤️ by <b>Nikhil</b>
    </div>
""", unsafe_allow_html=True)