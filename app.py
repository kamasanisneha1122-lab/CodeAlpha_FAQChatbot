import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="AI FAQChatbot",
    page_icon="🤖",
    layout="centered"
)

# -----------------------
# CUSTOM CSS
# -----------------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    color: #00E5FF;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: white;
    font-size: 18px;
    margin-bottom:20px;
}

.user-box {
    background-color: #1976D2;
    padding: 12px;
    border-radius: 12px;
    color: white;
    margin-top: 10px;
}

.bot-box {
    background-color: #00C853;
    padding: 12px;
    border-radius: 12px;
    color: white;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# HEADER
# -----------------------
st.markdown(
    "<div class='title'>🤖 AI FAQChatbot</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>CodeAlpha AI Internship Project</div>",
    unsafe_allow_html=True
)

# -----------------------
# SIDEBAR
# -----------------------
st.sidebar.title("📌 Project Information")

st.sidebar.success("AI FAQ Chatbot")

st.sidebar.write("""
### Features
✅ NLP Based Matching

✅ Cosine Similarity

✅ Confidence Score

✅ Chat History

✅ Smart Response System
""")

# -----------------------
# FAQ DATASET
# -----------------------

faq_data = {
    "question":[
        "what is ai",
        "what is machine learning",
        "what is deep learning",
        "what is nlp",
        "what is computer vision",
        "what is python",
        "what is data science",
        "what is chatbot",
        "what is github",
        "what is streamlit",
        "what is openai",
        "what is chatgpt"
    ],

    "answer":[
        "Artificial Intelligence enables machines to simulate human intelligence.",
        "Machine Learning is a subset of AI that learns from data.",
        "Deep Learning uses neural networks with multiple layers.",
        "NLP stands for Natural Language Processing.",
        "Computer Vision helps machines understand images and videos.",
        "Python is a powerful programming language.",
        "Data Science extracts insights from data.",
        "A chatbot interacts with users through conversations.",
        "GitHub is a platform for version control and collaboration.",
        "Streamlit helps build web apps using Python.",
        "OpenAI is an artificial intelligence research company.",
        "ChatGPT is an AI chatbot developed by OpenAI."
    ]
}

faq = pd.DataFrame(faq_data)

# -----------------------
# NLP MODEL
# -----------------------
vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(
    faq["question"]
)

# -----------------------
# CHAT HISTORY
# -----------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------
# INPUT
# -----------------------
user_question = st.text_input(
    "💬 Ask Your Question"
)

if st.button("Send 🚀"):

    if user_question:

        user_vector = vectorizer.transform(
            [user_question.lower()]
        )

        similarity = cosine_similarity(
            user_vector,
            question_vectors
        )

        best_match = similarity.argmax()

        confidence = similarity[0][best_match]

        # Confidence Threshold
        if confidence < 0.20:

            answer = (
                "❌ Sorry, I don't know the answer to that question."
            )

        else:

            answer = faq.iloc[best_match]["answer"]

        st.session_state.history.append(
            (user_question, answer, confidence)
        )

# -----------------------
# DISPLAY CHAT
# -----------------------
for question, answer, score in reversed(
        st.session_state.history):

    st.markdown(
        f"<div class='user-box'><b>You:</b> {question}</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<div class='bot-box'><b>Bot:</b> {answer}</div>",
        unsafe_allow_html=True
    )

    st.progress(float(score))

    st.write(
        f"🎯 Confidence Score: {score:.2f}"
    )

# -----------------------
# CLEAR CHAT
# -----------------------
if st.button("🗑️ Clear Chat"):
    st.session_state.history = []

# -----------------------
# FOOTER
# -----------------------
st.markdown("---")
st.caption(
    "🚀 Developed for CodeAlpha Artificial Intelligence Internship"
)