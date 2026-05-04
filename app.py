import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="AskYourDocs AI", page_icon="🤖", layout="wide")

# API KEY
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("API Key not found. Add it in Streamlit secrets.")
    st.stop()

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel("models/gemini-flash-latest")

# =========================
# FUNCTIONS
# =========================
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def create_vector_store(text_chunks):
    embeddings = get_embeddings()
    db = FAISS.from_texts(text_chunks, embedding=embeddings)
    db.save_local("faiss_index")

def load_db():
    embeddings = get_embeddings()
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

# =========================
# MODERN UI STYLE
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

/* Title */
.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: bold;
    background: linear-gradient(90deg, #10b981, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
}

/* Upload box */
.upload-box {
    border: 2px dashed #334155;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    background: rgba(30, 41, 59, 0.5);
}

/* Chat bubble */
.user-msg {
    background: #10b981;
    padding: 12px;
    border-radius: 12px;
    margin: 10px 0;
    color: white;
}

.bot-msg {
    background: #1e293b;
    padding: 12px;
    border-radius: 12px;
    margin: 10px 0;
    border: 1px solid #334155;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #10b981, #059669);
    color: white;
    border-radius: 10px;
    padding: 10px;
    border: none;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown('<div class="main-title">AskYourDocs AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ارفع ملفاتك واسأل عنها مباشرة</div>', unsafe_allow_html=True)

# =========================
# UPLOAD SECTION
# =========================
st.markdown('<div class="upload-box">', unsafe_allow_html=True)

pdf_docs = st.file_uploader(
    "📂 اسحب ملفات PDF هنا أو اضغط للرفع",
    accept_multiple_files=True
)

if st.button("🚀 معالجة الملفات"):
    if pdf_docs:
        with st.spinner("جارٍ تحليل الملفات..."):
            raw_text = get_pdf_text(pdf_docs)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = splitter.split_text(raw_text)

            create_vector_store(chunks)

        st.success("تم تجهيز الملفات بنجاح ✅")
    else:
        st.warning("ارفع ملفات الأول")

st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# =========================
# CHAT
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض الرسائل
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

# إدخال المستخدم
if prompt := st.chat_input("اكتب سؤالك هنا..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        try:
            db = load_db()
            docs = db.similarity_search(prompt)
            context = "\n\n".join([d.page_content for d in docs])
        except:
            context = "لا توجد ملفات مرفوعة"

        full_prompt = f"""
        أنت مساعد ذكي يجيب فقط من الملفات.

        السياق:
        {context}

        السؤال:
        {prompt}
        """

        response = model.generate_content(full_prompt)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response.text
        })

    st.rerun()
