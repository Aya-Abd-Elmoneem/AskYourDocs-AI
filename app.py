import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================
# 1. CONFIGURATION
# =========================
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel("models/gemini-flash-latest")

st.set_page_config(
    page_title="AskYourDocs AI",
    page_icon="🤖",
    layout="wide"
)

# =========================
# 2. FUNCTIONS
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
# 3. UI (ONE PAGE)
# =========================
st.title("🤖 AskYourDocs AI")
st.markdown("اسأل عن أي ملف PDF وسيقوم الذكاء الاصطناعي بالإجابة بناءً عليه")

# Upload Section
pdf_docs = st.file_uploader(
    "📂 Upload your PDF files",
    accept_multiple_files=True
)

if st.button("📊 Process Files"):
    if pdf_docs:
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = splitter.split_text(raw_text)

            create_vector_store(chunks)

        st.success("جاهز للإجابة ✅")
    else:
        st.warning("ارفع ملفات الأول")

st.divider()

# =========================
# 4. CHAT
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("اسأل عن ملفاتك..."):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                db = load_db()
                docs = db.similarity_search(prompt)

                context = "\n\n".join(
                    [doc.page_content for doc in docs]
                )
            except:
                context = "لا توجد ملفات مرفوعة"

            sys_msg = "أنت مساعد ذكي يجيب فقط من الملفات المرفوعة. أجب باحتراف وباختصار."

            full_query = f"""
            {sys_msg}

            السياق:
            {context}

            السؤال:
            {prompt}
            """

            response = model.generate_content(full_query)

            st.markdown(response.text)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response.text
            })
