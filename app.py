import os
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# LangChain core
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate

# Gemini LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# Data & store
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ============================
# Config
# ============================
load_dotenv()
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

DATA_FILES = [("data/salary.txt", "salary"), ("data/insurance.txt", "insurance")]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "./.chroma_multi_agent"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

ANSWER_TEMPLATE = (
    "Always answer in this format:\n"
    "1) A short direct sentence that answers the question.\n"
    "2) Then 2–4 bullet points with key details, examples, or steps.\n"
    "Do NOT mention tools, retrievals, scratchpads, or say thank you.\n"
)

st.set_page_config(page_title="Multi-Agent RAG (Salary + Insurance)", layout="centered", initial_sidebar_state="expanded")

# ✅ Initialize session state variables
if "chat" not in st.session_state:
    st.session_state.chat = []
if "history_msgs" not in st.session_state:
    st.session_state.history_msgs = []
if "pending_q" not in st.session_state:
    st.session_state.pending_q = None

# ============================
# Helpers: data & store
# ============================

def load_docs(files=None) -> list[Document]:
    docs: list[Document] = []
    if files:
        for uploaded_file in files:
            file_path = os.path.join("data", uploaded_file.name)
            os.makedirs("data", exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            topic = "salary" if "salary" in uploaded_file.name.lower() else (
                "insurance" if "insurance" in uploaded_file.name.lower() else "general"
            )

            loader = TextLoader(file_path, encoding="utf-8")
            loaded = loader.load()
            for d in loaded:
                d.metadata["topic"] = topic
                d.metadata["source"] = uploaded_file.name
            docs.extend(loaded)
    else:
        for path, topic in DATA_FILES:
            if os.path.exists(path):
                loader = TextLoader(path, encoding="utf-8")
                loaded = loader.load()
                for d in loaded:
                    d.metadata["topic"] = topic
                    d.metadata["source"] = os.path.basename(path)
                docs.extend(loaded)
    return docs


def build_store(docs: list[Document]) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=PERSIST_DIR)
    return vs


def get_retrievers(vs: Chroma):
    salary_ret = vs.as_retriever(search_kwargs={"k": 2, "filter": {"topic": "salary"}})
    insur_ret = vs.as_retriever(search_kwargs={"k": 2, "filter": {"topic": "insurance"}})
    return salary_ret, insur_ret

# ============================
# Helpers: answer formatting
# ============================

def run_agent(query: str, retriever, llm: ChatGoogleGenerativeAI) -> str:
    docs = retriever.invoke(query)
    context = "\n".join(d.page_content for d in docs[:2])  # keep short
    tmpl = PromptTemplate.from_template(
        "You are a helpful assistant.\n" + ANSWER_TEMPLATE +
        f"\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:\n"
    )
    return llm.invoke(tmpl.format(context=context, question=query)).content.strip()

# ============================
# Build agents
# ============================

def build_agents(uploaded_files=None):
    docs = load_docs(uploaded_files)
    if not docs:
        st.stop()

    vs = build_store(docs)
    salary_ret, insur_ret = get_retrievers(vs)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2, google_api_key=GOOGLE_API_KEY)
    return llm, vs, salary_ret, insur_ret

# ============================
# Coordinator (keyword routing)
# ============================

def keyword_router(query: str, salary_ret, insur_ret, llm):
    q = query.lower()

    salary_keywords = ["salary", "ctc", "in-hand", "gross", "net", "payslip", "deduction", "pf", "professional tax"]
    insurance_keywords = ["insurance", "claim", "hospital", "premium", "co-pay", "tpa", "dependent", "network", "coverage"]

    if any(kw in q for kw in salary_keywords):
        return run_agent(query, salary_ret, llm), "Salary Agent"
    elif any(kw in q for kw in insurance_keywords):
        return run_agent(query, insur_ret, llm), "Insurance Agent"
    else:
        return (
            "**Could you clarify whether your question is about salary or insurance?**\n"
            "- For salary, ask about gross, in-hand, PF, CTC, deductions, tax.\n"
            "- For insurance, ask about claims, coverage, premiums, co-pay, hospitalizations.",
            "Coordinator"
        )

# ============================
# UI
# ============================

st.title("Salary & Insurance Q&A Assistant")
st.caption("Smart multi-agent app that answers your salary and insurance questions using uploaded or default context files.")


with st.sidebar:
    st.markdown("### ⚙️ Context Setup")
    st.caption(
            "By default, the app uses **data/salary.txt** and **data/insurance.txt**.\n\n"
            "👉 You can also upload your own documents to override or extend the context."
        )
    

    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload salary/insurance docs", type=["txt", "docx"], accept_multiple_files=True
    )
    rebuild = st.button("Rebuild Vector Store")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat = []
        st.session_state.history_msgs = []

    # ✅ Download chat
    if "chat" in st.session_state and st.session_state.chat:
        chat_text = "\n".join([f"{r}: {m}" for r, m in st.session_state.chat])
        fname = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        st.download_button("📤 Download chat.txt", chat_text, file_name=fname)

    # ✅ Debug: show retriever snippets
    with st.expander("🔎 Show retrieved snippets (top 4)") as exp:
        chat = st.session_state.get("chat", [])
        if chat:
            last_user_q = next((q for r, q in reversed(chat) if r == "user"), None)
            last_agent_role = next((r for r, _ in reversed(chat) if r != "user"), None)
            if last_user_q:
                if last_agent_role == "Salary Agent":
                    docs = st.session_state.salary_ret.invoke(last_user_q)
                    st.markdown("**Salary retriever results:**")
                elif last_agent_role == "Insurance Agent":
                    docs = st.session_state.insur_ret.invoke(last_user_q)
                    st.markdown("**Insurance retriever results:**")
                else:
                    docs = (
                        st.session_state.salary_ret.invoke(last_user_q)
                        + st.session_state.insur_ret.invoke(last_user_q)
                    )
                    st.markdown("**Mixed/Unknown route: showing both retrievers**")

                for i, d in enumerate(docs[:4], 1):
                    st.write(f"{i}. {d.page_content[:400]}…")
                    meta = d.metadata
                    st.caption(f"Source: {meta.get('source','?')} | Topic: {meta.get('topic','?')}")
        else:
            st.write("Ask a question to see retrieved snippets.")

    # ============================
# Initialize agents + retrievers
# ============================

if "llm" not in st.session_state or rebuild:
    with st.spinner("⚡ Building vector store & retrievers..."):
        llm, vs, salary_ret, insur_ret = build_agents(uploaded_files)
        st.session_state.llm = llm
        st.session_state.vs = vs
        st.session_state.salary_ret = salary_ret
        st.session_state.insur_ret = insur_ret

# ============================
# Chat input + keyword routing
# ============================

user_q = st.chat_input("Ask about salary or insurance…")

if "pending_q" in st.session_state and st.session_state.pending_q:
    user_q = st.session_state.pending_q
    st.session_state.pending_q = None

if user_q:
    with st.spinner("🤔 Thinking..."):
        answer, agent_label = keyword_router(
            user_q,
            st.session_state.salary_ret,
            st.session_state.insur_ret,
            st.session_state.llm,
        )

    # ✅ Bold first line, keep original bullets
    lines = answer.splitlines()
    if lines:
        first = lines[0].lstrip("•-1234567890)").strip()
        rest=[ln.lstrip("•-1234567890.").strip() for ln in lines[1:] if ln.strip()]

        formatted_answer = f"**{first}**"
        for ln in lines[1:]:
            if ln.strip():
                formatted_answer += f"\n{ln}"
    else:
        formatted_answer = f"**{answer.strip()}**"

    st.session_state.chat.append(("user", user_q))
    st.session_state.chat.append((agent_label, formatted_answer))
    st.session_state.history_msgs.append(HumanMessage(content=user_q))
    st.session_state.history_msgs.append(AIMessage(content=formatted_answer))


# ============================
# Render transcript
# ============================

for role, msg in st.session_state.get("chat", []):
    if role == "user":
        with st.chat_message("user"):
            st.markdown(msg)
    else:
        with st.chat_message("assistant"):
            st.markdown(
                f"<span style='background-color:#e5e7eb; color:#111827; "
                f"padding:2px 8px; border-radius:8px; font-size:12px; font-weight:600;'>{role}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(msg)

            # ✅ Follow-up suggestions with unique keys
            if role == "Salary Agent":
                for idx, s in enumerate(["What is gross salary?", "Is employer PF part of in-hand?"]):
                    if st.button(s, key=f"follow_{role}_{idx}_{hash(msg)}"):
                        st.session_state.pending_q = s
                        st.rerun()
            elif role == "Insurance Agent":
                for idx, s in enumerate(["What is co-pay?", "How is premium decided?"]):
                    if st.button(s, key=f"follow_{role}_{idx}_{hash(msg)}"):
                        st.session_state.pending_q = s
                        st.rerun()
