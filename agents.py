import os
import streamlit as st
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv

# LangChain core
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.callbacks.streamlit import StreamlitCallbackHandler
# LLM
from langchain_groq import ChatGroq

# Data & store
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Agents & tools
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

# ============================
# Config
# ============================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

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

st.set_page_config(page_title="Multi-Agent RAG (Salary + Insurance)", layout="wide")

# ============================
# Helpers: data & store
# ============================

def load_docs(files=None) -> list[Document]:
    docs: list[Document] = []

    # Case 1: uploaded files
    if files:
        for uploaded_file in files:
            # save temporarily
            file_path = os.path.join("data", uploaded_file.name)
            os.makedirs("data", exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # assign topic by filename (simple heuristic)
            topic = "salary" if "salary" in uploaded_file.name.lower() else (
                "insurance" if "insurance" in uploaded_file.name.lower() else "general"
            )

            loader = TextLoader(file_path, encoding="utf-8")
            loaded = loader.load()
            for d in loaded:
                d.metadata["topic"] = topic
                d.metadata["source"] = uploaded_file.name
            docs.extend(loaded)

    # Case 2: fallback → use defaults
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # ✅ If store already exists, reuse it
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        vs = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        # Optional: check if retriever has docs, if empty then rebuild
        if not vs.get()["ids"]:
            vs = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=PERSIST_DIR)
    else:
        vs = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=PERSIST_DIR)

    return vs


def get_retrievers(vs: Chroma):
    salary_ret = vs.as_retriever(search_kwargs={"k": 2, "filter": {"topic": "salary"}})
    insur_ret = vs.as_retriever(search_kwargs={"k": 2, "filter": {"topic": "insurance"}})
    return salary_ret, insur_ret

# ============================
# Helpers: tools & agents
# ============================

def make_answering_tool(name: str, description: str, retriever, llm: ChatGroq) -> StructuredTool:
    """A tool that retrieves context and returns a final, formatted answer."""
    tmpl = PromptTemplate.from_template(
        "You are a helpful assistant.\n" + ANSWER_TEMPLATE + "\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:\n"
    )

    class QueryInput(BaseModel):
        query: str = Field(..., description="The user's question to answer")

    def _run(query: str) -> str:
        docs = retriever.invoke(query)
        context = "\n".join(d.page_content for d in docs)
        prompt = tmpl.format(context=context, question=query)
        return llm.invoke(prompt).content.strip()

    return StructuredTool.from_function(
        name=name,
        description=description,
        func=_run,
        args_schema=QueryInput,
    )


def bind_tools_to_prompt(prompt: ChatPromptTemplate, tools: list[StructuredTool]) -> ChatPromptTemplate:
    lines = [f"- {t.name}: {t.description}" for t in tools]
    tools_desc = "\n".join(lines) if lines else "None"
    tool_names = ", ".join([t.name for t in tools]) if tools else "None"
    return prompt.partial(tools=tools_desc, tool_names=tool_names)


def make_specialist_agents(llm: ChatGroq, salary_ret, insur_ret):
    salary_tool = make_answering_tool(
        name="salary_answer",
        description="Answer salary questions (gross, in-hand, PF, deductions) using the salary explainer.",
        retriever=salary_ret,
        llm=llm,
    )
    insur_tool = make_answering_tool(
        name="insurance_answer",
        description="Answer insurance questions (coverage, premium, co-pay, claims) using the insurance explainer.",
        retriever=insur_ret,
        llm=llm,
    )

    salary_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are the Salary Agent. ONLY answer salary/pay/CTC/deductions/tax/payslip questions.\n"
         "You have exactly these tools available:\n{tools}\n\n"
         "Valid tool names are: {tool_names}\n"
         "Use tools to ground your answers.\n"
         "Always distinguish Employer PF (CTC, not deducted) vs Employee PF (deducted from in-hand).\n"
         + ANSWER_TEMPLATE
         ),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    salary_prompt = bind_tools_to_prompt(salary_prompt, [salary_tool])

    insurance_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are the Insurance Agent. ONLY answer insurance questions: coverage, premiums, co-pay, claims, networks.\n"
         "You have exactly these tools available:\n{tools}\n\n"
         "Valid tool names are: {tool_names}\n"
         "Use tools to ground your answers.\n"
         + ANSWER_TEMPLATE
         ),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    insurance_prompt = bind_tools_to_prompt(insurance_prompt, [insur_tool])

    salary_agent = create_tool_calling_agent(llm, tools=[salary_tool], prompt=salary_prompt)
    insur_agent = create_tool_calling_agent(llm, tools=[insur_tool], prompt=insurance_prompt)

    salary_exec = AgentExecutor(agent=salary_agent, tools=[salary_tool], handle_parsing_errors=True, verbose=False)
    insur_exec = AgentExecutor(agent=insur_agent, tools=[insur_tool], handle_parsing_errors=True, verbose=False)
    return salary_exec, insur_exec


def make_coordinator(llm: ChatGroq, salary_exec: AgentExecutor, insur_exec: AgentExecutor):
    class QueryInput(BaseModel):
        query: str = Field(..., description="The user's question to route")

    def _call_salary(query: str) -> str:
        res = salary_exec.invoke({"input": query, "chat_history": []})
        return res["output"] if isinstance(res, dict) and "output" in res else str(res)

    def _call_insurance(query: str) -> str:
        res = insur_exec.invoke({"input": query, "chat_history": []})
        return res["output"] if isinstance(res, dict) and "output" in res else str(res)

    call_salary = StructuredTool.from_function(
        name="call_salary_agent",
        description="Route questions about salary, in-hand pay, CTC, gross/net salary, PF, tax, deductions, payslips.",
        func=_call_salary,
        args_schema=QueryInput,
    )
    call_insur = StructuredTool.from_function(
        name="call_insurance_agent",
        description="Route questions about insurance: coverage, premium, co-pay, claim process, hospitalization, network.",
        func=_call_insurance,
        args_schema=QueryInput,
    )

    coord_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are the Coordinator. Decide which agent to call.\n"
         "You have these tools:\n{tools}\n\n"
         "Rules:\n"
         "1. ALWAYS pick exactly ONE: call_salary_agent OR call_insurance_agent.\n"
         "2. NEVER answer yourself.\n"
         "3. If unclear, ask ONE clarifying question.\n\n"
         "💡 Routing rules:\n"
         "- Salary Agent: salary, in-hand, CTC, PF, tax, professional tax, gross, net, deductions, payslips.\n"
         "- Insurance Agent: coverage, premium, co-pay, claims, hospitalization, dependents, exclusions, TPA, network.\n"
         ),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    coord_prompt = bind_tools_to_prompt(coord_prompt, [call_salary, call_insur])

    coordinator = create_tool_calling_agent(llm, tools=[call_salary, call_insur], prompt=coord_prompt)
    return AgentExecutor(
        agent=coordinator,
        tools=[call_salary, call_insur],
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        verbose=False,
    )

# ============================
# Guardrail (patched)
# ============================
BANNED_SNIPPETS = (
    "thank you for the", 
    "thank you for the information", 
    "thank you for the clarification",
    "the tool", 
    "based on the tool", 
    "as per the tool", 
    "tool call", 
    "observation"
)
MIN_LEN = 60

def sanitize_or_fallback(answer: str, question: str, retriever, llm_google) -> str:
    ans = (answer or "").strip()
    low = ans.lower()

    bad = (
        not ans
        or len(ans) < MIN_LEN
        or any(snip in low for snip in BANNED_SNIPPETS)
        or "•" not in ans
    )
    if not bad:
        return ans

    # ✅ Only fetch top 2 docs for speed
    docs = retriever.invoke(question)[:2]
    context = "\n".join(d.page_content[:500] for d in docs)

    fallback = (
        "You are a helpful assistant.\n" + ANSWER_TEMPLATE +
        f"\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:\n"
    )

    # ⚡ Always fallback to Gemini for safe answer
    response = llm_google.generate_content(fallback)
    return response.text.strip()

# ============================
# Build agents
# ============================
def make_google_llm(model_name="gemini-1.5-pro", temperature=0.2):
    """Wrapper for Google Generative AI LLM"""
    return genai.GenerativeModel(model_name=model_name)

def build_agents(uploaded_files=None):
    docs = load_docs(uploaded_files)  # handles both uploads + fallback
    if not docs:
        st.stop()

    vs = build_store(docs)
    salary_ret, insur_ret = get_retrievers(vs)

    # ⚡ Coordinator → still use Groq (fast, low tokens)
    llm_coord = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        groq_api_key=GROQ_API_KEY,
    )

    # 🧠 Specialist agents → Google Gemini (high quota, accurate)
    llm_agent = make_google_llm("gemini-1.5-pro")

    # Salary + Insurance agents use Gemini
    salary_exec, insur_exec = make_specialist_agents(llm_agent, salary_ret, insur_ret)

    # Coordinator uses Groq
    coordinator = make_coordinator(llm_coord, salary_exec, insur_exec)

    # Return both
    return llm_agent, vs, salary_ret, insur_ret, coordinator
# ============================
# UI
# ============================

st.title("🤝 Multi-Agent RAG: Salary + Insurance")

with st.sidebar:
    st.markdown("### Setup")
    st.caption("Uses data/salary.txt and data/insurance.txt")
    rebuild = st.button("Rebuild Vector Store")
    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader("Upload salary/insurance docs", type=["txt", "docx"], accept_multiple_files=True)
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat = []
        st.session_state.history_msgs = []
    if "chat" in st.session_state and st.session_state.chat:
        chat_text = "\n".join([f"{r}: {m}" for r, m in st.session_state.chat])
        fname = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        st.download_button("📤 Download chat.txt", chat_text, file_name=fname)

    
    with st.expander("🔎 Show retrieved snippets (top 4)"):
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
                    docs = st.session_state.salary_ret.invoke(last_user_q) + st.session_state.insur_ret.invoke(last_user_q)
                    st.markdown("**Mixed/Unknown route: showing both retrievers**")

                for i, d in enumerate(docs[:4], 1):
                    st.write(f"{i}. {d.page_content[:500]}{'…' if len(d.page_content) > 500 else ''}")
                    meta = d.metadata
                    st.caption(f"Source: {meta.get('source','?')}  |  Topic: {meta.get('topic','?')}")
        else:
            st.write("Ask a question to see retrieved snippets.")

if "coordinator" not in st.session_state or rebuild:
    if not GROQ_API_KEY:
        st.error("❌ Missing GROQ_API_KEY in .env")
        st.stop()
    with st.spinner("Building vector store & agents…"):
        llm, vs, salary_ret, insur_ret, coordinator = build_agents(uploaded_files)
        st.session_state.llm = llm
        st.session_state.vs = vs
        st.session_state.salary_ret = salary_ret
        st.session_state.insur_ret = insur_ret
        st.session_state.coordinator = coordinator
        if "history_msgs" not in st.session_state:
            st.session_state.history_msgs = []
        if "chat" not in st.session_state:
            st.session_state.chat = []



user_q = st.chat_input("Ask about salary or insurance…")

# if follow-up clicked
if "pending_q" in st.session_state and st.session_state.pending_q:
    user_q = st.session_state.pending_q
    st.session_state.pending_q = None

if user_q:
    # ✅ keep only last 3 turns (6 msgs)
    recent_history = st.session_state.history_msgs[-6:]
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            result = st.session_state.coordinator.invoke({
                "input": user_q,
                "chat_history": recent_history,
            })

    answer = result.get("output", "").strip()
    used_tool = None
    for (action, _obs) in result.get("intermediate_steps", []):
        used_tool = getattr(action, "tool", used_tool)

    if not used_tool:
        # 🚨 Coordinator misbehaved → force fallback based on query
        retriever = (
            st.session_state.salary_ret if "salary" in user_q.lower()
            else st.session_state.insur_ret
        )
        answer = sanitize_or_fallback(answer, user_q, retriever, st.session_state.llm)
        agent_label = "Fallback"
    else:
        # ✅ Normal tool-based route
        retriever = (
            st.session_state.salary_ret
            if used_tool == "call_salary_agent"
            else st.session_state.insur_ret
        )
        answer = sanitize_or_fallback(answer, user_q, retriever, st.session_state.llm)
        agent_label = (
            "Salary Agent" if used_tool == "call_salary_agent" else "Insurance Agent"
        )

    # ✅ transcript + memory
    st.session_state.chat.append(("user", user_q))
    st.session_state.chat.append((agent_label, answer))
    st.session_state.history_msgs.append(HumanMessage(content=user_q))
    st.session_state.history_msgs.append(AIMessage(content=answer))

# ✅ Render transcript (unchanged)
for role, msg in st.session_state.get("chat", []):
    if role == "user":
        with st.chat_message("user"):
            st.markdown(msg)
    else:
        with st.chat_message("assistant"):
            st.markdown(
                f"<span style='background-color:#e5e7eb; color:#111827; "
                f"padding:2px 8px; border-radius:8px; font-size:12px; "
                f"font-weight:600;'>{role}</span>",
                unsafe_allow_html=True,
            )
            lines = msg.split("•")
            if lines:
                st.markdown(f"**{lines[0].strip()}**")
                for ln in lines[1:]:
                    clean = ln.strip()
                    if clean:
                        st.markdown(f"- {clean}")

            if role == "Salary Agent":
                for idx, s in enumerate(["What is gross salary?", "Is employer PF part of in-hand?"]):
                    if st.button(s, key=f"follow_{role}_{idx}_{len(st.session_state.chat)}"):
                        st.session_state.pending_q = s
                        st.rerun()

            elif role == "Insurance Agent":
                for idx, s in enumerate(["What is co-pay?", "How is premium decided?"]):
                    if st.button(s, key=f"follow_{role}_{idx}_{len(st.session_state.chat)}"):
                        st.session_state.pending_q = s
                        st.rerun()


