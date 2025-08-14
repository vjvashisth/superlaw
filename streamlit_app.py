# streamlit_app.py
import os
import sys
import pathlib
import traceback
import streamlit as st

# ---------------------------------------------------------
# Paths & flexible import strategy
# ---------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent

# Try package-style imports first (requires src/__init__.py),
# then fall back to adding a 'src' folder directly to sys.path.
def _import_rag_modules():
    # First attempt: package style
    try:
        from src.rag_chat import generate_rag_response
        from src.embed_and_store import load_faiss_index
        return generate_rag_response, load_faiss_index, "package"
    except ModuleNotFoundError:
        pass

    # Second attempt: add a flat src path to sys.path and import modules
    candidate_src_dirs = [
        ROOT / "src",                  # layout: ./src/*.py
        ROOT / "superlaw" / "src",     # layout: ./superlaw/src/*.py
    ]
    for p in candidate_src_dirs:
        if (p / "rag_chat.py").exists() and (p / "embed_and_store.py").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            try:
                from rag_chat import generate_rag_response
                from embed_and_store import load_faiss_index
                return generate_rag_response, load_faiss_index, "flat"
            except Exception:
                traceback.print_exc()
                raise

    raise RuntimeError(
        "Could not locate rag_chat.py and embed_and_store.py in ./src or ./superlaw/src.\n"
        "Make sure your repository structure is one of:\n"
        "  superlaw/\n"
        "    streamlit_app.py\n"
        "    src/\n"
        "      rag_chat.py\n"
        "      embed_and_store.py\n"
        "OR\n"
        "  superlaw/\n"
        "    streamlit_app.py\n"
        "    superlaw/\n"
        "      src/\n"
        "        rag_chat.py\n"
        "        embed_and_store.py\n"
        "If using package imports, ensure src/__init__.py exists."
    )

generate_rag_response, load_faiss_index, IMPORT_STYLE = _import_rag_modules()

# ---------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------
st.set_page_config(page_title="Superlaw RAG", page_icon="🧠", layout="wide")
st.title("🧠 Superlaw — RAG Chat")

with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Top K chunks", 1, 20, 5)
    st.caption("Uses FAISS index at ./vector_index/")

# ---------------------------------------------------------
# Secrets / API key
# ---------------------------------------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))  # noqa: S105
if OPENAI_API_KEY and not os.getenv("OPENAI_API_KEY"):
    # Ensure OpenAI client (inside your files) can find it
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if not os.getenv("OPENAI_API_KEY"):
    st.warning("Set OPENAI_API_KEY in Streamlit secrets or environment.", icon="⚠️")

# ---------------------------------------------------------
# Load FAISS index lazily (once per session)
# ---------------------------------------------------------
if "faiss_loaded" not in st.session_state:
    st.session_state.faiss_loaded = False
    st.session_state.index = None
    st.session_state.metadata = None
    st.session_state.load_error = None

if not st.session_state.faiss_loaded:
    idx_path = ROOT / "vector_index" / "faiss.index"
    meta_path = ROOT / "vector_index" / "index_metadata.pkl"
    if idx_path.exists() and meta_path.exists():
        try:
            index, metadata = load_faiss_index(str(idx_path), str(meta_path))
            st.session_state.index = index
            st.session_state.metadata = metadata
            st.session_state.faiss_loaded = True
        except Exception as e:
            st.session_state.load_error = f"Failed to load FAISS index: {e}"
    else:
        st.session_state.load_error = "Missing FAISS index or metadata. Build them under ./vector_index/"

# ---------------------------------------------------------
# Tabs: Chat / Index
# ---------------------------------------------------------
tab_chat, tab_index = st.tabs(["💬 Chat", "🗂️ Index"])

with tab_chat:
    st.subheader("Ask questions about your documents")

    if st.session_state.load_error:
        st.error(st.session_state.load_error)
        st.stop()

    # Chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

    user_q = st.chat_input("Type your question...")
    if user_q:
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Some versions of your rag_chat may accept (question, k, index, metadata, api_key)
                # Others may only accept (question). We'll try the richer signature first.
                try:
                    answer = generate_rag_response(
                        user_q,
                        k=top_k,
                        index=st.session_state.index,
                        metadata=st.session_state.metadata,
                        api_key=os.getenv("OPENAI_API_KEY"),
                    )
                except TypeError:
                    # Fallback to simplest signature
                    answer = generate_rag_response(user_q)
                except Exception as e:
                    st.error(f"Error generating answer:\n\n{e}")
                    st.stop()

                st.markdown(answer)

        st.session_state.history.append(("user", user_q))
        st.session_state.history.append(("assistant", answer))

with tab_index:
    st.subheader("Index status")
    idx_path = ROOT / "vector_index" / "faiss.index"
    meta_path = ROOT / "vector_index" / "index_metadata.pkl"
    st.write(f"Import style: `{IMPORT_STYLE}`")
    if idx_path.exists() and meta_path.exists():
        st.success("FAISS index found.")
        st.code(f"{idx_path}\n{meta_path}")
        try:
            size_bytes = idx_path.stat().st_size
            st.write(f"Index size: {size_bytes:,} bytes")
        except Exception:
            pass
    else:
        st.info(
            "No FAISS index found. Generate it using your existing pipeline "
            "(embed_and_store.py) so the chat can retrieve context."
        )
