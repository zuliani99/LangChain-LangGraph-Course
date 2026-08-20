from typing import Any, Dict, List

import streamlit as st

from backend.core import run_llm

# helper function to format the sources of the retrieved documents
def _format_sources(context_docs: List[Any]) -> List[str]:
    return [
        str(meta.get("source", "Unknown"))
        for doc in (context_docs or [])
        if(meta := (getattr(doc, "metadata", None) or {})) is not None
    ]


st.set_page_config(page_title="LangChain Documentation Assistant", page_icon=":books:", layout="centered")
st.title("LangChain Documentation Assistant :books:")

with st.sidebar:
    st.subheader("Session")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Ask me anything about LangChain docs. I'll retreive relevant context and cite sources.",
            "sources": [], # list of sources for the assistant's response
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}") # "-" is used to create a bullet point in markdown

prompt = st.chat_input("Ask me anything about LangChain docs...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retreiving docs and generating response..."):
                result: Dict[str, Any] = run_llm(prompt)
                answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
                sources = _format_sources(result.get("context", []))

            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)