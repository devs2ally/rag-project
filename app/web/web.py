import streamlit as st
from app.engine import get_rag_chain

st.set_page_config(page_title="나만의 RAG 챗봇", page_icon="🤖")

st.title("🤖 지식 기반 AI 챗봇")
st.caption("PostgreSQL과 Llama 3.1로 구동되는 RAG 시스템입니다.")

# RAG 체인 초기화 (캐싱하여 속도 향상)
@st.cache_resource
def load_chain():
    return get_rag_chain()


chain = load_chain()

# 채팅 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 내용 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 받기
if prompt := st.chat_input("질문을 입력하세요..."):
    # 1. 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. AI 답변 생성 및 표시
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            response = chain.invoke(prompt)
            # RetrievalQA는 딕셔너리를 반환하므로 'result' 키값을 가져옵니다.
            answer = response['result'] if isinstance(response, dict) else response
            st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})