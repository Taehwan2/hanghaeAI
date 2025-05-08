import streamlit as st
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

model = ChatOpenAI(model="gpt-4.1-mini", api_key=OPEN_API)
st.title("GPTBOT")

if images := st.file_uploader("본인의 전신이 보이는 사진을 올려주세요!", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True):
    image_url = []
    for image in images:
        st.image(image)
        image_url.append(base64.b64encode(image.read()).decode("utf-8"))

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"][0]['text'])
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("강아지와 고양이 사진을 올리고 아무 질문이나 던져주세요!"):
    with st.chat_message("user"):
        st.markdown(prompt)
    content = [
        {"type": "text", "text": prompt}
    ]
    for image in image_url:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"}
        })
    st.session_state.messages.append({"role": "user", "content": content})
    with st.chat_message("assistant"):
        messages = []
        for m in st.session_state.messages:
            if m["role"] == "user":
                messages.append(HumanMessage(content=m["content"]))
            else:
                messages.append(AIMessage(content=m["content"]))

        result = model.invoke(messages)
        response = result.content

        st.markdown(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })