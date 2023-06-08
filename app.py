
import streamlit as st
from ingestion import run_llm
from streamlit_chat import message

st.set_page_config(page_title="Geregè", page_icon=":robot_face:")
st.header("Chat with Geregè")



prompt = st.text_input("Prompt", placeholder="Enter your question here...")

#Leveraging Streamlit session_state, which will work as memory in our case. It is unique for each session.

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )

        st.session_state["user_prompt_history"].append(prompt) #Appending user input (prompt)
        st.session_state["chat_answers_history"].append(generated_response["answer"]) #Appending LLM answer
        st.session_state["chat_history"].append((prompt, generated_response["answer"])) #Combining both into one empty list, which includes a tuple that takes a string as 1st input & Str 2nd.



#ChatGPT fix
if st.session_state["chat_answers_history"]:
    for i, (generated_response, user_query) in enumerate(
        zip(
            st.session_state["chat_answers_history"],
            st.session_state["user_prompt_history"],
        )
    ):
        message(user_query, is_user=True, key=f"user_msg_{i}",avatar_style="croodles-neutral") # align's the message to the right
        message(generated_response, key=f"bot_msg_{i}")
