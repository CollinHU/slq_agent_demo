import streamlit as st

import uuid
import threading
from http import HTTPStatus
import time
import json

from common.realtime_record import TestSt
from common.utils import Utils
from common.ContentModeration import ContentModeration

from dashscope import Generation
import dashscope
from alibaba_sdk.alibaba_text_moderation import TextAutoRoute

dashscope.api_key = Utils.get_tongyi_key()
VOCIE_TOKEN=Utils.get_alibaba_voice_recognition_token()

st.set_page_config(page_title="Qwen Demo", page_icon="Qwen", layout="centered", initial_sidebar_state="auto", menu_items=None)
#openai.api_key = st.secrets.openai_key
st.title("Qwen Demo")
#st.info("Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")

st.sidebar.title('Sidebar')
model_name = st.sidebar.radio("Choose a model:", ("Qwen-max-0428","Qwen2-72b-instruct (Open Source)"))
voice_recognition = st.sidebar.radio("Choose a language for voice recognition:", ("Mandarin","English", "Cantonese"))
counter_placeholder = st.sidebar.empty()
clear_button = st.sidebar.button("Clear Conversation", key='Clear')



if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

if model_name == "Qwen-max-0428":
    model = "qwen-max-0428"
else:
    model = "qwen2-72b-instruct"

if voice_recognition == "Mandarin":
    VOCIE_APPKEY=Utils.get_alibaba_voice_recognition_appKey()
elif voice_recognition == "English":
    VOCIE_APPKEY=Utils.get_alibaba_voice_recognition_EnglishAppKey()
else:
    VOCIE_APPKEY=Utils.get_alibaba_voice_recognition_CantoneseAppKey()

if clear_button:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

@st.cache_resource
def return_flag():
    event = threading.Event()
    flag = [0, False, False, event, '', ['']]
    return flag

@st.cache_resource
def load_content_moderation():
    clt = ContentModeration()
    return clt

moderation_client = load_content_moderation()

flag = return_flag()

#if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
#        st.session_state.chat_engine = query_engine

if prompt := st.chat_input("Input Your Question Here"): # Prompt for user input and save to chat history
    # add moderation part
    flag[4] += prompt

    moderation_res = moderation_client.text_moderation(text = flag[4], service= 'llm_query_moderation')
    if moderation_res == 'TRUE':
        st.session_state.messages.append({"role": "user", "content":flag[4]})
    else:
        st.session_state.messages.append({"role": "user", "content": flag[4]})
        st.session_state.messages.append({"role": "assistant", "content": moderation_res})
        flag[4] = ""


for message in st.session_state.messages: # Display the prior chat messages
    if message['role'] == 'system':
        continue
    with st.chat_message(message["role"]):
        #if message['role'] == 'user':
        st.markdown(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        #\nIf user asked question in Chinese, you should generate query in English then give the final answer in Chinese.
        #If user asked question in English, you should generate English query in English then give the final answer in English.
        try:
            streaming_response = Generation.call(
                model= model,
                messages = [
                    {'role': msg['role'], 'content':msg['content']}
                    for msg in st.session_state.messages
                ],
                result_format='message',
                stream=True,
                incremental_output=True
            )
            full_response = ""
            markdown_placeholder = st.empty()
            sessionId = str(uuid.uuid1())
            for response in streaming_response:
                if response.status_code == HTTPStatus.OK:
                    chunck = response.output.choices[0]['message']['content']
                    #print(sessionId)
                    moderation_res = moderation_client.text_moderation_streaming(text = chunck, sessionId=sessionId, service= 'llm_response_moderation')
                    #print(moderation_res)
                    if moderation_res != 'TRUE':
                        full_response = moderation_res
                        break

                    full_response += chunck
                    markdown_placeholder.markdown(full_response + '✍️')
                else:
                    #print('Not 200',response.message)
                    full_response = response.message
        except Exception as e:
            print(f'error {e}')

            full_response = "Your question is not clear, please re-orgonize your question again"
            #st.write(step_response)
        
        markdown_placeholder.markdown(full_response)
        flag[4] = ""
        flag[5] = ['']
        #time.sleep(2)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message) # Add response to message history

st.markdown("""
            <style>
                div[data-testid="column"] {
                    width: fit-content !important;
                    flex: unset;
                }
                div[data-testid="column"] * {
                    width: fit-content !important;
                }
            </style>
            """, unsafe_allow_html=True)

if st.session_state.messages[-1]["role"] != "user":
    with st.chat_message("user"):
        full_response =""
        
        col1, col2 = st.columns(2)
        with col1:
            voice_start = st.button('Microphone Button')
        
        with col2:
            send_record = st.button('Send Input')

        with st.spinner("Voice Inputing..."):
            
            markdown_placeholder = st.empty()
            start = time.time()

            messages = []
            msg_len = 0
            t = TestSt(0, flag[3], messages, VOCIE_TOKEN, VOCIE_APPKEY)

            if voice_start:
                if not flag[1]:
                    print('---voice start---')
                    flag[3].clear()
                    flag[1] = True
                else:
                    flag[3].set()
                    time.sleep(0.1)
                    print('---voice end---')
                    flag[1] = False
                    flag[0] = 0
            
            while flag[1]:
                if flag[0] == 0:
                    print('***start thread***')
                    
                    flag[0] = 1
                    flag[4] = ""
                    flag[5] = ['']
                    start = time.time()
                    msg_len = len(messages)
                    t.start()
                if time.time() - start > 60:
                    flag[3].set()
                    flag[0] = 0
                    
                    print('time out')
                    flag[1] = False
                    break
                time.sleep(0.1)
                if len(messages) > msg_len:
                    msg_len = len(messages)
                    chunck = messages[-1]
                    msg = list(chunck.values())[0]
                    if chunck.get('sentence'):
                        flag[5][-1] = msg
                        flag[4] = ''.join(flag[5])
                        flag[5].append('')
                    else:
                        flag[5][-1] = msg
                        flag[4] = ''.join(flag[5])
                    markdown_placeholder.markdown(flag[4] + "✍️")

            markdown_placeholder.markdown(flag[4])
            if send_record and flag[4] != "":
                markdown_placeholder.markdown('')

                print('save result')
                moderation_res = moderation_client.text_moderation(text = flag[4], service= 'llm_query_moderation')
                if moderation_res == 'TRUE':
                    st.session_state.messages.append({"role": "user", "content":flag[4]})
                else:
                    print('ERROR')
                    st.session_state.messages.append({"role": "user", "content": flag[4]})
                    st.session_state.messages.append({"role": "assistant", "content": moderation_res})
                    flag[4] = ""
