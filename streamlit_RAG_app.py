import streamlit as st
import uuid
import threading
from common.realtime_record import TestSt
#from llama_index import VectorStoreIndex, ServiceContext, Document
##from llama_index.llms import OpenAI
#import openai
#from llama_index import SimpleDirectoryReader
from alibaba_sdk.alibaba_text_moderation import TextAutoRoute

st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
#openai.api_key = st.secrets.openai_key
st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")
st.info("Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about DBS annual report in 2022!"}
    ]

import time
def streaming_print(txt, markdown_placeholder):
    full_txt = ""
    for chunk in txt.split(" "):
        full_txt += chunk + " "
        markdown_placeholder.markdown(full_txt + "🐌")
        time.sleep(0.15)
    markdown_placeholder.markdown(full_txt)

@st.cache_resource(show_spinner=False)
def load_data():
    #from langchain_community.llms import Tongyi
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        from common.utils import Utils
        import os
        os.environ["DASHSCOPE_API_KEY"] = Utils.get_tongyi_key()
        #llm = Tongyi(model_name= 'qwen-max-1201', temperature = 0)
        from langchain_community.llms import Tongyi
        qwen = Tongyi()

        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        model_name = "BAAI/bge-base-en-v1.5"
        encode_kwargs = {'normalize_embeddings' : True}
        embedding_model = HuggingFaceBgeEmbeddings(
            model_name = model_name,
            encode_kwargs = encode_kwargs,
            query_instruction = "Represent this sentence for searching relevant passages:"
        )

        from llama_index.core import Settings
        Settings.llm = qwen
        Settings.embed_model = embedding_model

    #with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        from llama_index.core import StorageContext, load_index_from_storage

        storage_dbs_doc = StorageContext.from_defaults(persist_dir = './data/subquery/dbs')
        storage_dbs_index = load_index_from_storage(storage_dbs_doc)
        query_engine = storage_dbs_index.as_query_engine(streaming = True)
        return query_engine

@st.cache_resource
def return_flag():
    event = threading.Event()
    flag = [0, False, False, event, '', ['']]
    return flag

query_engine = load_data()
flag = return_flag()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = query_engine

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    flag[4] += prompt
    st.session_state.messages.append({"role": "user", "content":flag[4]})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        #if message['role'] == 'user':
        st.markdown(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        #\nIf user asked question in Chinese, you should generate query in English then give the final answer in Chinese.
        #If user asked question in English, you should generate English query in English then give the final answer in English.
        try:
            streaming_response = query_engine.query(flag[4])
            full_response = ""
            markdown_placeholder = st.empty()
            for chunk in streaming_response.response_gen:
                #st.write(step_response)
                if not isinstance(chunk, str):
                    chunk = f"**Final Answer**: {chunk.response}"
                chunk = chunk.replace('$', '\$')
                #streaming_print(chunk, markdown_placeholder)
                full_response += chunk
                markdown_placeholder.markdown(full_response + "✍️")
                time.sleep(0.15)
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
        #with col2:
        #    voice_end = st.button('End voice')
        with col2:
            send_record = st.button('Send Input')

        #print(f'step 0 {voice_start, voice_end}')
        with st.spinner("Voice Inputing..."):
            #voice_start = st.button('microphone')
            #print(f'step 1 {voice_start, voice_end, flag[4]}')
            markdown_placeholder = st.empty()
            #full_response = ""
            start = time.time()
            #print(f'flag {flag[-1]}')
            #print(f'step 2 {voice_start, voice_end}')

            messages = []
            msg_len = 0
            t = TestSt(0, flag[3], messages)

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
            #if voice_end:
            #    flag[3].set()
            #    time.sleep(0.5)
            #    print('---voice end---')
            #    flag[1] = False
            #    flag[0] = 0
            
            while flag[1]:
                if flag[0] == 0:
                    print('***start thread***')
                    #print(f'voice_start 3 {voice_start, voice_end}')
                    flag[0] = 1
                    flag[4] = ""
                    flag[5] = ['']
                    start = time.time()
                    msg_len = len(messages)
                    t.start()
                if time.time() - start > 60:
                    flag[3].set()
                    flag[0] = 0
                    #flag[4] += ''.join([chunk['sentence'] for chunk in messages if chunk.get('sentence')])
                    #print(f'voice_start 3 {voice_start, voice_end}')
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
                    markdown_placeholder.markdown(flag[4] + "🐌")

            markdown_placeholder.markdown(flag[4])
            #send_record = st.button('send input')
            #print(f'message: {send_record} {flag[4]}')
            #print(f'send_record 3 {send_record},6 {full_response}')
            if send_record and flag[4] != "":
                #print(f'button {voice_start}, {voice_end}, {send_record}')
                print('save result')
                st.session_state.messages.append({"role": "user", "content":flag[4]})
                #send_record =False