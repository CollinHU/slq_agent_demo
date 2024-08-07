import streamlit as st
import threading
from common.realtime_record import TestSt
#from llama_index import VectorStoreIndex, ServiceContext, Document
##from llama_index.llms import OpenAI
#import openai
#from llama_index import SimpleDirectoryReader

st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
#openai.api_key = st.secrets.openai_key
st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")
st.info("Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about HSBC annual report in 2022!"}
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
        from common.CustomLLM import QwenLLM
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        token = Utils.get_tongyi_key()

        qwen = QwenLLM(url= url, model='qwen-max-1201', token=token, temperature=0.15)
        #from langchain.embeddings import HuggingFaceBgeEmbeddings
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        model_name = "/Users/collin/.cache/modelscope/hub/AI-ModelScope/bge-base-zh-v1.5"

        encode_kwargs = {'normalize_embeddings' : True}
        embedding_model = HuggingFaceBgeEmbeddings(
            model_name = model_name,
            encode_kwargs = encode_kwargs,
            query_instruction = "为这个句子生成表示用于检索相关文章：",
        )
        from llama_index import ServiceContext
        from llama_index import StorageContext, load_index_from_storage

        #print('test1')
        service_context = ServiceContext.from_defaults(llm = qwen, embed_model = embedding_model)
        storage_dbs_doc = StorageContext.from_defaults(persist_dir = './data/subquery/hsbc')
        storage_dbs_index = load_index_from_storage(storage_dbs_doc, service_context=service_context)
        v_db_retriever = storage_dbs_index.as_retriever()
        #print('test2')
        from common.SubQuestionEngine import SubQuestionQueryEngine

        doc_des = '匯豐控股公司2022年報及帳目'
        qwen = QwenLLM(url= url, model='qwen-max-1201', token=token, temperature=0.2)
        qa_chatbot = SubQuestionQueryEngine(retriever=v_db_retriever, doc_des=doc_des, llm=qwen)
        return qa_chatbot

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
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            #\nIf user asked question in Chinese, you should generate query in English then give the final answer in Chinese.
            #If user asked question in English, you should generate English query in English then give the final answer in English.
            requirements_prompt = """
            \n If applicable, give a table listing metric and other in text when giving the final result. You final answer should be the same language as the question
            \nIf user asked question in Chinese, you should generate query in English then give the final answer in Chinese."""
            try:
                streaming_response = query_engine.query(flag[4])
                full_response = []
                markdown_placeholder = st.empty()
                for chunk in streaming_response:
                    #st.write(step_response)
                    if not isinstance(chunk, str):
                        chunk = f"**Final Answer**: {chunk.response}"
                    chunk = chunk.replace('$', '\$')
                    #streaming_print(chunk, markdown_placeholder)
                    full_response.append(chunk)
                    step_response = ""
                    for word in chunk.split(" "):
                        step_response += word + " "
                        markdown_placeholder.markdown('\n'.join(full_response[:-1]) + '\n' + step_response + "🐌")
                        time.sleep(0.1)
            except Exception as e:
                print(f'error {e}')
                full_response = ["Your question is not clear, please re-orgonize your question again"]
                #st.write(step_response)
            #st.write(full_response)
            markdown_placeholder.markdown('\n'.join(full_response))
            flag[4] = ""
            flag[5] = ['']
            #time.sleep(2)
            #markdown_placeholder.markdown("\n".join(full_response))
            message = {"role": "assistant", "content": "\n".join(full_response)}
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