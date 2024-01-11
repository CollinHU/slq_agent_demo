import streamlit as st

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
        {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
    ]

import time
def streaming_print(txt, markdown_placeholder):
    full_txt = ""
    for chunk in txt.split(" "):
        full_txt += chunk + " "
        markdown_placeholder.markdown(full_txt + "🐌")
        time.sleep(0.15)
    markdown_placeholder.markdown(full_txt)

def parsing_result(chunk, markdown_placeholder):
        full_response = ""
        if "actions" in chunk:
            for action in chunk["actions"]:

                thought = action.log
                response = ""
                pure_thought = ""
                for chunk in thought.split('\n'):
                    if 'Action' not in chunk:
                        pure_thought += chunk + '\n'
                
                if len(pure_thought) > 0:
                    response += f"**Thought**: {pure_thought}"
                response += f"\n**Action**: ```{action.tool}``` with input ```{action.tool_input}```"
                full_response += response + '\n'
                streaming_print(response, markdown_placeholder)
                #st.write(response)
            # Observation
        elif "steps" in chunk:
            for step in chunk["steps"]:
                response = f"**Observation**: ```{step.observation}```"
                full_response += response + '\n'
                #st.write(response)
                streaming_print(response, markdown_placeholder)
            # Final result
        elif "output" in chunk:
            response = f"**Final Result**: {chunk['output']}"
            full_response += response
            streaming_print(response, markdown_placeholder)
            #st.write(f"Final Result: {chunk['output']}")
        else:
            raise ValueError
        return full_response

@st.cache_resource(show_spinner=False)
def load_data():
    from langchain_community.llms import Tongyi
    from common.utils import Utils
    import os
    os.environ["DASHSCOPE_API_KEY"] = Utils.get_tongyi_key()
    #llm = Tongyi(model_name= 'qwen-max-1201', temperature = 0)
    from common.CustomLLM import QwenLLM
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    token = Utils.get_tongyi_key()

    qwen = QwenLLM(url= url, model='qwen-max-1201', token=token, temperature=0.2)
    from langchain.embeddings import HuggingFaceBgeEmbeddings
    model_name = "BAAI/bge-base-en-v1.5"

    encode_kwargs = {'normalize_embeddings' : True}
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name = model_name,
        encode_kwargs = encode_kwargs,
        query_instruction = "Represent this sentence for searching relevant passages:"
    )

    from llama_index import VectorStoreIndex, SimpleDirectoryReader
    from llama_index.tools import QueryEngineTool, ToolMetadata
    from llama_index.query_engine.sub_question_query_engine import SubQuestionQueryEngineCustom
    from llama_index.callbacks import CallbackManager, LlamaDebugHandler
    from llama_index import ServiceContext
    service_context = ServiceContext.from_defaults(llm = qwen, embed_model = embedding_model)
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        from llama_index import StorageContext, load_index_from_storage

        storage_dbs_doc = StorageContext.from_defaults(persist_dir = './data/subquery/dbs')
        storage_dbs_index = load_index_from_storage(storage_dbs_doc, service_context=service_context)
        query_engine_tools = [
            QueryEngineTool(
                query_engine=storage_dbs_index.as_query_engine(),
                metadata=ToolMetadata(
                    name="DBS Holdings plc Annual Report and Accounts 2022",
                    description="Provide information about DBS Group Holdings Ltd financials for year 2022",
                ),
            ),
        ]

        query_engine = SubQuestionQueryEngineCustom.from_defaults(
            query_engine_tools=query_engine_tools,
            service_context=service_context,
            verbose=True,
            use_async=False,
        )
        return query_engine

query_engine = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = query_engine

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        #if message['role'] == 'user':
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            streaming_response = query_engine.query(prompt + ". if applicable, give the final result in markdown table format")
            full_response = []
            step_response = ""
            for chunk in streaming_response:
                #st.write(step_response)
                if not isinstance(chunk, str):
                    print('ss')
                    chunk = f"**Final Answer**: {chunk.response}"
                chunk = chunk.replace('$', '')
                markdown_placeholder = st.empty()
                streaming_print(chunk, markdown_placeholder)
                full_response.append(chunk)
                #st.write(step_response)
            #st.write(full_response)
            message = {"role": "assistant", "content": "\n".join(full_response)}
            st.session_state.messages.append(message) # Add response to message history
