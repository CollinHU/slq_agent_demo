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
    #import openai
    #openai.api_base = "https://api.duckgpt.top/v1"
    import os
    os.environ['OPENAI_API_BASE']='https://api.chatanywhere.tech/v1'

    from langchain.agents import create_sql_agent
    from langchain_community.agent_toolkits import SQLDatabaseToolkit
    #from langchain.agents.agent_toolkits import SQLDatabaseToolkit
    from langchain.sql_database import SQLDatabase
    #from langchain.llms.openai import OpenAI
    #from langchain.agents import AgentExecutor
    from langchain.agents.agent_types import AgentType
    #from langchain.chat_models import ChatOpenAI
    from langchain_openai import ChatOpenAI
    
    from common.utils import Utils
    from common.agent_utils import base_suffix, custom_suffix_filter, custom_suffix_sim, SQL_SUFFIX_CUSTOM
    from common.agent_utils import create_retriever_filter, create_retriever_sim
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        db = SQLDatabase.from_uri('postgresql+psycopg2://flowise:flowise@localhost/metastore')
        llm = ChatOpenAI(model='gpt-4-1106-preview', temperature=0, openai_api_key = Utils.get_openai_key())
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        custom_tool_list_1 = [ create_retriever_sim(openai_key=Utils.get_openai_key())]
        custom_tool_list_2 = [ create_retriever_filter(opai_key=Utils.get_openai_key())]
        agent_compose = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            extra_tools=custom_tool_list_1 + custom_tool_list_2,
            suffix=custom_suffix_sim + custom_suffix_filter + base_suffix + SQL_SUFFIX_CUSTOM,
        )
        return agent_compose

sql_agent = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = sql_agent

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
            streaming_response = st.session_state.chat_engine.stream({'input': prompt})
            full_response = []
            step_response = ""
            for chunk in streaming_response:
                #st.write(step_response)
                markdown_placeholder = st.empty()
                step_response = parsing_result(chunk, markdown_placeholder)
                full_response.append(step_response)
                #st.write(step_response)
            #st.write(full_response)
            message = {"role": "assistant", "content": "\n".join(full_response)}
            st.session_state.messages.append(message) # Add response to message history
