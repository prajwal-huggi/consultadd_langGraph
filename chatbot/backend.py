from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import os
import sqlite3

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    huggingfacehub_api_token= os.getenv("HUGGINGFACE_API_ENDPOINT"),
    max_new_tokens= 500,
    temperature= 0.3
)

model= ChatHuggingFace(llm= llm)

# Defining the state of the langgraph
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
def chat_node(state: ChatState):
    messages= state['messages']
    response= model.invoke(messages)
    return {'messages': [response]}

conn= sqlite3.connect(database='chatbot.db', check_same_thread= False)

checkpointer= SqliteSaver(conn= conn)
    
graph= StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot= graph.compile(checkpointer= checkpointer)

# For streaming instead of .invoke we use .stream
CONFIG= {'configurable': {'thread_id': 'thread-1'}}
stream= chatbot.stream(
    {'messages': [HumanMessage(content='What is my name?')]},
    config= CONFIG,
    stream_mode='messages'
)

print(type(stream))

for message_chunk, metadata in stream:
    if message_chunk.content:
        print(message_chunk.content, end=" ", flush= True)

# to get the chats with the help of thread id
print("************************************************************************************")
# print(chatbot.get_state(config= CONFIG))
