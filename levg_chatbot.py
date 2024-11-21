import os

os.environ["GROQ_API_KEY"]

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

#Managing Conversation History
from langchain_core.messages import SystemMessage, trim_messages

model = ChatGroq(model="llama3-8b-8192")
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter= model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Add a prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Set class
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str



# Define the function that calls the model
def call_model(state: MessagesState):
    chain = prompt | model
    trimmed_messages = trimmer.invoke(state["messages"])
    response = chain.invoke(
        {"messages": trimmed_messages, "language":language}
    )
    return {"messages": response}



# Define the (single) node in the graph
workflow = StateGraph(state_schema=MessagesState)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# To support multiple conversations
config = {"configurable": {"thread_id": "abc123"}}



result= model.invoke([HumanMessage(content="Hi! I'm Levent")])
result= model.invoke([HumanMessage(content="What's my name?")])
result= model.invoke(
    [
        HumanMessage(content="Hi! I'm Levent"),
        AIMessage(content="Hello Levent! How can I assist you today?"),
        HumanMessage(content="What's my name?")
        
    ]
)

query="Hi! I'm Levent. Please tell me a joke."
language="English"
input_messages = [HumanMessage(query)]
for chunk, metadata in app.stream(
    {"messages": input_messages, "language": language},
    config,
    stream_mode= "messages",
):
    if isinstance(chunk, AIMessage): #Filter to just model responses
        print(chunk.content, end="|")

output = app.invoke({"messages": input_messages, "language": language}, config)
output["messages"][-1].pretty_print()  # output contains all messages in state
#print(result.content)
