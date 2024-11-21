from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import pprint

chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

system = "You are a helpful assistant."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat
result = chain.invoke({"text": "Explain the importance of low latency LLMs."})

print(result.content)
print(result.response_metadata)
pprint.pprint(result.response_metadata)