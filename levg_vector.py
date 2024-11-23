import getpass
import os
from openai import OpenAI


from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import pprint



LANGCHAIN_TRACING_V2=True
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="LANGCHAIN_API_KEY"
LANGCHAIN_PROJECT="levg_vector"

os.environ["GROQ_API_KEY"]

llm = ChatGroq(model="llama3-8b-8192")
client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=self.model_name,
        messages=conversation,
        temperature=0,
        top_p=1,
        frequency_penalty=0,    
        presence_penalty=0
    )
    return response.choices[0].message.content

# Generate Sample Documents
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

vectorstore = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(), 
)

# Return documents based on similarity to a string query
vectorstore.similarity_search("cat")

# Return documents based on similarity to an embedded query

embedding = OpenAIEmbeddings().embed_query("cat")
vectorstore.similarity_search_by_vector(embedding)

# Retrieve Documents
retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)  # select top result
retriever.batch(["cat", "shark"])
                
# To use a VectorStoreRetriever
retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": 1},)

retriever.batch(["cat", "shark"])

# Example
message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

response = rag_chain.invoke("tell me about cats")

print(response.content)