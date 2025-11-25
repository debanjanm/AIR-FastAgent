#------------------------------------------------------------------------------##
import pickle
# from langchain_community.retrievers import BM25Retriever # Import is still needed for loading

# # Define the file path where the retriever is saved
# file_path = "bm25_retriever.pkl"

# # Load the BM25Retriever using pickle
# with open(file_path, "rb") as f:
#     loaded_bm25_retriever = pickle.load(f)

# print(f"BM25Retriever loaded from {file_path}")

# # You can now use the loaded_bm25_retriever
# # query = "lazy dog"
# query = "Explain PopularityAdjusted Block Model (PABM)"
# relevant_docs = loaded_bm25_retriever.invoke(query)
# for doc in relevant_docs:
#     print(doc.page_content)

##------------------------------------------------------------------------------##
## Set up environment variables for LM Studio
import os
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1/"
os.environ["OPENAI_API_KEY"] = "test"

##------------------------------------------------------------------------------##
# Set up a LangChain pipeline with prompt template and LLM
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# choose model name exactly as LM Studio exposes it (check LM Studio UI)
# llm = ChatOpenAI(model="qwen/qwen3-4b-thinking-2507", temperature=0.2)  

llm = ChatOpenAI(model="qwen/qwen3-4b-2507", temperature=0)  


# template = "Answer the Question based on the context below.\n\nContext: {context}\n\nQ: {question}\nA:"
# prompt = PromptTemplate.from_template(template)

# # prompt = PromptTemplate(input_variables=["q"], template="Context: Q: {q}\nA:")
# chain = prompt | llm | StrOutputParser()

# print(chain.run("Explain recursion in 2 sentences."))

# print(chain.invoke({"context": relevant_docs, "question": "Explain PopularityAdjusted Block Model (PABM)"}))

##------------------------------------------------------------------------------##

# pip install -qU "langchain[anthropic]" to call the model

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver 

# def get_weather(city: str) -> str:
#     """Get weather for a given city."""
#     return f"It's always sunny in {city}!"

def retrieve_context(query: str) -> str:
    """Retrieve context for a given query using the loaded BM25Retriever."""

    file_path = "bm25_retriever.pkl"

    # Load the BM25Retriever using pickle
    with open(file_path, "rb") as f:
        loaded_bm25_retriever = pickle.load(f)
    docs = loaded_bm25_retriever.invoke(query)
    return "\n".join(doc.page_content for doc in docs)

agent = create_agent(
    model=llm,
    tools=[retrieve_context],
    system_prompt="You are a helpful assistant. you have access to a tool that retrieves relevant context based on user queries.",
    checkpointer=InMemorySaver(),
)

# Run the agent
sl = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain PopularityAdjusted Block Model (PABM)"}]},
    {"configurable": {"thread_id": "1"}},
)

print("Agent Response:{}".format(sl))

response = sl.get('messages')[1]

print("------------------------------")
print("Agent Response:{}".format(response.content))


##------------------------------------------------------------------------------##

