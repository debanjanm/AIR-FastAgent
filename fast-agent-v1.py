##------------------------------------------------------------------------------##
import os
import pickle
import sqlite3

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

##------------------------------------------------------------------------------##
## Set up environment variables for LM Studio
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1/"
os.environ["OPENAI_API_KEY"] = "test"

llm = ChatOpenAI(model="qwen/qwen3-4b-2507", temperature=0)


template = "Answer the Question based on the context below.\n\nContext: {context}\n\nQ: {question}\nA:"
prompt = PromptTemplate.from_template(template)

# prompt = PromptTemplate(input_variables=["q"], template="Context: Q: {q}\nA:")
chain = prompt | llm | StrOutputParser()


##------------------------------------------------------------------------------##
def get_sqlite_memory(db_path: str = "agent_memory.db") -> SqliteSaver:
    """
    Creates a SQLite checkpointer for LangChain agents.
    This handles storage and retrieval of AI/Human messages.
    """
    # check_same_thread=False is usually required for production/async environments
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)


##------------------------------------------------------------------------------##
@tool
def retrieve_context(query: str) -> str:
    """Retrieve context for a given query using the loaded BM25Retriever."""

    # Define the file path where the retriever is saved
    file_path = "bm25_retriever.pkl"

    # Load the BM25Retriever using pickle
    with open(file_path, "rb") as f:
        loaded_bm25_retriever = pickle.load(f)

    print(f"BM25Retriever loaded from {file_path}")

    docs = loaded_bm25_retriever.invoke(query)
    return "\n".join(doc.page_content for doc in docs)


##------------------------------------------------------------------------------##
memory = get_sqlite_memory("my_chat_history.db")

agent = create_agent(
    model=llm,
    tools=[retrieve_context],
    system_prompt="You are a helpful assistant. you have access to a tool that retrieves relevant context based on user queries.",
    checkpointer=memory,
)

##------------------------------------------------------------------------------##
# Session A: define a unique thread_id
config = {"configurable": {"thread_id": "session_A"}}


# print("--- Interaction 1 (Storing) ---")
# response1 = agent.invoke(
#     {
#         "messages": [
#             {"role": "user", "content": "Explain PopularityAdjusted Block Model (PABM)"}
#         ]
#     },
#     config=config,
# )
# print(f"AI: {response1['messages'][-1].content}")

# print("\n--- Interaction 2 (Retrieving) ---")
# # The agent automatically queries SQLite using 'session_A' to remember the name
# response2 = agent.invoke(
#     {"messages": [{"role": "user", "content": "Any other application examples?"}]},
#     config=config,
# )
# print(f"AI: {response2['messages'][-1].content}")



response2 = agent.invoke(
    {"messages": [{"role": "user", "content": "Who are you?"}]},
    config=config,
)
print(f"AI: {response2['messages'][-1].content}")

##------------------------------------------------------------------------------##
