import sqlite3

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver


# --- 1. The Memory Function ---
def get_sqlite_memory(db_path: str = "agent_memory.db") -> SqliteSaver:
    """
    Creates a SQLite checkpointer for LangChain agents.
    This handles storage and retrieval of AI/Human messages.
    """
    # check_same_thread=False is usually required for production/async environments
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)


# --- 2. Setup Tools & Model ---
@tool
def get_weather(city: str) -> str:
    """Get the weather for a specific city."""
    return f"The weather in {city} is sunny and 25°C."


import os

os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1/"
os.environ["OPENAI_API_KEY"] = "test"

llm = ChatOpenAI(model="qwen/qwen3-4b-2507", temperature=0)

# --- 3. Create Agent with Memory ---
# We initialize the memory using our function
memory = get_sqlite_memory("my_chat_history.db")

# We pass the 'checkpointer' (memory) directly to create_agent
agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
    checkpointer=memory,
)

# --- 4. Usage: Store & Retrieve ---

# Session A: define a unique thread_id
config = {"configurable": {"thread_id": "session_A"}}

print("--- Interaction 1 (Storing) ---")
response1 = agent.invoke(
    {"messages": [{"role": "user", "content": "Hi, my name is John."}]}, config=config
)
print(f"AI: {response1['messages'][-1].content}")

print("\n--- Interaction 2 (Retrieving) ---")
# The agent automatically queries SQLite using 'session_A' to remember the name
response2 = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my name?"}]}, config=config
)
print(f"AI: {response2['messages'][-1].content}")
