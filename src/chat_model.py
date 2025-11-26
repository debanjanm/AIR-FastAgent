import os

os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"
os.environ["OPENAI_API_KEY"] = "test"

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# choose model name exactly as LM Studio exposes it (check LM Studio UI)
llm = ChatOpenAI(model="qwen/qwen3-4b-thinking-2507", temperature=0.2)

prompt = PromptTemplate(input_variables=["q"], template="Q: {q}\nA:")
chain = prompt | llm | StrOutputParser()

# print(chain.run("Explain recursion in 2 sentences."))

print(chain.invoke("Who are you?"))
