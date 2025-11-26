##------------------------------------------------------------------------------##
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

##------------------------------------------------------------------------------##
import os

os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1/"
os.environ["OPENAI_API_KEY"] = "test"


from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-qwen3-embedding-4b",
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
    check_embedding_ctx_length=False,
)

# text = "LangChain is a framework for developing applications powered by language models."
# single_vector = embeddings.embed_query(text)
# print(single_vector)
# print(str(single_vector)[:100])  # Show the first 100 characters of the vector

texts = ["Hello world", "LangChain with LM Studio", "Local embeddings are great!"]

db = Chroma.from_texts(texts, embeddings)
results = db.similarity_search("hello world")

for d in results:
    print(d.page_content)

##------------------------------------------------------------------------------##
