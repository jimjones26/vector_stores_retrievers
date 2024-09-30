from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings


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

embedding = OllamaEmbeddings(
    base_url="http://a03a-216-147-123-78.ngrok-free.app",
    model="snowflake-arctic-embed:m-long",
)

# ------------------------------------------------------------------
# Vector stores
# ------------------------------------------------------------------

vectorstore = Chroma.from_documents(documents, embedding=embedding)

vectorstore.similarity_search("cat")

await vectorstore.asimilarity_search("cat")

vectorstore.similarity_search_with_score("cat")
