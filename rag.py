from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_community.document_loaders import UnstructuredURLLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os
from prompt import PROMPT, EXAMPLE_PROMPT

# Set tokenizers parallelism early to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()
CHUNK_SIZE = 1000
COLLECTION_NAME = 'real-estate'
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"

llm = None
vector_store = None

def initialize_components():
    global llm, ef, vector_store
    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)


    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'trust_remote_code': True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )

def process_urls(urls):
    '''
    This function scraps the data from the given urls and stores into vector DB
    :param urls: input urls
    :return:
    '''
    yield 'Initialize...✅'
    initialize_components()

    vector_store.reset_collection()

    loader = WebBaseLoader(urls)
    yield 'Loading Data...✅'
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',' '],
        chunk_size=CHUNK_SIZE
    )
    yield 'Splitting Text...✅'
    docs = text_splitter.split_documents(data)

    yield 'Saving Data...✅'

    uuids = [str(uuid4(  )) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    yield 'Processings URLs is done successfully!✅ '


def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector database is not initialized ")

    qa_chain = load_qa_with_sources_chain(llm, chain_type="stuff",
                                          prompt=PROMPT
                                          ,document_prompt=EXAMPLE_PROMPT)

    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
    result = chain.invoke({"question": query}, return_only_outputs=True)
    sources = result.get("sources", "")

    return result['answer'], sources

if __name__ == '__main__':
    urls = [
        'https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html'
        ,'https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html'
    ]

    process_urls(urls)
    answer, sources = generate_answer("Tell me what was the 30 year fixed mortagate rate along with the date?")
    print("Answer : ", answer)
    print("Sources : ", sources)
