from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import translators as ts

embeddings = OllamaEmbeddings()  # 初始化嵌入模型

class SaveEmbeddings:
    """
    Save the embeddings of the text or document to the local FAISS index.
    """
    
    def get_faiss_from_text(self, text):
        """
        Split TEXT into chunks and save the embeddings.

        Args:
            text (str): Text to be split into chunks and saved in FAISS.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_text(text)
        
        vectorstore = FAISS.from_texts(texts=splits, embedding=embeddings)
        vectorstore.save_local("faiss_index")

    def get_faiss_from_document(self, document):
        """
        Split DOCUMENT into chunks and save the embeddings.

        Args:
            document (Iterable[Document]): Documents to be split into chunks and saved in FAISS.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(document)
        
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        vectorstore.save_local("faiss_index")


class RagChat:
    """
    A Retrieval-Augmented Generation (RAG) chatbot using FAISS.
    """
    def __init__(self, query):
        """
        Initialize the RAG chatbot.

        Args:
            query (str): User query to be processed by the RAG model.
        """
        self.query = query
        self.vectorstore = FAISS.load_local("faiss_index", embeddings)

    def format_docs(self, docs):
        """
        Format retrieved documents into text.

        Args:
            docs (List[Document]): Documents retrieved from the vector store.

        Returns:
            str: Formatted document content.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    @property
    def rag_chain(self):
        """
        Set up the RAG chain with retriever, prompt, and language model.

        Returns:
            RAG_Chain: A chain that can be invoked with `self.rag_chain.invoke(self.query)`.
        """
        retriever = self.vectorstore.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")  # Load the RAG prompt

        llm = HuggingFacePipeline.from_model_id(
            model_id="llama2",
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 100},
        )

        return (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def chat(self):
        """
        Process the query and return a response.

        Returns:
            str: Response from the RAG model.
        """
        return self.rag_chain.invoke(self.query)


class Translation:
    """
    Translate text from one language to another.
    """
    def __init__(self, query, translator='google', from_language='en', to_language='zh-TW'):
        """
        Initialize the translation settings.

        Args:
            query (str): Text to be translated.
            translator (str, optional): Translator service (default: 'google').
            from_language (str, optional): Source language (default: 'en').
            to_language (str, optional): Target language (default: 'zh-TW').
        """
        self.translator = translator
        self.query = query
        self.from_language = from_language
        self.to_language = to_language

    @property
    def translate_text(self):
        """
        Perform the translation.

        Returns:
            str: Translated text.
        """
        return ts.translate_text(
            query_text=self.query,  # 修正變數名稱
            translator=self.translator,
            from_language=self.from_language,
            to_language=self.to_language
        )


class LongMemory():
    pass