from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import translators as ts

embeddings = OllamaEmbeddings()

class Save_embeddings():
    
    def get_faiss_from_text(self,text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_text(text)
        
        vectorstore = FAISS.from_texts(texts=splits, embedding=embeddings)
        vectorstore.save_local("fassi_index")

    def get_faiss_from_document(self,document):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(document)
        
        vectorstore = FAISS.from_documents(documents=splits,embeddings=embeddings)
        vectorstore.save_local("fassi_index")


class Rag_chat():    
    def __init__(self,query):
        self.query = query
        self.vectorstore = FAISS.load_local("faiss_index", embeddings)
        
    def format_docs(docs):
        return "\n\n".join( doc.page_content for doc in docs)
    
    @property
    def rag_chain(self):
        retriever = self.vectorstore.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")
        
        llm = HuggingFacePipeline.from_model_id(model_id="llama2",task="text-generation",pipeline_kwargs={"max_new_tokens": 100})
        
        return (
                {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
        
    def chat(self):
        self.rag_chain.invoke(self.query)

class Translation():
    def __init__(self,query,translator='google',from_language='en',to_language='zh-TW'):
        self.translator = translator
        self.query = query
        self.from_language = from_language
        self.to_language = to_language
    
    @property
    def get(self):
        return ts.translate_text(query_text=self.query, translator=self.translator, from_language= self.from_language, to_language=self.to_language)