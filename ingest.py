
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader 
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
from langchain.vectorstores.chroma import Chroma 
import os 

# Define the persist directory
persist_directory = "db"

def main():
    documents = [] 
    for root, dirs, files in os.walk("docs"): 
            for file in files: 
               if file.endswith(".pdf"): 
                loader = PyPDFLoader(os.path.join(root, file))
                documents.extend(loader.load())
                print(file)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
            texts = text_splitter.split_documents(documents)

            # Create HuggingFaceBgeEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            # Create a persistent Chroma client with the new configuration
            client = chromadb.PersistentClient(path= persist_directory)


            # Create a Chroma vector store
            db = Chroma(client=client)
            db.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory= persist_directory
            )
            db = None

               
                
                

if __name__ == "__main__":
    main()