from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

class UploadDataRagService:
    """
    Service to handle the upload of data for RAG (Retrieval-Augmented Generation).
    """
    def __init__(self, data_path_general=r"C:\Users\Ale\Downloads\Vector\general", data_path_countries=r"C:\Users\Ale\Downloads\Vector\teams"):
        self.data_path_general = data_path_general
        self.data_path_countries = data_path_countries

    def get_chunks(self):
        """
        Read the txt files from the path and return the chunks.
        """
        txts = []
        for root, dirs, files in os.walk(self.data_path_general):
            for file in files:
                if file.endswith(".txt"):
                    txts.append(os.path.join(root, file))
        docs = []
        for txt in txts:
            loader = TextLoader(txt)
            temp = loader.load()
            docs.extend(temp)
        txts = []
        for root, dirs, files in os.walk(self.data_path_countries):
            for file in files:
                if file.endswith(".txt"):
                    txts.append(os.path.join(root, file))
        for txt in txts:
            loader = TextLoader(txt)
            temp = loader.load()
            for doc in temp:
                doc.metadata = {"country": os.path.splitext(os.path.basename(txt))[0]}
            docs.extend(temp)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
        return text_splitter.split_documents(docs)