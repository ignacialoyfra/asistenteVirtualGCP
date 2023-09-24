
import vertexai
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import VertexAIEmbeddings
import pandas as pd


PROJECT_ID = "PROJECT_ID"
LOCATION = "LOCATION"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'CREDENTIALES'

vertexai.init(project=PROJECT_ID, location=LOCATION)

embeddings = VertexAIEmbeddings()

document = PyPDFLoader("GCP.pdf")
data = document.load()
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        length_function = len,
        chunk_overlap = 250
)

documents = text_splitter.split_documents(data)

# Creando el nuevo dataframe 

def dataframe_documentation(documents:list):
    data = pd.DataFrame()
    page_content = []
    metadata = []
    for line in range(len(documents)):
        page_content.append(documents[line].page_content)
        metadata.append(documents[line].metadata)
    data["page_content"] = page_content
    data["metadata"] = metadata
    return data

# Creando los nuevos embeddings 

def created_embeddings(data:object, embeddings):
    col_index = data.columns.get_loc('page_content')
    vectors = []
    for index in range(data.shape[0]):
        val = embeddings.embed_query(data.iloc[index, col_index])
        vectors.append(val)
    data["embeddings"] = vectors
    return data

data_ = dataframe_documentation(documents)
data_embeddings = created_embeddings(data_, embeddings)
#data_embeddings.to_pickle("data_embeddings_GCP.pickle")