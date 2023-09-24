import vertexai
import pandas as pd
import numpy as np
from langchain import PromptTemplate
from langchain.chat_models import ChatVertexAI
import os
from langchain.embeddings import VertexAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
import fitz
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langdetect import detect
from googletrans import Translator

from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import VertexAI
import pickle


# Función que traduce un string en caso de que el mensaje este en ingles.
def validar_y_traducir(texto):
    idioma = detect(texto)
    if idioma == 'en':
        traductor = Translator()
        texto_traducido = traductor.translate(texto, src='en', dest='es')   
        return texto_traducido.text
    else:
        return texto



# Inicialización de la API KEY de Serapi
os.environ["SERPAPI_API_KEY"] = "SERPAPI_API_KEY"
search = SerpAPIWrapper()

# Inicialización de Vertex AI junto a los Embeddings de Vertex AI. 
PROJECT_ID = "PROJECT_ID" 
LOCATION = "LOCATION"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'CREDENTIALES'

vertexai.init(project=PROJECT_ID, location=LOCATION)

embeddings = VertexAIEmbeddings()

# Se carga el archivo PDF y se divide en chunks.
# Se obtiene como resultado una lista de objetos Document
document = PyPDFLoader("GCP.pdf")
data = document.load()
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        length_function = len,
        chunk_overlap = 250
)

documents = text_splitter.split_documents(data)

# Se inicializa el LLM tipo conversacional de Vertex AI.
chat = ChatVertexAI(
    temperature=0.7,
    max_output_tokens=1024
)

# Se carga la documentación del contenido del PDF junto a su metadata y embeddings.
data_embeddings = pd.read_pickle("data_embeddings_GCP.pickle")

# Función que retorna una lista de las páginas sin repetir.
def pages_number(pags:list):
    list_pages = []
    for pag in range(len(pags)):
        list_pages.append(pags[pag]["page"])
    list_pages = list(set(list_pages))
    return list_pages

# Función que calcula el producto punto entre dos vectores
def distancia_vector(x, y):
    return np.dot(np.array(x), np.array(y))

# Función que retorna un dataframe con la pregunta del usuario junto a su vector.
def question_user(q:str):
  data_question = pd.DataFrame()
  emb = []
  q_list = []
  emb.append(embeddings.embed_query(q))
  q_list.append(q)
  data_question["pregunta"] = q_list
  data_question["embedding_pregunta"] = emb

  return data_question

# Función que retorna el mismo dataframe ingresado, pero con una columna más,
# que es la distancia entre el vector del usuario con todos los vectores almacenados del documento.
def data_metadata(data:object, p:str):
    data_p = question_user(p)
    data["distancia"] = data["embeddings"].apply(lambda x:distancia_vector(data_p.iloc[0,1],x))
    return data, p


# Función que retorna una lista ordenada de forma descendente y que está filtrada
# # por el parámetro ingresado por el usuario, además retorna la pregunta del usuario
def metadata_final(data:object, p:str, param:float):
    data_sorted = data.sort_values(by = "distancia",ascending=False)
    data_sorted = data_sorted[data_sorted["distancia"] >= param]
    content = data_sorted["metadata"].tolist()
    return content, p

# Función que invoca a las funciones anteriores.
def function_main_content(p:str, data:object):
    data, p = data_metadata(data, p)
    content, p = metadata_final(data, p, 0.7)
    return content, p

# Función que retorna los documentos que cumplen la condición del parámetro, 
# estos son los documentos  
def documents_prompt(documents:list, pages:list):
    docs = []
    for doc in range(len(documents)):
        val_aux = documents[doc].metadata["page"]
        if val_aux in pages:
            docs.append(documents[doc])
        else:
            continue
    return docs

# Función que invoca a las funciones anteriores y que retorma los documentos finales
# que va a recibir el LLM, además de retornar las páginas correspondientes a los documentos.
def documents_main(p:str, data:object, documents:list):
    pags, p = function_main_content(p,data)
    pags = pages_number(pags)
    docs = documents_prompt(documents, pags)
    pags = sorted(pags)
    return docs, pags   

        
# Se declaran las herramientas del agente
tools = [
	Tool(
	name = "Current Search",
	func=search.run,
	description="useful for when you need to answer questions about products/services of Google Cloud Platform"
	),
]

# Se inicializa el agente junto a su memoria
memory_ = ConversationBufferMemory(memory_key="chat_history")
agent_chain = initialize_agent(
        tools, 
        chat, 
	agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
        verbose=True,
        memory=memory_)


# Función final que se llama en el archivo principal para invocar al asistente.
def conversation_complete(query:str, chat, documents, data_embeddings):
    
    try:
        # Definición del prompt
        template = """Tienes la siguiente información: {context}
            y recibes una pregunta:{human_input} y tienes el siguiente contexto de la conversación {chat_history}.
            Responde solo en español y utilizando solo la información que recibes.
            Contesta de forma muy detallada.
            Limitate a responder solamente con la información que recibes al inicio y con el historial del chat.
            Si no sabes la respuesta, solo dilo."""
        prompt = PromptTemplate(
                    input_variables=["chat_history", "human_input", "context"], template=template
            )
        memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
        # Se crea la cadena
        chain = load_qa_chain(
        chat, chain_type="stuff", memory=memory, prompt=prompt)
        docs_, pags = documents_main(query,data_embeddings, documents) # Se obtienen los documentos filtrados con sus páginas correspondientes.
        # print("DOCUMENTOS:",docs_)
        
        # Se obtiene la respuesta del asistente
        respuesta = chain({"input_documents": docs_, "human_input": query}, return_only_outputs=True)
        # Se captura la salida final
        respuesta_final = respuesta['output_text']
        
        # Se válida la salida del asistente.
        if respuesta_final == '\n':
            respuesta_final = "Lo siento, puedes reformular la pregunta."
            pags = []
            return respuesta_final, pags
            
        elif docs_ == []:
            respuesta_final = agent_chain.run(query)
            pags = []
            respuesta_final = validar_y_traducir(respuesta_final)
            return respuesta_final, pags
        else:
            return respuesta_final, pags
    except Exception as e:
        pags = []
        respuesta_final = "Lo siento, pero tu pregunta me provocó confusión, ¿Puedes reiniciarme?"
        return respuesta_final, pags
            

# Función que convierte una hoja de pdf a imagen de tipo png.
def images_created(file:str, pags:list):
    document_pdf = fitz.open(file)
    for pag in range(len(pags)):
        pagina = document_pdf.load_page(pags[pag]-1)
        image = pagina.get_pixmap()
        image.save(f'pagina_{pags[pag]}.png','png')





