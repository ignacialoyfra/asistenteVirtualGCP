import streamlit as st
import time
import os,sys
import fitz

from chatbot_docs import documents, data_embeddings, conversation_complete, chat, images_created
archivo_pdf = 'GCP.pdf'
st.title("💬 Chatbot GCP")
st.write("""Soy un asistente virtual, que te ayuda a estudiar para productos de GCP""")  

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ingresa tu consulta: "): #Propmt es el mensaje del usuario
    while prompt is None:
        time.sleep()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        assistant_response, pags = conversation_complete(prompt, chat, documents, data_embeddings )
        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
       
        if pags != []:
            print("Páginas:",pags)
            expander = st.expander("Referencias")
            images_created(archivo_pdf,pags)
            for page in pags:
                path_image = f"pagina_{page}.png"
                expander.image(path_image,caption=f'Página {page}',use_column_width=True)
                time.sleep(2)
        elif assistant_response == "Lo siento, puedes reformular la pregunta.":
            st.write("Sin referencias.")
        else:
            st.write("Respuesta Agente.")
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})