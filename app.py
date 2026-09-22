import streamlit as st
import ollama
from PIL import Image

st.set_page_config(page_title="Fast Local VLM Chat", page_icon="⚡", layout="centered")
st.title("⚡ Fast Local Vision Chat")
st.caption("Optimized for Intel CPU Inference")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Image Uploading
st.sidebar.header("📸 Visual Input")
uploaded_file = st.sidebar.file_uploader("Drag & drop an image:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_preview = Image.open(uploaded_file)
    st.sidebar.image(image_preview, caption="Active Context", use_container_width=True)

# Display Existing Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Build the payload
    ollama_message = {'role': 'user', 'content': prompt}
    if uploaded_file:
        ollama_message['images'] = [uploaded_file.getvalue()]

    # Render assistant placeholder and STREAM the response text
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # SWITCHED MODEL TO 'moondream' AND ENABLED stream=True
            response_stream = ollama.chat(
                model='moondream', 
                messages=[ollama_message],
                stream=True 
            )
            
            # Catch tokens as they emit from the CPU engine
            for chunk in response_stream:
                token = chunk['message']['content']
                full_response += token
                response_placeholder.markdown(full_response + "▌") # Adds a typing cursor
            
            # Remove cursor when finished
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error: {e}")
