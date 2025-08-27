import streamlit as st
import requests

# --- Page Config ---
st.set_page_config(page_title="Hugging Face Chatbot", page_icon="🤖")
st.title("🤖 Hugging Face Multi-Model Chatbot")

# --- Sidebar: Model Selection ---
st.sidebar.header("Select a model")
model_choice = st.sidebar.selectbox(
    "Choose a model",
    [
        "facebook/blenderbot-90M",
        "microsoft/DialoGPT-medium",
        "google/flan-t5-large",
        "EleutherAI/gpt-neo-2.7B",
        "EleutherAI/gpt-j-6B"
    ]
)

# --- Hugging Face API Key from Streamlit Cloud Secrets ---
hf_api_key = st.secrets["HF_API_KEY"]

# --- Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Function to query Hugging Face Inference API ---
def query_hf_api(prompt, model):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_api_key}"}
    payload = {"inputs": prompt}
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        elif isinstance(result, dict) and "error" in result:
            return f"Error: {result['error']}"
        else:
            return str(result)
    else:
        return f"Error: {response.status_code} - {response.text}"

# --- User Input Form ---
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", "")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get bot response
    bot_response = query_hf_api(user_input, model_choice)
    st.session_state.messages.append({"role": "bot", "content": bot_response})

# --- Display Chat History ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")
