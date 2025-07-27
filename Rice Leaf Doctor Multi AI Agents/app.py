import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
from crewai_pipeline import get_diagnosis_agents_pipeline
from langchain_groq import ChatGroq
from rice_chatbot import get_response
from gtts import gTTS
import tempfile
import os
import datetime
import base64
from rice_chatbot import get_chatbot_block
import pandas as pd
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="CropDoctor - AI Rice Disease Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Header */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        color: white;
        text-align: center;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Cards */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
    }
    .card h3 {
    color: #2c3e50;  /* Dark color for visibility */
    }
    .card p {
    color: #6c757d;  /* Gray color for subtitle */
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Disease result card */
    .disease-result {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa726 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .disease-name {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Tip card */
    .tip-card {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .tip-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Animation keyframes */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model and mappings (exactly as in your original code)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

inv_map = {
    0: 'bacterial_leaf_blight',
    1: 'bacterial_leaf_streak',
    2: 'bacterial_panicle_blight',
    3: 'blast',
    4: 'brown_spot',
    5: 'dead_heart',
    6: 'downy_mildew',
    7: 'hispa',
    8: 'normal',
    9: 'tungro'
}

FARMER_TIPS = {
    "bacterial_leaf_blight": "Use disease-free seeds.\nAvoid overhead irrigation.\nApply appropriate copper-based bactericides.",
    "bacterial_leaf_streak": "Ensure good drainage in fields.\nAvoid excessive nitrogen fertilizer.\nRotate crops to prevent buildup.",
    "bacterial_panicle_blight": "Use resistant rice varieties.\nMaintain balanced fertilization.\nRemove and destroy infected panicles.",
    "blast": "Keep fields well-drained.\nApply recommended fungicides like tricyclazole.\nUse resistant rice varieties.",
    "brown_spot": "Use potassium-rich fertilizers.\nAvoid excess nitrogen.\nApply fungicides such as mancozeb when needed.",
    "dead_heart": "Check for stem borers regularly.\nUse pheromone traps.\nRemove and destroy affected tillers.",
    "downy_mildew": "Improve field airflow.\nAvoid waterlogging.\nApply appropriate fungicides at early stages.",
    "hispa": "Remove affected leaves early.\nSpray neem-based insecticide or recommended chemicals.\nPractice deep plowing after harvest.",
    "normal": "Your crop looks healthy!\nMaintain regular watering and nutrient schedule.\nMonitor weekly for early signs of disease or pests.",
    "tungro": "Use resistant rice varieties.\nRemove infected plants promptly.\nControl green leafhoppers using insecticides."
}

LANG_PROMPT_MAP = {
    "English": "Translate the diagnosis report into English.",
    "Urdu": "اس رپورٹ کو اردو میں ترجمہ کریں۔",
    "Hindi": "इस रिपोर्ट का हिंदी में अनुवाद करें।"
}

LANG_TTS_MAP = {
    "English": "en",
    "Urdu": "ur",
    "Hindi": "hi"
}

# Initialize session state for diagnosis history
if 'diagnosis_history' not in st.session_state:
    st.session_state.diagnosis_history = []

# Your original functions (keeping them exactly the same)
def predict_with_confidence(image_input):
    img = image_input.resize((64, 64))
    x = keras_image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    label_idx = np.argmax(preds)
    label = inv_map[label_idx]
    return label, preds

def plot_confidence(preds):
    fig, ax = plt.subplots(figsize=(10, 6))
    diseases = list(inv_map.values())
    
    # Sort diseases by confidence for better visualization
    disease_conf = list(zip(diseases, preds))
    disease_conf.sort(key=lambda x: x[1])
    sorted_diseases, sorted_preds = zip(*disease_conf)
    
    bars = ax.barh(sorted_diseases, sorted_preds, color='green')
    ax.set_xlabel("Confidence")
    ax.set_title("Prediction Confidence")
    
    # Add percentage labels
    for i, (bar, conf) in enumerate(zip(bars, sorted_preds)):
        ax.text(conf + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{conf:.1%}', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig

def translate_response(response_text, language):
    if language == "English":
        return type('obj', (object,), {'content': response_text})()
    
    try:
        llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model="llama3-8b-8192")
        prompt = f"{LANG_PROMPT_MAP[language]}\n\n{response_text}"
        translated = llm.invoke(prompt)
        return translated
    except Exception as e:
        st.error(f"Translation failed: {str(e)}")
        return type('obj', (object,), {'content': response_text})()

def generate_pdf(disease, explanation, lang):
    pdf = FPDF()
    pdf.add_page()
    
    # Try to use DejaVu font, fallback to Arial if not available
    try:
        font_path = "DejaVuSans.ttf"  
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", size=12)
    except:
        pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt=f"CropDoctor Report - {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=f"Detected Disease: {disease}")
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=f"Explanation:\n{explanation}")
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=f"Language: {lang}")
    
    # Return PDF as bytes for Streamlit download
    return bytes(pdf.output(dest='S'))

def run_diagnosis(inputs, language):
    try:
        label, preds = predict_with_confidence(inputs)
        result = get_diagnosis_agents_pipeline(label)
        full_text = f"Predicted Disease: {label}\n\n Diagnosis Report:\n\n{result}"
        translated = translate_response(full_text, language)
        translated_text = str(translated.content)

        # Add to session state history
        st.session_state.diagnosis_history.append({
            "label": label,
            "img": inputs,
            "text": translated_text,
            "timestamp": datetime.datetime.now()
        })

        # --- Audio Generation Change is Here ---
        # Generate TTS audio and prepare it for return
        tts = gTTS(text=translated_text, lang=LANG_TTS_MAP[language])
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        # Read the audio data into a variable to be returned for both playing and downloading
        audio_data_bytes = audio_buffer.read()
        # --- End of Change ---

        # Generate confidence plot
        conf_fig = plot_confidence(preds)
        
        # Generate PDF
        pdf_bytes = generate_pdf(label, translated_text, language)

        tip_text = FARMER_TIPS.get(label, "General farming advice: Ensure good water and pest management.")

        # Return the prepared audio data along with everything else
        return translated_text, audio_data_bytes, conf_fig, pdf_bytes, tip_text, label, np.max(preds)

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        return None, None, None, None, None, None, None

def create_header():
    st.markdown("""
    <div class="header-container fade-in">
        <div class="header-title">Rice Leaf Doctor</div>
        <div class="header-subtitle">Multi-Agent AI for Crop Disease Diagnosis</div>
    </div>
    """, unsafe_allow_html=True)

def diagnosis_page(language):
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>📷 Upload Leaf Image</h3>
            <p>Upload a clear image of the rice leaf for disease detection</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of rice leaf"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Run Diagnosis", type="primary"):
                with st.spinner("🔬 Analyzing image..."):
                    result = run_diagnosis(image, language)
                    
                    if result[0] is not None:
                        translated_text, audio_data, conf_fig, pdf_bytes, tip_text, label, confidence = result
                        
                        # Store results in session state for display in col2
                        st.session_state.current_result = {
                            'translated_text': translated_text,
                            'audio_data': audio_data,
                            'conf_fig': conf_fig,
                            'pdf_bytes': pdf_bytes,
                            'tip_text': tip_text,
                            'label': label,
                            'confidence': confidence
                        }

    with col2:
        if 'current_result' in st.session_state:
            result = st.session_state.current_result
            
            # Disease result
            if result['label'] == 'normal':
                st.success("✅ Healthy Crop Detected!")
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); color: white; padding: 1.5rem; border-radius: 15px; text-align: center; margin: 1rem 0;">
                    <div style="font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem;">✅ Healthy</div>
                    <div style="font-size: 1rem; opacity: 0.9;">Confidence: {result['confidence']:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"⚠️ Disease Detected: {result['label'].replace('_', ' ').title()}")
                st.markdown(f"""
                <div class="disease-result">
                    <div class="disease-name">⚠️ {result['label'].replace('_', ' ').title()}</div>
                    <div>Confidence: {result['confidence']:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Diagnosis text
            st.markdown("### 📝 Detailed Analysis")
            st.text_area("", value=result['translated_text'], height=200, disabled=True)
            
            # Farmer tips
            st.markdown(f"""
            <div class="tip-card">
                <div class="tip-title">🌿 Farmer Tips</div>
                <div>{result['tip_text'].replace(chr(10), '<br>')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Audio Player and Download
            if result['audio_data']:
                # This line plays the audio
                st.audio(result['audio_data'], format='audio/mp3')

                # This new button allows downloading the audio
                st.download_button(
                    label="🔊 Download Audio Report",
                    data=result['audio_data'],
                    file_name=f"report_{result['label']}.mp3",
                    mime="audio/mp3"
                )
            
            # PDF download
            if result['pdf_bytes']:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=result['pdf_bytes'],
                    file_name=f"diagnosis_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            
            # Confidence chart
            if result['conf_fig']:
                st.markdown("### 📊 Confidence Analysis")
                st.pyplot(result['conf_fig'], use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def history_page():
    st.markdown("## 🗂️ Diagnosis History")
    
    if st.session_state.diagnosis_history:
        st.markdown(f"**Total Diagnoses:** {len(st.session_state.diagnosis_history)}")
        
        # Display history items
        for i, diagnosis in enumerate(reversed(st.session_state.diagnosis_history)):
            with st.expander(f"{diagnosis['label'].replace('_', ' ').title()} - {diagnosis['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(diagnosis['img'], width=200)
                with col2:
                    st.write(f"**Disease:** {diagnosis['label'].replace('_', ' ').title()}")
                    st.write(f"**Date:** {diagnosis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    
                st.text_area("Diagnosis Details:", value=diagnosis['text'], height=150, disabled=True, key=f"history_{i}")
        
        # Show history gallery similar to Gradio version
        if st.button("🔄 Refresh Gallery"):
            st.markdown("### 🖼️ Images & Labels")
            history_data = [(entry["img"], f"{entry['label'].replace('_', ' ').title()} — {entry['text'][:80]}...") for entry in st.session_state.diagnosis_history]
            
            # Display in grid format
            cols = st.columns(3)
            for i, (img, caption) in enumerate(history_data):
                with cols[i % 3]:
                    st.image(img, caption=caption, use_container_width=True)
    else:
        st.info("📷 No diagnosis history available. Start by analyzing some images!")

def chatbot_page():
    st.markdown("## 🤖 AI Agronomist Assistant")
    st.markdown("Chat with our AI-powered agricultural expert for personalized farming advice.")

    # Initialize chat history in Streamlit's session state if it doesn't exist
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hello! How can I help you with your rice crops today?"}
        ]

    # Display past messages from history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get new user input from the chat box
    if prompt := st.chat_input("Ask about rice diseases, pests, or cultivation..."):
        # Add user's message to history and display it
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display the AI's response
        with st.chat_message("assistant"):
            with st.spinner("🌱 The Agronomist is thinking..."):
                # --- This is where we call your AI logic ---
                
                # 1. Recreate the history in the format your AI function expects (list of tuples)
                # It takes all previous user messages and pairs them with the assistant's responses.
                history_tuples = []
                user_msgs = [msg["content"] for msg in st.session_state.chat_messages if msg["role"] == "user"]
                assistant_msgs = [msg["content"] for msg in st.session_state.chat_messages if msg["role"] == "assistant"]
                
                # The first assistant message is the greeting, so we skip it when creating pairs.
                if len(user_msgs) > 0:
                    history_tuples = list(zip(user_msgs, assistant_msgs[1:]))

                # 2. Call your actual AI function from rice_chatbot.py
                response = get_response(prompt, history_tuples)
                
                # 3. Display the response
                st.markdown(response)
        
        # Add the AI's new response to the chat history
        st.session_state.chat_messages.append({"role": "assistant", "content": response})

def main():
    load_css()
    create_header()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        page = st.selectbox(
            "Choose a section:",
            ["🩺 Diagnose", "🗂️ History", "🤖 AI Agronomist"],
            key="navigation"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        language = st.selectbox(
            "🌐 Language:",
            ["English", "Urdu", "Hindi"],
            key="language_setting"
        )
        
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
            <h4>📊 Statistics</h4>
            <p style="margin: 5px 0;">Total Diagnoses: {len(st.session_state.diagnosis_history)}</p>
            <p style="margin: 5px 0;">Healthy Crops: {sum(1 for d in st.session_state.diagnosis_history if d['label'] == 'normal')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content based on navigation
    if page == "🩺 Diagnose":
        diagnosis_page(language)
    elif page == "🗂️ History":
        history_page()
    elif page == "🤖 AI Agronomist":
        chatbot_page()

if __name__ == "__main__":
    main()