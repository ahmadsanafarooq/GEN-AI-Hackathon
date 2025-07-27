import gradio as gr
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
from crewai_pipeline import get_diagnosis_agents_pipeline
from langchain_groq import ChatGroq
from gtts import gTTS
import tempfile
import os
import datetime
import base64
from transformers import pipeline
from rice_chatbot import get_chatbot_block
from transformers import MarianMTModel, MarianTokenizer

model = tf.keras.models.load_model("model.h5")

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
    "Urdu": "اس رٹ کو اردو میں ترجمہ کریں۔",
    "Hindi": "इस रिपोर्ट का हिंदी में अनुवाद करें।"
}

LANG_TTS_MAP = {
    "English": "en",
    "Urdu": "ur",
    "Hindi": "hi"
}

diagnosis_history = []

def predict_with_confidence(image_input):
    img = image_input.resize((64, 64))
    x = keras_image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    label_idx = np.argmax(preds)
    label = inv_map[label_idx]
    return label, preds

def plot_confidence(preds):
    fig, ax = plt.subplots(figsize=(6, 3))
    diseases = list(inv_map.values())
    ax.barh(diseases, preds, color='green')
    ax.set_xlabel("Confidence")
    ax.set_title("Prediction Confidence")
    plt.tight_layout()

    temp_path = tempfile.mktemp(suffix=".png")
    plt.savefig(temp_path)
    plt.close(fig)
    return temp_path


def translate_response(response_text, language):
    llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model="llama3-8b-8192")
    prompt = f"{LANG_PROMPT_MAP[language]}\n\n{response_text}"
    translated = llm.invoke(prompt)
    return translated

def generate_pdf(disease, explanation, lang):
    pdf = FPDF()
    pdf.add_page()
    font_path = "DejaVuSans.ttf"  
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", size=12)
    pdf.cell(200, 10, txt=f"CropDoctor Report - {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=f"Detected Disease: {disease}")
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=f"Explanation:\n{explanation}")
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=f"Language: {lang}")
    pdf_path = tempfile.mktemp(suffix=".pdf")
    pdf.output(pdf_path)
    return pdf_path

def pipeline(inputs, language):
    try:
        label, preds = predict_with_confidence(inputs)
        result = get_diagnosis_agents_pipeline(label)
        full_text = f"Predicted Disease: {label}\n\n Diagnosis Report:\n\n{result}"
        translated = translate_response(full_text, language)
        translated_text = str(translated.content)

        diagnosis_history.append({
            "label": label,
            "img": inputs,
            "text": translated_text
        })

        tts = gTTS(text=translated_text, lang=LANG_TTS_MAP[language])
        temp_audio_path = tempfile.mktemp(suffix=".mp3")
        tts.save(temp_audio_path)

        conf_plot = plot_confidence(preds)
        pdf_path = generate_pdf(label, translated_text, language)

        tip_text = FARMER_TIPS.get(label, "General farming advice: Ensure good water and pest management.")

        return translated_text, temp_audio_path, conf_plot, pdf_path, tip_text

    except Exception as e:
        print("Error:", e)
        return f"Error occurred:\n{str(e)}", None, None, None, None

custom_css = """
footer {display:none !important;}
.gradio-container {
    background-color: var(--background, #f5fdf5);
    color: var(--body-text-color, #111111);
}
.gr-box {
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    padding: 24px;
    background-color: var(--block-background-fill, #ffffff);
}
button {
    background-color: #228B22 !important;
    color: white !important;
    border-radius: 8px;
}
"""

with gr.Blocks(css=custom_css) as app:
    gr.Markdown("""
    <div style='text-align:center'>
        <h1 style='font-size:2.5em; color:#228B22;'>🌾 CropDoctor</h1>
        <p style='font-size:1.1em;'>Detect Rice Diseases, Hear Diagnosis & Get Farming Tips</p>
    </div>
    """)

    with gr.Tab(" Diagnose Disease"):
        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(type="pil", label= "Upload Rice Leaf Image")
                lang_input = gr.Radio(["English", "Urdu", "Hindi"], label="Select Language", value="English")
                diagnose_btn = gr.Button(" Run Diagnosis", variant="primary")
            with gr.Column(scale=2):
                diagnosis_text = gr.Textbox(label=" Diagnosis", lines=6, interactive=False)
                audio_out = gr.Audio(label=" Audio Explanation", interactive=False)
                conf_graph = gr.Image(label=" Diagnosis Confidence", interactive=False)
                pdf_output = gr.File(label=" Download PDF")
                tip_card = gr.Textbox(label=" Farmer Tip", lines=3, interactive=False)

        diagnose_btn.click(
            pipeline,
            inputs=[img_input, lang_input],
            outputs=[diagnosis_text, audio_out, conf_graph, pdf_output, tip_card]
        )

    with gr.Tab(" Diagnosis History"):
        def show_history():
            items = []
            for entry in diagnosis_history:
                items.append((entry["img"], f"{entry['label'].upper()} - {entry['text'][:100]}..."))
            return items

        hist_gallery = gr.Gallery(label=" Past Diagnoses", columns=2, rows=3, object_fit="cover")
        refresh_btn = gr.Button(" Refresh History")
        refresh_btn.click(fn=show_history, outputs=hist_gallery)

    with gr.Tab("AI Agronomist Chat"):
        get_chatbot_block()

if __name__ == "__main__":
    app.launch()