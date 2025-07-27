import os
import gradio as gr
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

# Use Hugging Face's secret environment variable for Groq
groq_api_key = os.environ.get("GROQ_API_KEY")

# Setup ChatGroq with LangChain
llm = ChatGroq(
    model="llama3-70b-8192",
    groq_api_key=groq_api_key
)

# Prompt template
system_msg = SystemMessage(
    content="You are an expert agricultural assistant specialized in rice plant diseases. Provide helpful, clear, and accurate answers."
)

# Chat function
def get_response(message, history):
    try:
        messages = [system_msg]
        for user_msg, bot_msg in history:
            messages.append(HumanMessage(content=user_msg))
            messages.append(SystemMessage(content=bot_msg))
        messages.append(HumanMessage(content=message))
        response = llm(messages)
        return response.content
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Custom CSS for agriculture-themed look
custom_css = """
#chatbot {background-color: #f0fff0;}
.gradio-container {font-family: 'Segoe UI', sans-serif;}
h1 {color: #2e7d32;}
footer {display: none !important;}
"""

# Wrap UI inside a function so it can be reused
def get_chatbot_block():
    with gr.Blocks(css=custom_css) as demo:
        gr.Markdown("<h1>🌾 Agro Rice Disease Expert</h1>")
        chatbot = gr.Chatbot(label="👨‍🌾 Ask about Rice Plant Diseases", elem_id="chatbot", height=400)
        msg = gr.Textbox(placeholder="Ask your question here...", label="Your Question")
        clear = gr.Button("Clear")

        def respond(user_message, chat_history):
            response = get_response(user_message, chat_history)
            chat_history.append((user_message, response))
            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)
    return demo