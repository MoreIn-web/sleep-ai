import argparse
import gradio as gr
from llama_cpp import Llama

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Path to GGUF model")
parser.add_argument("--ctx", type=int, default=4096, help="Context length")
parser.add_argument("--gpu", type=int, default=-1, help="GPU layers (-1 for CPU only)")
args = parser.parse_args()

# Load model
print(f"🚀 Loading model from {args.model} ... (This may take a while)")
llm = Llama(model_path=args.model, n_ctx=args.ctx, n_gpu_layers=args.gpu, verbose=True)

# AI response function
def chat_with_ai(message, history):
    prompt = ""
    for user, bot in history:
        prompt += f"User: {user}\nDew Sleep: {bot}\n"
    prompt += f"User: {message}\nDew Sleep:"

    output = llm(prompt, max_tokens=512, stop=["User:", "Dew Sleep:"], echo=False)
    answer = output["choices"][0]["text"].strip()
    return answer

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 💬 Dew Sleep AI — Friendly Bangla Chatbot")
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="Type your question here...")
    clear = gr.Button("Clear")

    def user_input(user_message, chat_history):
        response = chat_with_ai(user_message, chat_history)
        chat_history.append((user_message, response))
        return "", chat_history

    msg.submit(user_input, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

# Run app
demo.launch(server_name="0.0.0.0", server_port=7860)
