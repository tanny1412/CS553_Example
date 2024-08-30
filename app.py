import gradio as gr
from huggingface_hub import InferenceClient
import time
import torch
from transformers import pipeline

# Inference client setup
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
# pipe = pipeline("text-generation", "microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.bfloat16, device_map="auto")

# Global flag to handle cancellation
stop_inference = False

def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    use_local_model,
):
    global stop_inference
    stop_inference = False  # Reset cancellation flag

    if use_local_model:
        # Simulate local inference
        messages = [{"role": "system", "content": system_message}]

        for val in history:
            if val[0]:
                messages.append({"role": "user", "content": val[0]})
            if val[1]:
                messages.append({"role": "assistant", "content": val[1]})

        messages.append({"role": "user", "content": message})

        response = ""
        for message in pipe(
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
        ):
            token = message['generated_text'][-1]['content']
            response += token
            yield response  # Yielding response directly

        # Ensure the history is updated after generating the response
        history[-1] = (message, response)  # Update the last tuple in history with the full response
        yield history  # Yield the updated history

    else:
        # API-based inference
        messages = [{"role": "system", "content": system_message}]
        for val in history:
            if val[0]:
                messages.append({"role": "user", "content": val[0]})
            if val[1]:
                messages.append({"role": "assistant", "content": val[1]})
        messages.append({"role": "user", "content": message})

        response = ""
        for message_chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            if stop_inference:
                response = "Inference cancelled."
                break
            token = message_chunk.choices[0].delta.content
            response += token
            yield response  # Yielding response directly

        # Ensure the history is updated after generating the response
        history[-1] = (message, response)  # Update the last tuple in history with the full response
        yield history  # Yield the updated history

def cancel_inference():
    global stop_inference
    stop_inference = True

# Custom CSS for a fancy look
custom_css = """
#main-container {
    background-color: #f0f0f0;
    font-family: 'Arial', sans-serif;
}

.gradio-container {
    max-width: 700px;
    margin: 0 auto;
    padding: 20px;
    background: white;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    border-radius: 10px;
}

.gr-button {
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.gr-button:hover {
    background-color: #45a049;
}

.gr-slider input {
    color: #4CAF50;
}

.gr-chat {
    font-size: 16px;
}

#title {
    text-align: center;
    font-size: 2em;
    margin-bottom: 20px;
    color: #333;
}
"""

# Define the interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1 style='text-align: center;'>🌟 Fancy AI Chatbot 🌟</h1>")
    gr.Markdown("Interact with the AI chatbot using customizable settings below.")

    with gr.Row():
        system_message = gr.Textbox(value="You are a friendly Chatbot.", label="System message", interactive=True)
        use_local_model = gr.Checkbox(label="Use Local Model", value=False)

    with gr.Row():
        max_tokens = gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens")
        temperature = gr.Slider(minimum=0.1, maximum = 4.0, value=0.7, step=0.1, label="Temperature")
        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)")

    chat_history = gr.Chatbot(label="Chat")

    user_input = gr.Textbox(show_label=False, placeholder="Type your message here...")

    cancel_button = gr.Button("Cancel Inference", variant="danger")

    def chat_fn(message, history):
        history.append((message, ""))  # Initialize with empty response
        response_gen = respond(
            message,
            history,
            system_message.value,
            max_tokens.value,
            temperature.value,
            top_p.value,
            use_local_model.value,
        )
        full_response = ""
        for response in response_gen:
            full_response += response  # Accumulate the full response

        # Replace the last history tuple with the complete message-response pair
        history[-1] = (message, full_response)
        yield history

    user_input.submit(chat_fn, [user_input, chat_history], chat_history)
    cancel_button.click(cancel_inference)

if __name__ == "__main__":
    demo.launch(share=False)  # Remove share=True because it's not supported on HF Spaces
