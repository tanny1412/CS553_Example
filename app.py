import gradio as gr
from huggingface_hub import InferenceClient
import time

# Inference client setup
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

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
        time.sleep(2)  # simulate a delay
        response = "This is a response from the local model."
        history.append((message, response))
        yield history
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
        for message in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            if stop_inference:
                history.append((message, "Inference cancelled."))
                yield history
                break
            token = message.choices[0].delta.content
            response += token
            history.append((message, response))
            yield history

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
        temperature = gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature")
        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)")

    chat_history = gr.Chatbot(label="Chat")

    user_input = gr.Textbox(show_label=False, placeholder="Type your message here...")

    cancel_button = gr.Button("Cancel Inference", variant="danger")

    def chat_fn(message, history):
        return respond(
            message,
            history,
            system_message.value,
            max_tokens.value,
            temperature.value,
            top_p.value,
            use_local_model.value,
        )

    user_input.submit(chat_fn, [user_input, chat_history], chat_history)
    cancel_button.click(cancel_inference)

if __name__ == "__main__":
    demo.launch(share=True)
