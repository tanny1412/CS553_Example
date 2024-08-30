import gradio as gr
from huggingface_hub import InferenceClient
import time
import threading

# Inference client setup
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Global flag to determine the mode and to handle cancellation
use_local = False
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
    global use_local, stop_inference
    use_local = use_local_model
    stop_inference = False  # Reset cancellation flag

    if use_local:
        # Simulate local inference
        time.sleep(2)  # simulate a delay
        response = "This is a response from the local model."
        yield response
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
                yield "Inference cancelled."
                break
            token = message.choices[0].delta.content
            response += token
            yield response

def cancel_inference():
    global stop_inference
    stop_inference = True
    return gr.update(label="Inference cancelled.")

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
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message", interactive=True),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
        gr.Checkbox(label="Use Local Model", value=False),
        gr.Button("Cancel Inference"),
    ],
    css=custom_css,
    title="🌟 Fancy AI Chatbot 🌟",
    description="Interact with the AI chatbot using customizable settings below."
)

cancel_button = demo.add_button("Cancel Inference", variant="danger", elem_id="cancel_button")
cancel_button.click(cancel_inference, None, None)

if __name__ == "__main__":
    demo.launch()
