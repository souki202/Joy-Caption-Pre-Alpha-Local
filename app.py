import spaces
import gradio as gr
from huggingface_hub import InferenceClient
from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM, BitsAndBytesConfig
from pathlib import Path
import torch
import torch.amp.autocast_mode
from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("--use-8bit", action="store_true", help="Running the model in 8-bit")
parser.add_argument("--not-wsl", action="store_true", help="Not using WSL")
parser.add_argument("--device", type=str, default="cuda:0", help='"cuda:0", "cuda:1", or "cpu". "cuda:0" for a normal environment with a single GPU.')
parser.add_argument("--token", type=str, default="", help="HuggingFace token")

args = parser.parse_args()
use_8bit = args.use_8bit
hf_token = args.token
device = args.device
not_wsl = args.not_wsl

print(f"Device: {device}, Use 8bit: {str(use_8bit)}")

CLIP_PATH = "google/siglip-so400m-patch14-384"
VLM_PROMPT = 'A descriptive caption for this image:\n'
MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B"
CHECKPOINT_PATH = Path("wpkklhc6")
TITLE = "<h1><center>JoyCaption Pre-Alpha (2024-07-30a)</center></h1>"

class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
    
    def forward(self, vision_outputs: torch.Tensor):
        x = self.linear1(vision_outputs)
        x = self.activation(x)
        x = self.linear2(x)
        return x

# Windowsのパスをwslのパスに変換。ドライブレターは任意のものに対応する
def windows_to_wsl_path(windows_path):
    drive_letter = windows_path[0]
    path = windows_path[3:].replace('\\', '/')
    return f'/mnt/{drive_letter.lower()}/{path}'

# Not sure why, but True makes it slower; False does not change memory usage.
quantization_config = BitsAndBytesConfig(load_in_8bit=False) if use_8bit else None

# Load CLIP
print("Loading CLIP")
clip_processor = AutoProcessor.from_pretrained(CLIP_PATH, quantization_config=quantization_config)
clip_model = AutoModel.from_pretrained(CLIP_PATH)
clip_model = clip_model.vision_model
clip_model.eval().to(device)
clip_model.requires_grad_(False)


# Tokenizer
print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, token=hf_token, quantization_config=quantization_config)
assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast), f"Tokenizer is of type {type(tokenizer)}"

# LLM
print("Loading LLM")
text_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map=device, torch_dtype=torch.bfloat16, token=hf_token, quantization_config=quantization_config)
text_model.eval()

# Image Adapter
print("Loading image adapter")
image_adapter = ImageAdapter(clip_model.config.hidden_size, text_model.config.hidden_size)
image_adapter.load_state_dict(torch.load(CHECKPOINT_PATH / "image_adapter.pt", map_location=device))
# image_adapter.load_state_dict(torch.load(CHECKPOINT_PATH / "image_adapter.pt", map_location="cpu"))
image_adapter.eval().to(device)


@spaces.GPU()
@torch.no_grad()
def stream_chat(input_image: Image.Image, temperature: float, top_k: int, input_prompt: str, max_length: int) -> str:
    torch.cuda.empty_cache()

    # Preprocess image
    image = clip_processor(images=input_image, return_tensors='pt').to(device).pixel_values
    image = image.to(device)

    # Tokenize the prompt
    prompt = tokenizer.encode(input_prompt, return_tensors='pt', padding=False, truncation=False, add_special_tokens=False)

    # Embed image
    with torch.amp.autocast_mode.autocast(device, enabled=True):
        vision_outputs = clip_model(pixel_values=image, output_hidden_states=True)
        image_features = vision_outputs.hidden_states[-2]
        embedded_images = image_adapter(image_features)
        embedded_images = embedded_images.to(device)
    
    # Embed prompt
    prompt_embeds = text_model.model.embed_tokens(prompt.to(device))
    assert prompt_embeds.shape == (1, prompt.shape[1], text_model.config.hidden_size), f"Prompt shape is {prompt_embeds.shape}, expected {(1, prompt.shape[1], text_model.config.hidden_size)}"
    embedded_bos = text_model.model.embed_tokens(torch.tensor([[tokenizer.bos_token_id]], device=text_model.device, dtype=torch.int64))

    # Construct prompts
    inputs_embeds = torch.cat([
        embedded_bos.expand(embedded_images.shape[0], -1, -1),
        embedded_images.to(dtype=embedded_bos.dtype),
        prompt_embeds.expand(embedded_images.shape[0], -1, -1),
    ], dim=1)

    input_ids = torch.cat([
        torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long),
        torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
        prompt,
    ], dim=1).to(device)
    attention_mask = torch.ones_like(input_ids)

    #generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=300, do_sample=False, suppress_tokens=None)
    generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=max_length, do_sample=True, top_k=top_k, temperature=temperature, suppress_tokens=None)

    # Trim off the prompt
    generate_ids = generate_ids[:, input_ids.shape[1]:]
    if generate_ids[0][-1] == tokenizer.eos_token_id:
        generate_ids = generate_ids[:, :-1]

    caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

    return caption.strip().strip("\n").strip("'").strip('"')

def process_folder(folder_path: str, temperature: float, top_k: int, input_prompt: str, max_length: int) -> str:
    results = []

    # Windowsの任意のドライブのパスをwslで使えるように変換
    folder_path = folder_path if not_wsl else windows_to_wsl_path(folder_path)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert("RGB")
            result = stream_chat(image, temperature, top_k, input_prompt, max_length)

            if (result == "") or (result == None):
                print(f"Error processing. Result is empty. {filename}")
                continue

            output_file = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            results.append(f"{filename}: {result}")
    return "\n".join(results)

with gr.Blocks() as demo:
    gr.HTML(TITLE)

    with gr.Tab("Single Image Processing"):
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image")
                with gr.Row():
                    temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.75, step=0.05, label="Temperature")
                    top_k = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="Top K")
                max_length = gr.Slider(label="Max Length", minimum=5, maximum=1000, step=5, value=300)
                input_prompt = gr.Textbox(label="Prompt (Almost no effect)", value=VLM_PROMPT)
                run_button = gr.Button("Caption")
            
            with gr.Column():
                output_caption = gr.Textbox(label="Caption")
            run_button.click(fn=stream_chat, inputs=[input_image, temperature, top_k, input_prompt, max_length], outputs=[output_caption])

    with gr.Tab("Batch Processing"):
        with gr.Column():
            folder_path = gr.Textbox(lines=1, placeholder="For example: C:\\...", label="Folder Path")
            with gr.Row():
                temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.75, step=0.05, label="Temperature")
                top_k = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="Top K")
            max_length = gr.Slider(label="Max Length", minimum=5, maximum=1000, step=5, value=3-0)
            input_prompt = gr.Textbox(label="Prompt (Almost no effect)", value=VLM_PROMPT)
            batch_button = gr.Button("Caption")
        with gr.Column():
            output_result = gr.Textbox(label="Caption")
        batch_button.click(fn=process_folder, inputs=[folder_path, temperature, top_k, input_prompt, max_length], outputs=[output_result])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
