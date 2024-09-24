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
import torchvision.transforms.functional as TVF

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
CHECKPOINT_PATH = Path("9em124t2-499968")
TITLE = "<h1><center>JoyCaption Pre-Alpha (2024-07-30a)</center></h1>"

class ImageAdapter(nn.Module):
	def __init__(self, input_features: int, output_features: int, ln1: bool, pos_emb: bool, num_image_tokens: int, deep_extract: bool):
		super().__init__()
		self.deep_extract = deep_extract

		if self.deep_extract:
			input_features = input_features * 5

		self.linear1 = nn.Linear(input_features, output_features)
		self.activation = nn.GELU()
		self.linear2 = nn.Linear(output_features, output_features)
		self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features)
		self.pos_emb = None if not pos_emb else nn.Parameter(torch.zeros(num_image_tokens, input_features))

		# Mode token
		#self.mode_token = nn.Embedding(n_modes, output_features)
		#self.mode_token.weight.data.normal_(mean=0.0, std=0.02)   # Matches HF's implementation of llama3

		# Other tokens (<|image_start|>, <|image_end|>, <|eot_id|>)
		self.other_tokens = nn.Embedding(3, output_features)
		self.other_tokens.weight.data.normal_(mean=0.0, std=0.02)   # Matches HF's implementation of llama3

	def forward(self, vision_outputs: torch.Tensor):
		if self.deep_extract:
			x = torch.concat((
				vision_outputs[-2],
				vision_outputs[3],
				vision_outputs[7],
				vision_outputs[13],
				vision_outputs[20],
			), dim=-1)
			assert len(x.shape) == 3, f"Expected 3, got {len(x.shape)}"  # batch, tokens, features
			assert x.shape[-1] == vision_outputs[-2].shape[-1] * 5, f"Expected {vision_outputs[-2].shape[-1] * 5}, got {x.shape[-1]}"
		else:
			x = vision_outputs[-2]

		x = self.ln1(x)

		if self.pos_emb is not None:
			assert x.shape[-2:] == self.pos_emb.shape, f"Expected {self.pos_emb.shape}, got {x.shape[-2:]}"
			x = x + self.pos_emb

		x = self.linear1(x)
		x = self.activation(x)
		x = self.linear2(x)

		# Mode token
		#mode_token = self.mode_token(mode)
		#assert mode_token.shape == (x.shape[0], mode_token.shape[1], x.shape[2]), f"Expected {(x.shape[0], 1, x.shape[2])}, got {mode_token.shape}"
		#x = torch.cat((x, mode_token), dim=1)

		# <|image_start|>, IMAGE, <|image_end|>
		other_tokens = self.other_tokens(torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(x.shape[0], -1))
		assert other_tokens.shape == (x.shape[0], 2, x.shape[2]), f"Expected {(x.shape[0], 2, x.shape[2])}, got {other_tokens.shape}"
		x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)

		return x

	def get_eot_embedding(self):
		return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)

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

if (CHECKPOINT_PATH / "text_model").exists:
    print("Loading VLM's custom text model")
    print("./" + (CHECKPOINT_PATH / "text_model/").__str__())
    text_model = AutoModelForCausalLM.from_pretrained("./" + (CHECKPOINT_PATH / "text_model/").__str__(), local_files_only=True, device_map=device, torch_dtype=torch.bfloat16, token=hf_token, quantization_config=quantization_config, trust_remote_code=True)
    text_model.eval()
else:
    text_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map=device, torch_dtype=torch.bfloat16, token=hf_token, quantization_config=quantization_config)
    text_model.eval()

# Image Adapter
print("Loading image adapter")
image_adapter = ImageAdapter(clip_model.config.hidden_size, text_model.config.hidden_size, False, False, 38, False)
image_adapter.load_state_dict(torch.load(CHECKPOINT_PATH / "image_adapter.pt", map_location=device))
# image_adapter.load_state_dict(torch.load(CHECKPOINT_PATH / "image_adapter.pt", map_location="cpu"))
image_adapter.eval().to(device)


@spaces.GPU()
@torch.no_grad()
def stream_chat(input_image: Image.Image, temperature: float, top_k: int, input_prompt: str, max_length: int) -> str:
    torch.cuda.empty_cache()

    # Preprocess image
    # image = clip_processor(images=input_image, return_tensors='pt').to(device).pixel_values
    # image = image.to(device)
    image = input_image.resize((384, 384), Image.LANCZOS)
    pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
    pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
    pixel_values = pixel_values.to(device)

    # Tokenize the prompt
    prompt = tokenizer.encode(input_prompt, return_tensors='pt', padding=False, truncation=False, add_special_tokens=False)

    # Embed image
    with torch.amp.autocast_mode.autocast(device, enabled=True):
        vision_outputs = clip_model(pixel_values=pixel_values, output_hidden_states=True)
        image_features = vision_outputs.hidden_states
        embedded_images = image_adapter(image_features)
        embedded_images = embedded_images.to(device)
    
    # Embed prompt
    prompt_embeds = text_model.model.embed_tokens(prompt.to(device))
    assert prompt_embeds.shape == (1, prompt.shape[1], text_model.config.hidden_size), f"Prompt shape is {prompt_embeds.shape}, expected {(1, prompt.shape[1], text_model.config.hidden_size)}"
    embedded_bos = text_model.model.embed_tokens(torch.tensor([[tokenizer.bos_token_id]], device=text_model.device, dtype=torch.int64))
    eot_embed = image_adapter.get_eot_embedding().unsqueeze(0).to(dtype=text_model.dtype)
    
    # Construct prompts
    inputs_embeds = torch.cat([
        embedded_bos.expand(embedded_images.shape[0], -1, -1),
        embedded_images.to(dtype=embedded_bos.dtype),
        prompt_embeds.expand(embedded_images.shape[0], -1, -1),
        eot_embed.expand(embedded_images.shape[0], -1, -1),
    ], dim=1)

    input_ids = torch.cat([
        torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long),
        torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
        prompt,
        torch.tensor([[tokenizer.convert_tokens_to_ids("<|eot_id|>")]], dtype=torch.long),
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
            output_file = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.txt")

            # 既に処理済みのファイルはスキップ
            if os.path.exists(output_file):
                continue

            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert("RGB")
            result = stream_chat(image, temperature, top_k, input_prompt, max_length)

            if (result == "") or (result == None):
                print(f"Error processing. Result is empty. {filename}")
                continue

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            results.append(f"{filename}: {result}")
    return "\n".join(results)

CAPTION_TEMPLATE_MAP = {
	"formal": ["Write a descriptive caption for this image in a formal tone."],
	"formal with word count": ["Write a descriptive caption for this image in a formal tone within {word_count} words."],
	"formal with length": ["Write a {length} descriptive caption for this image in a formal tone."],
	"informal": ["Write a descriptive caption for this image in a casual tone."],
	"informal with word count": ["Write a descriptive caption for this image in a casual tone within {word_count} words."],
	"informal with length": ["Write a {length} descriptive caption for this image in a casual tone."],

	"training_prompt": ["Write a stable diffusion prompt for this image."],
	"training_prompt with word count": ["Write a stable diffusion prompt for this image within {word_count} words."],
	"training_prompt with length": ["Write a {length} stable diffusion prompt for this image."],

	"rng-tags": ["Write a list of Booru tags for this image."],
	"rng-tags with word count": ["Write a list of Booru tags for this image within {word_count} words."],
	"rng-tags with length": ["Write a {length} list of Booru tags for this image."],
}

def update_prompt(caption_type):
    return CAPTION_TEMPLATE_MAP[caption_type][0]

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
                input_prompt = gr.Textbox(label="Prompt", value=VLM_PROMPT)
                caption_template_input = gr.Dropdown(choices=list(CAPTION_TEMPLATE_MAP.keys()), label="Caption Templates")
                run_button = gr.Button("Caption")
            with gr.Column():
                output_caption = gr.Textbox(label="Caption")

            caption_template_input.change(
                fn=update_prompt,
                inputs=caption_template_input,
                outputs=input_prompt
            )
            run_button.click(fn=stream_chat, inputs=[input_image, temperature, top_k, input_prompt, max_length], outputs=[output_caption])

    with gr.Tab("Batch Processing"):
        with gr.Column():
            folder_path = gr.Textbox(lines=1, placeholder="For example: C:\\...", label="Folder Path")
            with gr.Row():
                temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.75, step=0.05, label="Temperature")
                top_k = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="Top K")
            max_length = gr.Slider(label="Max Length", minimum=5, maximum=1000, step=5, value=300)
            input_prompt = gr.Textbox(label="Prompt", value=VLM_PROMPT)
            caption_template_input = gr.Dropdown(choices=list(CAPTION_TEMPLATE_MAP.keys()), label="Caption Templates")
            batch_button = gr.Button("Caption")
        with gr.Column():
            output_result = gr.Textbox(label="Caption")

        caption_template_input.change(
            fn=update_prompt,
            inputs=caption_template_input,
            outputs=input_prompt
        )
        batch_button.click(fn=process_folder, inputs=[folder_path, temperature, top_k, input_prompt, max_length], outputs=[output_result])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
