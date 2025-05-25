from diffusers_helper.hf_login import login

import os
import re
import copy

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math
import time
from PIL import Image 
import torchvision.transforms as T
image_transform = T.ToPILImage()

import random

import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import argostranslate.package
import argostranslate.translate

from torch.utils.data import DataLoader

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, save_bcthw_as_png, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete, unload_specific_models
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.lora_utils import load_lora, unload_all_loras

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePack_F1_I2V_HY_20250503', torch_dtype=torch.bfloat16).cpu()

from_code = "en"
to_code = "zh"

# Download and install Argos Translate package
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
package_to_install = next(
    filter(
        lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
    )
)
argostranslate.package.install_from_path(package_to_install.download())


vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()


if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = False
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)

else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

stream = AsyncStream()

prompt_g = [['',0]]*4
start_latent = []
input_image_g = np.empty([1])
loaded_lora = []

data_unit = {
    'llama_vec':        [],
    'clip_l_pooler':    [],
    'ordered_segments': [],
    'prompt_latent':    [],
    'prompt_frame_len': [],
    'last_image':       []
    }

data = []
data.append(data_unit.copy())
data.append(data_unit.copy())
data.append(data_unit.copy())
data.append(data_unit.copy())

prompt_n_g = ""



outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)


@torch.no_grad()
def worker(input_image, n_prompt, seed, total_second_length, steps, shift, cfg, gs, rs, gpu_memory_preservation, use_teacache, resolution, mp4_crf, max_segment_length, prompt0, length0, prompt1, length1, prompt2, length2, prompt3, length3, render0, render1, render2, render3, decode_latent, randomize, quick_preview, lora_strength, loaded_lora):

    job_id = generate_timestamp()    
    global prompt_g, data, input_image_g, start_latent, image_encoder_last_hidden_state, transformer
    final_list = []  
    stream.output_queue.push(('progress', (False, 0, None, '', make_progress_bar_html(0, 'Starting ...'))))
    try:

        # load_lora        
        # load_lora(transformer, "D:/SD/!Paths/loras/Hunyuan/aman4.safetensors")
        
        # Text encoding       
        prompt = [[prompt0, length0, render0], [prompt1, length1, render1], [prompt2, length2, render2], [prompt3, length3, render3]]
        
        # Encode text if it is new or updated
        # 3 items: prompts from UI, promptlists with prompts+encodings, final_list with promptlist plus expansions 
        
        # update promptlist if reencoding is necessary, Checking only if text has changed
        text_encoders_are_loaded = False

        for i, (pmt, pmt_g) in enumerate(zip(prompt, prompt_g)):
            if pmt[0] != pmt_g[0]:
                if text_encoders_are_loaded == False:
                    stream.output_queue.push(('progress', (False, 0, None, '', make_progress_bar_html(0, 'Text encoding ...'))))
                    offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=17.5)
                    load_model_as_complete(text_encoder, target_device=gpu, unload=False)            
                    load_model_as_complete(text_encoder_2, target_device=gpu, unload=False)
                    text_encoders_are_loaded = True
                # translatedText = argostranslate.translate.translate(pmt[0], from_code, to_code)
                # move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=1)
                l, c = encode_prompt_conds(f'Describe subject and changes. {pmt[0]}.', text_encoder, text_encoder_2, tokenizer, tokenizer_2)
                
                #l, c = encode_prompt_conds(f'{pmt[0]}', text_encoder, text_encoder_2, tokenizer, tokenizer_2)
                # print(l)
                # print(c)

                data[i]['llama_vec'] = l
                data[i]['clip_l_pooler'] = c
                data[i]['last_image'] = []
                # stream.output_queue.push(('latent', (0, False, i, None)))  
        
        prompt_g = prompt

       
        if text_encoders_are_loaded:
            unload_specific_models(text_encoder)
            unload_specific_models(text_encoder_2)

        # Need to get latent segment times properly ordered

        history_length = prompt[0][1]  

        for i, unit in enumerate(data): 
            
            # if the time is greater than zero, store it as the remainder
            
            if prompt[i][1] > 0:
                remainder = prompt[i][1]
                temp_list = []

                # if the remainder is larger than history time that has passed so far, subtract history time and append it

                while remainder > history_length:
                    temp_list.append(history_length)
                    remainder -= history_length
                    history_length += history_length

                # then append whatever is left over
                
                if remainder > 0:
                    temp_list.append(remainder)
                    # Dont add the first time, since we initialized history_length with it
                    
                    if i>0:
                        history_length += remainder

                # Now each time in this unit is smaller than the sum of all previous times
                # now constrain times to be no larger than the max_length 

                temp_list2 = []
                for temp_list_item in temp_list:
                    divisions = int(temp_list_item // max_segment_length)

                    if divisions > 0:
                        for j in range(divisions):
                            temp_list2.append(max_segment_length)

                    remainder = temp_list_item % max_segment_length

                    if remainder > 0:
                        temp_list2.append(remainder)
     
                # final_list += temp_list2
                unit['ordered_segments'] = temp_list2



        
        # Also check if we need to clear any of the existing latent renders    
        last_render = False
        for i, p in enumerate(prompt):

            if p[2]:
                # stream.output_queue.push(('latent', (0, False, i, None)))  
                last_render = True
            else:
                if last_render:
                    data[i]['ordered_segments'] = []
                    data[i]['prompt_latent'] = []
                    data[i]['prompt_frame_len'] = []   
                    data[i]['last_image'] = []
                    stream.output_queue.push(('latent', (0, False, i, None)))    

        # if cfg == 1:
            # llama_vec_n, clip_l_pooler_n = torch.zeros_like(promptlist[0][0]), torch.zeros_like(promptlist[0][1])
        # else:
            # llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        llama_vec_n, clip_l_pooler_n = torch.zeros_like(data[0]['llama_vec']), torch.zeros_like(data[0]['clip_l_pooler'])
        

        # Processing input image

        stream.output_queue.push(('progress', (False, 0, None, '', make_progress_bar_html(0, 'Image processing ...'))))
        
        if input_image is None:
            resolution = int(((resolution)//16)*16)
            height = width = resolution
            input_image = np.zeros((resolution, resolution, 3), dtype=np.uint8)
            input_image_np = np.array(input_image)
        else:        
            H, W, C = input_image.shape
            scale = resolution/pow(H*W,0.5)
            height, width = int(((scale*H)//16)*16), int(((scale*W)//16)*16)

            input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
            # Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))
        
        if not np.array_equal(input_image_np, input_image_g):

            input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
        
        
            # VAE encoding

            stream.output_queue.push(('progress', (False, 0, None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

            load_model_as_complete(vae, target_device=gpu, unload=False)
            start_latent = vae_encode(input_image_pt, vae)
            unload_specific_models(vae)
            
            # CLIP Vision

            stream.output_queue.push(('progress', (False, 0, None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

            load_model_as_complete(image_encoder, target_device=gpu, unload=False)
            image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
            unload_specific_models(image_encoder)
            
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
            input_image_g = input_image_np


        # Sampling

        stream.output_queue.push(('progress', (False, 0, None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu")


        if use_teacache:
            transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
        else:
            transformer.initialize_teacache(enable_teacache=False)
            
        initial_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32).cpu()
        initial_latents = torch.cat([initial_latents, start_latent.to(initial_latents)], dim=2)     

        history_latents = []

        for i, prmpt in enumerate(data):
            

            history_latents = initial_latents
            prompt_frames = 0
            
            if i > 0:
                
                # Build the history latents
                
                for j in range (i): # need to iterate over the latent list
                    for generated_latent in data[j]['prompt_latent']:
                        history_latents = torch.cat([history_latents, generated_latent.to(history_latents)], dim=2)
                
                if prompt[i][2] == True and gpu_memory_preservation:
                    if prmpt['last_image'] == []:
                        # CLIP Vision

                        
                        offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=9.2)
                        load_model_as_complete(vae, target_device=gpu, unload=False)
                        input_image_np = vae_decode(history_latents[:, :, -1:, :, :], vae)
                        unload_specific_models(vae)
                       
                        input_image_np = torch.clamp(input_image_np.float(), -1., 1.) * 127.5 + 127.5
                        input_image_np = input_image_np.detach().cpu().to(torch.uint8)
                        input_image_np = np.array(einops.rearrange(input_image_np, 'b c t h w -> (t w) (b h) c'))

                        load_model_as_complete(image_encoder, target_device=gpu, unload=False)
                        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)

                        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
                        prmpt['last_image'] = image_encoder_last_hidden_state
                        unload_specific_models(image_encoder)
                        
                    else:
                        image_encoder_last_hidden_state = prmpt['last_image']
                
            # if the prmpt is marked to render
           
            if prompt[i][2] == True:     

                prompt_latents = []
                image_latents = torch.zeros(size=(1, 16, 1, height // 8, width // 8), dtype=torch.float32, device='cuda')
                
                if loaded_lora:
                    transformer = load_lora(transformer, loaded_lora, lora_strength)                    
                
                for latent_len_i, latent_len in enumerate(prmpt['ordered_segments']):

                    # Check if we reached the end of th eprocessing buffer
                    if stream.input_queue.top() == 'end':
                        stream.output_queue.push(('end', None))
                        return

                    latent_window_size = int((float(latent_len)*30)//4)

                    llama_vec, llama_attention_mask = crop_or_pad_yield_mask(prmpt['llama_vec'], length=512)
                    llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
                    
                    # Dtype            
                    llama_vec = llama_vec.to(transformer.dtype)
                    llama_vec_n = llama_vec_n.to(transformer.dtype)
                    clip_l_pooler = prmpt['clip_l_pooler'].to(transformer.dtype)
                    clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
                    image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

                    # unload_complete_models()
                    # Estimate memory reservation for inference
                    offset_mem = resolution*0.0024 #GB   
                    size_in_bytes = 4 * 16 * (latent_window_size * 4) // 4 * height // 8 * width // 8 # float 32            
                    latent_size = (size_in_bytes / (1024*1024)) #GB
                    gpu_memory_preservation = latent_size + offset_mem
                    print('size: ', gpu_memory_preservation)
                    
                    offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation+1)
                    move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation+1)

                    def callback(d):
                        preview = d['denoised']
                        preview = vae_decode_fake(preview)

                        preview = (preview * 127.5 + 127.5).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                        preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')
                        
                        if stream.input_queue.top() == 'end':
                            stream.output_queue.push(('end', None))
                            raise KeyboardInterrupt('User ends the task.')

                        current_step = d['i'] + 1
                        percentage = int(100.0 * current_step / steps)
                        hint = f'Sampling {current_step}/{steps}'
                        desc = f'Currently generating: {prompt[i][0]}'
                        latent_height, _, _ = preview.shape
                        stream.output_queue.push(('progress', (True, latent_height, preview, desc, make_progress_bar_html(percentage, hint))))
                        return

                    indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
                    clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
                    clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

                    clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
                    clean_latents = torch.cat([history_latents[:, :, -1:, :, :].to(history_latents), clean_latents_1x], dim=2)
                    
                    # Genereated products
                    
                    if randomize:
                        seed = random.randint(1, 100000000)
                        stream.output_queue.push(('seed', (seed)))

                    rnd.manual_seed(seed)

                    
                    generated_latents = sample_hunyuan(
                        transformer=transformer,
                        quick_preview = quick_preview,
                        sampler='unipc',
                        width=width,
                        height=height,
                        frames=latent_window_size * 4 - 3,
                        real_guidance_scale=cfg,
                        distilled_guidance_scale=gs,
                        guidance_rescale=rs,
                        #shift=shift,
                        num_inference_steps=steps,
                        generator=rnd,
                        prompt_embeds=llama_vec,
                        prompt_embeds_mask=llama_attention_mask,
                        prompt_poolers=clip_l_pooler,
                        negative_prompt_embeds=llama_vec_n,
                        negative_prompt_embeds_mask=llama_attention_mask_n,
                        negative_prompt_poolers=clip_l_pooler_n,
                        device=gpu,
                        dtype=torch.bfloat16,
                        image_embeddings=image_encoder_last_hidden_state,
                        latent_indices=latent_indices,
                        clean_latents=clean_latents,
                        clean_latent_indices=clean_latent_indices,
                        clean_latents_2x=clean_latents_2x,
                        clean_latent_2x_indices=clean_latent_2x_indices,
                        clean_latents_4x=clean_latents_4x,
                        clean_latent_4x_indices=clean_latent_4x_indices,
                        callback=callback,
                    )
                    
                   
                    # Prompt products
                    

                    image_latents = torch.cat([image_latents, generated_latents], dim=2)                        
                    latent0 = vae_decode_fake(image_latents[:, :, 1:, :, :])
                    latent0 = (latent0 * 127.5 + 127.5).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                    latent0 = einops.rearrange(latent0, 'b c t h w -> (b h) (t w) c')
                    latent_height, _, _ = latent0.shape
                    stream.output_queue.push(('latent', (latent_height, True, i, latent0)))         

                    
                    prompt_frames += int(generated_latents.shape[2])
                    prompt_latents.append(generated_latents)
                    history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)    
                prmpt['prompt_latent'] = prompt_latents #latent  
                prmpt['prompt_frame_len'] = prompt_frames  #total frames 
                
                if loaded_lora:
                    transformer = unload_all_loras(transformer)
            
 
        if decode_latent:
            final_latents = initial_latents
            
            total_generated_latent_frames = 1
            for prmpt in data:
                if prmpt['prompt_frame_len']:
                    total_generated_latent_frames += prmpt['prompt_frame_len']
                    for generated_latent in prmpt['prompt_latent']:
                        final_latents = torch.cat([final_latents, generated_latent.to(history_latents)], dim=2)                    

            offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=12)
            load_model_as_complete(vae, target_device=gpu, unload=False)
            stream.output_queue.push(('progress', (False, 0, None, '', make_progress_bar_html(0, 'VAE decoding ...'))))

            
            real_history_latents = final_latents[:, :, -total_generated_latent_frames:, :, :]
            history_pixels = vae_decode(real_history_latents, vae).cpu()
            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            stream.output_queue.push(('file', output_filename))
            unload_specific_models(vae)
                    


            
    except:
        traceback.print_exc()

    stream.output_queue.push(('end', None))
    return


def process(input_image, n_prompt, seed, total_second_length, steps, shift, cfg, gs, rs, gpu_memory_preservation, use_teacache, resolution, mp4_crf, max_segment_length, prompt0, length0, prompt1, length1, prompt2, length2, prompt3, length3, render0, render1, render2, render3, decode_latent, randomize, quick_preview, lora_strength, loaded_lora ):
    global stream
    # assert input_image is not None, 'No input image!'

    yield None, gr.update(), '', '', gr.update(interactive=False), gr.update(interactive=True), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    stream = AsyncStream()
    temp_file = []

    async_run(worker, input_image, n_prompt, seed, total_second_length, steps, shift, cfg, gs, rs, gpu_memory_preservation, use_teacache, resolution, mp4_crf, max_segment_length, prompt0, length0, prompt1, length1, prompt2, length2, prompt3, length3, render0, render1, render2, render3, decode_latent, randomize, quick_preview, lora_strength, loaded_lora  )

    output_filename = None

    while True: # maybe add a wait to slow this down
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            temp_file.append(data)
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        if flag == 'progress':
            visible, latent_height, preview, desc, html = data
            yield gr.update(), gr.update(height=latent_height, visible=visible, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                
        if flag == 'seed':
            seed = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(value=seed)   
            
        if flag == 'latent':
            latent_height, isvisible, i, latent0 = data
            if i==0:
                yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(height=latent_height, visible=isvisible, value=latent0), gr.update(), gr.update(), gr.update(), gr.update()
            if i==1:
                yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(height=latent_height, visible=isvisible, value=latent0), gr.update(), gr.update(), gr.update()  
            if i==2:
                yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(height=latent_height, visible=isvisible, value=latent0), gr.update(), gr.update()   
            if i==3:
                yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(height=latent_height, visible=isvisible, value=latent0), gr.update()                   

        if flag == 'end':
            for file in temp_file[:-1]:
                print(file)
                os.remove(file)
            yield output_filename, gr.update(visible=True), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            break
        time.sleep(0.1)


def end_process():
    stream.input_queue.push('end')



css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack-F1')
    with gr.Row():

        with gr.Column():
            with gr.Group():
                result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=400, loop=True) 
                preview_image = gr.Image(label="Next Latents", height=0, visible=False)                
                progress_bar = gr.HTML('', elem_classes='no-generating-animation')
                progress_desc = gr.Markdown('', elem_classes='no-generating-animation')

            with gr.Accordion("Input Image", open=False):
                input_image = gr.Image(sources='upload', type="numpy", label="Image", height=400)                

            with gr.Row():
                start_button = gr.Button(value="Start Generation")
                end_button = gr.Button(value="End Generation", interactive=False)           

            with gr.Group():
                
                with gr.Column():               
                    with gr.Row(): 
                        render0 = gr.Checkbox(label="", value=True, min_width = 50, show_label=False)                            
                        prompt0 = gr.Textbox(label="", value='', scale=20, show_label=False)
                        length0 = gr.Number(label="", value=3, min_width=100, show_label=False)                    
                    latent0 = gr.Image(label="", interactive=False, height=100, visible=False)     
                with gr.Column():    
                    with gr.Row():  
                        render1 = gr.Checkbox(label="", value=False, min_width = 50)            
                        prompt1 = gr.Textbox(label="", value='', scale=20)
                        length1 = gr.Number(label="", value=0, min_width=100)   
                    latent1 = gr.Image(label="", interactive=False, height=100, visible=False) 
                with gr.Column(): 
                    with gr.Row():      
                        render2 = gr.Checkbox(label="", value=False, min_width = 50)  
                        prompt2 = gr.Textbox(label="", value='', scale=20)
                        length2 = gr.Number(label="", value=0, min_width=100)   
                    latent2 = gr.Image(label="", interactive=False, height=100, visible=False) 
                with gr.Column(): 
                    with gr.Row():      
                        render3 = gr.Checkbox(label="", value=False, min_width = 50)  
                        prompt3 = gr.Textbox(label="", value='', scale=20)
                        length3 = gr.Number(label="", value=0, min_width=100)                
                    latent3 = gr.Image(label="", interactive=False, height=100, visible=False) 
            with gr.Group():
                with gr.Row():  
                    with gr.Column():  
                        use_teacache = gr.Checkbox(label='Use TeaCache', value=True,)   
                        gpu_memory_preservation = gr.Checkbox(label="Use Recent Latent", value=False)  
                        decode_latent = gr.Checkbox(label='Decode Latent', value=False,)  
                    with gr.Column():  
                        randomize = gr.Checkbox(label='Randomize Next Run?', value=True,) 
                        seed = gr.Number(label="Seed", value=33, precision=0) 
                    with gr.Column(): 
                        quick_preview = gr.Checkbox(label='Quick Preview', value=False,)
                        resolution = gr.Number(label="Resolution", value=512, precision=0)
                    with gr.Column(): 
                        lora_strength = gr.Slider(label="Lora Stength", minimum=0, maximum=2, value=1, step=0.05)
            with gr.Accordion("Additional Parameters", open=False): 
                with gr.Column():
                    loaded_lora = gr.FileExplorer(root_dir="D:/SD/!Paths/loras/Hunyuan/")
                    # lora_strength
                    
            
            with gr.Accordion("Additional Parameters", open=False): 

                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # Not used
                total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1, visible=False)
                max_segment_length = gr.Slider(label="Maximum Segment Length [s]", minimum=0, maximum=10, value=1.5, step=0.1, visible=True)  # Should not change
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')
                shift = gr.Slider(label="shift", minimum=0, maximum=10, value=3, step=0.1, info='Changing this value is not recommended.')

                cfg = gr.Slider(label="text noise", minimum=0, maximum=10, value=1, step=0.1, visible=True)  # Should not change
                gs = gr.Slider(label="Distilled Guidance", minimum=0, maximum=30, value=10, step=0.5, info='Changing this value is not recommended.')
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=True)  # Should not change

                

                mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")

        # with gr.Column():



    ips = [input_image, n_prompt, seed, total_second_length, steps, shift, cfg, gs, rs, gpu_memory_preservation, use_teacache, resolution, mp4_crf, max_segment_length, prompt0, length0, prompt1, length1, prompt2, length2, prompt3, length3, render0, render1, render2, render3, decode_latent, randomize, quick_preview, lora_strength, loaded_lora]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button, latent0, latent1, latent2, latent3, seed])
    end_button.click(fn=end_process)


block.launch(
    server_name="0.0.0.0",#args.server,
    server_port=7860, #args.port,
    share=False, #args.share,
    inbrowser=args.inbrowser,
)
