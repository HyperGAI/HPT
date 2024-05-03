import os
import os.path as osp
import string
import sys
import warnings
import re
import torch.nn as nn

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          SiglipImageProcessor, 
                          GenerationConfig, StoppingCriteriaList)

from ..smp import cn_string, get_cache_path
from ..utils import DATASET_TYPE, CustomPrompt
from .modeling_siglip import SiglipVisionModel

def interpolate_pos_embed_siglip(model, new_size):
    pos_emb = model.vision_model.embeddings.position_embedding.weight.float()
    ori_size = int((pos_emb.shape[0])**0.5)
    dim = pos_emb.shape[1]
    print("Position interpolate from %dx%d to %dx%d" % (ori_size, ori_size, new_size, new_size))
    pos_tokens = pos_emb
    pos_tokens = pos_tokens.reshape(-1, ori_size, ori_size, dim).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2).squeeze(0)
    new_pos_embed = pos_tokens #torch.cat((extra_tokens, pos_tokens), dim=0)
    new_pos_embed = new_pos_embed.to(torch.float16)
    return torch.nn.Parameter(new_pos_embed)

class HPT1_5(CustomPrompt):

    INSTALL_REQ = True

    def __init__(self,
                 global_model_path='HyperGAI/HPT1_5-Air-Llama-3-8B-Instruct-multimodal',
                 vis_scale=448,
                 visual_select_layer=-2,
                 prompt_template='llama3_chat',
                 stop_words=[],
                 torch_dtype=torch.float16):
        from vlmeval.utils import PROMPT_TEMPLATE, StopWordStoppingCriteria

        llm = AutoModelForCausalLM.from_pretrained(global_model_path,
                                                   subfolder='llm',
                                                   trust_remote_code=True,
                                                   use_safetensors=True,
                                                   torch_dtype=torch_dtype,
                                                   device_map='cpu')

        tokenizer = AutoTokenizer.from_pretrained(global_model_path,
                                                  subfolder='llm',
                                                  trust_remote_code=True,
                                                  encode_special_tokens=True)
        print(f'Load LLM')

        # build visual_encoder
        image_size = vis_scale 
        self.image_size = vis_scale
        
        image_processor = SiglipImageProcessor.from_pretrained(global_model_path,
                                                    subfolder='visual_encoder',
                                                    size={"height": image_size, "width": image_size})
        
        visual_encoder = SiglipVisionModel.from_pretrained(global_model_path,
                                                subfolder='visual_encoder',
                                                torch_dtype=torch_dtype, device_map='cpu')

        patch_size = visual_encoder.vision_model.embeddings.patch_size
        num_positions = (image_size//patch_size)**2 
        new_size = image_size//patch_size
        visual_encoder.vision_model.embeddings.num_patches = (image_size//patch_size)**2
        visual_encoder.vision_model.embeddings.num_positions = num_positions
        visual_encoder.vision_model.embeddings.position_ids = torch.arange(num_positions).expand((1, -1))
        visual_encoder.vision_model.embeddings.position_embedding.weight = interpolate_pos_embed_siglip(visual_encoder, new_size)
        visual_encoder.config.image_size = image_size
        
        print(f'Load visual_encoder')
        

        projector = AutoModel.from_pretrained(global_model_path,
                                              subfolder='projector',
                                              use_safetensors=True,
                                              trust_remote_code=True,
                                              torch_dtype=torch_dtype,
                                              )
        print(f'Load projector')

        llm.eval()
        visual_encoder.eval()
        projector.eval()

        self.llm = llm.cuda()
        self.tokenizer = tokenizer
        self.visual_encoder = visual_encoder.cuda()
        self.image_processor = image_processor
        self.projector = projector.cuda()
        self.visual_select_layer = visual_select_layer
        if prompt_template is not None:
            self.prompt_template = PROMPT_TEMPLATE[prompt_template]
            stop_words += self.prompt_template.get('STOP_WORDS', [])
        else:
            self.prompt_template = None

        self.stop_criteria = StoppingCriteriaList()
        for word in stop_words:
            self.stop_criteria.append(
                StopWordStoppingCriteria(self.tokenizer, word))

    def build_gen_config(self, dataset, qtype):
        gen_kwargs = dict(max_new_tokens=1024,
                          do_sample=True,
                          temperature=0.6,
                          num_beams=5,
                          top_p=0.9,
                          eos_token_id=self.tokenizer.eos_token_id,
                          pad_token_id=self.tokenizer.pad_token_id
                          if self.tokenizer.pad_token_id is not None else
                          self.tokenizer.eos_token_id)
        # For single word generation
        if (dataset is not None and DATASET_TYPE(dataset) in ['multi-choice', 'Y/N']):
            if qtype == '':
                gen_kwargs.update(
                    dict(max_new_tokens=5, do_sample=False, num_beams=1))
            elif qtype == 'open':
                gen_kwargs.update(
                    dict(max_new_tokens=1024, do_sample=False, num_beams=1))

        return GenerationConfig(**gen_kwargs)

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'multi-choice':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if ('hint' in line
                                and not pd.isna(line['hint'])) else None

        if hint is not None:
            try:
                question = hint + '\n' + question
            except:
                print(hint)

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        qtype = ''
        if not cn_string(question):
            if 'question_type' in line.keys() and 'choice' not in line['question_type']:
                prompt = question + '\n' + ("Answer with succinct phrase or a single word, incorporating professional terms when necessary, to address inquiries regarding scene description, key elements identification, and potential activities in the provided image.") #("Answer the question using a single word or phrase.")
                qtype = 'open'
            else:
                prompt = question + '\n' + ("Answer with the option's letter from the given choices directly.")
        else:
            prompt = question + '\n' + '请直接回答选项字母。'

        return {'image': tgt_path, 'text': prompt, 'qtype': qtype}

    def generate(self, image_path, prompt, dataset=None, qtype=''):
        from vlmeval.utils import expand2square, prepare_inputs_labels_for_multimodal, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        
        if isinstance(image_path, list) and len(image_path) > 1 and prompt.count('<image 1>') > 0 and prompt.count('<image 2>') > 0:
            image_list = [Image.open(image_path_cur).convert('RGB') for image_path_cur in image_path]
            image1 = image_list[0]
            w1, h1 = image1.size 
           
            image2 = image_list[1]
            w2, h2 = image2.size 
            
            w_sum, h_sum = w1 + w2, h1 + h2
            if w_sum > h_sum:
                h2_new = int(h2 * w2 / w1)
                image2 =image2.resize((w1, h2_new))
                image = Image.new('RGB', (w1, h1 + h2_new))
                image.paste(image1, (0, 0))
                image.paste(image2, (0, h1))
                
                prompt = prompt.replace('<image 1>', '<up image>')
                prompt = prompt.replace('<image 2>', '<down image>')
            else:
                w2_new = int(w2 * h2 / h1)
                image2 = image2.resize((w2_new, h2))
                image = Image.new('RGB', (w1 + w2_new, h2))
                image.paste(image1, (0, 0))
                image.paste(image2, (w1, 0))
               
                prompt = prompt.replace('<image 1>', '<left image>')
                prompt = prompt.replace('<image 2>', '<right image>')

            image_size = 448
            image = image.resize((image_size, image_size), 3)
        elif isinstance(image_path, list):
            image = Image.open(image_path[0]).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')

        image = self.image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        image = image.cuda().unsqueeze(0)
        visual_outputs = self.visual_encoder(image, output_hidden_states=True)
        pixel_values = self.projector(
            visual_outputs.hidden_states[self.visual_select_layer][:, 1:])

        inputs = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        if self.prompt_template:
            inputs = self.prompt_template['INSTRUCTION'].format(input=inputs)

        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = self.tokenizer(chunk)
            else:
                cur_encode = self.tokenizer(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        if len(chunk_encode) != 2:
            print(prompt)

        assert len(chunk_encode) == 2
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode['input_ids'])
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids).cuda().unsqueeze(0)
        mm_inputs = prepare_inputs_labels_for_multimodal(
            llm=self.llm, input_ids=ids, pixel_values=pixel_values)

        gen_config = self.build_gen_config(dataset, qtype)

        generate_output = self.llm.generate(
            **mm_inputs,
            generation_config=gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria)
        predict = self.tokenizer.decode(generate_output[0],
                                        skip_special_tokens=True).strip()
        return predict
