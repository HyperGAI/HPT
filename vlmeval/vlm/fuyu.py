import string
import pandas as pd
import torch
from transformers import GenerationConfig
from ..smp import cn_string
from ..utils import DATASET_TYPE, CustomPrompt
from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image


class Fuyu8B(CustomPrompt):
    INSTALL_REQ = True

    def __init__(self, model_id="adept/fuyu-8b"):
        self.processor = FuyuProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.model = FuyuForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16).cuda()
        self.model.eval()

    def build_gen_config(self, dataset, qtype):
        gen_kwargs = dict(max_new_tokens=1024,
                          do_sample=False,
                          # temperature=0.7,
                          num_beams=1,
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
        elif dataset == 'demo':
            # for demo test
            gen_kwargs = dict(max_new_tokens=1024,
                              do_sample=False,
                              # temperature=0.2,
                              # top_p=0.7,
                              num_beams=1,
                              eos_token_id=self.tokenizer.eos_token_id,
                              pad_token_id=self.tokenizer.pad_token_id
                              if self.tokenizer.pad_token_id is not None else
                              self.tokenizer.eos_token_id)

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
                prompt = question + '\n' + (
                    "Answer with succinct phrase or a single word, incorporating professional terms when necessary, to address inquiries regarding scene description, key elements identification, and potential activities in the provided image.")  # ("Answer the question using a single word or phrase.")
                qtype = 'open'
            else:
                prompt = question + '\n' + ("Answer with the option's letter from the given choices directly.")
        else:
            prompt = question + '\n' + '请直接回答选项字母。'

        return {'image': tgt_path, 'text': prompt, 'qtype': qtype}

    def generate(self, image_path, prompt, dataset=None, qtype=''):
        image = Image.open(image_path).convert('RGB')
        # import pdb; pdb.set_trace()
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
        gen_config = self.build_gen_config(dataset, qtype)

        # autoregressively generate text
        generation_output = self.model.generate(**inputs,
                                                bos_token_id=self.tokenizer.bos_token_id,
                                                generation_config=gen_config)
        import pdb; pdb.set_trace()
        predict = self.processor.batch_decode(generation_output[:, inputs.input_ids.shape[1]:],
                                              skip_special_tokens=True)[0].strip()
        return predict
