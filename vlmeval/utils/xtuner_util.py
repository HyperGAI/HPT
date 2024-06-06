from PIL import Image
import torch
from torch import nn
from transformers import PreTrainedModel
from typing import List, Optional

from mmengine.config import ConfigDict
from transformers import StoppingCriteria


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN_INDEX = 0
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = '<image>'


class StopWordStoppingCriteria(StoppingCriteria):
    """StopWord stopping criteria."""

    def __init__(self, tokenizer, stop_word):
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.length = len(self.stop_word)

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        cur_text = self.tokenizer.decode(input_ids[0])
        cur_text = cur_text.replace('\r', '').replace('\n', '')
        return cur_text[-self.length:] == self.stop_word


PROMPT_TEMPLATE = ConfigDict(
    phi3_chat=dict(
        SYSTEM='<|system|>\n{system}<|end|>\n',
        INSTRUCTION='<|user|>\n{input}<|end|>\n<|assistant|>\n',
        SUFFIX='<|end|>',
        SUFFIX_AS_EOS=True,
        SEP='\n',
        STOP_WORDS=['<|end|>']),
    default=dict(
        SYSTEM='<|System|>:{system}\n',
        INSTRUCTION='<|User|>:{input}\n<|Bot|>:',
        SEP='\n'),
    zephyr=dict(
        SYSTEM='<|system|>\n{system}\n',
        INSTRUCTION='<|user|>\n{input}\n<|assistant|>\n',
        SEP='\n'),
    internlm_chat=dict(
        SYSTEM='<|System|>:{system}\n',
        INSTRUCTION='<|User|>:{input}<eoh>\n<|Bot|>:',
        SUFFIX='<eoa>',
        SUFFIX_AS_EOS=True,
        SEP='\n',
        STOP_WORDS=['<eoa>']),
    internlm2_chat=dict(
        SYSTEM='<|im_start|>system\n{system}<|im_end|>\n',
        INSTRUCTION=('<|im_start|>user\n{input}<|im_end|>\n'
                     '<|im_start|>assistant\n'),
        SUFFIX='<|im_end|>',
        SUFFIX_AS_EOS=True,
        SEP='\n',
        STOP_WORDS=['<|im_end|>', '<|endoftext|>']),
    llama3_chat=dict(
        SYSTEM='',
        INSTRUCTION=('<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nAnswer the questions carefully!<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'),
        SUFFIX='<|end_of_text|>',
        SUFFIX_AS_EOS=True,
        SEP='\n\n',
        STOP_WORDS=['<|eot_id|>', '<|end_of_text|>']),
    moss_sft=dict(
        SYSTEM='{system}\n',
        INSTRUCTION='<|Human|>: {input}<eoh>\n',
        SEP='\n',
        STOP_WORDS=['<eoc>', '<eom>']),
    llama2_chat=dict(
        SYSTEM=(
            '[INST] <<SYS>>\n You are a helpful, respectful and honest '
            'assistant. Always answer as helpfully as possible, while being '
            'safe. Your answers should not include any harmful, unethical, '
            'racist, sexist, toxic, dangerous, or illegal content. Please '
            'ensure that your responses are socially unbiased and positive in '
            'nature.\n{system}\n<</SYS>>\n [/INST] '),
        INSTRUCTION='[INST] {input} [/INST]',
        SEP='\n'),
    code_llama_chat=dict(
        SYSTEM='{system}\n', INSTRUCTION='[INST] {input} [/INST]'),
    chatglm2=dict(
        SYSTEM='{system}\n',
        INSTRUCTION='[Round {round}]\n\n问：{input}\n\n答：',
        SEP='\n\n'),
    chatglm3=dict(
        SYSTEM='<|system|>\n{system}',
        INSTRUCTION='<|user|>\n{input}<|assistant|>\n',
        SEP='\n'),
    qwen_chat=dict(
        SYSTEM=('<|im_start|>system\n{system}<|im_end|>\n'),
        INSTRUCTION=('<|im_start|>user\n{input}<|im_end|>\n'
                     '<|im_start|>assistant\n'),
        SUFFIX='<|im_end|>',
        SUFFIX_AS_EOS=True,
        SEP='\n',
        STOP_WORDS=['<|im_end|>', '<|endoftext|>']),
    baichuan_chat=dict(
        SYSTEM='{system}\n',
        INSTRUCTION='<reserved_102>{input}<reserved_103>',
        SEP='\n'),
    baichuan2_chat=dict(
        SYSTEM='{system}\n',
        INSTRUCTION='<reserved_106>{input}<reserved_107>',
        SEP='\n'),
    wizardlm=dict(
        SYSTEM=('A chat between a curious user and an artificial '
                'intelligence assistant. The assistant gives '
                'helpful, detailed, and polite answers to the '
                'user\'s questions. {system}\n '),
        INSTRUCTION=('USER: {input} ASSISTANT:'),
        SEP='\n'),
    wizardcoder=dict(
        SYSTEM=(
            'Below is an instruction that describes a task. '
            'Write a response that appropriately completes the request.\n\n'
            '{system}\n '),
        INSTRUCTION=('### Instruction:\n{input}\n\n### Response:'),
        SEP='\n\n'),
    vicuna=dict(
        SYSTEM=('A chat between a curious user and an artificial '
                'intelligence assistant. The assistant gives '
                'helpful, detailed, and polite answers to the '
                'user\'s questions. {system}\n '),
        INSTRUCTION=('USER: {input} ASSISTANT:'),
        SEP='\n'),
    deepseek_coder=dict(
        SYSTEM=('You are an AI programming assistant, utilizing '
                'the DeepSeek Coder model, developed by DeepSeek'
                'Company, and you only answer questions related '
                'to computer science. For politically sensitive '
                'questions, security and privacy issues, and '
                'other non-computer science questions, you will '
                'refuse to answer. {system}\n'),
        INSTRUCTION=('### Instruction:\n{input}\n### Response:\n'),
        SEP='\n'),
    # TODO: deprecation, v0.2.0
    deepseekcoder=dict(
        SYSTEM=('You are an AI programming assistant, utilizing '
                'the DeepSeek Coder model, developed by DeepSeek'
                'Company, and you only answer questions related '
                'to computer science. For politically sensitive '
                'questions, security and privacy issues, and '
                'other non-computer science questions, you will '
                'refuse to answer. {system}\n'),
        INSTRUCTION=('### Instruction:\n{input}\n### Response:\n'),
        SEP='\n'),
    deepseek_moe=dict(
        SYSTEM=('[INST] {system} [/INST]\n'),
        INSTRUCTION=('[INST] {input} [/INST]'),
        SEP='\n'),
    mistral=dict(
        SYSTEM=('[INST] {system} [/INST]\n'),
        INSTRUCTION=('[INST] {input} [/INST]'),
        SEP='\n'),
    mixtral=dict(
        SYSTEM=('[INST] {system} [/INST]\n'),
        INSTRUCTION=('[INST] {input} [/INST]'),
        SEP='\n'),
)

SYSTEM_TEMPLATE = ConfigDict(
    moss_sft=('You are an AI assistant whose name is {bot_name}.\n'
              'Capabilities and tools that {bot_name} can possess.\n'
              '- Inner thoughts: enabled.\n'
              '- Web search: enabled. API: Search(query)\n'
              '- Calculator: enabled. API: Calculate(expression)\n'
              '- Equation solver: enabled. API: Solve(equation)\n'
              '- Text-to-image: disabled.\n'
              '- Image edition: disabled.\n'
              '- Text-to-speech: disabled.\n'),
    alpaca=('Below is an instruction that describes a task. '
            'Write a response that appropriately completes the request.\n'),
    arxiv_gentile=('If you are an expert in writing papers, please generate '
                   "a good paper title for this paper based on other authors' "
                   'descriptions of their abstracts.\n'),
    colorist=('You are a professional color designer. Please provide the '
              'corresponding colors based on the description of Human.\n'),
    coder=('You are a professional programer. Please provide the '
           'corresponding code based on the description of Human.\n'),
    lawyer='你现在是一名专业的中国律师，请根据用户的问题给出准确、有理有据的回复。\n',
    medical='如果你是一名医生，请根据患者的描述回答医学问题。\n',
    sql=('If you are an expert in SQL, please generate a good SQL Query '
         'for Question based on the CREATE TABLE statement.\n'),
)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def prepare_inputs_labels_for_multimodal(
        llm: PreTrainedModel,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None):
    if pixel_values is None:
        return {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'inputs_embeds': None,
            'labels': labels
        }

    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(
            0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # remove the padding using attention_mask -- TODO: double check
    input_ids = [
        cur_input_ids[cur_attention_mask]
        for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
    ]
    labels = [
        cur_labels[cur_attention_mask]
        for cur_labels, cur_attention_mask in zip(labels, attention_mask)
    ]

    new_inputs_embeds = []
    new_labels = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            cur_pixel_values = pixel_values[cur_image_idx]
            
            cur_inputs_embeds_1 = llm.get_input_embeddings()(cur_input_ids)
            cur_inputs_embeds = torch.cat(
                [cur_inputs_embeds_1, cur_pixel_values[0:0]], dim=0)
            new_inputs_embeds.append(cur_inputs_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        image_token_indices = [-1] + torch.where(
            cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]
            ]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] +
                                                    1:image_token_indices[i +
                                                                          1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] +
                                              1:image_token_indices[i + 1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_inputs_embeds = llm.get_input_embeddings()(
            torch.cat(cur_input_ids_noim))
        cur_inputs_embeds_no_im = torch.split(
            cur_inputs_embeds, split_sizes, dim=0)
        cur_new_inputs_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_inputs_embeds.append(cur_inputs_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                cur_pixel_values = pixel_values[cur_image_idx]
                cur_image_idx += 1
                cur_new_inputs_embeds.append(cur_pixel_values)
                cur_new_labels.append(
                    torch.full((cur_pixel_values.shape[0], ),
                               IGNORE_INDEX,
                               device=cur_labels.device,
                               dtype=cur_labels.dtype))

        cur_new_inputs_embeds = torch.cat(cur_new_inputs_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_inputs_embeds.append(cur_new_inputs_embeds)
        new_labels.append(cur_new_labels)

    # Combine them
    max_len = max(x.shape[0] for x in new_inputs_embeds)
    batch_size = len(new_inputs_embeds)

    new_inputs_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len),
                                   IGNORE_INDEX,
                                   dtype=new_labels[0].dtype,
                                   device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len),
                                 dtype=attention_mask.dtype,
                                 device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len),
                               dtype=position_ids.dtype,
                               device=position_ids.device)

    for i, (cur_new_embed,
            cur_new_labels) in enumerate(zip(new_inputs_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        new_inputs_embeds_padded.append(
            torch.cat((cur_new_embed,
                       torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                                   dtype=cur_new_embed.dtype,
                                   device=cur_new_embed.device)),
                      dim=0))
        if cur_len > 0:
            new_labels_padded[i, :cur_len] = cur_new_labels
            attention_mask[i, :cur_len] = True
            position_ids[i, :cur_len] = torch.arange(
                0,
                cur_len,
                dtype=position_ids.dtype,
                device=position_ids.device)

    new_inputs_embeds = torch.stack(new_inputs_embeds_padded, dim=0)

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None

    return {
        'input_ids': None,
        'position_ids': position_ids,
        'attention_mask': attention_mask,
        'past_key_values': past_key_values,
        'inputs_embeds': new_inputs_embeds,
        'labels': new_labels
    }
