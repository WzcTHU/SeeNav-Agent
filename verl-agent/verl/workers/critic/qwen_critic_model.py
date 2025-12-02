import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen2_5_VLForConditionalGeneration

@dataclass
class CausalLMOutputWithValue(CausalLMOutputWithPast):
    values: torch.FloatTensor = None

class Qwen2_5_VLForConditionalGenerationWithValueHead(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # 通常 transformer hidden_size 在 config.hidden_size
        self.value_head = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        output_value=True, # 新增
        **kwargs
    ):
        print('################################## HIT FORWARD ##################################')
        # 要保证 output_hidden_states True
        if output_hidden_states is None:
            output_hidden_states = True
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True, # 强制dict
            **kwargs
        )

        # 取最后一层 hidden_state (batch, seq, hidden)
        # 视模型决定输出为 tuple 还是 dict
        last_hidden_state = outputs.hidden_states[-1]
        values = self.value_head(last_hidden_state)  # (batch, seq, 1)

        if output_value:
            # 复制所有原始输出，添加 values
            return CausalLMOutputWithValue(
                loss=outputs.get('loss', None),
                # logits=outputs['logits'],
                logits=values,
                past_key_values=outputs.get('past_key_values', None),
                hidden_states=outputs.get('hidden_states', None),
                attentions=outputs.get('attentions', None),
                values=values
            )
        else:
            return outputs