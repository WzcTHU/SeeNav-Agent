import os
import anthropic
from openai import OpenAI
from lmdeploy import pipeline, PytorchEngineConfig
from planner.utils.EBNav_planner_config import llm_generation_guide, vlm_generation_guide
from planner.utils.EBNav_planner_utils import convert_format_2claude, convert_format_2gemini, ActionPlan_1, ActionPlan, ActionPlan_lang, fix_json, convert_format_2qwen25vl

from openai import AzureOpenAI
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM, LlavaForConditionalGeneration, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

from peft import PeftModel, PeftConfig
import torch

endpoint = "your_end_point"
subscription_key = os.getenv('AZURE_OPENAI_API_KEY')
api_version = "your_api_version"

temperature = 0
max_completion_tokens = 2048
remote_url = os.environ.get('remote_url')

class RemoteModel:
    def __init__(
        self,
        model_name,
        model_type='remote',
        language_only=False,
        tp=1,
        task_type=None, # used to distinguish between manipulation and other environments
        local_model_path=None
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.language_only = language_only
        self.task_type = task_type
        self.processor = None
        if self.model_type == 'local':
            model_path = local_model_path
            if model_name in ['Qwen2.5-VL-3B-Instruct', 'Qwen2.5-VL-7B-Instruct']:
                # model_path = '/mnt/adtfs/jensencwang/research/verl-agent/checkpoints/verl_agent_ebnav/srgpo_qwen2.5_vl_7b-sft_step_w-0.5-json/global_step_150/actor/huggingface'
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
                self.processor = AutoProcessor.from_pretrained(model_path)

                # peft_config = PeftConfig.from_pretrained('/mnt/kaiwu-user-jensencwang/research/LLama-Factory/LLaMA-Factory/saves/qwen2_5vl-7b/lora/SeeNav-SFT-lr16')
                # base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
                # self.model = PeftModel.from_pretrained(base_model, '/mnt/kaiwu-user-jensencwang/research/LLama-Factory/LLaMA-Factory/saves/qwen2_5vl-7b/lora/SeeNav-SFT-lr16')
                # self.processor = AutoProcessor.from_pretrained(peft_config.base_model_name_or_path)
            elif model_name in ['deepseek-vl2-tiny']:
                self.processor = AutoProcessor.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
            elif model_name in ['llava-1.5-7b-hf']:
                self.processor = AutoProcessor.from_pretrained(model_path)
                self.model = LlavaForConditionalGeneration.from_pretrained(model_path)
            elif model_name in ['InternVL3_5-2B-HF']:
                self.processor = AutoProcessor.from_pretrained(model_path)
                self.model = AutoModelForImageTextToText.from_pretrained(model_path)
            else:
                backend_config = PytorchEngineConfig(session_len=12000, dtype='float16', tp=tp)
                self.model = pipeline(self.model_name, backend_config=backend_config)

        else:
            if "claude" in self.model_name:
                self.model = anthropic.Anthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY"),
                )
            elif "gemini" in self.model_name:
                self.model = OpenAI(
                    api_key=os.environ.get("GEMINI_API_KEY"),
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
            elif "gpt" in self.model_name:
                self.model = OpenAI()
            elif 'qwen' in self.model_name:
                self.model = OpenAI(
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
            elif "Qwen2-VL" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "Qwen2.5-VL" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif 'GPT4.1' in self.model_name:
                self.model = AzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    api_key=subscription_key,
                )
            elif "Llama-3.2-11B-Vision-Instruct" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "OpenGVLab/InternVL" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "meta-llama/Llama-3.2-90B-Vision-Instruct" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "90b-vision-instruct" in self.model_name: # you can use fireworks to inference
                self.model = OpenAI(base_url='https://api.fireworks.ai/inference/v1',
                                    api_key=os.environ.get("firework_API_KEY"))
            else:
                try:
                    self.model = OpenAI(base_url = remote_url)
                except:
                    raise ValueError(f"Unsupported model name: {model_name}")


    def respond(self, message_history: list):
        if self.model_type == 'local':
            return self._call_local(message_history)
        else:
            if "claude" in self.model_name:
                return self._call_claude(message_history)
            elif "gemini" in self.model_name:
                return self._call_gemini(message_history)
            elif "gpt" in self.model_name:
                return self._call_gpt(message_history)
            elif 'qwen' in self.model_name:
                return self._call_gpt(message_history)
            elif "Qwen2-VL-7B-Instruct" in self.model_name:
                return self._call_qwen7b(message_history)
            elif "Qwen2.5-VL-7B-Instruct" in self.model_name:
                return self._call_qwen7b(message_history)
            elif "Qwen2-VL-72B-Instruct" in self.model_name:
                return self._call_qwen72b(message_history)
            elif "Qwen2.5-VL-72B-Instruct" in self.model_name:
                return self._call_qwen72b(message_history)
            elif "Llama-3.2-11B-Vision-Instruct" in self.model_name:
                return self._call_llama11b(message_history)
            elif "meta-llama/Llama-3.2-90B-Vision-Instruct" in self.model_name:
                return self._call_qwen72b(message_history)
            elif "90b-vision-instruct" in self.model_name:
                return self._call_llama90(message_history)
            elif "OpenGVLab/InternVL" in self.model_name:
                return self._call_intern38b(message_history)
            elif 'GPT4.1' in self.model_name:
                return self._call_gpt41(message_history)
            else:
                raise ValueError(f"Unsupported model name: {self.model_name}")

    def _call_local(self, message_history: list):
        if not self.language_only:
            message_history = convert_format_2qwen25vl(message_history)

        if not self.language_only:
            response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))


        # response = self.model.chat.completions.create(
        #     model=self.model_name,
        #     messages=message_history,
        #     response_format=response_format,
        #     temperature=0.4,
        #     max_tokens=max_completion_tokens
        # )

        # out = response.choices[0].message.content

        text = self.processor.apply_chat_template(message_history, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(message_history)

        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, temperature=0.001, max_new_tokens=max_completion_tokens)
        # generated_ids = self.model.generate(**inputs, temperature=0.4, max_new_tokens=max_completion_tokens)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        out = output[0]

        out = fix_json(out)
        return out
    
    def _call_claude(self, message_history: list):

        if not self.language_only:
            message_history = convert_format_2claude(message_history)

        response = self.model.messages.create(
            model=self.model_name,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            messages=message_history
        )

        return response.content[0].text 

    def _call_gemini(self, message_history: list):

        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        response = self.model.beta.chat.completions.parse(
            model=self.model_name, 
            messages=message_history,
            response_format= ActionPlan_lang if self.language_only else ActionPlan,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )
        tokens = response.usage.prompt_tokens

        return str(response.choices[0].message.parsed.model_dump_json())

    def _call_gpt(self, message_history: list):

        if not self.language_only:
            response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )
        out = response.choices[0].message.content

        return out
    
    def _call_gpt41(self, message_history: list):
        if not self.language_only:
            response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))

        response = self.model.chat.completions.create(
            # model="gpt-4.1",
            model="gpt-4o",
            messages=message_history,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )
        out = response.choices[0].message.content
        out = fix_json(out)
        return out
    
    def _call_qwen7b(self, message_history: list):

        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if not self.language_only:
            response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )

        out = response.choices[0].message.content
        return out
    
    def _call_llama90(self, message_history: list):
        response = self.model.chat.completions.create(
            model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
            messages=message_history,
            response_format={"type": "json_object", "schema": ActionPlan_1.model_json_schema()},
            temperature = temperature
        )
        out = response.choices[0].message.content
        return out
    
    def _call_llama11b(self, message_history):

        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if not self.language_only:
            response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )
        out = response.choices[0].message.content
        return out
    

    def _call_qwen72b(self, message_history):
        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if not self.language_only:
            response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))
        
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )

        # easy to meet json errors
        out = response.choices[0].message.content
        out = fix_json(out)
        return out
    
    def _call_intern38b(self, message_history):

        # if not self.language_only:
        #     message_history = convert_format_2gemini(message_history)

        # no use, lmdeploy use support json schema only if it is pytorch-backended
        if not self.language_only:
            response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            # response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens,
        )

        # easy to meet json errors
        out = response.choices[0].message.content
        out = fix_json(out)
        return out



if __name__ == "__main__":
    pass
