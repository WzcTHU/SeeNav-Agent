import copy
import json
import re
import numpy as np
import os
from envs.utils.EBNav_env_utils import logger
from planner.utils.EBNav_planner_utils import image_to_data_url, truncate_message_prompts, template, template_lang, template_one_action_each_step, concat_images, strategy
from planner.models.remote_model import RemoteModel
from envs.examples.ebnav_icl_examples.ebnav_visual_icl import create_example_json_list
from PIL import Image
template = template
template_lang = template_lang
template_one_action_each_step = template_one_action_each_step
strategy = strategy
MESSAGE_WINDOW_LEN = 5

class EBNavigationPlanner():
    def __init__(self, model_name = '', model_type = 'remote', actions = [], system_prompt = '', examples = [], n_shot=1, obs_key='auto', chat_history=False, language_only=False, multiview = False, multistep = False, visual_icl = False, tp=1, truncate=False, config=None, kwargs={}):
        self.model_name = model_name
        self.model_type = model_type
        # self.obs_key = obs_key
        self.system_prompt = system_prompt
        self.n_shot = n_shot
        self.chat_history = chat_history # whether to includ all the chat history for prompting
        self.truncate = truncate # whether to truncate message history when chat_history is True
        self.set_actions(actions)
        self.planner_steps = 0
        self.output_json_error = 0

        self.kwargs = kwargs
        self.action_key = kwargs.pop('action_key', 'action_id')

        self.multiview = multiview
        self.multistep = multistep
        self.visual_icl = visual_icl
        self.config = config

        if not self.visual_icl:
            if examples != []:
                self.examples = examples[:n_shot]
            else:
                self.examples = []
            self.language_only = language_only
        else:
            self.examples = []
            self.language_only = False
            if language_only:
                self.icl_text_only = True
            else:
                self.icl_text_only = False

        if self.config['image_concat']:
            self.first_prompt = f'''To achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy {'and Examples provided' if self.examples != [] else ''} \
\nAim for about 1-2 actions in this step. !!!Notice: you cannot assess the situation until the whole plan in this planning step is finished executed, so plan accordingly.\
\nAt last, output the action id(s) (0 ~ {len(self.actions)-1}) from the available actions to execute. 

Your input is a concatenation of your top-down view and the first-person view image, with the top-down image on the left and the first-person view on the right.\
{'The colored circle in the top-down view represents your current position, with the YELLOW side indicating your LEFT and the PURPLE side indicating your RIGHT. The GREEN arrow shows your current camera orientation. ' if self.config['agent_mark'] else ''}{'In the first person view and the overhead view, the red bounding box in both views highlights the object you need to navigate to.' if self.config['fv_bbox'] and self.config['bev_bbox'] else ''}\
Plan accordingly based on the visual observation. {'Notice that you can consider using information from the top-down view to adjust your actions when you lose sight of the target object from the first-person perspective or get stuck by obstacles.' if self.multiview else ''}

You are supposed to output in JSON.{template_lang if self.language_only else template}'''
            
            self.following_prompt = f'''To achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy {'and Examples provided' if self.examples != [] else ''} \
\nAim for about 5-6 actions in this step to be closer to the target object. !!!Notice: you cannot assess the situation until the whole plan in this planning step is finished executed, so plan accordingly.\
\nAt last, output the action id(s) (0 ~ {len(self.actions)-1}) from the available actions to execute. 

Your input is a concatenation of your top-down view and the first-person view image, with the top-down image on the left and the first-person view on the right.\
{'The colored circle in the top-down view represents your current position, with the YELLOW side indicating your LEFT and the PURPLE side indicating your RIGHT. The GREEN arrow shows your current camera orientation. ' if self.config['agent_mark'] else ''}{'In the first person view and the overhead view, the red bounding box in both views highlights the object you need to navigate to.' if self.config['fv_bbox'] and self.config['bev_bbox'] else ''}\
Plan accordingly based on the visual observation. {'Notice that you can consider using information from the top-down view to adjust your actions when you lose sight of the target object from the first-person perspective or get stuck by obstacles.' if self.multiview else ''}

You are supposed to output in JSON.{template_lang if self.language_only else template}'''

        else:
            self.first_prompt = f'''To achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy {'and Examples provided' if self.examples != [] else ''} \
\nAim for about 1-2 actions in this step. !!!Notice: you cannot assess the situation until the whole plan in this planning step is finished executed, so plan accordingly.\
\nAt last, output the action id(s) (0 ~ {len(self.actions)-1}) from the available actions to execute. 

The input given to you is {'an first person view observation (the first image)' if not self.multistep else 'latest 3 steps of the first person view observations'} {'and a overhead view of the house (the second image) where the green triangle shows your location, and the sharpest vertex of the triangle in the overhead view shows your current camera orientation, matching the direction of your first-person view.' if self.multiview else ''} \
In the first person view {'and the overhead view' if self.multiview else ''}, the red bounding box {'in both views ' if self.multiview else ''}highlights the object you need to navigate to. \
Plan accordingly based on the visual observation. {'Notice that you can consider using information from the top-down view to adjust your actions when you lose sight of the target object from the first-person perspective or get stuck by obstacles.' if self.multiview else ''}

You are supposed to output in JSON.{template_lang if self.language_only else template}'''

            self.following_prompt = f'''To achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy {'and Examples provided' if self.examples != [] else ''} \
\nAim for about 5-6 actions in this step to be closer to the target object. !!!Notice: you cannot assess the situation until the whole plan in this planning step is finished executed, so plan accordingly.\
\nAt last, output the action id(s) (0 ~ {len(self.actions)-1}) from the available actions to execute. 

The input given to you is {'an first person view observation (the first image)' if not self.multistep else 'latest 3 steps of the first person view observations'} {'and a overhead view of the house (the second image) where the green triangle shows your location, and the sharpest vertex of the triangle in the overhead view shows your current camera orientation, matching the direction of your first-person view.' if self.multiview else ''} \
In the first person view {'and the overhead view' if self.multiview else ''}, the red bounding box {'in both views ' if self.multiview else ''}highlights the object you need to navigate to. \
Plan accordingly based on the visual observation. {'Notice that you can consider using information from the top-down view to adjust your actions when you lose sight of the target object from the first-person perspective or get stuck by obstacles.' if self.multiview else ''}

You are supposed to output in JSON.{template_lang if self.language_only else template}'''
        
        if self.config['image_concat']:
            self.one_action_each_step_prompt = f'''To achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy {'and Examples provided' if self.examples != [] else ''} \
\nYou should analyse the action space to find out what action might help you to be closer to the target object or bypass obstacles. !!!Notice: you cannot assess the situation until the whole plan in this planning step is finished executed, so plan accordingly\
\nAt last, output the action id(s) (0 ~ {len(self.actions)-1}) from the available actions to execute. !!!Notice: You are required to strictly output {str(self.config['topK_action'])} DIFFERENT candidate actions in this step, repeated actions are not allowed.

Your input is a concatenation of your top-down view and your first-person view image, with the top-down view on the left and the first-person image on the right. The GREEN TRIANGLE in the top-down view shows your location, and the sharpest vertex of the triangle in the top-down view shows your current camera orientation, matching the direction of your first-person view. \
In the first person view {'and the overhead view' if self.multiview else ''}, the red bounding box {'in both views ' if self.multiview else ''}highlights the object you need to navigate to. \
Plan accordingly based on the visual observation. {'Notice that you can consider using information from the top-down view to adjust your actions when you lose sight of the target object from the first-person perspective or get stuck by obstacles.' if self.multiview else ''}

You are supposed to output in JSON.{template_one_action_each_step}'''

        else:
            self.one_action_each_step_prompt = f'''To achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy {'and Examples provided' if self.examples != [] else ''} \
\nAim for output {str(self.config['topK_action'])} different candidate actions in this step that might help to be closer to the target object or bypass obstacles. !!!Notice: you cannot assess the situation until the whole plan in this planning step is finished executed, so plan accordingly.\
\nAt last, output the action id(s) (0 ~ {len(self.actions)-1}) from the available actions to execute.

The input given to you is {'an first person view observation (the first image)' if not self.multistep else 'latest 3 steps of the first person view observations'} {'and a overhead view of the house (the second image) where the green triangle shows your location, and the sharpest vertex of the triangle in the overhead view shows your current camera orientation, matching the direction of your first-person view.' if self.multiview else ''} \
In the first person view {'and the overhead view' if self.multiview else ''}, the red bounding box {'in both views ' if self.multiview else ''}highlights the object you need to navigate to. \
Plan accordingly based on the visual observation. {'Notice that you can consider using information from the top-down view to adjust your actions when you lose sight of the target object from the first-person perspective or get stuck by obstacles.' if self.multiview else ''}

You are supposed to output in JSON.{template_one_action_each_step}'''

        self.model = RemoteModel(model_name=model_name, model_type=model_type, language_only=language_only, tp=tp, local_model_path=self.config['local_model_path'])

        self.now_input = None
        self.now_output = None


    def set_actions(self, actions):
        self.actions = actions
        self.available_action_str = self.get_availabel_action_prompt(actions)

    def get_availabel_action_prompt(self, available_actions):
        available_action_str = ''
        for i in range(len(available_actions)):
            available_action_str += '\naction id ' + str(i) + ': ' + str(available_actions[i]) 
            if i < len(available_actions) - 1:
                available_action_str += ', '
        return available_action_str


    def process_prompt(self, user_instruction, prev_act_feedback=[]):

        user_instruction = user_instruction.rstrip('.')

        if len(prev_act_feedback) == 0:
            if self.n_shot >= 1:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example {i}: \n {x}' for i,x in enumerate(self.examples)])) 
            else:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')

            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'

            if self.config['action_project']:
                prompt += self.one_action_each_step_prompt
            else:
                prompt += self.first_prompt
     
        elif self.chat_history:

            # This is to support the sliding window feature
            if self.n_shot >= 1:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example  {i}: \n {x}' for i,x in enumerate(self.examples)])) 
            else:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')

            prompt += f'\n\n## The human instruction is: {user_instruction}.'

            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                prompt += '\n Step {}, action id {}, {}, env feedback: {}'.format(i, action_feedback[0], self.actions[action_feedback[0]], action_feedback[1])

            if self.config['action_project']:
                prompt += f"\n\n{self.one_action_each_step_prompt}"
            else:
                prompt += f"\n\n{self.following_prompt}"

        else:
            if self.n_shot >= 1:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example  {i}: \n {x}' for i,x in enumerate(self.examples)])) 
            else:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')

            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'

            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                prompt += '\n Step {}, action id {}, {}, env feedback: {}'.format(i, action_feedback[0], self.actions[action_feedback[0]], action_feedback[1])
            
            if self.config['action_project']:
                prompt += f"\n\n{self.one_action_each_step_prompt}"
            else:
                prompt += f"\n\n{self.following_prompt}"

        return prompt
    

    def get_message(self, observation, prompt, messages=[]):
        if self.language_only:
            current_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}],
            }
        elif self.multiview:
            if self.config['image_concat']:
                # img = concat_images(observation['fv_rgb'], observation['bev_rgb'])
                img = concat_images(observation['bev_rgb'], observation['fv_rgb'])

                data_url = image_to_data_url(image=img)

                current_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                            }
                        }, 
                        {"type": "text", "text": prompt}],
                }
            else:
                data_url1 = image_to_data_url(image=observation['fv_rgb'])
                data_url2 = image_to_data_url(image=observation['bev_rgb'])
                current_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url1,
                            }
                        }, 
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url2,
                            }
                        },
                        {"type": "text", "text": prompt}],
                }
        else:
            data_url = image_to_data_url(image=observation['fv_rgb'])
            current_message = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        }
                    }, 
                    {"type": "text", "text": prompt}],
            }

        messages = messages + [current_message]

        return messages[-MESSAGE_WINDOW_LEN:]


    def reset(self):
        # at the beginning of the episode
        self.episode_messages = []
        self.episode_act_feedback = []
        self.planner_steps = 0
        self.output_json_error = 0

    def language_to_action(self, output_text):
        pattern = r'\*\*\d+\*\*'
        match = re.search(pattern, output_text)
        if match:
            action = int(match.group().strip('*'))
        else:
            print('random action')
            action = np.random.randint(len(self.actions))
        return action
    
    def json_to_action(self, output_text, json_key='executable_plan'):
        valid = True
        try:
            json_object = json.loads(output_text)
            action = [x[self.action_key] for x in json_object[json_key]]
            if not len(action):
                print('empty plan, move forward instead')
                # action = [np.random.randint(len(self.actions))]
                action = [0]
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
            print('random action')
            self.output_json_error += 1
            action = [np.random.randint(len(self.actions))]
            valid = False
        except Exception as e:
            # Catch-all for any other unexpected errors not handled specifically
            print("An unexpected error occurred:", e)
            print('Using random action due to an unexpected error')
            action = [np.random.randint(len(self.actions))]
            valid = False
        return action, valid

        
    def act_custom(self, prompt, obs):
        assert type(obs) == str # input image path
        out = self.model.respond(prompt, obs)
        out = out.replace("'",'"')
        out = out.replace('\"s ', "\'s ")
        out = out.replace('```json', '').replace('```', '')
        logger.debug(f"Model Output:\n{out}\n")
        self.planner_steps += 1
        action, valid = self.json_to_action(out)
        if valid:
            return action, out
        else:
            out = '''{"visual_state_description":"invalid json, random action", "reasoning_and_reflection":"invalid json, random action",
                   "language_plan":"invalid json, random action"}'''
            return action, out

    def save_in_out_pair(self, save_path):
        to_save = {
            'input': self.now_input,
            'output': self.now_output
        }
        with open(save_path, 'w') as f:
            json.dump(to_save, f, ensure_ascii=False, indent=1)

    def act(self, observation, user_instruction, projector=None):
        if self.config['planning']:
            if self.config['align'] or self.config['agent_mark']:
                fv, bev = projector.draw_action(action_ids=None, obs=observation, agent_mark=self.config['agent_mark'], align=self.config['align'])
                observation['fv_rgb'] = fv
                observation['bev_rgb'] = bev
                if self.config['save_image']:
                    img = concat_images(observation['bev_rgb'], observation['fv_rgb'])
                    image_path = os.path.join(self.config['log_path'], 'episode_{}_step_{}_action.png'.format(projector.episode_idx, projector.planner_steps))
                    img.save(image_path)

            prompt = self.process_prompt(user_instruction, prev_act_feedback=self.episode_act_feedback)
            if self.model_type == 'custom':
                return self.act_custom(prompt, observation)

            if len(self.episode_messages) == 0:
                self.episode_messages = self.get_message(observation, prompt)
            else:
                if self.chat_history:
                    self.episode_messages = self.get_message(observation, prompt, self.episode_messages)
                else:
                    self.episode_messages = self.get_message(observation, prompt)
            
            # Apply truncation if chat_history and truncate are both True
            messages_to_send = self.episode_messages
            if self.chat_history and self.truncate:
                messages_to_send = truncate_message_prompts(self.episode_messages)
            
            for entry in messages_to_send:
                for content_item in entry["content"]:
                    if content_item["type"] == "text":
                        text_content = content_item["text"]
                        # logger.debug(f"Model Input:\n{text_content}\n")

            try:
                out = self.model.respond(messages_to_send)
                # 再说self相关属性中保存messages_to_send和out，方便外部存储，支撑后续SFT
                self.now_input = copy.deepcopy(messages_to_send)
                self.now_output = copy.deepcopy(out)
            
            except Exception as e:
                if 'qwen' in self.model_name:
                    return -2,'''{"visual_state_description":"qwen model generate empty action due to inappropriate content check", "reasoning_and_reflection":"invalid json, random action",
                    "language_plan":"invalid json, random action"}'''

            if self.chat_history:
                self.episode_messages.append(
                    {
                    "role": "assistant",
                    "content": [{"type": "text", "text": out}],
                    }
                )    
            logger.debug(f"Model Output:\n{out}\n")

            print('############################### planning阶段 ###############################')
            print('模型输出:', out)
            action, valid = self.json_to_action(out)
            print('解析后动作输出:', action)

        # 动作投影交互
        if self.config['action_project']:
            action = [0, 1, 2, 3, 4, 5, 6, 7]
            # img = concat_images(observation['fv_rgb'], observation['bev_rgb'])
            # image_path = os.path.join(self.config['log_path'], 'episode_{}_step_{}_action.png'.format(projector.episode_idx, projector.planner_steps))
            # img.save(image_path)
            action, out, valid = self.action_with_projection(observation, user_instruction, action, projector)

        self.planner_steps += 1
        if valid:
            return action, out
        else:
            logger.debug(f"Action Invalid Info:\n{action}\n{valid}\n")
            out = '''{"visual_state_description":"invalid json, random action", "reasoning_and_reflection":"invalid json, random action",
                   "language_plan":"invalid json, random action"}'''
            return action, out
        
    def construct_sft_sample(self, prompt: str, out: str, image_path: str, image: Image = None, image_url: str = None):
    # {
    #     "conversations": [
    #         {
    #             "from": "human",
    #             "value": "<image>\n<image>\nYou are an indoor navigation robot. You can currently observe your first-person view (FV) image (the first image) and a bird’s-eye view (BEV) image (the second image) of the indoor environment. In the BEV, your location is indicated by a green triangle; the sharpest vertex of the triangle shows your current camera orientation, matching the direction of your first-person view. Your first-person field of view is limited, approximately 100 degrees. Your task is to find the target object: 'Plate' in both the FV image and the BEV image, and provide the bounding box coordinates of the target in both images in JSON format with a template as: {'FV': [x1, y1, x2, y2], 'BEV': [x1', y1', x2', y2']}, where each bounding box consists of two coordinate pairs representing the top-left corner [x1, y1] and the bottom-right corner [x2, y2] of the box in each view.When detecting the target, please use information from your first-person view as well as your current position and orientation to assist with target localization in the BEV. If the target is not visible in your first-person view, the output JSON should be like: {'FV': null, 'BEV': [x1', y1', x2', y2']}. English and Chinese characters are not allowed in the output bounding box coordinates, only Arabic numerals are permitted."
    #         },
    #         {
    #             "from": "gpt",
    #             "value": "{'FV': [349, 304, 369, 308], 'BEV': [50, 141, 72, 162]}"
    #         }
    #     ],
    #     "images": [
    #         "/mnt/kaiwu-user-jensencwang/research/data/DualViewGrounding/dataset-0904/train/figs/FloorPlan11_Plate_20250904_222753_step_9[fv].png",
    #         "/mnt/kaiwu-user-jensencwang/research/data/DualViewGrounding/dataset-0904/train/figs/FloorPlan11_Plate_20250904_222753_step_9[bev-agent].png"
    #     ]
    # }
        self.sft_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{prompt}"
                },
                {
                    "from": "gpt",
                    "value": f"{out}"
                }
            ],
            "image_path": image_path,
            "image_url": image_url,
            "image": image
        }
        
    def action_with_projection(self, observation, user_instruction, candidate_action, projector=None):
        fv, bev = projector.draw_action(action_ids=list(set(candidate_action)), obs=observation, agent_mark=self.config['agent_mark'], align=self.config['align'])

        template = '''You are supposed to output in JSON.
The output json format should be {'visual_state_description':str, 'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':List[{'action_id':int, 'action_name':str}]}
The fields in above JSON follows the purpose below:
1. visual_state_description is for description of current state of the first-person view and the top-down view image, 
2. reasoning_and_reflection is for summarizing the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task, and providing reasoning process of why you choose the current action, 
3. language_plan is for describing the best action you choose from all the action arrows, which is started by the step number and the action name, 
4. executable_plan is the best action you choose that having an action ID and a name.
5. keep your plan efficient and concise.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.
'''

        history = '''\n\n The action history:'''
        num_total = len(self.episode_act_feedback)
        HISTORY_WINDOW = 5
        # 如果不足5条，slice到全部即可
        for i, action_feedback in enumerate(self.episode_act_feedback[-HISTORY_WINDOW:]):
            step_num = num_total - len(self.episode_act_feedback[-HISTORY_WINDOW:]) + i
            history += '\n Step {}, action id {}, {}, env feedback: {}'.format(
                step_num,
                action_feedback[0],
                self.actions[action_feedback[0]],
                action_feedback[1]
            )
        
        print(history)

        if self.config['image_concat']:
            prompt = f'''You are a robot operating in a home. Now the human instruction is: {user_instruction}. Your input is a concatenation of your top-down view and the first-person view image, with the top-down image on the left and the first-person view on the right.\
{'The colored circle in the top-down view represents your current position, with the YELLOW side indicating your LEFT and the PURPLE side indicating your RIGHT. The GREEN arrow shows your current camera orientation. ' if self.config['agent_mark'] else ''}{'In the first person view and the overhead view, the red bounding box in both views highlights the object you need to navigate to.' if self.config['fv_bbox'] and self.config['bev_bbox'] else ''}\
The candidate actions are shown in the overhead view and the first person view images, and each action is represented by a blue arrow, with the corresponding action ID at the end of the arrow. 

The meaning of each action ID corresponds to the following actions:
action id 0: Move forward by 0.25, 
action id 1: Move backward by 0.25, 
action id 2: Move rightward by 0.25, 
action id 3: Move leftward by 0.25, 
action id 4: Rotate to the right by 90 degrees., 
action id 5: Rotate to the left by 90 degrees., 
action id 6: Tilt the camera upward by 30 degrees., 
action id 7: Tilt the camera downward by 30 degrees.

Among these actions, actions 0-3 will only be annotated in the top-down view, while actions 4-7 will only be annotated in the first-person view.

!!!Notice: You should use rotation (action 4-5) or camera tilt (action 6-7) sparingly, only when you lose track of the target object and it's NOT IN YOUR VIEW. If so, plan nothing but ONE ROTATION OR TILT at a step until that object appears in your view. After the target object appears, start navigation and avoid using rotation until you lose sight of the target again.
{history}
Now your task is to first identify the {str(self.config['topK_action'])} selectable action IDs from the action arrows in both the top view and the front view. Then, based on the information in the images, determine which arrow's action will best help you reach the navigation target{' with a red bounding box' if self.config['fv_bbox'] and self.config['bev_bbox'] else ''} (if you encounter any obstacles, prioritize bypassing the obstacles currently blocking your way), and choose the optimal action. !!!Notice: you can only select from the action IDs that are labeled in the images.
{strategy}
To achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of historical actions. 3. Use what you learn from the Strategies
{template}
'''
        else:
            prompt = f'''You are a robot operating in a home. Now the human instruction is: {user_instruction}. The input given to you is an first person view observation (the first image) and a overhead view of the house (the second image). \
{'The colored circle in the top-down view represents your current position, with the YELLOW side indicating your LEFT and the PURPLE side indicating your RIGHT. The GREEN arrow shows your current camera orientation. ' if self.config['agent_mark'] else ''}{'In the first person view and the overhead view, the red bounding box in both views highlights the object you need to navigate to.' if self.config['fv_bbox'] and self.config['bev_bbox'] else ''}\
The candidate actions are also shown in the first person view and the overhead view images, and each action is represented by a red arrow, with the corresponding action ID at the end of the arrow. 

The valid action id (0 ~ 7) and action names are: 
action id 0: Move forward by 0.25, 
action id 1: Move backward by 0.25, 
action id 2: Move rightward by 0.25, 
action id 3: Move leftward by 0.25, 
action id 4: Rotate to the right by 90 degrees., 
action id 5: Rotate to the left by 90 degrees., 
action id 6: Tilt the camera upward by 30 degrees., 
action id 7: Tilt the camera downward by 30 degrees.

Among these actions, actions 0-3 will only be annotated in the top-down view, while actions 4-7 will only be annotated in the first-person view.

!!!Notice: You should use rotation (action 4-5) or camera tilt (action 6-7) sparingly, only when you lose track of the target object and it's NOT IN YOUR VIEW. If so, plan nothing but ONE ROTATION OR TILT at a step until that object appears in your view. After the target object appears, start navigation and avoid using rotation until you lose sight of the target again.
{history}
Now your task is to first identify the {str(self.config['topK_action'])} selectable action IDs from the action arrows in both the top view and the front view. Then, based on the information in the images, determine which arrow's action will best help you reach the navigation target{' with a red bounding box' if self.config['fv_bbox'] and self.config['bev_bbox'] else ''} (if you encounter any obstacles, prioritize bypassing the obstacles currently blocking your way), and choose the optimal action. !!!Notice: you can only select from the action IDs that are labeled in the images.
{strategy}
To achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of historical actions. 3. Summarize how you learn from the Strategy {'and Examples provided' if self.config['visual_icl'] else ''}
{template}
'''

        if self.config['image_concat']:
            
            content = []
            content.append({"type": "text", "text": prompt})
            if self.config['visual_icl']:
                visual_example = create_example_json_list(True)
                content.extend(visual_example)
                content.append({"type": "text", "text": "Below is your current step observation, please starting planning to navigate to the target object by learning from the above-mentioned strategy and in-context learning examples. ### Output nothing else but a JSON string following the above mentioned format ###"})

            # img = concat_images(fv, bev)
            img = concat_images(bev, fv)
            data_url = image_to_data_url(image=img)
            content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        }
                    })
            
            message = {
                "role":"user",
                "content":content
            }

            # message = {
            #     "role": "user",
            #     "content": [
            #         {
            #             "type": "image_url",
            #             "image_url": {
            #                 "url": data_url,
            #             }
            #         }, 
            #         {"type": "text", "text": prompt}],
            # }

            messages = [message]
            if self.chat_history and self.truncate:
                message = truncate_message_prompts(message)
        else:
            data_url1 = image_to_data_url(image=fv)
            data_url2 = image_to_data_url(image=bev)
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url1,
                        }
                    }, 
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url2,
                        }
                    },
                    {"type": "text", "text": prompt}],
            }
            messages = [message]
        if self.config['save_image']:
            if self.config['image_concat']:
                image_path = os.path.join(self.config['log_path'], 'episode_{}_step_{}_action.png'.format(projector.episode_idx, projector.planner_steps))
                img.save(image_path)
            else:
                image_path1 = os.path.join(self.config['log_path'], 'episode_{}_step_{}_front-action.png'.format(projector.episode_idx, projector.planner_steps))
                image_path2 = os.path.join(self.config['log_path'], 'episode_{}_step_{}_top-action.png'.format(projector.episode_idx, projector.planner_steps))
                fv.save(image_path1)
                bev.save(image_path2)

        out = self.model.respond(messages)

        print('############################### 动作投影阶段 ###############################')
        print('模型输出:', out)
        action, valid = self.json_to_action(out)
        action = [action[0]]        # 限制只取第一个action
        print('解析后动作输出:', action)

        if self.config['record_sft_sample']:
            if not os.path.exists(os.path.join(self.config['sft_sample_path'], 'images')):
                os.makedirs(os.path.join(self.config['sft_sample_path'], 'images'))
            image_path = os.path.join(self.config['sft_sample_path'], 'images', 'episode_{}_step_{}_action.png'.format(projector.episode_idx, projector.planner_steps))
            self.construct_sft_sample(prompt=prompt, out=out, image_path=image_path, image=img, image_url=data_url)

        if valid:
            return action, out, valid
        else:
            logger.debug(f"Action Invalid Info:\n{action}\n{valid}\n")
            out = '''{"visual_state_description":"invalid json, random action", "reasoning_and_reflection":"invalid json, random action",
                   "language_plan":"invalid json, random action"}'''
            return action, out, valid



    def update_info(self, info):
        """Update episode feedback history."""
        self.episode_act_feedback.append([
            info['action_id'],
            info['env_feedback']
        ])

if __name__ == '__main__':
    out = '''{"visual_state_description": "The bread is located on the countertop near the sink, which is visible in the overhead view.", "reasoning_and_reflection": "The bread is clearly visible on the countertop near the sink in the overhead view. The first-person view confirms the presence of the bread in the same location. The bread is within reach, indicating no further navigation is needed to locate it.", "language_plan": "0. Move forward by 0.25\n1. Move rightward by 0.25", "executable_plan": "{{"action_id": 0, "action_name": "Move forward by 0.25"}, {"action_id": 1, "action_name": "Move rightward by 0.25"}}"}'''
    # action = json_to_action(out)
