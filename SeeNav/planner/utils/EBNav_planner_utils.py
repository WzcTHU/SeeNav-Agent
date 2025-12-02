from PIL import Image
from pydantic import BaseModel, Field
import io
import base64
import typing_extensions as typing
import re

eb_navigation_system_prompt = '''## You are a robot operating in a home. You can do various tasks and output a sequence of actions to accomplish a given task with images of your status.

## The available action id (0 ~ {}) and action names are: {}.

*** Strategy ***

1. Locate the Target Object Type: Clearly describe the spatial location of the target object 
from the observation image (i.e. in the front left side, a few steps from current standing point).

2. Navigate by *** Using Move forward and Move right/left as main strategy ***, since any point can be reached through a combination of those. \
When planning for movement, reason based on target object's location and obstacles around you. \

3. Focus on primary goal: Only address invalid action when it blocks you from moving closer in the direction to target object. In other words, \
do not overly focus on correcting invalid actions when direct movement towards target object can still bring you closer. \

4. *** Use Rotation Sparingly ***, only when you lose track of the target object and it's not in your view. If so, plan nothing but ONE ROTATION at a step until that object appears in your view. \
After the target object appears, start navigation and avoid using rotation until you lose sight of the target again.

5. *** Do not complete task too early until you can not move any closer to the object, i.e. try to be as close as possible.

6. ***  If an invalid action has occurred in the action history, please do not perform this action again unless you have already performed a rotation operation.
{}

----------

'''

# eb_navigation_system_prompt = '''## You are a robot operating in a home. You can do various tasks and output a sequence of actions to accomplish a given task with images of your status.

# ## The available action id (0 ~ {}) and action names are: {}.

# *** Strategy ***

# 1. Locate the Target Object Type: Clearly describe the spatial location of the target object 
# from the observation image (i.e. in the front left side, a few steps from current standing point).

# 2. Navigate by *** Using Move forward and Move right/left as main strategy ***, since any point can be reached through a combination of those. \
# When planning for movement, reason based on target object's location and obstacles around you. \

# 3. Focus on primary goal: Only address invalid action when it blocks you from moving closer in the direction to target object. In other words, \
# do not overly focus on correcting invalid actions when direct movement towards target object can still bring you closer. \

# 4. *** Use Rotation Sparingly ***, only when you lose track of the target object and it's not in your view. If so, plan nothing but ONE ROTATION at a step until that object appears in your view. \
# After the target object appears, start navigation and avoid using rotation until you lose sight of the target again.

# 5. *** Do not complete task too early until you can not move any closer to the object, i.e. try to be as close as possible.

# 6. *** The red bounding box marks your navigation target. Please pay special attention to whether there is a corresponding red bounding box and a red nevigation arrow in your FIRST-PERSON VIEW. Avoid mistakenly judging an existing box as absent, or assuming a non-existent box is present.

# 7. *** There is a red navigation arrow in the top-down view and the first-person view that point from the agent to the target, you can use this arrow to assist the current navigation task. If the red navigation arrow is not visible in the first-person view, it means that the target is not visible in the first-person view.

# 8. ***  If an invalid action has occurred in the action history, please do not perform this action again unless you have already performed a rotation operation.
# {}

# ----------

# '''


template_lang = '''\
The output json format should be {'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':List[{'action_id':int, 'action_name':str}...]}
The fields in above JSON follows the purpose below:
1. reasoning_and_reflection is for summarizing the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task, \
2. language_plan is for describing a list of actions to achieve the user instruction. Each action is started by the step number and the action name, \
3. executable_plan is a list of actions needed to achieve the user instruction, with each action having an action ID and a name.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!
'''

template = '''
The output json format should be {'visual_state_description':str, 'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':List[{'action_id':int, 'action_name':str}...]}
The fields in above JSON follows the purpose below:
1. visual_state_description is for description of current state from the visual image, 
2. reasoning_and_reflection is for summarizing the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task, 
3. language_plan is for describing a list of actions to achieve the user instruction. Each action is started by the step number and the action name, 
4. executable_plan is a list of actions needed to achieve the user instruction, with each action having an action ID and a name.
5. keep your plan efficient and concise.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.
'''

template_one_action_each_step = '''
The output json format should be {'visual_state_description':str, 'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':List[{'action_id':int, 'action_name':str}...]}
The fields in above JSON follows the purpose below:
1. visual_state_description is for description of current state from the visual image, 
2. reasoning_and_reflection is for summarizing the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task, 
3. language_plan is for describing a list of different candidate actions that might be able to help to achieve the user instruction according to the current observation. Each action is started by the step number and the action name, and there is no sequential relationship between these actions in terms of execution.
4. executable_plan is a list of different candidate actions that might be able to help to achieve the user instruction according to the current observation, with each action having an action ID and a name, and there is no sequential relationship between these actions in terms of execution.
5. keep your plan efficient and concise.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.
'''

# strategy = '''
# ## Strategy1: The action arrows on the images indicate the forward direction of the agent after performing the corresponding actions (actions 0-3), or the direction of the agent's view rotation (actions 4-7). You can use this information to help decide which action to take.
# ## Strategy2: When determining the relative left-right position between the target and the agent, do not simply look at whether the target is on the left or right side of the top-down view. Instead, you need to take the agent’s orientation into account (for example, when the agent is facing downward in the image, objects that appear to be on the left side in the image are actually on the agent’s right side).
# ## Strategy3: If an invalid action has occurred in the action history, please do not perform this action again unless you have already performed a rotation operation.
# '''

# strategy = '''
# ## Strategy1: The action arrows on the images indicate the forward direction of the agent after performing the corresponding actions (actions 0-3), or the direction of the agent's view rotation (actions 4-7). You can use this information to help decide which action to take.
# ## Strategy2: The red bounding box marks your navigation target. Please pay special attention to whether there is a corresponding red bounding box and a red nevigation arrow in your FIRST-PERSON VIEW. Avoid mistakenly judging an existing box as absent, or assuming a non-existent box is present.
# ## Strategy3: There is a red navigation arrow in the top-down view and the first-person view that point from the agent to the target, you can use this arrow to assist the current navigation task. If the red navigation arrow is not visible in the first-person view, it means that the target is not visible in the first-person view.
# ## Strategy4: When you choose an action, please pay attention to the relationship between the action arrows in the top-down view and the red navigation arrow. Your movement direction should help shorten the red navigation arrow.
# ## Strategy5: If an invalid action has occurred in the action history, please do not perform this action again unless you have already performed a rotation operation.
# '''

strategy = '''
## Strategy1: The action arrows on the images indicate the forward direction of the agent after performing the corresponding actions (actions 0-3), or the direction of the agent's view rotation (actions 4-7). You can use this information to help decide which action to take.
## Strategy2: When determining the relative left-right position between the target and the agent, do not simply look at whether the target is on the left or right side of the top-down view. Instead, you need to take the agent’s orientation into account (for example, when the agent is facing downward in the image, objects that appear to be on the left side in the image are actually on the agent’s right side).
## Strategy3: The red bounding box marks your navigation target. Please pay special attention to whether there is a corresponding red bounding box and a red nevigation arrow in your FIRST-PERSON VIEW. Avoid mistakenly judging an existing box as absent, or assuming a non-existent box is present.
## Strategy4: There is a red navigation arrow in the top-down view and the first-person view that point from the agent to the target, you can use this arrow to assist the current navigation task. If the red navigation arrow is not visible in the first-person view, it means that the target is not visible in the first-person view.
## Strategy5: When you choose an action, please pay attention to the relationship between the action arrows in the top-down view and the red navigation arrow. Your movement direction should help shorten the red navigation arrow and align the green arrow with the red arrow.
## Strategy6: If an invalid action has occurred in the action history, please do not perform this action again unless you have already performed a rotation operation.
'''

def image_to_data_url(image: Image.Image, format='PNG'):
    # 指定格式，常见为 PNG/JPEG
    mime_type = f"image/{format.lower()}"
    
    # 保存到缓存区
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    base64_encoded_data = base64.b64encode(buffer.read()).decode('utf-8')
    
    # 构造 data URL
    data_url = f"data:{mime_type};base64,{base64_encoded_data}"
    return data_url


def concat_images(image1: Image.Image, image2: Image.Image):
    width, height = image1.size  # 获取单张图片的宽高
    # 创建一张新图片，宽度是两张拼起来，高度保持不变
    new_image = Image.new('RGB', (width * 2, height))
    # 把第一张图片贴到新图的左边
    new_image.paste(image1, (0, 0))
    # 把第二张图片贴到新图的右边
    new_image.paste(image2, (width, 0))
    return new_image

def truncate_message_prompts(message_history: list):
    """
    Traverse the message list and truncate the part before "------------" in the text content of all messages except the last one
    
    Args:
        message_history: Message list, each message contains role and content
        
    Returns:
        list: Processed message list
    """
    if not message_history:
        return message_history
        
    # Create a deep copy to avoid modifying the original data
    processed_messages = []
    
    # Process all messages except the last one
    for i, message in enumerate(message_history):
        if i == len(message_history) - 1:
            # Keep the last message unchanged
            processed_messages.append(message)
        else:
            # Process current message
            processed_message = {
                "role": message.get("role", ""),
                "content": []
            }
            
            # Traverse content list
            for content_item in message.get("content", []):
                if content_item.get("type") == "text":
                    # Process text type content
                    text_content = content_item.get("text", "")
                    
                    # Look for "----------" separator
                    if "----------" in text_content:
                        # Truncate content before separator, keep content after separator
                        truncated_text = text_content.split("----------")[1]
                    else:
                        # If no separator found, keep original text
                        truncated_text = text_content
                        
                    processed_content_item = content_item.copy()
                    processed_content_item["text"] = truncated_text
                    processed_message["content"].append(processed_content_item)
                else:
                    # Directly copy non-text type content
                    processed_message["content"].append(content_item.copy())
            
            processed_messages.append(processed_message)
    
    return processed_messages

# def fix_json(json_str):
#     """
#     Locates the substring between the keys "reasoning_and_reflection" and "language_plan"
#     and escapes any inner double quotes that are not already escaped.
    
#     The regex uses a positive lookahead to stop matching when reaching the delimiter for the next key.
#     """
#     # first fix common errors
#     json_str = json_str.replace("'",'"')
#     json_str = json_str.replace('\"s ', "\'s ")
#     json_str = json_str.replace('\"re ', "\'re ")
#     json_str = json_str.replace('\"ll ', "\'ll ")
#     json_str = json_str.replace('\"t ', "\'t ")
#     json_str = json_str.replace('\"d ', "\'d ")
#     json_str = json_str.replace('\"m ', "\'m ")
#     json_str = json_str.replace('\"ve ', "\'ve ")
#     json_str = json_str.replace('```json', '').replace('```', '')

#     # Then fix some situations. Pattern explanation:
#     # 1. ("reasoning_and_reflection"\s*:\s*") matches the key and the opening quote.
#     # 2. (?P<value>.*?) lazily captures everything in a group named 'value'.
#     # 3. (?=",\s*"language_plan") is a positive lookahead that stops matching before the closing quote
#     #    that comes before the "language_plan" key.
#     pattern = r'("reasoning_and_reflection"\s*:\s*")(?P<value>.*?)(?=",\s*"language_plan")'
    
#     def replacer(match):
#         prefix = match.group(1)            # Contains the key and the opening quote.
#         value = match.group("value")         # The raw value that might contain unescaped quotes.
#         # Escape any double quote that is not already escaped.
#         fixed_value = re.sub(r'(?<!\\)"', r'\\"', value)
#         return prefix + fixed_value

#     # Use re.DOTALL so that newlines in the value are included.
#     fixed_json = re.sub(pattern, replacer, json_str, flags=re.DOTALL)
#     fixed_json = re.sub(r'("executable_plan"\s*:\s*)"( ?\[[\s\S]*?\])"', r'\1\2', fixed_json)
#     return fixed_json

def convert_format_2claude(messages):
    new_messages = []
    
    for message in messages:
        if message["role"] == "user":
            new_content = []
    
            for item in message["content"]:
                if item.get("type") == "image_url":
                    base64_data = item["image_url"]["url"][22:]
                    new_item = {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_data
                        }
                    }
                    new_content.append(new_item)
                else:
                    new_content.append(item)

            new_message = message.copy()
            new_message["content"] = new_content
            new_messages.append(new_message)

        else:
            new_messages.append(message)

    return new_messages

def convert_format_2gemini(messages):
    new_messages = []
    
    for message in messages:
        if message["role"] == "user":

            new_content = []
            for item in message["content"]:
                if item.get("type") == "image_url":
                    base64_data = item["image_url"]["url"][22:]
                    new_item = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_data}"
                        }
                    }
                    new_content.append(new_item)
                else:
                    new_content.append(item)

            new_message = message.copy()
            new_message["content"] = new_content
            new_messages.append(new_message)

        else:
            new_messages.append(message)
        
    return new_messages


def convert_format_2qwen25vl(messages):
    new_messages = []
    
    for message in messages:
        if message["role"] == "user":

            new_content = []
            for item in message["content"]:
                if item.get("type") == "image_url":
                    base64_data = item["image_url"]["url"][22:]
                    new_item = {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{base64_data}"
                    }
                    new_content.append(new_item)
                else:
                    new_content.append(item)

            new_message = message.copy()
            new_message["content"] = new_content
            new_messages.append(new_message)

        else:
            new_messages.append(message)
        
    return new_messages


class ExecutableAction_1(typing.TypedDict): 
    action_id: int = Field(
        description="The action ID to select from the available actions given by the prompt"
    )
    action_name: str = Field(
        description="The name of the action"
    )

class ActionPlan_1(BaseModel):
    visual_state_description: str = Field(
        description="Description of current state from the visual image"
    )
    reasoning_and_reflection: str = Field(
        description="summarize the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task"
    )
    language_plan: str = Field(
        description="The list of actions to achieve the user instruction. Each action is started by the step number and the action name"
    )
    executable_plan: list[ExecutableAction_1] = Field(
        description="A list of actions needed to achieve the user instruction, with each action having an action ID and a name."
    )

class ExecutableAction(typing.TypedDict): 
    action_id: int
    action_name: str

class ActionPlan(BaseModel):
    visual_state_description: str
    reasoning_and_reflection: str
    language_plan: str
    executable_plan: list[ExecutableAction]

class ExecutableAction_lang(typing.TypedDict): 
    action_id: int
    action_name: str

class ActionPlan_lang(BaseModel):
    reasoning_and_reflection: str
    language_plan: str
    executable_plan: list[ExecutableAction_lang]



def fix_json(json_str):
    """
    Locates the substring between the keys "reasoning_and_reflection" and "language_plan"
    and escapes any inner double quotes that are not already escaped.
    Works even when the value contains unescaped double quotes.
    """
    # first fix common errors
    json_str = json_str.replace("'",'"')
    json_str = json_str.replace('\"s ', "\'s ")
    json_str = json_str.replace('\"re ', "\'re ")
    json_str = json_str.replace('\"ll ', "\'ll ")
    json_str = json_str.replace('\"t ', "\'t ")
    json_str = json_str.replace('\"d ', "\'d ")
    json_str = json_str.replace('\"m ', "\'m ")
    json_str = json_str.replace('\"ve ', "\'ve ")
    json_str = json_str.replace('```json', '').replace('```', '')

    # 用文本方式锁定区间，然后替换其中未转义引号
    key_start = '"reasoning_and_reflection":"'
    key_end = ',"language_plan"'
    start_idx = json_str.find(key_start)
    end_idx = json_str.find(key_end, start_idx)
    if start_idx != -1 and end_idx != -1:
        value_start = start_idx + len(key_start)
        value_end = end_idx
        value = json_str[value_start:value_end]

        # 只转义未转义的双引号
        fixed_value = re.sub(r'(?<!\\)"', r'\\"', value)

        # ----------- 新增：修正结尾被多转义情况 -----------
        # 部分模型可能会输出 ...\\" 作为字符串的结束，应消掉最后一处 (结尾的 \" 改 ")
        if fixed_value.endswith('\\"'):
            fixed_value = fixed_value[:-2] + '"'
        # ----------------------------------------------

        # 拼回 json_str
        json_str = json_str[:value_start] + fixed_value + json_str[value_end:]

    # 若需要的话再修复 plan 字段，使其能被 json.loads 解析
    json_str = re.sub(r'("executable_plan"\s*:\s*)"( ?\[[\s\S]*?\])"', r'\1\2', json_str)
    return json_str

if __name__ == '__main__':
    s = '''{"visual_state_description":"In the top-down view, the robot is positioned near the top center of the room, facing left. The Pot, highlighted with a red bounding box, is located on the left side of the room, close to the wall. In the first-person view, the Pot is directly ahead and clearly visible, indicating that the robot is facing the correct direction to approach the Pot.","reasoning_and_reflection":"The previous two actions were both "Rotate to the left by 90 degrees", which successfully reoriented the robot so that the Pot is now directly in front of it. The task was not completed previously because the robot needed to rotate to face the Pot. Now, the Pot is visible and directly ahead, so the next step should be to move forward towards it. The optimal action is to move forward, as there are no visible obstacles between the robot and the Pot.","language_plan":"Step 2: Move forward by 0.25 to get closer to the Pot.","candidate_actions":[0,1,2,3,4,5,6,7],"executable_plan":[{"action_id":0,"action_name":"Move forward by 0.25"}]}'''
    print(s)
    out = fix_json(s)
    print(out)