EBNAV_TEMPLATE = """
<image>
## You are a robot operating in a home. You can do various tasks and output a sequence of actions to accomplish a given task with images of your status. 
The input given to you is an first person view observation. 
Now the human instruction is: {user_instruction}.

## The available action id (0 ~ 7) and action names are:
action id 0: Move forward by 0.25, 
action id 1: Move backward by 0.25, 
action id 2: Move rightward by 0.25, 
action id 3: Move leftward by 0.25, 
action id 4: Rotate to the right by 90 degrees, 
action id 5: Rotate to the left by 90 degrees, 
action id 6: Tilt the camera upward by 30 degrees, 
action id 7: Tilt the camera downward by 30 degrees.

*** Strategy ***

1. Locate the Target Object Type: Clearly describe the spatial location of the target object 
from the observation image (i.e. in the front left side, a few steps from current standing point).

2. Navigate by *** Using Move forward and Move right/left as main strategy ***, since any point can be reached through a combination of those.
When planning for movement, reason based on target object's location and obstacles around you.

3. Focus on primary goal: Only address invalid action when it blocks you from moving closer in the direction to target object. In other words, do not overly focus on correcting invalid actions when direct movement towards target object can still bring you closer.

4. *** Use Rotation Sparingly ***, only when you lose track of the target object and it's not in your view. If so, plan nothing but ONE ROTATION at a step until that object appears in your view.
After the target object appears, start navigation and avoid using rotation until you lose sight of the target again.

5. *** Do not complete task too early until you can not move any closer to the object, i.e. try to be as close as possible.

6. ***  If an invalid action has occurred in the action history, please do not perform this action again unless you have already performed a rotation operation.

To achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy. Aim for one action per step. At last, output the action id from the available actions to execute.

{history}

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. The reasoning process should include your analysis of the current input image and the thought process behind the actions you take. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action id (just one Arabic numeral) for current step and present it within <action> </action> tags, e.g. <action>0</action>.
"""

EBNAV_TEMPLATE_VP = """
<image>
## You are a robot operating in a home. You can do various tasks and output a sequence of actions to accomplish a given task with images of your status. 
Your input is a concatenation of your top-down view and the first-person view image, with the top-down image on the left and the first-person view on the right.
The colored circle in the top-down view represents your current position, with the YELLOW side indicating your LEFT and the PURPLE side indicating your RIGHT. The GREEN arrow shows your current camera orientation. In the first person view and the overhead view, the red bounding box in both views highlights the object you need to navigate to.
The candidate actions are shown in the overhead view and the first person view images, and each action is represented by a blue arrow, with the corresponding action ID at the end of the arrow. 

Now the human instruction is: {user_instruction}.

## The available action id (0 ~ 7) and action names are:
action id 0: Move forward by 0.25, 
action id 1: Move backward by 0.25, 
action id 2: Move rightward by 0.25, 
action id 3: Move leftward by 0.25, 
action id 4: Rotate to the right by 90 degrees, 
action id 5: Rotate to the left by 90 degrees, 
action id 6: Tilt the camera upward by 30 degrees, 
action id 7: Tilt the camera downward by 30 degrees.

Among these actions, actions 0-3 will only be annotated in the top-down view, while actions 4-7 will only be annotated in the first-person view.

*** Strategy ***
To achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy. Aim for one action per step. At last, output the action id from the available actions to execute.
1. The action arrows on the images indicate the forward direction of the agent after performing the corresponding actions (actions 0-3), or the direction of the agent's view rotation (actions 4-7). You can use this information to help decide which action to take.

2. Strategy2: When determining the relative left-right position between the target and the agent, do not simply look at whether the target is on the left or right side of the top-down view. Instead, you need to take the agent’s orientation into account (for example, when the agent is facing downward in the image, objects that appear to be on the left side in the image are actually on the agent’s right side).

3. Strategy3: The red bounding box marks your navigation target. Please pay special attention to whether there is a corresponding red bounding box and a red nevigation arrow in your FIRST-PERSON VIEW. Avoid mistakenly judging an existing box as absent, or assuming a non-existent box is present.

4. Strategy4: There is a red navigation arrow in the top-down view and the first-person view that point from the agent to the target, you can use this arrow to assist the current navigation task. If the red navigation arrow is not visible in the first-person view, it means that the target is not visible in the first-person view.

5. Strategy5: When you choose an action, please pay attention to the relationship between the action arrows in the top-down view and the red navigation arrow. Your movement direction should help shorten the red navigation arrow and align the green arrow with the red arrow.

6. Strategy6: If an invalid action has occurred in the action history, please do not perform this action again unless you have already performed a rotation operation.

7. Strategy7: Based on the information in the images, determine which arrow's action will best help you reach the navigation target with a red bounding box (if you encounter any obstacles, prioritize bypassing the obstacles currently blocking your way), and choose the optimal action.

8. Strategy8: You should use rotation (action 4-5) or camera tilt (action 6-7) sparingly, only when you lose track of the target object and it's NOT IN YOUR VIEW. If so, plan nothing but ONE ROTATION OR TILT at a step until that object appears in your view. After the target object appears, start navigation and avoid using rotation until you lose sight of the target again.

{history}

Now it's your turn to take an action.
While ensuring that your thought process is complete, please describe it as concisely as possible. Your full response should not exceed 1024 tokens.
You should first reason step-by-step about the current situation. The reasoning process should include your analysis of the current input image and the thought process behind the actions you take. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action id (just one Arabic numeral) for current step and present it within <action> </action> tags, e.g. <action>0</action>.
"""


EBNAV_TEMPLATE_VP_JSON = """
<image>
## You are a robot operating in a home. You can do various tasks and output a sequence of actions to accomplish a given task with images of your status. 
Your input is a concatenation of your top-down view and the first-person view image, with the top-down image on the left and the first-person view on the right.
The colored circle in the top-down view represents your current position, with the YELLOW side indicating your LEFT and the PURPLE side indicating your RIGHT. The GREEN arrow shows your current camera orientation. In the first person view and the overhead view, the red bounding box in both views highlights the object you need to navigate to.
The candidate actions are shown in the overhead view and the first person view images, and each action is represented by a blue arrow, with the corresponding action ID at the end of the arrow. 

Now the human instruction is: {user_instruction}.

## The available action id (0 ~ 7) and action names are:
action id 0: Move forward by 0.25, 
action id 1: Move backward by 0.25, 
action id 2: Move rightward by 0.25, 
action id 3: Move leftward by 0.25, 
action id 4: Rotate to the right by 90 degrees, 
action id 5: Rotate to the left by 90 degrees, 
action id 6: Tilt the camera upward by 30 degrees, 
action id 7: Tilt the camera downward by 30 degrees.

Among these actions, actions 0-3 will only be annotated in the top-down view, while actions 4-7 will only be annotated in the first-person view.

{history}

Now your task is to determine which arrow's action will best help you reach the navigation target with a red bounding box (if you encounter any obstacles, prioritize bypassing the obstacles currently blocking your way), and choose the optimal action id.

*** Strategy ***
To achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy. Aim for one action per step. At last, output the action id from the available actions to execute.
1. The action arrows on the images indicate the forward direction of the agent after performing the corresponding actions (actions 0-3), or the direction of the agent's view rotation (actions 4-7). You can use this information to help decide which action to take.

2. Strategy2: When determining the relative left-right position between the target and the agent, do not simply look at whether the target is on the left or right side of the top-down view. Instead, you need to take the agent’s orientation into account (for example, when the agent is facing downward in the image, objects that appear to be on the left side in the image are actually on the agent’s right side).

3. Strategy3: The red bounding box marks your navigation target. Please pay special attention to whether there is a corresponding red bounding box and a red nevigation arrow in your FIRST-PERSON VIEW. Avoid mistakenly judging an existing box as absent, or assuming a non-existent box is present.

4. Strategy4: There is a red navigation arrow in the top-down view and the first-person view that point from the agent to the target, you can use this arrow to assist the current navigation task. If the red navigation arrow is not visible in the first-person view, it means that the target is not visible in the first-person view.

5. Strategy5: When you choose an action, please pay attention to the relationship between the action arrows in the top-down view and the red navigation arrow. Your movement direction should help shorten the red navigation arrow and align the green arrow with the red arrow.

6. Strategy6: If an invalid action has occurred in the action history, please do not perform this action again unless you have already performed a rotation operation.

7. Strategy7: Based on the information in the images, determine which arrow's action will best help you reach the navigation target with a red bounding box (if you encounter any obstacles, prioritize bypassing the obstacles currently blocking your way), and choose the optimal action.

8. Strategy8: You should use rotation (action 4-5) or camera tilt (action 6-7) sparingly, only when you lose track of the target object and it's NOT IN YOUR VIEW. If so, plan nothing but ONE ROTATION OR TILT at a step until that object appears in your view. After the target object appears, start navigation and avoid using rotation until you lose sight of the target again.

You are supposed to output in JSON.
The output json format should be {{'visual_state_description': str, 'reasoning_and_reflection': str, 'language_plan': str, 'executable_plan': List[{{'action_id': int, 'action_name': str}}]}}
The fields in above JSON follows the purpose below:
1. visual_state_description is for description of current state of the first-person view and the top-down view image, 
2. reasoning_and_reflection is for summarizing the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task, and providing reasoning process of why you choose the current action, 
3. language_plan is for describing the best action you choose from all the action arrows, which is started by the step number and the action name, 
4. executable_plan is the best action you choose that having an action ID and a name.
5. keep your plan efficient and concise.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.
"""