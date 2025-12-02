import ai2thor.controller
import gym
import numpy as np
import time
from PIL import Image
import json
import os
import cv2
import sys
import math
import random
from ai2thor.platform import CloudRendering
from DualViewGrounding.utils.bbox_utils import get_scene_bounds, draw_agent_arrow, world_to_birdview, draw_target_box, birdview_draw_object
# from embodiedbench.main import logger
import copy


import logging
logging.basicConfig(level=logging.DEBUG)


SUCCESS_THRESHOLD = 1

ValidEvalSets = ['base', 'common_sense', 'complex_instruction', 'visual_appearance', 'long_horizon', 'SFT_base', 'ICL']

DISCRETE_SKILLSET = [
    "Move forward by 0.25",
    "Move backward by 0.25",
    "Move rightward by 0.25",
    "Move leftward by 0.25",
    "Rotate to the right by 90 degrees.",
    "Rotate to the left by 90 degrees.",
    "Tilt the camera upward by 30 degrees.",
    "Tilt the camera downward by 30 degrees.",
    # "Crouch to be lower",
    # "Stand to be taller",
    # "Complete the current task."
]

# "Move forward by 0.25 meter.",
# "Move backward by 0.25 meter.",
# "Move right by 0.25 meter.",
# "Move left by 0.25 meter.",


class EBNavigationEnv(gym.Env):
    def __init__(self, eval_set='base', exp_name='test_base', down_sample_ratio=1.0, fov = 100, multiview = False, boundingbox = False, multistep = False,  resolution = 500, selected_indexes =[], target_only=True, log_path=None, full_config=None):
        """
        A wrapper for AI2-THOR ManipulaTHOR environment.

        :param config: Dictionary containing initialization parameters for the controller.
        """
        self.full_config = full_config
        self.resolution = resolution
        self.config = {
            "agentMode": "default",
            "gridSize": 0.1,
            "visibilityDistance": 10,
            "renderDepthImage": True,
            "renderInstanceSegmentation": True,
            "width": self.resolution,
            "height": self.resolution,
            "fieldOfView": fov,
            "platform": CloudRendering,
            # "gpu_device": 0,
            "server_timeout": 100,
            "server_start_timeout": 100,
        }
        print('start initialize controller')
        self.env = ai2thor.controller.Controller(**self.config)
        print('controller initialized')
        # load dataset
        assert eval_set in ValidEvalSets
        self.down_sample_ratio = down_sample_ratio
        self.data_path = os.path.join(os.path.dirname(__file__), f"datasets/{eval_set}.json")
        self.dataset = self._load_dataset()

        if len(selected_indexes):
            self.dataset = [self.dataset[i] for i in selected_indexes]

        self.selected_indexes = selected_indexes
        self.target_only = target_only
        # Episode tracking
        self.number_of_episodes = len(self.dataset)
        self._reset = False
        self._current_episode_num = 0
        self._current_step = 0
        self._max_episode_steps = 20
        self._episode_start_time = 0
        self.is_holding = False
        self.episode_log = []
        self.episode_language_instruction = ""
        self.episode_data = None

        self._last_event = None

        self.standing = True

        # set action space
        self.language_skill_set = DISCRETE_SKILLSET
        self.action_space = gym.spaces.Discrete(len(self.language_skill_set))

        # set log and verbosity(0 for concise)
        self.feedback_verbosity = 0
        if not log_path:
            self.log_path = 'running/eb_nav/{}'.format(exp_name)
        else:
            self.log_path = log_path

        self.multiview = multiview
        self.boundingbox = boundingbox
        self.multistep = multistep
        self.img_paths = []

    def _load_dataset(self):
        with open(self.data_path) as f:
            dataset_split = json.load(f)
        dataset = dataset_split["tasks"]
        if 0 <= self.down_sample_ratio < 1:
            select_every = round(1 / self.down_sample_ratio)
            dataset = dataset[0:len(dataset):select_every]
        # bad_case_idx = [4,14,15,18,26,35,36,39,41,43,46,47,48,49,54,55,56,57,58]
        # dataset_bad_case = [dataset[i-1] for i in bad_case_idx]
        return dataset
    
    def __calculate_distance(self, pos1, pos2):
        return np.sqrt((pos1['x'] - pos2['x'])**2 + (pos1['z'] - pos2['z'])**2)
    
    def get_valid_pose(self, target_object_id):
        event = self.env.step(
            action="GetInteractablePoses",
            objectId=target_object_id,
            horizons=[0]
            # rotations=DEFAULT_ROTATIONS,
            # standings=[True]
        )

        if not event.metadata["actionReturn"]:
            return None
        
        # Get object position
        obj_metadata = next(obj for obj in self.env.last_event.metadata["objects"] 
                        if obj["objectId"] == target_object_id)

        obj_position = {
            'x': obj_metadata['position']['x'],
            'z': obj_metadata['position']['z']
        }

        poses = event.metadata["actionReturn"]
        random.shuffle(poses)
        for pose in poses:
            pos = {'x': pose['x'], 'z': pose['z']}
            dis = self.__calculate_distance(pos, obj_position)
            if dis >= 2.5 and dis <= 3:
                return pose
        return None
    
    def get_object_ids_by_type(self, metadata, object_type):
        objects = []
        for obj in metadata["objects"]:
            if obj["objectType"] == object_type:
                objects.append(obj["objectId"])
        return objects

    def random_reset(self):
        train_scene_target_pairs = None
        with open(os.path.join(os.path.dirname(__file__), 'datasets', 'sft_scene_obj_pair.json'), 'r') as f:
            train_scene_target_pairs = json.load(f)
        self.number_of_episodes = len(train_scene_target_pairs)
        while True:
            chosen_pair = random.choice(train_scene_target_pairs)
            scene, target_type = chosen_pair[0], chosen_pair[1]

            self.env.reset(scene=scene)

            # 执行一系列场景随机化操作

            self.env.step(action="RandomizeMaterials")
            self.env.step(
                action="RandomizeLighting",
                brightness=(0.5, 1.5),
                randomizeColor=True,
                hue=(0, 1),
                saturation=(0.5, 1),
                synchronized=False
            )
            self.env.step(action="InitialRandomSpawn", forceVisible=True)


            target_objects = self.get_object_ids_by_type(self.env.last_event.metadata, target_type)
            if target_objects == []:
                print(f"Warning: No {target_type} found in {scene}")
                continue
            else:
                print(f'Found valid scene {scene} and target {target_type}')
                target_object_id = target_objects[0]
                for obj in self.env.last_event.metadata["objects"]:
                    if obj["objectId"] == target_object_id:
                        self.target_position = obj["position"]
                        break
                # Get other objects to hide (all objects of same type except the target)
                # objects_to_hide = target_objects[1:] if len(target_objects) > 1 else []

                # Get valid initial pose
                pose = self.get_valid_pose(target_object_id)
                if not pose:
                    print(f"Warning: Could not find valid pose in {scene}")
                    continue
                break

        traj_data = {
            "targetObjectType": target_type,
            "targetObjectIds": target_object_id,
            "target_position": {
                "x": self.target_position["x"],
                "y": self.target_position["y"],
                "z": self.target_position["z"]
            },
            "agentPose": {
                "position": {
                    "x": pose["x"],
                    "y": pose["y"],
                    "z": pose["z"]
                },
                "rotation": pose["rotation"],
                "horizon": pose["horizon"]
            },
            "scene": scene,
            "object_to_hide": [],
            "instruction": f"navigate to the {target_type} in the room and be as close as possible to it"
        }

        self.episode_data = traj_data
        self.episode_language_instruction = traj_data["instruction"]
        if self.multiview:
            event = self.env.step(action="GetMapViewCameraProperties", raise_for_failure=True)
            pose = copy.deepcopy(event.metadata["actionReturn"])
            pose["orthographic"] = True

            # add the camera to the scene
            self.env.step(
                action="AddThirdPartyCamera",
                **pose,
                skyboxColor="white",
                raise_for_failure=True,
            )
        pose = traj_data["agentPose"]
        self.env.step(
            action="Teleport",
            position={
                "x": pose["position"]["x"],
                "y": pose["position"]["y"],
                "z": pose["position"]["z"]
            },
            rotation={
                "x": 0,
                "y": pose["rotation"],
                "z": 0
            },
            horizon=pose["horizon"],
            standing=True
        )

        # finish reset environment 
        # reset episode information
        self._current_episode_num += 1
        self._current_step = 0

        self.standing = True
        obs = {
            'fv_rgb': Image.fromarray(self.env.last_event.frame)
        }
        if self.multiview:
            obs['bev_rgb'] = Image.fromarray(self.env.last_event.third_party_camera_frames[-1])

        self._reset = True
        self.episode_log = []
        self._episode_start_time = time.time()

        self.img_paths = []

        return obs

    def reset(self, **kwargs):
        if self.full_config['random_reset']:
            return self.random_reset()

        """
        Reset the environment.

        :param scene: Optionally set the scene for reset.
        :return: The initial observation.
        """
        # self.save_episode_log()
        assert self._current_episode_num < self.number_of_episodes

        # start reset environment 
        traj_data = self.dataset[self._current_episode_num]
        self.episode_data = traj_data
        self.episode_language_instruction = traj_data["instruction"]

        scene_name = traj_data["scene"]
        # logger.info(f"Restoring scene {scene_name}...")
        self._last_event = self.env.reset(
            scene=scene_name
        )

        if self.multiview:
            event = self.env.step(action="GetMapViewCameraProperties", raise_for_failure=True)
            pose = copy.deepcopy(event.metadata["actionReturn"])
            pose["orthographic"] = True

            # add the camera to the scene
            self.env.step(
                action="AddThirdPartyCamera",
                **pose,
                skyboxColor="white",
                raise_for_failure=True,
            )

        pose = traj_data["agentPose"]
        self.env.step(
            action="Teleport",
            position={
                "x": pose["position"]["x"],
                "y": pose["position"]["y"],
                "z": pose["position"]["z"]
            },
            rotation={
                "x": 0,
                "y": pose["rotation"],
                "z": 0
            },
            horizon=pose["horizon"],
            standing=True
        )

        # finish reset environment 
        # reset episode information
        self._current_episode_num += 1
        self._current_step = 0

        self.standing = True
        obs = {
            'fv_rgb': Image.fromarray(self.env.last_event.frame)
        }
        if self.multiview:
            obs['bev_rgb'] = Image.fromarray(self.env.last_event.third_party_camera_frames[-1])

        self._reset = True
        self.episode_log = []
        self._episode_start_time = time.time()

        self.img_paths = []

        return obs
    
    def discrete_action_mapper(self, action_index):
        """
        Maps a discrete action index to the corresponding iTHOR environment action.

        Parameters:
            env: The AI2-THOR environment object.
            action_index: An integer representing the action index.

        Raises:
            ValueError: If the action index is invalid.
        """

        if action_index == 0:  # Move forward by 0.25 meter
            self._last_event = self.env.step(action="MoveAhead", moveMagnitude=0.25)
        elif action_index == 1:  # Move backward by 0.25 meter
            self._last_event = self.env.step(action="MoveBack", moveMagnitude=0.25)
        elif action_index == 2:  # Move right by 0.25 meter
            self._last_event = self.env.step(action="MoveRight", moveMagnitude=0.25)
        elif action_index == 3:  # Move left by 0.25 meter
            self._last_event = self.env.step(action="MoveLeft", moveMagnitude=0.25)
        elif action_index == 4:  # Rotate clockwise by 45 degrees
            self._last_event = self.env.step(action="RotateRight", degrees=90)
        elif action_index == 5:  # Rotate counterclockwise by 45 degrees
            self._last_event = self.env.step(action="RotateLeft", degrees=90)
        elif action_index == 6:  # Tilt the camera upward by 30 degrees
            self._last_event = self.env.step(action="LookUp", degrees=30)
        elif action_index == 7:  # Tilt the camera downward by 30 degrees
            self._last_event = self.env.step(action="LookDown", degrees=30)
        # elif action_index == 8:  # Crouch to be lower
        #     self._last_event = self.env.step(action="Crouch")
        #     self.standing = False
        # elif action_index == 9:  # Stand to be taller
        #     self._last_event = self.env.step(action="Stand")
        #     self.standing = True
        # elif action_index == 8:  # Complete the current task
        #     self._last_event = self.env.step(action="Done")
        else:
            print(f"Invalid action index: {action_index}")

    def measure_success(self):
        # success measurement
        agent_position = self.env.last_event.metadata["agent"]["position"]
        target_object_id = self.episode_data["targetObjectIds"]
        target_position = self.episode_data["target_position"]

        # for obj in self.env.last_event.metadata["objects"]:
        #     if obj["objectId"] == target_object_id:
        #         target_position = obj["position"]
        #         break

        dist = math.sqrt(
            (agent_position["x"] - target_position["x"])**2 +
            (agent_position["z"] - target_position["z"])**2
        )
        success = (dist <= SUCCESS_THRESHOLD)
        return float(success), dist

        

    def step(self, action: int, reasoning, i_flag):
        """
        Perform an action in the environment.

        :param action: The name of the action to perform.
        :param kwargs: Additional parameters for the action.
        :return: Event.
        """

        assert self._reset, 'Reset env before stepping'
        info = {}

        self._current_step += 1

        if self._current_step>=self._max_episode_steps:

            if type(action)!=int or action > 7 or action < 0:
                action = np.random.randint(8)

            self.discrete_action_mapper(action)
            reward, distance = self.measure_success()
            done = True
            info['action_description'] = self.language_skill_set[action]

        else:
            if type(action)!=int or action > 7 or action < 0:
                action = np.random.randint(8)

            self.discrete_action_mapper(action)
            reward, distance = self.measure_success()
            if reward>0:
                done = True
            else:
                done = False
            info['action_description'] = self.language_skill_set[action]

        #info['action_description'] = self.language_skill_set[action]

        obs = {'fv_rgb': Image.fromarray(self.env.last_event.frame)}
        if self.multiview:
            obs['bev_rgb'] = Image.fromarray(self.env.last_event.third_party_camera_frames[-1])
        reward, distance = self.measure_success()

        ## test calculate reward
        info['distance'] = distance
        info['env_feedback'] = self.get_env_feedback(self._last_event)
        info['reasoning'] = reasoning
        # info['reflection'] = reasoning['reasoning_and_reflection']
        # info['plan'] = reasoning['language_plan']
        info['instruction'] = self.episode_language_instruction
        info['env_step'] = self._current_step
        info['episode_elapsed_seconds'] = time.time() - self._episode_start_time
        info['task_success'] = reward
        info['last_action_success'] = self.env.last_event.metadata['lastActionSuccess']
        info['action_id'] = action
        # info['reasoning'] = reasoning

        self.episode_log.append(info)

        if i_flag == 1:
            self.save_episode_log_per_step(1)
        else:
            self.save_episode_log_per_step(0)
        
        self.episode_log = []

        return obs, reward, done, info
        
    def get_env_feedback(self, event):
        """
        To extract relevant information from the event to construct a feedback dictionary.

        :param event: self._last_event
        :return: A dictionary containing structured feedback.
        """
        if self.feedback_verbosity == 1:
            feedback = {
                "lastActionSuccess": event.metadata.get("lastActionSuccess", None),
                "errorMessage": event.metadata.get("errorMessage", None),
                "lastAction": event.metadata.get("lastAction", None),

                "agent": {
                    "position": event.metadata.get("agent", {}).get("position", {}),
                    "rotation": event.metadata.get("agent", {}).get("rotation", {}),
                    "is_standing": self.standing
                }
            }
        else:
            # Does not provide the specific reason why the action fails if so
            feedback = {
                "lastActionSuccess": event.metadata.get("lastActionSuccess", None),
                "lastAction": event.metadata.get("lastAction", None),
                "errorMessage": event.metadata.get("errorMessage", None),

                "agent": {
                    "is_standing": self.standing
                }
            }

        msg = ''
        if feedback["lastActionSuccess"]:
            msg += f"Last action {feedback['lastAction']} executed successfully."
        else:
            msg += f"Last action {feedback['lastAction']} is invalid. {feedback['errorMessage']}"
        return msg

    def seed(self, seed=None):
        self.env.random_initilize(seed)

    def draw_navigation_line_bev(self, image, agent_pos, target_pos_pt, bounds, img_width, img_height, line_width, color=(255,0,0)):
        '''绘制agent中心和target中心的连线'''
        # 兼容PIL和np格式
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # agent中心像素坐标
        agent_u, agent_v = world_to_birdview(agent_pos['x'], agent_pos['z'], bounds, img_width, img_height)
        target_u, target_v = target_pos_pt

        start_pt = (int(round(agent_u)), int(round(agent_v)))
        end_pt = (int(round(target_u)), int(round(target_v)))

        # 绘制连线
        cv2.line(
            image,
            start_pt,
            end_pt,
            color=color,
            thickness=line_width,
            lineType=cv2.LINE_AA
        )

        # 计算箭头长度（像素），可根据线宽自适应
        arrow_length = max(line_width * 4, 12)
        # 计算总线方向向量
        vec = np.array([target_u - agent_u, target_v - agent_v], dtype=np.float32)
        l = np.linalg.norm(vec)
        if l < 1e-5:
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # agent和target重合，直接返回

        direction = vec / l
        # 箭头起点: 目标点往回箭头长度的距离
        arrow_start = (
            int(round(target_u - direction[0]*arrow_length)),
            int(round(target_v - direction[1]*arrow_length))
        )
        # 在终点加箭头
        cv2.arrowedLine(
            image,
            arrow_start,
            end_pt,
            color=color,
            thickness=line_width,
            line_type=cv2.LINE_AA,
            tipLength=0.7  # 箭头尖占箭头线段长度的比例
        )

        # 转回PIL格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)
    
    def draw_navigation_line_fv(self, image, target_pos_pt, img_width, img_height, line_width, color=(255, 0, 0)):
        '''
        在主视图中绘制导航线，起点为图像下方中央，终点为传入的目标像素点
        参数:
            image: PIL.Image对象或numpy array
            target_pos_pt: (u, v)格式，目标在图像中的像素坐标(int/float均可)
            img_width, img_height: 图像宽高（int）
            line_width: 线宽
            color: BGR颜色元组  
        '''

        # 兼容PIL和np格式
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 图像下方中央点作为起点
        start_u = img_width // 2
        start_v = img_height - 1
        start_pt = (int(round(start_u)), int(round(start_v)))

        target_u, target_v = target_pos_pt
        end_pt = (int(round(target_u)), int(round(target_v)))

        # 绘制连线
        cv2.line(
            image,
            start_pt,
            end_pt,
            color=color,
            thickness=line_width,
            lineType=cv2.LINE_AA
        )

        # 箭头尺寸
        arrow_length = max(line_width * 4, 12)
        vec = np.array([target_u - start_u, target_v - start_v], dtype=np.float32)
        l = np.linalg.norm(vec)
        if l < 1e-5:
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 起终点重合

        direction = vec / l
        # 箭头起点，为目标点回退arrow_length像素
        arrow_start = (
            int(round(target_u - direction[0] * arrow_length)),
            int(round(target_v - direction[1] * arrow_length))
        )
        # 终点加箭头
        cv2.arrowedLine(
            image,
            arrow_start,
            end_pt,
            color=color,
            thickness=line_width,
            line_type=cv2.LINE_AA,
            tipLength=0.7
        )

        # 转回PIL格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)
    
    def check_target_visibility(self):
        if self.episode_data["targetObjectIds"] in self.env.last_event.instance_detections2D:
            return True
        else:
            return False

    def add_marks_on_obs(self, obs, fv_bbox=True, bev_bbox=True):
        new_obs = {'fv_rgb': obs['fv_rgb']}
        if fv_bbox:
            fv, fv_bbox_pt = draw_target_box(obs['fv_rgb'], self.env.last_event.instance_detections2D, self.episode_data["targetObjectIds"], color=(0, 0, 255), thickness=2)
            if self.full_config['nav_line']:
                if fv_bbox_pt:
                    target_pos_pt = (int((fv_bbox_pt[0] + fv_bbox_pt[2]) / 2), int((fv_bbox_pt[1] + fv_bbox_pt[3]) / 2))
                    fv = self.draw_navigation_line_fv(fv, target_pos_pt, self.config['width'], self.config['height'], line_width=2, color=(0,0,255))
            new_obs['fv_rgb'] = fv
        if 'bev_rgb' in obs.keys():
            bev = obs['bev_rgb']
            scene_bounds_raw = self.env.last_event.metadata['sceneBounds']
            scene_bounds = get_scene_bounds(scene_bounds_raw)
            agent_pos = self.env.last_event.metadata['agent']['position']
            agent_rot = self.env.last_event.metadata['agent']['rotation']
            # if agent_arrow:
            #     bev = draw_agent_arrow(agent_pos, agent_rot, scene_bounds, bev, img_width=self.config['width'], img_height=self.config['height'], color=(0,255,0))
            if bev_bbox:
                bbox_raw = []
                for obj in self.env.last_event.metadata['objects']:
                    if obj['objectId'] == self.episode_data["targetObjectIds"]:
                        bbox_raw = obj['axisAlignedBoundingBox']
                        break
                bev, bev_bbox_pt = birdview_draw_object(scene_bounds_raw, bbox_raw, img_width=500, img_height=500, color=(255,0,0), background=bev)
                if self.full_config['nav_line']:
                    target_pos_pt = (int((bev_bbox_pt[0] + bev_bbox_pt[2]) / 2), int((bev_bbox_pt[1] + bev_bbox_pt[3]) / 2))
                    bev = self.draw_navigation_line_bev(bev, agent_pos, target_pos_pt, scene_bounds, self.config['width'], self.config['height'], line_width=2, color=(0,0,255))

            new_obs['bev_rgb'] = bev
        return new_obs


    def save_image(self, obs):
        """Save current agent view as a PNG image."""
        episode_idx = self._current_episode_num if not len(self.selected_indexes) else self.selected_indexes[self._current_episode_num - 1] + 1

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        if 'fv_rgb' in obs.keys():
            image_path1 = os.path.join(self.log_path, 'episode_{}_step_{}_{}_front.png'.format(episode_idx, self._current_step, time_stamp))
            obs['fv_rgb'].save(image_path1)
        if 'bev_rgb' in obs.keys():
            image_path2 = os.path.join(self.log_path, 'episode_{}_step_{}_{}_top.png'.format(episode_idx, self._current_step, time_stamp))
            obs['bev_rgb'].save(image_path2)
        

    def save_episode_log_per_step(self, flag):

        episode_idx = self._current_episode_num if not len(self.selected_indexes) else self.selected_indexes[self._current_episode_num - 1] + 1

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        filename = 'episode_{}.json'.format(episode_idx)
        if len(self.episode_log):
            with open(os.path.join(self.log_path, filename), 'a') as f:
                if flag == 1:
                    f.write('\n\n')
                    for item in self.episode_log:
                        if 'object_states' in item:
                            item.pop('object_states')
                        try:
                            json.dump(item, f, ensure_ascii=False)
                        except:
                            import pdb;pdb.set_trace()
                        f.write('\n') 
                else:
                    for item in self.episode_log:
                        if 'object_states' in item:
                            item.pop('object_states')
                        try:
                            json.dump(item, f, ensure_ascii=False)
                        except:
                            import pdb;pdb.set_trace()
                        f.write('\n') 

    # def save_episode_log(self):
    #     if not os.path.exists(self.log_path):
    #         os.makedirs(self.log_path)
    #     time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    #     filename = 'episode_{}_step_{}_{}.json'.format(self._current_episode_num, self._current_step, time_stamp)
    #     if len(self.episode_log):
    #         with open(os.path.join(self.log_path, filename), 'w') as f:
    #             for item in self.episode_log:
    #                 if 'object_states' in item:
    #                     item.pop('object_states')
    #                 try:
    #                     json.dump(item, f, ensure_ascii=False)
    #                 except:
    #                     import pdb;pdb.set_trace()
    #                 f.write('\n') 

    def close(self):
        """Close the environment."""
        self.env.stop()


if __name__ == "__main__":
    env = EBNavigationEnv("base")
    print('done')
    print('reset...')
    env.reset()
    print('done')
    print([(i, name) for i, name in enumerate(env.language_skill_set)])
    for _ in range(30):
        # Select  action
        action = int(input('action id: ')) #env.action_space.sample()
        if action in env.language_skill_set:
            action = env.language_skill_set.index(action)
        else:
            action = int(action)
            if action < 0:
                break
        
        print(env.language_skill_set[action])
        
        # Execute action
        obs, reward, done, info = env.step(action, "", 1)
        print(reward, done, info)
        # Optional rendering and image saving
        env.save_image()
        if done:
            break
    env.close()
    env.close()