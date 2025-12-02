import os
import yaml
import ray
import random
import numpy as np
import math
import json
import time
import torch
import copy
from PIL import Image
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

from agent_system.environments.prompts.ebnav import *
from .utils.utils import *
from .utils.ActionMarker import ActionMarker

# train_scenes = ["FloorPlan11", "FloorPlan12", "FloorPlan13", "FloorPlan14", "FloorPlan15", "FloorPlan16", "FloorPlan17", "FloorPlan18", "FloorPlan19", "FloorPlan20", "FloorPlan21", "FloorPlan22",
#                 "FloorPlan23", "FloorPlan24", "FloorPlan25", "FloorPlan26", "FloorPlan27", "FloorPlan28", "FloorPlan29", "FloorPlan30", "FloorPlan211", "FloorPlan212", "FloorPlan213", "FloorPlan214",
#                 "FloorPlan215", "FloorPlan216", "FloorPlan217", "FloorPlan218", "FloorPlan219", "FloorPlan220", "FloorPlan221", "FloorPlan222", "FloorPlan223", "FloorPlan224", "FloorPlan225", "FloorPlan226",
#                 "FloorPlan227", "FloorPlan228", "FloorPlan229", "FloorPlan230", "FloorPlan311", "FloorPlan312", "FloorPlan313", "FloorPlan314", "FloorPlan315", "FloorPlan316", "FloorPlan317", "FloorPlan318",
#                 "FloorPlan319", "FloorPlan320", "FloorPlan321", "FloorPlan322", "FloorPlan323", "FloorPlan324", "FloorPlan325", "FloorPlan326", "FloorPlan327", "FloorPlan328", "FloorPlan329", "FloorPlan330"]

# valid_objs = ['Knife', 'Spatula', 'HousePlant', 'Shelf', 'Candle', 'SprayBottle', 'RemoteControl', 'Pot', 'AluminumFoil', 'Footstool', 'CellPhone', 'Newspaper', 'DeskLamp', 'WineBottle', 'Pan', 
#               'Book', 'SaltShaker', 'Desk', 'VacuumCleaner', 'Box', 'TVStand', 'DogBed', 'Sink', 'Stool', 'StoveBurner', 'AlarmClock', 'GarbageBag', 'Bowl', 'Pillow', 'DiningTable', 'SideTable', 
#               'Microwave', 'PepperShaker', 'Curtains', 'TeddyBear', 'Bread', 'Painting', 'Mug', 'Statue', 'Poster', 'Blinds', 'TissueBox', 'Bottle', 'Chair', 'PaperTowelRoll', 'GarbageCan', 
#               'BasketBall', 'ArmChair', 'SoapBottle', 'Cup', 'Dresser', 'FloorLamp', 'Kettle', 'Cabinet', 'Lettuce', 'LightSwitch', 'Tomato', 'Ladle', 'ButterKnife', 'CoffeeTable', 'Dumbbell', 
#               'BaseballBat', 'Mirror', 'TennisRacket', 'ShelvingUnit', 'LaundryHamper', 'Apple', 'Safe', 'Window', 'SinkBasin', 'Potato', 'DishSponge', 'Vase', 'Toaster', 'KeyChain', 'Laptop', 
#               'Television', 'Fridge', 'Bed', 'Drawer', 'CD', 'Cloth', 'Boots', 'Sofa', 'Faucet', 'CoffeeMachine', 'Plate', 'Watch', 'WateringCan']

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def compute_reward(info):
    success_reward = info['won']
    return success_reward

def compute_step_reward(info):
    step_reward = 0
    if (info['distance'] < info['last_distance'] - 0.01) or (info['visibility'] and not info['last_visibility']):
        step_reward = 1

    # '''不需要在这里添加invalid action的惩罚，因为在ray_trainer中已经有了apply_invalid_action_penalty的功能'''
    # action_format_penalty = 0
    # if info['action_id'] == -1:
    #     action_format_penalty = 1
    # reward = step_reward - action_format_penalty
    return step_reward

class EBNavBaseEnv(object):
    def __init__(self, config, is_train):
        self.env = None
        self.config = config
        self.seed = 42
        self.target_position = None
        self.steps = 0
        self.is_train = is_train
        self.user_instruction = ''
        self.history_act_feedback = []
        self.actions = ["Move forward by 0.25",
                        "Move backward by 0.25",
                        "Move rightward by 0.25",
                        "Move leftward by 0.25",
                        "Rotate to the right by 90 degrees.",
                        "Rotate to the left by 90 degrees.",
                        "Tilt the camera upward by 30 degrees.",
                        "Tilt the camera downward by 30 degrees."]
        self.valid_actions = [0, 1, 2, 3, 4, 5, 6, 7]
        self.obs = {'fv_rgb': None, 'bev_rgb': None}

    def __get_object_ids_by_type(self, metadata, object_type):
        objects = []
        for obj in metadata["objects"]:
            if obj["objectType"] == object_type:
                objects.append(obj["objectId"])
        return objects
    
    def __calculate_distance(self, pos1, pos2):
        return np.sqrt((pos1['x'] - pos2['x'])**2 + (pos1['z'] - pos2['z'])**2)
    
    def __get_valid_pose(self, pos_seed=0):
        event = self.env.step(
            action="GetInteractablePoses",
            objectId=self.target_object_id,
            horizons=[0]
            # rotations=DEFAULT_ROTATIONS,
            # standings=[True]
        )

        if not event.metadata["actionReturn"]:
            return None
        
        # Get object position
        obj_metadata = next(obj for obj in self.env.last_event.metadata["objects"] 
                        if obj["objectId"] == self.target_object_id)

        obj_position = {
            'x': obj_metadata['position']['x'],
            'z': obj_metadata['position']['z']
        }

        poses = event.metadata["actionReturn"]
        random.seed(pos_seed)
        random.shuffle(poses)
        for pose in poses:
            pos = {'x': pose['x'], 'z': pose['z']}
            dis = self.__calculate_distance(pos, obj_position)
            if dis >= self.config['dataset']['min_distance'] and dis <= self.config['dataset']['max_distance']:
                return pose
        
        return None
    
    def __check_target_visibility(self):
        if self.target_object_id in self.env.last_event.instance_detections2D:
            return True
        else:
            return False

    def __measure_success(self):
        # success measurement
        agent_position = self.env.last_event.metadata["agent"]["position"]

        dist = math.sqrt(
            (agent_position["x"] - self.target_position["x"])**2 +
            (agent_position["z"] - self.target_position["z"])**2
        )
        success = (dist <= self.config['dataset']['sueeess_threshold'])
        return float(success), dist
    
    def __image2tensor(self, image: Image.Image):
        image_array = np.array(image)           # H X W X 3
        # image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0   # 3 X H X W
        image_tensor = torch.from_numpy(image_array).float() / 255.0   # 3 X H X W
        return image_tensor.cpu()
    
    def __draw_navigation_line_bev(self, image, agent_pos, target_pos_pt, bounds, img_width, img_height, line_width, color=(255,0,0)):
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
    
    def __draw_navigation_line_fv(self, image, target_pos_pt, img_width, img_height, line_width, color=(255, 0, 0)):
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
    
    def __add_marks_on_obs(self, fv_bbox=True, bev_bbox=True):
        '''在视图中添加导航线、bounding box'''
        new_obs = {'fv_rgb': self.obs['fv_rgb']}
        if fv_bbox:
            fv, fv_bbox_pt = draw_target_box(self.obs['fv_rgb'], self.env.last_event.instance_detections2D, self.target_object_id, color=(0, 0, 255), thickness=2)
            if fv_bbox_pt:
                target_pos_pt = (int((fv_bbox_pt[0] + fv_bbox_pt[2]) / 2), int((fv_bbox_pt[1] + fv_bbox_pt[3]) / 2))
                fv = self.__draw_navigation_line_fv(fv, target_pos_pt, self.config['envs']['width'], self.config['envs']['height'], line_width=2, color=(0,0,255))
            new_obs['fv_rgb'] = fv
        if 'bev_rgb' in self.obs.keys():
            bev = self.obs['bev_rgb']
            scene_bounds_raw = self.env.last_event.metadata['sceneBounds']
            scene_bounds = get_scene_bounds(scene_bounds_raw)
            agent_pos = self.env.last_event.metadata['agent']['position']
            # agent_rot = self.env.last_event.metadata['agent']['rotation']
            if bev_bbox:
                bbox_raw = []
                for obj in self.env.last_event.metadata['objects']:
                    if obj['objectId'] == self.target_object_id:
                        bbox_raw = obj['axisAlignedBoundingBox']
                        break
                bev, bev_bbox_pt = birdview_draw_object(scene_bounds_raw, bbox_raw, img_width=self.config['envs']['width'], img_height=self.config['envs']['height'], color=(255,0,0), background=bev)
                if bev_bbox_pt:
                    target_pos_pt = (int((bev_bbox_pt[0] + bev_bbox_pt[2]) / 2), int((bev_bbox_pt[1] + bev_bbox_pt[3]) / 2))
                    bev = self.__draw_navigation_line_bev(bev, agent_pos, target_pos_pt, scene_bounds, self.config['envs']['width'], self.config['envs']['height'], line_width=2, color=(0,0,255))

            new_obs['bev_rgb'] = bev

        '''添加动作投影和智能体箭头'''
        projector = ActionMarker(env=self.env, env_name='EB-Navigation', config={'img_height': self.config['envs']['height'], 'img_width': self.config['envs']['width']})
        
        fv, bev = projector.draw_action(action_ids=self.valid_actions, obs=new_obs, agent_mark=True, align=True)
        self.obs['fv_rgb'] = fv
        self.obs['bev_rgb'] = bev

    
    def init_env(self):
        # self.seed = seed
        # random.seed(self.seed)
        env_init_config = {
            "agentMode": self.config['envs']['agentMode'],
            "gridSize": self.config['envs']['gridSize'],
            "visibilityDistance": self.config['envs']['visibilityDistance'],
            "renderDepthImage": self.config['envs']['renderDepthImage'],
            "renderInstanceSegmentation": self.config['envs']['renderInstanceSegmentation'],
            "width": self.config['envs']['width'],
            "height": self.config['envs']['height'],
            "fieldOfView": self.config['envs']['fieldOfView'],
            "platform": CloudRendering
        }
        # print('#################### init_env #####################')
        for _ in range(self.config['envs']['ai2thor_retry_num']):
            try:
                self.env = Controller(**env_init_config)
                # print('#################### init_env done #####################')
                return self
            except Exception as e:
                print('ERROR IN ENV INITIALIZATION:\n', e)
                if self.env is not None:
                    self.env.stop()
                time.sleep(5)
                continue
        raise Exception("Failed to initialize environment")

    
    def update_history(self, info):
        """Update episode feedback history."""
        self.history_act_feedback.append([
            info['action_id'],
            info['env_feedback']
        ])
    
    def reset(self, scene_seed=0, pos_seed=0, spawn_seed=0, val_idx=None):
        # print('################### reset ###################')
        if not self.is_train:
            print('###### reset val scenes ######')
            data_path = os.path.join(os.path.dirname(__file__), 'dataset', self.config['dataset']['eval_dataset'])
            with open(data_path) as f:
                dataset_split = json.load(f)
                dataset = dataset_split["tasks"]

            if val_idx:
                traj_data = dataset[val_idx]
            else:
                random.seed(scene_seed)
                traj_data = random.choice(dataset)
            for i in range(self.config['envs']['ai2thor_retry_num']):
                try:
                    self.env.reset(traj_data["scene"])
                    break
                except Exception as e:
                    print('ERROR IN ENV RESET TO SCENE:\n', e)
                    time.sleep(5)
                    if i >= self.config['envs']['ai2thor_retry_num'] - 1:
                        raise Exception("Failed to reset environment")

            pose = traj_data["agentPose"]
            self.target_object_id = traj_data["targetObjectIds"]
            visibility = self.__check_target_visibility()
            self.target_position = traj_data["target_position"]
            self.user_instruction = traj_data["instruction"]

        else:
            # 初始化场景
            print('###### reset train scenes ######')
            train_scene_target_pairs = None
            with open(os.path.join(os.path.dirname(__file__), 'dataset', self.config['dataset']['train_scene_obj_pair']), 'r') as f:
                train_scene_target_pairs = json.load(f)
            random.seed(scene_seed)
            while True:
                chosen_pair = random.choice(train_scene_target_pairs)
                scene, target_type = chosen_pair[0], chosen_pair[1]
                for i in range(self.config['envs']['ai2thor_retry_num']):
                    try:
                        self.env.reset(scene=scene)
                        break
                    except Exception as e:
                        print('ERROR IN ENV RESET TO SCENE:\n', e)
                        time.sleep(5)
                        if i >= self.config['envs']['ai2thor_retry_num'] - 1:
                            raise Exception("Failed to reset environment")

                # 执行一系列场景随机化操作
                for i in range(self.config['envs']['ai2thor_retry_num']):
                    try:
                        # if self.config['dataset']['if_RandomizeMaterials']:
                        #     self.env.step(action="RandomizeMaterials")
                        # if self.config['dataset']['if_RandomizeLighting']:
                        #     self.env.step(
                        #     action="RandomizeLighting",
                        #     brightness=(0.5, 1.5),
                        #     randomizeColor=True,
                        #     hue=(0, 1),
                        #     saturation=(0.5, 1),
                        #     synchronized=False
                        #     )
                        if self.config['dataset']['if_InitialRandomSpawn']:
                            self.env.step(
                                action="InitialRandomSpawn",
                                randomSeed=spawn_seed,
                                forceVisible=True
                                )
                        break
                    except Exception as e:
                        print('ERROR IN SCENE RANDOMIZATION:\n', e)
                        time.sleep(5)
                        if i >= self.config['envs']['ai2thor_retry_num'] - 1:
                            raise Exception("Failed to randomize environment")

                target_objects = self.__get_object_ids_by_type(self.env.last_event.metadata, target_type)
                if target_objects == []:
                    print(f"Warning: No {target_type} found in {scene}")
                    continue
                else:
                    # print(f'Found valid scene {scene} and target {target_type}')
                    self.target_object_id = target_objects[0]
                    for obj in self.env.last_event.metadata["objects"]:
                        if obj["objectId"] == self.target_object_id:
                            self.target_position = obj["position"]
                            break
                    # Get other objects to hide (all objects of same type except the target)
                    # objects_to_hide = target_objects[1:] if len(target_objects) > 1 else []

                    # Get valid initial pose
                    pose = self.__get_valid_pose(pos_seed)
                    visibility = self.__check_target_visibility()
                    if not pose:
                        print(f"Warning: Could not find valid pose in {scene}")
                        continue
                    self.user_instruction = f"navigate to the {target_type} in the room and be as close as possible to it"
                    break
        
        for i in range(self.config['envs']['ai2thor_retry_num']):
            try:
                event = self.env.step(action="GetMapViewCameraProperties", raise_for_failure=True)
                third_party_pose = copy.deepcopy(event.metadata["actionReturn"])
                third_party_pose["orthographic"] = True
                self.env.step(action="AddThirdPartyCamera", **third_party_pose, skyboxColor="white", raise_for_failure=True)
                break
            except Exception as e:
                print('ERROR IN ADDING THIRD PARTY CAMERA:\n', e)
                time.sleep(5)
                if i >= self.config['envs']['ai2thor_retry_num'] - 1:
                    raise Exception("Failed to add third-party camera")

        if "position" not in pose.keys():
            pose = {
                "position": {"x": pose["x"], "y": pose["y"], "z": pose["z"]},
                "rotation": pose["rotation"],
                "horizon": pose["horizon"],
                "standing": pose["standing"]
            }
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
        self.last_visibility = visibility
        _, self.last_distance = self.__measure_success()
        if self.config['envs']['visual_prompt']:
            if self.config['output_format'] == 'json':
                text_obs = EBNAV_TEMPLATE_VP_JSON.format(user_instruction=self.user_instruction, history='')
            else:
                text_obs = EBNAV_TEMPLATE_VP.format(user_instruction=self.user_instruction, history='')
        else:
            text_obs = EBNAV_TEMPLATE.format(user_instruction=self.user_instruction, history='')
        
        self.obs['fv_rgb'] = Image.fromarray(self.env.last_event.frame)
        self.obs['bev_rgb'] = Image.fromarray(self.env.last_event.third_party_camera_frames[-1])
        if self.config['envs']['visual_prompt']:
            self.__add_marks_on_obs()
        image_obs = concat_images(self.obs['bev_rgb'], self.obs['fv_rgb'])

        
        # import uuid
        # uid = uuid.uuid4().hex
        # os.makedirs(f'/mnt/kaiwu-user-jensencwang/research/verl-agent/debug_dir/images', exist_ok=True)
        # image_obs.save(f'/mnt/kaiwu-user-jensencwang/research/verl-agent/debug_dir/images/image_{scene_seed}_{pos_seed}_{spawn_seed}_{uid}.png')
        
        info = {}
        self.steps = 0
        # print('################### reset success ###################')
        
        return text_obs, self.__image2tensor(image_obs), info
    
    def get_env_feedback(self, event):
        """
        To extract relevant information from the event to construct a feedback dictionary.

        :param event: self._last_event
        :return: A dictionary containing structured feedback.
        """

        feedback = {
            "lastActionSuccess": event.metadata.get("lastActionSuccess", None),
            "errorMessage": event.metadata.get("errorMessage", None),
            "lastAction": event.metadata.get("lastAction", None),

            "agent": {
                "position": event.metadata.get("agent", {}).get("position", {}),
                "rotation": event.metadata.get("agent", {}).get("rotation", {}),
                "is_standing": True
            }
        }

        msg = ''
        if feedback["lastActionSuccess"]:
            msg += f"Last action {feedback['lastAction']} executed successfully."
        else:
            msg += f"Last action {feedback['lastAction']} is invalid. {feedback['errorMessage']}"
        return msg

    def step(self, action: int):
        # print(f'################### step {self.steps} ###################')
        history = '''\nThe action history:'''
        HISTORY_WINDOW = self.config['envs']['history_window']
        # 如果不足5条，slice到全部即可
        for i, action_feedback in enumerate(self.history_act_feedback[-HISTORY_WINDOW:]):
            step_num = self.steps - len(self.history_act_feedback[-HISTORY_WINDOW:]) + i
            history += '\n Step {}, action id {}, {}, env feedback: {}'.format(
                step_num,
                action_feedback[0],
                self.actions[action_feedback[0]],
                action_feedback[1]
            )
        if self.config['envs']['visual_prompt']:
            if self.config['output_format'] == 'json':
                text_obs = EBNAV_TEMPLATE_VP_JSON.format(user_instruction=self.user_instruction, history=history)
            else:
                text_obs = EBNAV_TEMPLATE_VP.format(user_instruction=self.user_instruction, history=history)
        else:
            text_obs = EBNAV_TEMPLATE.format(user_instruction=self.user_instruction, history=history)
        
        if action not in self.valid_actions:
            _, distance = self.__measure_success()
            info = {'distance': distance, 'last_distance': self.last_distance + 0, 'visibility': False, 'last_visibility': copy.deepcopy(self.last_visibility), 
                    'won': 0, 'env_feedback': '', 'last_action_success': False, 'action_id': -1, 'env_step': self.steps,
                    'step_reward': 0}

            self.obs['fv_rgb'] = Image.fromarray(self.env.last_event.frame)
            self.obs['bev_rgb'] = Image.fromarray(self.env.last_event.third_party_camera_frames[-1])
            if self.config['envs']['visual_prompt']:
                self.__add_marks_on_obs()
            image_obs = concat_images(self.obs['bev_rgb'], self.obs['fv_rgb'])

            return text_obs, self.__image2tensor(image_obs), -1, False, info

        if action == 0:  # Move forward
            last_event = self.env.step(action="MoveAhead", moveMagnitude=self.config['envs']['moveMagnitude'])
        elif action == 1:  # Move backward
            last_event = self.env.step(action="MoveBack", moveMagnitude=self.config['envs']['moveMagnitude'])
        elif action == 2:  # Move right
            last_event = self.env.step(action="MoveRight", moveMagnitude=self.config['envs']['moveMagnitude'])
        elif action == 3:  # Move left
            last_event = self.env.step(action="MoveLeft", moveMagnitude=self.config['envs']['moveMagnitude'])
        elif action == 4:  # Rotate clockwise
            last_event = self.env.step(action="RotateRight", degrees=self.config['envs']['rotateDegree'])
        elif action == 5:  # Rotate counterclockwise
            last_event = self.env.step(action="RotateLeft", degrees=self.config['envs']['rotateDegree'])
        elif action == 6:  # Tilt the camera upward
            last_event = self.env.step(action="LookUp", degrees=self.config['envs']['tiltDegree'])
        elif action == 7:  # Tilt the camera downward
            last_event = self.env.step(action="LookDown", degrees=self.config['envs']['tiltDegree'])

        visibility = self.__check_target_visibility()
        success, distance = self.__measure_success()
        self.steps += 1

        info = {}

        info['distance'] = distance + 0
        info['last_distance'] = self.last_distance + 0
        info['visibility'] = copy.deepcopy(visibility)
        info['last_visibility'] = copy.deepcopy(self.last_visibility)
        info['won'] = success
        info['env_feedback'] = self.get_env_feedback(last_event)
        info['last_action_success'] = self.env.last_event.metadata['lastActionSuccess']
        info['action_id'] = action
        info['env_step'] = self.steps
        info['step_reward'] = compute_step_reward(info)
        self.update_history(info)

        self.last_distance = distance
        self.last_visibility = visibility

        if self.steps >= self.config['envs']['max_steps'] or success:
            done = True
        else:
            done = False
        reward = success

        self.obs['fv_rgb'] = Image.fromarray(self.env.last_event.frame)
        self.obs['bev_rgb'] = Image.fromarray(self.env.last_event.third_party_camera_frames[-1])
        if self.config['envs']['visual_prompt']:
            self.__add_marks_on_obs()
        image_obs = concat_images(self.obs['bev_rgb'], self.obs['fv_rgb'])
        return text_obs, self.__image2tensor(image_obs), reward, done, info
    
    def close(self):
        self.env.stop()


class EBNavWorker:
    def __init__(self, base_env):
        self.base_env = base_env
        try:
            self.env = self.base_env.init_env()
        except Exception as e:
            print("Error in init: ", e)
            if self.env is not None:
                self.env.close()
                self.env = None
    def step(self, action):
        if self.env is not None:
            try:
                text_obs, image_obs, reward, done, info = self.env.step(action)
            except Exception as e:
                print("Error in step: ", e)
                if self.env is not None:
                    self.env.close()
                text_obs, image_obs, reward, done, info = None, None, None, None, None
        else:
            text_obs, image_obs, reward, done, info = None, None, None, None, None
        return text_obs, image_obs, reward, done, info
    
    def reset(self, scene_seed, pos_seed, spawn_seed, val_idx=None):
        if self.env is not None:
            try:
                text_obs, image_obs, info = self.env.reset(scene_seed=scene_seed, pos_seed=pos_seed, spawn_seed=spawn_seed, val_idx=val_idx)
            except Exception as e:
                print("Error in reset: ", e)
                if self.env is not None:
                    self.env.close()
                    print("Env closed, perform re-init one time")
                    try:
                        self.env = self.base_env.init_env()
                        text_obs, image_obs, info = self.env.reset(scene_seed=scene_seed, pos_seed=pos_seed, spawn_seed=spawn_seed, val_idx=val_idx)
                    except Exception as e:
                        print("Error in init: ", e)
                        if self.env is not None:
                            self.env.close()
                            self.env = None
                        text_obs, image_obs, info = None, None, None
        else:
            print("self.env is None, perform re-init one time")
            try:
                self.env = self.base_env.init_env()
                text_obs, image_obs, info = self.env.reset(scene_seed=scene_seed, pos_seed=pos_seed, spawn_seed=spawn_seed, val_idx=val_idx)
            except Exception as e:
                print("Error in init: ", e)
                if self.env is not None:
                    self.env.close()
                    self.env = None
                text_obs, image_obs, info = None, None, None
        return text_obs, image_obs, info
    
    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()

class EBNavMultiProcessEnvs():
    def __init__(self, ebnav_config_path, seed, env_num, group_n, resources_per_worker, is_train, config):
        super().__init__()

        env_config = load_config_file(ebnav_config_path)
        env_config['output_format'] = config.actor_rollout_ref.actor.output_format
        base_env = EBNavBaseEnv(env_config, is_train)
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = self.env_num * self.group_n      # env_num = batch_size
        self._rng = np.random.RandomState(seed)
        env_worker = ray.remote(**resources_per_worker)(EBNavWorker)
        self.workers = []

        self.val_start_idx = None      # 避免val时batchsize过大产生oom，绘图用，后续要移除

        for i in range(self.num_processes):
            worker = env_worker.remote(base_env)
            self.workers.append(worker)

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"
        
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)
        
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (text_obs, image_obs, reward, done, info) in enumerate(results):
            text_obs_list.append(text_obs)
            image_obs_list.append(image_obs)
            dones_list.append(done)
            info_list.append(info)
            if info:
                rewards_list.append(compute_reward(info))
            else:
                rewards_list.append(None)

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        scene_seed = self._rng.randint(0, 10000, size=self.env_num)
        repeated_scene_seed = np.repeat(scene_seed, self.group_n)
        scene_seeds = repeated_scene_seed.tolist()

        pos_seed = self._rng.randint(0, 10000, size=self.env_num)
        repeated_pos_seed = np.repeat(pos_seed, self.group_n)
        pos_seeds = repeated_pos_seed.tolist()

        spawn_seed = self._rng.randint(0, 10000, size=self.env_num)
        repeated_spawn_seed = np.repeat(spawn_seed, self.group_n)
        spawn_seeds = repeated_spawn_seed.tolist()

        text_obs_list = []
        image_obs_list = []
        info_list = []
        futures = []
        if self.val_start_idx:
            val_idx = self.val_start_idx
        else:
            val_idx = 0
        for worker, scene_seed, pos_seed, spawn_seed in zip(self.workers, scene_seeds, pos_seeds, spawn_seeds):
            future = worker.reset.remote(scene_seed, pos_seed, spawn_seed, None)
            futures.append(future)
            # val_idx += 1

        try:
            results = ray.get(futures)
        except Exception as e:
            print('error:', e)
        # results = ray.get(futures)
        for i, (text_obs, image_obs, info) in enumerate(results):
            text_obs_list.append(text_obs)
            image_obs_list.append(image_obs)
            info_list.append(info)
        return text_obs_list, image_obs_list, info_list

    def close(self):
        """Close all workers."""
        # Send close commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.close.remote()
            futures.append(future)
        
        # Wait for all workers to close
        ray.get(futures)
        
        # Shutdown Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_ebnav_envs(ebnav_config_path, seed, env_num, group_n, resources_per_worker, is_train, config=None):
    return EBNavMultiProcessEnvs(ebnav_config_path, seed, env_num, group_n, resources_per_worker, is_train, config)
