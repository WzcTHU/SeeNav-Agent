# import copy
# import json
# import time
# import os
import math
import cv2
import numpy as np
from PIL import Image

# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .utils import get_scene_bounds, draw_agent_arrow, world_to_birdview

# from ai2thor.controller import Controller
# from PIL import Image, ImageDraw, ImageFont
# from ai2thor.platform import CloudRendering
# from .utils import get_object_ids_by_type, get_valid_pose


def rotate_point(x, y, center_x, center_y, angle_deg):
    """
    将点(x, y)以(center_x, center_y)为中心，顺时针旋转angle_deg度，适配opencv/PIL坐标系左上为原点
    angle_deg: 正值代表顺时针
    """
    angle_rad = math.radians(angle_deg)
    x0, y0 = x - center_x, y - center_y
    # 画图推导可得像素空间顺时针旋转公式如下
    x1 =  x0 * math.cos(angle_rad) + y0 * math.sin(angle_rad)
    y1 = -x0 * math.sin(angle_rad) + y0 * math.cos(angle_rad)
    return (x1 + center_x, y1 + center_y)

def draw_action_arrow_bev(image, bounds, img_width, img_height, agent_pos, agent_rot, arrow_length_px=100, action_id='0', offset_px=32, color=(255,0,0), align=False):
    """
    在底图上绘制箭头与动作ID
    现在arrow_length_px和offset_px都是以像素为单位
    """
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    id_str = str(action_id)
    
    # agent中心像素坐标
    agent_u, agent_v = world_to_birdview(agent_pos['x'], agent_pos['z'], bounds, img_width, img_height)

    # theta_deg = agent_rot['y']
    # theta_rad = math.radians(theta_deg)
    dir_map = {
        '0': 0,
        '1': 180,
        '2': 90,
        '3': 270
    }
    offset_deg = dir_map.get(id_str, 0)

    if align:
        # agent已经朝上，箭头的方向只与动作有关
        arrow_deg = offset_deg
        h, w = image.shape[0:2]
        center_x, center_y = w/2, h/2
        agent_u, agent_v = rotate_point(agent_u, agent_v, center_x, center_y, agent_rot['y'])
    else:
        # agent可能任意朝向，需要额外叠加agent的yaw
        arrow_deg = agent_rot['y'] + offset_deg

    arrow_angle = math.radians(arrow_deg)
    # arrow_angle = theta_rad + math.radians(offset_deg)

    # 在像素空间计算箭头起点和终点
    # offset_px表示离agent圆心的距离，arrow_length_px表示箭头长度
    start_u = int(round(agent_u + offset_px * math.sin(arrow_angle)))
    start_v = int(round(agent_v - offset_px * math.cos(arrow_angle)))
    end_u   = int(round(agent_u + (offset_px + arrow_length_px) * math.sin(arrow_angle)))
    end_v   = int(round(agent_v - (offset_px + arrow_length_px) * math.cos(arrow_angle)))

    # 画箭头
    cv2.arrowedLine(
        image, (start_u, start_v), (end_u, end_v),
        color=color, thickness=2, tipLength=0.2
    )

    # 箭头单位像素方向
    dir_px = np.array([end_u - start_u, end_v - start_v])
    norm = np.linalg.norm(dir_px)
    if norm == 0:
        unit_dir_px = np.array([0, 0])
    else:
        unit_dir_px = dir_px / norm

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(id_str, fontFace, fontScale, thickness)
    text_offset_px = 15  # 距离箭头head的像素距离，可适度调大让不被箭头盖住
    perp_unit_px = np.array([-unit_dir_px[1], unit_dir_px[0]])

    text_center = np.array([end_u, end_v]) + unit_dir_px * text_offset_px

    # 美观处理：文本整体垂直于箭头微偏（可改成其它正交量以微调视觉效果）
    text_center = text_center + perp_unit_px * 0  # 可以调成非0，看实际需求

    text_org = (
        int(round(text_center[0] - text_width / 2)),
        int(round(text_center[1] + text_height / 2))
    )
    
    cv2.putText(
        image, id_str, text_org,
        fontFace=fontFace, fontScale=fontScale,
        color=color, thickness=thickness, lineType=cv2.LINE_AA
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return image

def draw_action_arrow_fv(image, action_id, arrow_length=100, offset=10, text_dist=20, color=(255,0,0)):
    """
    用cv2在图像上绘制箭头和动作数字，箭头方向/起点偏移等参数可自定义。
    image: PIL.Image对象
    action_id: '4'-右, '5'-左, '6'-上, '7'-下（也可int）
    arrow_length: 箭头长度
    offset: 箭头起点离中心偏移
    text_dist: 文字距离箭头末端
    返回: PIL.Image对象
    """
    # PIL转np.array
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    id_str = str(action_id)
    # 方向字典
    dir_map = {
        '4': (1, 0),   # 右
        '5': (-1, 0),  # 左
        '6': (0, -1),  # 上
        '7': (0, 1),   # 下
    }
    if id_str not in dir_map:
        raise ValueError("action_id必须为4,5,6,7的一种")
    dx, dy = dir_map[id_str]
    # 箭头起点=中心+偏移 终点=起点+箭头长度
    start_u = int(cx + dx * offset)
    start_v = int(cy + dy * offset)
    end_u = int(start_u + dx * arrow_length)
    end_v = int(start_v + dy * arrow_length)
    # 绘箭头
    cv2.arrowedLine(
        image, (start_u, start_v), (end_u, end_v),
        color=color, thickness=2, tipLength=0.2
    )

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.0
    thickness = 2
    # 文字大小
    (text_width, text_height), baseline = cv2.getTextSize(id_str, fontFace, fontScale, thickness)
    txt_x = end_u + dx*text_dist
    txt_y = end_v + dy*text_dist
    # 对齐修正
    if dx == 0 and dy == -1:  # 上
        txt_x -= text_width // 2
        txt_y -= baseline + 1
    elif dx == 0 and dy == 1:  # 下
        txt_x -= text_width // 2
        txt_y += text_height + 1
    elif dx == 1 and dy == 0:  # 右
        # 靠右箭头头部右侧，垂直居中
        txt_x += 1
        txt_y += text_height // 2
    elif dx == -1 and dy == 0:  # 左
        txt_x -= text_width + 1
        txt_y += text_height // 2

    # 箭头末端外text位置
    text_pos = (int(end_u + dx * text_dist), int(end_v + dy * text_dist))
    cv2.putText(
        image, id_str, (int(txt_x), int(txt_y)), fontFace, fontScale,
        color, thickness, cv2.LINE_AA
    )
    # np.array转回PIL
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)

class ActionMarker:
    def __init__(self, env=None, env_name='EB-Navigation', config=None):
        self.env = env
        self.env_name = env_name
        self.config = config
        self.episode_idx = -1
        self.planner_steps = -1

    def draw_action(self, action_ids=None, obs=None, agent_mark=False, align=True):
        if self.env_name == "EB-Navigation":
            bev_with_action = self.__draw_eb_navigation(action_ids, obs, agent_mark, align)
        return bev_with_action
    

    def __draw_eb_navigation(self, action_ids=None, obs=None, agent_mark=False, align=True):
        fv = obs['fv_rgb']
        bev = obs['bev_rgb']
        scene_bounds_raw = self.env.last_event.metadata['sceneBounds']
        scene_bounds = get_scene_bounds(scene_bounds_raw)
        agent_pos = self.env.last_event.metadata['agent']['position']
        agent_rot = self.env.last_event.metadata['agent']['rotation']

        if agent_mark:
            bev = draw_agent_arrow(agent_pos, agent_rot, scene_bounds, bev, img_width=self.config['img_width'], img_height=self.config['img_height'], color=(0,255,0))
        if align:
            bev = bev.rotate(agent_rot['y'])
        # 0: Move forward by 0.25
        # 1: Move backward by 0.25
        # 2: Move rightward by 0.25
        # 3: Move leftward by 0.25
        # 4: Rotate to the right by 90 degrees
        # 5: Rotate to the left by 90 degrees
        # 6: Tilt the camera upward by 30 degrees
        # 7: Tilt the camera downward by 30 degrees
        if action_ids:
            for action_id in action_ids:
                action_id = str(action_id)
                if action_id in ['0', '1', '2', '3']:
                    bev = draw_action_arrow_bev(image=bev, bounds=scene_bounds, img_width=self.config['img_width'], img_height=self.config['img_height'], 
                                            agent_pos=agent_pos, agent_rot=agent_rot, arrow_length_px=40, action_id=action_id, offset_px=10, color=(255,0,0), align=align)
                if action_id in ['4', '5', '6', '7']:
                    fv = draw_action_arrow_fv(image=fv, action_id=action_id, arrow_length=40, offset=10, text_dist=10, color=(255,0,0))

        return fv, bev

if __name__ == '__main__':
    pass

    # env_config = {
    #         "agentMode": "default",
    #         "gridSize": 0.1,
    #         "visibilityDistance": 10,
    #         "renderDepthImage": True,
    #         "renderInstanceSegmentation": True,
    #         "width": 500,
    #         "height": 500,
    #         "fieldOfView": 100,
    #         "scene": "FloorPlan11",
    #         "platform": CloudRendering
    #     }

    # print('init controller')
    # env = Controller(**env_config)

    # target_type = 'Bread'
    # scene = 'FloorPlan11'
    # env.reset(scene=scene)
    # event = env.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    # pose = copy.deepcopy(event.metadata["actionReturn"])
    # pose["orthographic"] = True
    # env.step(action="AddThirdPartyCamera", **pose, skyboxColor="white", raise_for_failure=True)

    # target_objects = get_object_ids_by_type(env.last_event.metadata, target_type)

    # # Select first target object
    # target_object_id = target_objects[0]

    # objects_to_hide = target_objects[1:] if len(target_objects) > 1 else []
        
    # # Get valid initial pose
    # pose = get_valid_pose(env, target_object_id)
    # if not pose:
    #     print(f"Warning: Could not find valid pose in {scene}")

    # env.step(
    #     action="Teleport",
    #     position={
    #         "x": pose["x"],
    #         "y": pose["y"],
    #         "z": pose["z"]
    #     },
    #     rotation={
    #         "x": 0,
    #         "y": pose["rotation"],
    #         "z": 0
    #     },
    #     horizon=pose["horizon"],
    #     standing=True
    # )

    # marker_config = {
    #         "img_width": 500,
    #         "img_height": 500
    #     }
    # obs = {
    #     'fv_rgb': Image.fromarray(env.last_event.frame),
    #     'bev_rgb': Image.fromarray(env.last_event.third_party_camera_frames[-1])
    # }
    # marker = ActionMarker(env, 'EB-Navigation', marker_config)
    # fv, bev = marker.draw_action(action_ids=['0', '1', '2', '3', '4', '5', '6', '7'], obs=obs)
    # fv.save('fv.png')
    # bev.save('bev.png')