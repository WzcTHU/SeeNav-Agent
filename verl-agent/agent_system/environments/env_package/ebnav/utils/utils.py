import numpy as np
import cv2
import math
from PIL import Image
from PIL import ImageDraw
from typing import Dict

def get_scene_bounds(sceneBounds):
    """
    提取场景边界，输出xmin, xmax, zmin, zmax
    """
    xs = [pt[0] for pt in sceneBounds['cornerPoints']]
    zs = [pt[2] for pt in sceneBounds['cornerPoints']]
    return {
        'xmin': min(xs),
        'xmax': max(xs),
        'zmin': min(zs),
        'zmax': max(zs)
    }

def concat_images(image1: Image.Image, image2: Image.Image):
    width, height = image1.size  # 获取单张图片的宽高
    # 创建一张新图片，宽度是两张拼起来，高度保持不变
    new_image = Image.new('RGB', (width * 2, height))
    # 把第一张图片贴到新图的左边
    new_image.paste(image1, (0, 0))
    # 把第二张图片贴到新图的右边
    new_image.paste(image2, (width, 0))
    return new_image


def draw_target_box(
    image,
    instance_detections: Dict[str, np.ndarray],
    object_id: str,
    color: tuple = (0, 255, 0),  # Default color: green
    thickness = 1
):
    if object_id in instance_detections:

        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if image is None:
            raise ValueError(f"Could not read image")
        
        # Get coordinates for the specified object ID
        bbox = instance_detections[object_id]
        start_point = (int(bbox[0]), int(bbox[1]))  # Upper left corner
        end_point = (int(bbox[2]), int(bbox[3]))    # Lower right corner
        
        # Draw the rectangle
        cv2.rectangle(
            image,
            start_point,
            end_point,
            color,
            thickness
        )
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output_image = Image.fromarray(image_rgb)
        return output_image, bbox

    else:
        return image, None
    

def world_to_birdview(x, z, scene_bounds, img_width, img_height):
    """
    将世界坐标 (x,z) 映射到鸟瞰图像素坐标 (u,v)
    scene_bounds: dict，有 xmin/xmax, zmin/zmax
    x从左到右增加
    z从下往上增加
    u, v的原点在图像左上角, u向右递增, v向下递增
    """
    x_min = scene_bounds['xmin']
    x_max = scene_bounds['xmax']
    z_min = scene_bounds['zmin']
    z_max = scene_bounds['zmax']

    # 由于俯视图中很多房间是长方形，因此只有当图像的长宽比例与俯视图房间比例一致的时候才能直接使用(x_max - x_min)和(z_max - z_min)进行scale，
    # 其他情况下应该根据长宽比进行调整
    roi_w = x_max - x_min
    roi_h = z_max - z_min
    scale = min(img_width / roi_w, img_height / roi_h)
    pad_w = img_width - roi_w * scale
    pad_h = img_height - roi_h * scale
    u = (x - x_min) * scale + pad_w / 2
    v = (z_max - z) * scale + pad_h / 2  # z_max在上
    return (int(round(u)), int(round(v)))


def draw_object_bounding_box_birdview(scene_bounds, obj_bbox, img_width=500, img_height=500, color=(255,0,0), img=None):
    """
    在鸟瞰图上绘制单个物体的bounding box（axisAlignedBoundingBox）
    scene_bounds: dict，有 xmin/xmax/zmin/zmax
    obj_bbox: dict, 包含 'cornerPoints' (8x3 array)
    img: PIL Image，如果为None则新建白底
    返回绘制后的图片
    """
    corners = obj_bbox['cornerPoints']
    xs = [pt[0] for pt in corners]
    zs = [pt[2] for pt in corners]
    # 左上角 （x_min, z_max），右下角 （x_max, z_min）
    x1, z1 = min(xs), max(zs)
    x2, z2 = max(xs), min(zs)
    (u1, v1) = world_to_birdview(x1, z1, scene_bounds, img_width, img_height)
    (u2, v2) = world_to_birdview(x2, z2, scene_bounds, img_width, img_height)
    # 新建图片或使用已有
    if img is None:
        img = Image.new("RGB", (img_width, img_height), (255,255,255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([u1, v1, u2, v2], outline=color, width=2)
    return img, (u1, v1, u2, v2)


def birdview_draw_object(sceneBounds, obj_bbox, img_width=500, img_height=500, color=(255,0,0), background=None):
    bounds = get_scene_bounds(sceneBounds)
    img = None
    if background is not None:
        img = background
    img, bbox = draw_object_bounding_box_birdview(bounds, obj_bbox, img_width, img_height, color=color, img=img)
    return img, bbox


def draw_agent_arrow(agent_pos, agent_rot, scene_bounds, img,
                    img_width=500, img_height=500, 
                    color=(0,255,0), radius=15, 
                    arrow_length=14, arrow_width=8, 
                    arrow_scale=1.5):
    """
    arrow_scale: 对箭头长度和宽度的放大倍数
    """

    u, v = world_to_birdview(agent_pos['x'], agent_pos['z'], scene_bounds, img_width, img_height)

    theta_deg = agent_rot['y']
    theta_rad = math.radians(theta_deg)

    draw = ImageDraw.Draw(img)

    # 1. 画agent位置的圆
    # left_up_point = (u - radius, v - radius)
    # right_down_point = (u + radius, v + radius)
    # draw.ellipse([left_up_point, right_down_point], fill=color)

    # ====使用放大倍数====
    scaled_arrow_length = arrow_length * arrow_scale
    scaled_arrow_width = arrow_width * arrow_scale

    # 2. 箭头的起点（圆心）和终点
    arrow_tip = (
        u + (radius + scaled_arrow_length) * math.sin(theta_rad),
        v - (radius + scaled_arrow_length) * math.cos(theta_rad)
    )
    draw.line([ (u, v), arrow_tip ], fill=color, width=int(4*arrow_scale))

    # 3. 箭头的两侧（头部宽度与长度也扩大）
    arrow_head_length = scaled_arrow_length * 0.35
    arrow_head_width = scaled_arrow_width

    dir_x = math.sin(theta_rad)
    dir_y = -math.cos(theta_rad)
    perp_x = -dir_y
    perp_y = dir_x

    left_head = (
        arrow_tip[0] - arrow_head_length * dir_x + arrow_head_width * perp_x,
        arrow_tip[1] - arrow_head_length * dir_y + arrow_head_width * perp_y
    )
    right_head = (
        arrow_tip[0] - arrow_head_length * dir_x - arrow_head_width * perp_x,
        arrow_tip[1] - arrow_head_length * dir_y - arrow_head_width * perp_y
    )
    draw.polygon([arrow_tip, left_head, right_head], fill=color)



    # 画彩色半圆
    left_up_point = (u - radius, v - radius)
    right_down_point = (u + radius, v + radius)

    # Pie角度为顺时针0度为水平右侧，注意y为下为正，所以和我们的theta定义不一样
    # ai2thor朝上为0度，朝右为90度
    # 需要换算
    # 假设agent面朝direction, 那么其朝向正前方线与“水平右”线之间的夹角为90-theta_deg
    # “右半”是从前进方向顺时针90度到前进方向逆时针90度
    # PIL pieslice从水平右为0度，顺时针增加
    # agent朝向角对应的起点 = -theta_deg + 90

    # 右侧(紫色): agent的右侧是从朝向到朝向+180度（即agent前进方向的右侧），因为PIL的角度定义, 要换算
    start_angle = theta_deg - 90  # 朝向正前方的角度对应PIL的起点
    end_angle_right = start_angle + 180  # 右半圆（紫色）

    # 画右半部（紫色）
    draw.pieslice([left_up_point, right_down_point], start=start_angle, end=end_angle_right, fill=(128, 0, 128))  # 紫色

    # 左半部（黄色）：相邻的180度
    end_angle_left = start_angle
    start_angle_left = start_angle + 180
    draw.pieslice([left_up_point, right_down_point], start=start_angle_left, end=end_angle_left+360, fill=(255,255,0))  # 黄色

    return img