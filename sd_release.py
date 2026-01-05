"""
Stable Diffusion 3 图像生成与目标检测自动化流水线

本脚本的主要功能：
1. 从 COCO 或 VOC 数据集中加载图像描述和场景信息
2. 使用 GPT 模型增强图像描述，添加更多对象信息
3. 使用 Stable Diffusion 3 生成合成图像
4. 使用 Faster R-CNN 对生成的图像进行目标检测
5. 生成 YOLO 格式的标注文件和可视化结果

工作流程：
数据加载 -> 提示词增强 -> 图像生成 -> 目标检测 -> 标注保存
"""

from pycocotools.coco import COCO

import torch,yaml,pickle
import os,json,random
from os.path import join
import argparse
import os
import matplotlib.pyplot as plt
from filelock import FileLock
from diffusers import StableDiffusion3Pipeline
from mmdet.apis import init_detector, inference_detector
import mmcv
import time
from itertools import chain
import logging  # 添加缺失的 logging 模块
from PIL import Image  # 用于图像拼接


def str2bool(v):
    """
    将字符串转换为布尔值
    
    参数:
        v: 输入值（可以是字符串或布尔值）
    
    返回:
        bool: 转换后的布尔值
    
    支持的值:
        True: 'yes', 'true', 't', 'y', '1'
        False: 'no', 'false', 'f', 'n', '0'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def gen(pipe, prompt, inferstep=50):
    """
    使用 Stable Diffusion 3 生成图像（基础版本）
    
    参数:
        pipe: Stable Diffusion 3 的管道对象
        prompt: 图像生成的文本提示词
        inferstep: 推理步数，默认50步（步数越多质量越好但速度越慢）
    
    返回:
        PIL.Image: 生成的图像对象
    """
    # 使用当前时间戳生成随机种子，确保每次生成不同的图像
    # 修改cuda为cpu/mps（Mac M1 Pro 使用 mps）
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    generator = torch.Generator(device).manual_seed(int(time.time() * 1000) % 100000)
    # 使用相同的提示词作为 prompt 和 prompt_3（SD3 支持双提示词）
    out = pipe(prompt, prompt_3=prompt, num_inference_steps=inferstep, generator=generator)
    return out.images[0]

def gen2(pipe, prompt, prompt3=None, inferstep=50):
    """
    使用 Stable Diffusion 3 生成图像（增强版本，支持独立 prompt_3）
    
    参数:
        pipe: Stable Diffusion 3 的管道对象
        prompt: 主要的图像生成文本提示词
        prompt3: 可选的第三个提示词（SD3 特有），如果为 None 则只使用 prompt
        inferstep: 推理步数，默认50步
    
    返回:
        PIL.Image: 生成的图像对象
    """
    # 使用当前时间戳生成随机种子
    # 修改cuda为cpu/mps（Mac M1 Pro 使用 mps）
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    generator = torch.Generator(device).manual_seed(int(time.time() * 1000) % 100000)
    if prompt3 is None:
        # 只使用单个提示词
        out = pipe(prompt, num_inference_steps=inferstep, generator=generator)
    else:
        # 使用双提示词（prompt 和 prompt_3）
        out = pipe(prompt, prompt_3=prompt3, num_inference_steps=inferstep, generator=generator)
    return out.images[0]

def gen_quadrant_image(pipe, prompt, inferstep=50, seed_offset=0):
    """
    生成单个象限的图像（256x256）
    
    参数:
        pipe: Stable Diffusion 3 的管道对象
        prompt: 该象限的图像生成文本提示词
        inferstep: 推理步数
        seed_offset: 种子偏移量，用于确保不同象限有不同的随机性
    
    返回:
        PIL.Image: 生成的256x256图像对象
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # 使用时间戳和偏移量生成不同的随机种子
    generator = torch.Generator(device).manual_seed((int(time.time() * 1000) + seed_offset) % 100000)
    out = pipe(prompt, prompt_3=prompt, num_inference_steps=inferstep, generator=generator)
    # 将生成的图像调整为256x256（每个象限的大小）
    return out.images[0].resize((256, 256))

def combine_quadrants(top_left, top_right, bottom_left, bottom_right):
    """
    将4个象限的图像拼接成一张完整的512x512图像
    
    参数:
        top_left: 左上象限图像（PIL.Image）
        top_right: 右上象限图像（PIL.Image）
        bottom_left: 左下象限图像（PIL.Image）
        bottom_right: 右下象限图像（PIL.Image）
    
    返回:
        PIL.Image: 拼接后的512x512完整图像
    """
    # 创建一张512x512的空白图像
    combined = Image.new('RGB', (512, 512))
    # 将4个象限的图像粘贴到对应位置
    combined.paste(top_left, (0, 0))      # 左上：x=0, y=0
    combined.paste(top_right, (256, 0))   # 右上：x=256, y=0
    combined.paste(bottom_left, (0, 256)) # 左下：x=0, y=256
    combined.paste(bottom_right, (256, 256)) # 右下：x=256, y=256
    return combined

def parse_quadrant_captions(llm_output):
    """
    从LLM输出中解析4个象限的描述
    
    参数:
        llm_output: LLM返回的文本，应包含"Top Left:", "Top Right:", "Bottom Left:", "Bottom Right:"标记
    
    返回:
        dict: 包含4个象限描述的字典，键为'top_left', 'top_right', 'bottom_left', 'bottom_right'
    """
    import re
    quadrants = {}
    
    # 清理输入文本：移除多余的空格和换行
    text = llm_output.strip()
    
    # 尝试提取各个象限的描述（支持多种格式）
    patterns = {
        'top_left': [
            r'Top Left:\s*(.+?)(?=Top Right:|Bottom Left:|Bottom Right:|Caption:|$)',
            r'top left:\s*(.+?)(?=top right:|bottom left:|bottom right:|$)',
            r'左上[：:]\s*(.+?)(?=右上|左下|右下|$)'
        ],
        'top_right': [
            r'Top Right:\s*(.+?)(?=Bottom Left:|Bottom Right:|$)',
            r'top right:\s*(.+?)(?=bottom left:|bottom right:|$)',
            r'右上[：:]\s*(.+?)(?=左下|右下|$)'
        ],
        'bottom_left': [
            r'Bottom Left:\s*(.+?)(?=Bottom Right:|$)',
            r'bottom left:\s*(.+?)(?=bottom right:|$)',
            r'左下[：:]\s*(.+?)(?=右下|$)'
        ],
        'bottom_right': [
            r'Bottom Right:\s*(.+?)(?=$)',
            r'bottom right:\s*(.+?)(?=$)',
            r'右下[：:]\s*(.+?)(?=$)'
        ]
    }
    
    # 不区分大小写，允许跨行匹配
    for key, pattern_list in patterns.items():
        found = False
        for pattern in pattern_list:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                quadrants[key] = match.group(1).strip()
                # 清理描述：移除多余的换行和空格
                quadrants[key] = ' '.join(quadrants[key].split())
                found = True
                break
        
        if not found:
            # 如果找不到，使用默认描述
            quadrants[key] = "A scene with various objects."
            print(f"警告: 无法解析 {key} 象限的描述，使用默认描述")
    
    return quadrants


def get_llm_output(client, system_msg, user_prompt, model: str) -> str:
    """
    调用大语言模型（LLM）API 获取文本输出
    
    功能：
        使用 OpenAI API 或兼容的 API 调用 GPT 模型来增强图像描述
        支持 GPT-3.5、GPT-4.1、GPT-5 和 Vicuna 模型
    
    参数:
        client: OpenAI 客户端对象
        system_msg: 系统提示词，定义模型的角色和任务
        user_prompt: 用户提示词，包含需要处理的具体内容
        model: 模型名称（如 'gpt-4.1', 'gpt-3.5-turbo', 'vicuna'）
    
    返回:
        str: LLM 生成的文本响应
    
    异常:
        ValueError: 如果 API 调用失败
    """
    # 构建消息列表（用于 Chat 模型）
    if model in ["gpt-3.5", "gpt-3.5-turbo", "gpt-4.1", 'gpt-5']:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ]
    else:
        # 对于非 Chat 模型，直接使用 prompt
        messages = user_prompt
    
    # 生成缓存键（可用于后续的缓存功能）
    key = json.dumps([model, messages])
    
    # 尝试调用 API（目前只尝试一次）
    for _ in range(1):
        try:
            if model in ["gpt-3.5", "gpt-3.5-turbo", "gpt-4.1", 'gpt-5']:
                # 使用 Chat Completions API
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                response = completion.choices[0].message.content
            elif model == "vicuna":
                # 使用 Completions API（适用于 Vicuna 等模型）
                completion = client.completions.create(
                    model="lmsys/vicuna-7b-v1.5",
                    prompt=user_prompt,  # Vicuna 使用 prompt 而不是 messages
                    max_tokens=512,
                    temperature=0,  # 温度设为0，使用贪婪解码
                )
                response = completion.choices[0].text
            # 可以在这里添加缓存保存功能
            # save_to_cache(key, response, llm_cache)
            return response

        except Exception as e:
            # 记录错误并继续尝试
            logging.error(f"LLM Error: {e}")
            continue
    
    # 如果所有尝试都失败，抛出异常
    raise ValueError("Failed to get LLM output after retries")


def get_cap():
    """
    从数据集中加载图像描述（caption）和场景-对象映射关系
    
    功能：
        1. 根据数据集类型（COCO 或 VOC）加载相应的数据
        2. 提取每张图像的文本描述和包含的对象类别
        3. 构建场景到对象的映射字典（scene2obj），用于后续的对象增强
    
    返回:
        tuple: (all_captions, scene2obj)
            - all_captions: 包含所有图像描述信息的列表，每个元素包含：
                - 'text': 图像的文字描述
                - 'scene': 场景类别
                - 'classes': 图像中包含的对象类别列表
            - scene2obj: 字典，键为场景类别，值为该场景中可能出现的对象列表
    
    数据集支持:
        - COCO: 使用 COCO 数据集的标注文件
        - VOC: 使用 VOC 数据集的 BLIP 标注文件
    """
    if args.dataset == 'coco':
        # ========== COCO 数据集处理 ==========
        # 加载 COCO 实例标注文件（包含对象检测信息）
        with open("/local_dataset/coco_tamlt/annotations/instances_train2017.json", "r") as f:
                coco_data = json.load(f)
        
        # 创建类别ID到类别名称的映射字典
        category_map = {category['id']: category['name'] for category in coco_data['categories']}
        
        # 加载 YOLO 格式的类别名称文件
        with open('/NAS2/tamlt/tamlt/Code/LIC/YOLO/data/coco.yaml', 'r') as file:
            names = yaml.safe_load(file)['names']
        reverse_names = {name: class_id for class_id, name in names.items()}
        
        # 初始化 COCO API 对象
        coco = COCO("/local_dataset/coco_tamlt/annotations/instances_train2017.json")  # 对象检测标注
        coco_caps = COCO("/local_dataset/coco_tamlt/annotations/captions_train2017.json")  # 图像描述标注
        
        print('Here')
        # 获取所有图像IDµ
        original_id = coco.getImgIds()

        all_captions = []

        # 加载预处理的场景分类结果（每张图像对应的场景类别）
        allid = torch.load('/NAS2/tamlt/tamlt/Code/LIC/DCOD/results/characteristic/coco_all')
        # Places365 场景类别的统计信息（每个场景在数据集中的出现次数）
        # 这个字典定义了所有可能的场景类别
        probs = {'airfield': 157, 'airplane_cabin': 95, 'airport_terminal': 20, 'alcove': 1183, 'alley': 35, 'amphitheater': 8, 'amusement_arcade': 5, 'amusement_park': 380, 'apartment_building/outdoor': 13, 'aquarium': 202, 'aqueduct': 24, 'arcade': 13, 'arch': 54, 'archaelogical_excavation': 6, 'archive': 935, 'arena/hockey': 16, 'arena/performance': 18, 'arena/rodeo': 1085, 'army_base': 5, 'art_gallery': 1323, 'art_school': 0, 'art_studio': 317, 'artists_loft': 15, 'assembly_line': 29, 'athletic_field/outdoor': 156, 'atrium/public': 3, 'attic': 315, 'auditorium': 6, 'auto_factory': 58, 'auto_showroom': 5, 'badlands': 109, 'bakery/shop': 645, 'balcony/exterior': 4, 'balcony/interior': 23, 'ball_pit': 299, 'ballroom': 60, 'bamboo_forest': 8, 'bank_vault': 37, 'banquet_hall': 190, 'bar': 62, 'barn': 45, 'barndoor': 88, 'baseball_field': 2652, 'basement': 671, 'basketball_court/indoor': 76, 'bathroom': 4372, 'bazaar/indoor': 0, 'bazaar/outdoor': 51, 'beach': 1292, 'beach_house': 9, 'beauty_salon': 149, 'bedchamber': 17, 'bedroom': 758, 'beer_garden': 2, 'beer_hall': 1, 'berth': 194, 'biology_laboratory': 6, 'boardwalk': 74, 'boat_deck': 283, 'boathouse': 61, 'bookstore': 23, 'booth/indoor': 15, 'botanical_garden': 13, 'bow_window/indoor': 167, 'bowling_alley': 11, 'boxing_ring': 95, 'bridge': 398, 'building_facade': 26, 'bullring': 174, 'burial_chamber': 1830, 'bus_interior': 12, 'bus_station/indoor': 820, 'butchers_shop': 34, 'butte': 11, 'cabin/outdoor': 0, 'cafeteria': 128, 'campsite': 127, 'campus': 14, 'canal/natural': 35, 'canal/urban': 13, 'candy_store': 227, 'canyon': 0, 'car_interior': 28, 'carrousel': 10, 'castle': 8, 'catacomb': 26, 'cemetery': 646, 'chalet': 7, 'chemistry_lab': 1190, 'childs_room': 908, 'church/indoor': 27, 'church/outdoor': 28, 'classroom': 387, 'clean_room': 4503, 'cliff': 16, 'closet': 2238, 'clothing_store': 21, 'coast': 82, 'cockpit': 9, 'coffee_shop': 472, 'computer_room': 304, 'conference_center': 44, 'conference_room': 34, 'construction_site': 393, 'corn_field': 32, 'corral': 310, 'corridor': 53, 'cottage': 0, 'courthouse': 109, 'courtyard': 10, 'creek': 1, 'crevasse': 19, 'crosswalk': 429, 'dam': 35, 'delicatessen': 32, 'department_store': 11, 'desert/sand': 164, 'desert/vegetation': 6840, 'desert_road': 53, 'diner/outdoor': 0, 'dining_hall': 40, 'dining_room': 199, 'discotheque': 1748, 'doorway/outdoor': 13, 'dorm_room': 90, 'downtown': 87, 'dressing_room': 940, 'driveway': 199, 'drugstore': 404, 'elevator/door': 229, 'elevator_lobby': 0, 'elevator_shaft': 5, 'embassy': 62, 'engine_room': 157, 'entrance_hall': 10, 'escalator/indoor': 3, 'excavation': 27, 'fabric_store': 24, 'farm': 23, 'fastfood_restaurant': 0, 'field/cultivated': 142, 'field/wild': 62, 'field_road': 866, 'fire_escape': 198, 'fire_station': 748, 'fishpond': 196, 'flea_market/indoor': 0, 'florist_shop/indoor': 496, 'food_court': 20, 'football_field': 19, 'forest/broadleaf': 460, 'forest_path': 82, 'forest_road': 89, 'formal_garden': 0, 'fountain': 55, 'galley': 2, 'garage/indoor': 34, 'garage/outdoor': 1, 'gas_station': 198, 'gazebo/exterior': 16, 'general_store/indoor': 0, 'general_store/outdoor': 4, 'gift_shop': 12, 'glacier': 94, 'golf_course': 311, 'greenhouse/indoor': 20, 'greenhouse/outdoor': 8, 'grotto': 0, 'gymnasium/indoor': 2, 'hangar/indoor': 25, 'hangar/outdoor': 127, 'harbor': 785, 'hardware_store': 14, 'hayfield': 632, 'heliport': 142, 'highway': 769, 'home_office': 118, 'home_theater': 22, 'hospital': 42, 'hospital_room': 460, 'hot_spring': 2315, 'hotel/outdoor': 3, 'hotel_room': 305, 'house': 1, 'hunting_lodge/outdoor': 0, 'ice_cream_parlor': 1745, 'ice_floe': 2535, 'ice_shelf': 1601, 'ice_skating_rink/indoor': 1010, 'ice_skating_rink/outdoor': 657, 'iceberg': 536, 'igloo': 6991, 'industrial_area': 454, 'inn/outdoor': 2, 'islet': 45, 'jacuzzi/indoor': 224, 'jail_cell': 148, 'japanese_garden': 11, 'jewelry_shop': 127, 'junkyard': 85, 'kasbah': 90, 'kennel/outdoor': 209, 'kindergarden_classroom': 115, 'kitchen': 208, 'lagoon': 6, 'lake/natural': 194, 'landfill': 35, 'landing_deck': 1, 'laundromat': 61, 'lawn': 10, 'lecture_room': 60, 'legislative_chamber': 1, 'library/indoor': 18, 'library/outdoor': 3, 'lighthouse': 183, 'living_room': 695, 'loading_dock': 68, 'lobby': 2, 'lock_chamber': 13, 'locker_room': 244, 'mansion': 6, 'manufactured_home': 1, 'market/indoor': 0, 'market/outdoor': 0, 'marsh': 32, 'martial_arts_gym': 574, 'mausoleum': 36, 'medina': 69, 'mezzanine': 10, 'moat/water': 22, 'mosque/outdoor': 846, 'motel': 1790, 'mountain': 31, 'mountain_path': 19, 'mountain_snowy': 94, 'movie_theater/indoor': 2, 'museum/indoor': 4357, 'museum/outdoor': 1, 'music_studio': 109, 'natural_history_museum': 1368, 'nursery': 4453, 'nursing_home': 77, 'oast_house': 40, 'ocean': 386, 'office': 828, 'office_building': 27, 'office_cubicles': 69, 'oilrig': 234, 'operating_room': 22, 'orchard': 146, 'orchestra_pit': 0, 'pagoda': 111, 'palace': 1, 'pantry': 319, 'park': 87, 'parking_garage/indoor': 12, 'parking_garage/outdoor': 2, 'parking_lot': 738, 'pasture': 1015, 'patio': 172, 'pavilion': 13, 'pet_shop': 13, 'pharmacy': 325, 'phone_booth': 51, 'physics_laboratory': 2, 'picnic_area': 99, 'pier': 87, 'pizzeria': 250, 'playground': 797, 'playroom': 1227, 'plaza': 26, 'pond': 227, 'porch': 101, 'promenade': 186, 'pub/indoor': 0, 'racecourse': 659, 'raceway': 2390, 'raft': 49, 'railroad_track': 1176, 'rainforest': 68, 'reception': 7, 'recreation_room': 8, 'repair_shop': 31, 'residential_neighborhood': 4, 'restaurant': 57, 'restaurant_kitchen': 11, 'restaurant_patio': 0, 'rice_paddy': 243, 'river': 7, 'rock_arch': 3, 'roof_garden': 0, 'rope_bridge': 62, 'ruin': 1, 'runway': 1739, 'sandbox': 115, 'sauna': 1150, 'schoolhouse': 317, 'science_museum': 337, 'server_room': 38, 'shed': 34, 'shoe_shop': 66, 'shopfront': 8, 'shopping_mall/indoor': 0, 'shower': 1586, 'ski_resort': 18, 'ski_slope': 950, 'sky': 766, 'skyscraper': 129, 'slum': 436, 'snowfield': 301, 'soccer_field': 368, 'stable': 252, 'stadium/baseball': 176, 'stadium/football': 16, 'stadium/soccer': 59, 'stage/indoor': 294, 'stage/outdoor': 0, 'staircase': 223, 'storage_room': 19, 'street': 276, 'subway_station/platform': 207, 'supermarket': 47, 'sushi_bar': 0, 'swamp': 12, 'swimming_hole': 19, 'swimming_pool/indoor': 20, 'swimming_pool/outdoor': 1, 'synagogue/outdoor': 58, 'television_room': 4, 'television_studio': 1, 'temple/asia': 24, 'throne_room': 9, 'ticket_booth': 111, 'topiary_garden': 7, 'tower': 518, 'toyshop': 28, 'train_interior': 5, 'train_station/platform': 595, 'tree_farm': 572, 'tree_house': 21, 'trench': 216, 'tundra': 197, 'underwater/ocean_deep': 178, 'utility_room': 587, 'valley': 41, 'vegetable_garden': 7, 'veterinarians_office': 203, 'viaduct': 75, 'village': 6, 'vineyard': 11, 'volcano': 25, 'volleyball_court/outdoor': 127, 'waiting_room': 54, 'water_park': 3, 'water_tower': 307, 'waterfall': 0, 'watering_hole': 1050, 'wave': 750, 'wet_bar': 282, 'wheat_field': 275, 'wind_farm': 251, 'windmill': 285, 'yard': 147, 'youth_hostel': 69, 'zen_garden': 10}
        
        # 获取所有场景类别列表
        scene_all = list(probs.keys())
        # 初始化场景到对象的映射字典（每个场景对应一个对象列表）
        scene2obj = {_: [] for _ in scene_all}
        
        # 遍历所有图像，提取描述和对象信息
        for i, image_id in enumerate(original_id):
            # 获取该图像的所有描述标注ID
            caption_ids = coco_caps.getAnnIds(imgIds=image_id)
            # 加载描述标注（通常每张图像有多个描述）
            captions = coco_caps.loadAnns(caption_ids)
            
            # 获取该图像的所有对象检测标注ID
            ann_ids = coco.getAnnIds(imgIds=image_id)
            # 加载对象检测标注
            anns = coco.loadAnns(ann_ids)
            
            # 提取图像中所有对象的类别名称
            classes = []
            for ann in anns:
                coco_category_id = ann["category_id"]
                coco_class_name = category_map.get(coco_category_id)
                classes.append(coco_class_name)
            
            # 构建该图像的数据字典
            data = {
                'text': captions[0]['caption'],  # 使用第一个描述作为文本
                'scene': allid[image_id],  # 从预加载的场景分类结果中获取场景类别
                'classes': list(set(classes))  # 去重后的对象类别列表
            }
            
            # 更新场景到对象的映射：将该图像中的对象添加到对应场景的对象列表中
            v = list(set(scene2obj[allid[image_id]] + list(classes)))
            scene2obj[allid[image_id]] = v
            
            # 添加到总列表
            all_captions.append(data)
    elif args.dataset == 'voc':
        # ========== VOC 数据集处理 ==========
        categories_file = './results/categories_places365.txt'
        voc_blip_dir = './results/voc_blip'
        characteristic_file = './results/characteristic/voc_all'
        print("📁 使用 results/ 数据集")
        
        # 加载 Places365 场景类别文件
        with open(categories_file) as f:
            # 从每行中提取场景类别名称（跳过前3个字符，如 "/a/airfield" -> "airfield"）
            categories = [line.strip().split(' ')[0][3:] for line in f.readlines()]
        
        # 初始化场景到对象的映射字典
        scene2obj = {_: [] for _ in categories}
        
        # VOC 数据集的 BLIP 标注文件目录
        p = voc_blip_dir
        # 获取所有标注文件
        files = [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
        all_captions = []
        
        # 加载预处理的场景分类结果（图像文件名到场景类别的映射）
        allid = torch.load(characteristic_file)
        
        # 遍历所有标注文件
        for fi in files:
            p2 = join(p, fi)
            with open(p2, 'rb') as f:
                # 从文件名中提取图像标识（去掉前缀）
                ff = "_".join(fi.split("_")[1:])
                # 获取该图像对应的场景类别
                xx = allid[ff]
                
                # 加载 pickle 格式的标注数据（包含描述和对象信息）
                x = pickle.load(f)
                
                # 构建数据字典
                d = {
                    'text': x['cap'],  # BLIP 生成的图像描述
                    'classes': x['obj'],  # 图像中的对象类别列表
                    'scene': xx  # 场景类别
                     }
                
                # 更新场景到对象的映射
                v = list(set(scene2obj[xx] + list(x['obj'])))
                scene2obj[xx] = v
                
                # 添加到总列表
                all_captions.append(d)
                

    else:
        raise ValueError
    
    return all_captions,scene2obj

def save_predict(img_path, results, fn, output_img_dir, label_gen_path):
    """
    保存目标检测结果：生成可视化图像和 YOLO 格式的标注文件
    
    功能：
        1. 对检测结果应用多个置信度阈值（0.3-0.9）
        2. 为每个阈值生成带检测框的可视化图像
        3. 生成 YOLO 格式的标注文件（类别ID + 归一化的边界框坐标）
    
    参数:
        img_path: 输入图像路径
        results: MMDetection 的检测结果对象
        fn: 文件名（不含扩展名）
        output_img_dir: 输出图像目录
        label_gen_path: 输出标注文件目录
    
    输出文件格式:
        - 可视化图像: {output_dir}/t-{threshold}/{fn}-PRED.jpg
        - 标注文件: {label_dir}/t-{threshold}/{fn}-GEN.txt
          标注格式: "class_id x_center y_center width height" (所有值归一化到 [0,1])
    """
    # 定义多个置信度阈值，用于生成不同严格程度的检测结果
    target_conf = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    
    # 对每个置信度阈值分别处理
    for tc in target_conf:
        # 创建输出目录：可视化图像目录和标注文件目录
        odir = join(output_img_dir, 't-{}'.format(tc))  # 可视化图像目录
        lbdir = join(label_gen_path, 't-{}'.format(tc))  # 标注文件目录
        os.makedirs(odir, exist_ok=True)
        os.makedirs(lbdir, exist_ok=True)

        # 读取图像
        img = mmcv.imread(img_path)
        # 创建 matplotlib 图形用于绘制检测框
        plt.figure(figsize=(8, 8))
        plt.imshow(mmcv.bgr2rgb(img))  # 转换为 RGB 格式显示
        plt.axis('off')

        # 存储标注文本（YOLO 格式）
        msg = []
        
        # 从检测结果中提取信息
        instances = results.pred_instances  # InstanceData 对象
        boxes = instances.bboxes.cpu().numpy()  # 边界框坐标，形状 (N, 4)，格式 [x1, y1, x2, y2]
        scores = instances.scores.cpu().numpy()  # 置信度分数，形状 (N,)
        labels = instances.labels.cpu().numpy()  # 类别ID，形状 (N,)
        
        # 遍历所有检测结果
        for box, conf, cls_id in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
            
            # 如果置信度低于当前阈值，跳过该检测
            if conf < tc:
                    continue

            # 在图像上绘制检测框（红色边框）
                plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                                edgecolor='red', facecolor='none', linewidth=2))
            # 在检测框左上角添加类别ID和置信度文本
                plt.text(x1, y1, f'{cls_id}: {conf:.2f}', color='white',
                        bbox=dict(facecolor='red', edgecolor='none', pad=1))

            # 转换为 YOLO 格式：计算归一化的中心点坐标和宽高
            x_center = (x1 + x2) / 2.0  # 边界框中心 x 坐标
            y_center = (y1 + y2) / 2.0  # 边界框中心 y 坐标
            obj_w = x2 - x1  # 边界框宽度
            obj_h = y2 - y1  # 边界框高度

            # 归一化到图像尺寸（YOLO 格式要求所有坐标在 [0, 1] 范围内）
                h, w = img.shape[:2]
            # YOLO 格式：class_id x_center y_center width height（全部归一化）
                m = f'{cls_id} {x_center / w:.6f} {y_center / h:.6f} {obj_w / w:.6f} {obj_h / h:.6f}'
                msg.append(m)
            
        # 保存可视化图像
        plt.axis('off')
        ff = '{}/{}-PRED.jpg'.format(odir, fn)
        plt.savefig(ff)
        plt.close()

        # 保存 YOLO 格式的标注文件（每行一个检测框）
        label_f = open(join(lbdir, '{}-GEN.txt'.format(fn)), 'w')
        label_f.write('\n'.join(msg))
        label_f.close()


def gen_04():
    """
    主函数：完整的图像生成与检测流水线
    
    工作流程：
        1. 初始化 OpenAI API 客户端
        2. 根据参数确定数据集类型和微调配置
        3. 加载 Stable Diffusion 3 模型（可选加载 LoRA 权重）
        4. 加载 Faster R-CNN 目标检测模型
        5. 从数据集加载图像描述和场景-对象映射
        6. 对每个图像：
           a. 随机选择一个描述
           b. 根据场景获取可能的对象列表
           c. 使用 GPT 模型增强描述，添加对象信息
           d. 使用 SD3 生成图像
           e. 使用 Faster R-CNN 检测图像中的对象
           f. 保存图像、标注文件和增强后的描述
    """
    import openai
    
    # ========== 初始化 OpenAI API ==========
    # 定义不同模型的 API 基础 URL
    api_base = {
                "gpt-3.5-turbo": "https://api.openai.com/v1",
                "gpt-3.5": "https://api.openai.com/v1",
                "gpt-4.1": "https://api.openai.com/v1",
                "gpt-5": "https://api.openai.com/v1",
                }   
    
    # ========== 确定数据集类型和微调配置 ==========
    ft = ''  # 微调标识字符串
    
    # 根据微调数据名称自动确定数据集类型
    if 'coco' in args.ftdata:
        args.dataset = 'coco'
    else:
        args.dataset = 'voc'
        
    # 如果指定了微调步数，构建微调标识字符串
    if args.ftstep > 0:
        ft += '{}-ftstep-{}'.format(args.ftdata, args.ftstep)
    
    # ========== 设置 LLM 模型 ==========
    llmmodel = 'gpt-4.1'  # 使用 GPT-4.1 进行提示词增强
    args.llmmodel = llmmodel
    openai.api_base = api_base[args.llmmodel]
    # 创建 OpenAI 客户端
    client = openai.OpenAI(base_url='https://api.openai.com/v1')

    # ========== 构建输出路径 ==========
    cap_model = ''  # 描述模型标识（当前为空）
    
    # 构建输出目录路径，包含所有配置信息
    submsg = './results/sd-release/gen_04/stable-diffusion-3-medium-diffusers/{}{}/{}/{}/maxcap-{}-variou-tc'.format(
                args.dataset, cap_model, ft, llmmodel, args.maxcap)
        
    # 添加推理步数信息
    submsg += '-inferstep-{}'.format(args.inferstep)
    # 如果指定了特殊类型，添加类型和最大对象数信息
    if args.type != 'default':
        submsg += '-{}-{}'.format(args.type, args.maxobj)
    
    # 定义各种输出目录
    output_img_dir = os.path.join(submsg, 'images/train2017')  # 生成的图像目录
    label_gen_path = os.path.join(submsg, 'labels_gen/faster-rcnn/train2017')  # 检测标注目录
    caption_path = os.path.join(submsg, 'caption_merge/')  # 增强后的描述目录
    
    # 创建所有必要的输出目录
    for f in [output_img_dir, label_gen_path, caption_path]:
        os.makedirs(f, exist_ok=True)

    # ========== 加载 Stable Diffusion 3 模型 ==========
    print('Load SD')
    # 从 HuggingFace 加载 SD3 Medium 模型，使用半精度浮点数以节省显存
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    print('Load SD done')
    # 将模型移动到 GPU
    # 修改cuda为cpu/mps（Mac M1 Pro 使用 mps）
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = pipe.to(device)
    # 启用 CPU 卸载以节省显存（将不使用的模型组件移到 CPU）
    # 注意：MPS 模式下可能不需要或支持 CPU 卸载，如果出错可以注释掉
    if device == "cpu":
    pipe.enable_model_cpu_offload()
    pipe = pipe.to(device)
    
    # ========== 加载 LoRA 微调权重（如果指定） ==========
    if args.ftstep > 0:
        if args.dataset == 'voc':
            if args.ftdata == 'voc10k':
                # 加载 VOC 10k 数据集微调的 LoRA 权重
                pipe.load_lora_weights('./results/ftvoc/pytorch_lora_weights.safetensors')
            elif args.ftdata == 'voc10kmerge':
                # 加载 VOC 10k 合并数据集的 LoRA 权重
                pipe.load_lora_weights('xxx/pytorch_lora_weights.safetensors')
            else:
                raise ValueError(f"Unknown ftdata for VOC: {args.ftdata}")
        elif args.dataset == 'coco':
            if args.ftdata == 'coco20k':
                # 加载 COCO 20k 数据集微调的 LoRA 权重
                pipe.load_lora_weights('xx')
            elif args.ftdata == 'coco20kmerge':
                # 加载 COCO 20k 合并数据集的 LoRA 权重
                pipe.load_lora_weights('xs')
            else:
                raise ValueError(f"Unknown ftdata for COCO: {args.ftdata}")

    print('Here')

    # ========== 加载目标检测模型 ==========
    print('Init Detector')
    # 修改cuda为cpu/mps（Mac M1 Pro 使用 mps，但 MMDetection 可能不支持 mps，使用 cpu）
    # 注意：MMDetection 在 Mac 上通常只支持 CPU，MPS 支持可能不完整
    detector_device = "cpu"  # MMDetection 在 Mac 上建议使用 CPU
    if args.dataset == 'voc':
        # VOC 数据集的 Faster R-CNN 配置和权重
        config_file = './configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712.py'
        # 优先使用本地 dataset/voc/ 路径，如果不存在则使用 results/ 路径
        if os.path.exists('./dataset/voc/pretrain/faster-rcnn/faster_rcnn_r50_fpn_1x_voc0712.pth'):
            checkpoint_file = './dataset/voc/pretrain/faster-rcnn/faster_rcnn_r50_fpn_1x_voc0712.pth'
        else:
        checkpoint_file = './results/pretrain/voc/faster-rcnn/faster_rcnn_r50_fpn_1x_voc0712.pth'
        # model = init_detector(config_file, checkpoint_file, device='cuda:0')  # 原 CUDA 代码
        model = init_detector(config_file, checkpoint_file, device=detector_device)
    elif args.dataset == 'coco':
        # COCO 数据集的 Faster R-CNN 配置和权重
        config_file = 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
        checkpoint_file = './results/pretrain/coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        # model = init_detector(config_file, checkpoint_file, device='cuda:0')  # 原 CUDA 代码
        model = init_detector(config_file, checkpoint_file, device=detector_device)

    print('Init Detector Done')
    
    # 数据集名称映射（用于提示词）
    d = {'coco': 'COCO', 'voc': 'VOC'}

    # ========== 加载数据集描述和场景-对象映射 ==========
    all_captions, scene2obj = get_cap()
    
    # ========== 主循环：生成图像并检测 ==========
    for i in range(args.begin, args.end):
        # 设置随机种子以确保可重复性
        random.seed(i)
        # 生成输出文件名
        fn = '{}-GEN.jpg'.format(i+1)
        
        # 从所有描述中随机选择一个
        selected = random.sample(all_captions, 1)
        # 提取描述文本
        captions = [s['text'] for s in selected]
        print(captions)

        # ========== 获取可能的对象列表 ==========
        # 根据选中图像的场景，获取该场景中可能出现的所有对象
        possible_obj = [scene2obj[_['scene']] for _ in selected]
        # 将嵌套列表展平
        possible_obj = list(chain.from_iterable(possible_obj))
        # 如果对象太多，随机选择10个（避免提示词过长）
        if len(possible_obj) > 10:
            possible_obj = random.sample(possible_obj, 10)
        # 将对象列表转换为逗号分隔的字符串
        possible_obj = ", ".join(possible_obj)
        
        # ========== 构建 LLM 提示词（4象限版本） ==========
        # 系统提示词：定义 GPT 的角色和任务，要求生成4个象限的描述
        system_msg = (
            "You are a caption enhancement assistant specialized in generating object-rich image prompts for Stable Diffusion, "
            f"with a focus on datasets like {d[args.dataset]}. Your task is to create a 4-quadrant image composition by enriching a given caption "
            "and incorporating a provided list of possible objects. "
            "You must generate exactly 4 separate captions, one for each quadrant (Top Left, Top Right, Bottom Left, Bottom Right). "
            "Each caption should describe a coherent scene that fits naturally in its quadrant position. "
            "Distribute the objects from the provided list across the 4 quadrants in a logical and visually appealing way. "
            "Each caption should be a single, fluent sentence (ideally under 30 words), suitable for Stable Diffusion image generation. "
            "Format your response exactly as follows:\n"
            "Caption:\n"
            "Top Left: [description]\n"
            "Top Right: [description]\n"
            "Bottom Left: [description]\n"
            "Bottom Right: [description]"
            )

        # 构建用户提示词：包含原始描述和可能的对象列表
        caption_blocks = [f"Caption {i+1}:\n{captions[i]}\n" for i in range(1)]
        caption_text = "\n".join(caption_blocks)
        
        user_prompt = (
            f"Original Caption:\n{caption_text}\n\n"
            f"Here is a list of possible objects to consider: {possible_obj}. "
            f"Please create 4 quadrant captions by distributing these objects across Top Left, Top Right, Bottom Left, and Bottom Right quadrants. "
            "Each quadrant should have a coherent scene description that naturally incorporates some of the objects from the list. "
            "The 4 quadrants should work together to form a complete, visually interesting composition."
        )

        # ========== 使用 GPT 生成4象限描述 ==========
        rewrite_caption = get_llm_output(client, system_msg, user_prompt, args.llmmodel)
        print("原始描述:", captions)
        print("可能对象:", possible_obj)
        print("LLM生成的4象限描述:\n", rewrite_caption)
        
        # ========== 解析4象限描述 ==========
        quadrant_prompts = parse_quadrant_captions(rewrite_caption)
        print("解析后的象限描述:")
        for key, prompt in quadrant_prompts.items():
            print(f"  {key}: {prompt}")
        
        # ========== 为每个象限生成图像 ==========
        print("正在生成4个象限的图像...")
        top_left_img = gen_quadrant_image(pipe, quadrant_prompts['top_left'], args.inferstep, seed_offset=0)
        top_right_img = gen_quadrant_image(pipe, quadrant_prompts['top_right'], args.inferstep, seed_offset=1)
        bottom_left_img = gen_quadrant_image(pipe, quadrant_prompts['bottom_left'], args.inferstep, seed_offset=2)
        bottom_right_img = gen_quadrant_image(pipe, quadrant_prompts['bottom_right'], args.inferstep, seed_offset=3)
        
        # ========== 拼接4个象限成完整图像 ==========
        print("正在拼接4个象限...")
        combined_image = combine_quadrants(top_left_img, top_right_img, bottom_left_img, bottom_right_img)
        
        # ========== 保存拼接后的完整图像 ==========
        combined_image.save(join(output_img_dir, fn))
        print(f"已保存完整图像: {fn}")
        
        # ========== 对拼接后的完整图像进行目标检测 ==========
        print("正在进行目标检测...")
        results = inference_detector(model, join(output_img_dir, fn))
        # 保存检测结果（可视化图像和标注文件）
        save_predict(join(output_img_dir, fn), results, '{}'.format(i+1), output_img_dir, label_gen_path)
        
        # ========== 保存增强后的描述（包含4象限描述） ==========
        with open(join(caption_path, '{}.txt'.format(i+1)), 'w') as f:
            f.write("Caption:\n")
            f.write(f"Top Left: {quadrant_prompts['top_left']}\n")
            f.write(f"Top Right: {quadrant_prompts['top_right']}\n")
            f.write(f"Bottom Left: {quadrant_prompts['bottom_left']}\n")
            f.write(f"Bottom Right: {quadrant_prompts['bottom_right']}\n")


# ========== 命令行参数解析 ==========
parser = argparse.ArgumentParser(description='Stable Diffusion 3 图像生成与目标检测流水线')

# 基本参数
parser.add_argument("--maxcap", type=int, default=4, help="最大描述数量")
parser.add_argument("--begin", type=int, default=0, help="开始生成的图像索引")
parser.add_argument("--end", type=int, default=1, help="结束生成的图像索引（不包含）")
parser.add_argument("--randomsample", type=int, default=50, help="随机采样数量")
parser.add_argument("--llmmodel", type=str, default='gpt-3.5-turbo', help="LLM 模型名称（gpt-3.5-turbo, gpt-4.1 等）")
parser.add_argument("--topk", type=int, default=5, help="Top-K 采样参数")
parser.add_argument("--cluster", type=int, default=200, help="聚类数量")
parser.add_argument("--rebalance", type=str2bool, default=False, help="是否重新平衡数据")
parser.add_argument("--drop_prob", type=float, default=0.5, help="丢弃概率")
parser.add_argument("--maxobj", type=int, default=2, help="最大对象数量")
parser.add_argument("--inferstep", type=int, default=50, help="SD3 推理步数（越多质量越好但速度越慢）")
parser.add_argument("--ftstep", type=int, default=0, help="微调步数（0表示不使用微调）")
parser.add_argument("--pretrainstep", type=int, default=1000, help="预训练步数")
parser.add_argument("--capaug", type=str2bool, default=False, help="是否使用描述增强")
parser.add_argument("--dataset", type=str, default='coco', help="数据集类型（coco 或 voc）")

# 微调相关参数
parser.add_argument("--ftdata", type=str, default='voc10k', help="微调数据名称（voc10k, coco20k 等）")
parser.add_argument("--ncluster", type=int, default=128, help="聚类数量")

# 高级参数
parser.add_argument("--alpha", type=float, default=1.0, help="Alpha 参数")
parser.add_argument("--beta", type=float, default=1.0, help="Beta 参数")
parser.add_argument("--sampled", type=int, default=0, help="采样数量")
parser.add_argument("--method", type=str, default='topk', help="采样方法")
parser.add_argument("--k", type=int, default=0, help="K 值参数")
parser.add_argument("--portion", type=float, default=1.0, help="数据比例")
parser.add_argument("--lam", type=float, default=1.0, help="Lambda 参数")
parser.add_argument("--enrich", type=str2bool, default=False, help="是否启用增强")
parser.add_argument("--partition", type=str2bool, default=False, help="是否分区")
parser.add_argument("--cross", type=str2bool, default=False, help="是否交叉验证")
parser.add_argument("--sdft", type=str, default='coco', help="SD 微调数据集")
parser.add_argument("--type", type=str, default='default', help="生成类型")
parser.add_argument("--botk", type=int, default=10, help="Bottom-K 参数")

# 模型相关参数
parser.add_argument("--sd15", type=str2bool, default=False, help="是否使用 SD 1.5（当前使用 SD3）")
parser.add_argument("--lorarank", type=int, default=12, help="LoRA 秩（rank）")

# 解析参数
args = parser.parse_args()

# ========== 执行主函数 ==========
gen_04()


'''
# 示例代码（已注释）
# from diffusers import StableDiffusion3Pipeline
# import torch
# import time
# pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
# # 修改cuda为cpu/mps（Mac M1 Pro 使用 mps）
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# pipe = pipe.to(device)
# if device == "cpu":
#     pipe.enable_model_cpu_offload()
# 
# # 修改cuda为cpu/mps（Mac M1 Pro 使用 mps）
# generator = torch.Generator(device).manual_seed(int(time.time() * 1000) % 100000)

prompt='Busy city street with 6 persons, 3 cars, 1 bus, 2 bicycles. Person left of bicycle; car in front of bus; person partially occluded by car. Small and large instances, street-level, mid-shot. Crosswalk and storefronts in background. Overcast daylight, motion blur on legs; one person cropped at right edge'

out = pipe(prompt,num_inference_steps=50, generator=generator)

out.images[0].save('a.jpg')


prompt='A modern Christian poster design featuring a peaceful sunrise or sunset sky with soft pastel colors, rays of light shining through gentle clouds. A large white cross on the top-right side. Minimalist, inspirational, divine atmosphere, digital art, high resolution.'

prompt = 'A modern Christian poster with a serene, inspirational feel. The background features a beautiful pastel-colored sky during sunrise, soft golden sunlight breaking through fluffy clouds, creating radiant beams of light from the horizon. On the right side, a clean white cross glows subtly with a slight shadow for depth. On the left, bold, elegant Vietnamese text in a modern sans-serif font reads: "ĐỨC CHÚA TRỜI ĐÃ SAI CON MỘT NGÀI ĐẾN THẾ-GIAN, ĐẶNG CHÚNG TA NHỜ CON ĐƯỢC SỐNG". The text is navy blue with smooth edges and balanced spacing. The overall design is minimalist yet reverent, spiritual and uplifting, in a digital art style, ultra high resolution, perfect lighting, no clutter.'


prompt = "A modern Christian digital poster radiating hope and divine purpose. The background features a vivid, majestic sunrise over an open landscape — brilliant golden light bursting through layers of luminous, multi-toned clouds in the sky. Ethereal rays extend toward the horizon, symbolizing a bright future and spiritual awakening. In the distance, a glowing city of light or celestial architecture can be subtly seen, representing the Kingdom of God or divine destiny. Soft rolling hills and flowering fields gently lead the eye toward a glowing white cross on the right foreground, standing firm on a sunlit hill. Light particles shimmer in the air like holy embers. The entire scene is illuminated with warm, heavenly light, conveying clarity, purpose, and future hope. The composition is highly inspirational, spiritual, futuristic, and visually striking. Digital painting style, ultra high resolution, cinematic lighting, divine realism, 4K."

prompt='A powerful, spiritually symbolic Christian poster in a modern digital art style. The background shows a radiant, heavenly sky at dawn — brilliant golden light pouring through parted clouds, forming rays that stretch across the scene, representing divine presence. High above, a descending dove glows softly, symbolizing the Holy Spirit. In the center-right, a large white cross stands firmly on a grassy hill, bathed in holy light. At its base, a small, open Bible rests on a stone, its pages glowing with divine illumination. In the distance, the faint outline of the New Jerusalem — golden towers and shining gates — emerges from the horizon, suggesting the eternal promise of salvation. A narrow, glowing path leads from the viewer’s perspective toward the cross and onward to the heavenly city, symbolizing the journey of faith. Small, ethereal beams of light rise from the earth, representing prayers and the souls being guided. The overall design is layered, deeply symbolic, visionary, filled with hope and spiritual depth. Ultra high resolution, cinematic lighting, divine realism, perfect for digital posters.'



'''