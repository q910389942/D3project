"""
训练脚本 - 使用虚拟数据集训练，VOC2007_Test作为测试集

流程：
1. 将虚拟数据集（YOLO格式）转换为COCO格式，保存到 ./dataset/ToMMD
2. 将VOC2007_Test（VOC XML格式）转换为COCO格式
3. 使用虚拟数据集训练，VOC2007_Test测试评估
4. 定期评估并保存指标到CSV
5. 保存最终模型
"""

import os
import json
import re
import random
import xml.etree.ElementTree as ET
from datetime import datetime
from PIL import Image
import pandas as pd
import shutil
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.hooks import Hook
from mmdet.apis import init_detector

# ========== 路径配置 ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 虚拟数据集路径（YOLO格式）
# 使用带标注框的图像（t-0.5目录下的PRED图像）
VIRTUAL_IMAGES_DIR = os.path.join(SCRIPT_DIR, '../../D3/results/sd-release/gen_04/stable-diffusion-3-medium-diffusers/voc/gpt-4.1/maxcap-4-variou-tc-inferstep-50/images/train2017/t-0.5')
VIRTUAL_LABELS_DIR = os.path.join(SCRIPT_DIR, '../../D3/results/sd-release/gen_04/stable-diffusion-3-medium-diffusers/voc/gpt-4.1/maxcap-4-variou-tc-inferstep-50/labels_gen/faster-rcnn/train2017/t-0.5')

# 输出目录
DATASET_TOMMD_DIR = os.path.join(SCRIPT_DIR, 'dataset/ToMMD')
VOC_TEST_DIR = os.path.join(SCRIPT_DIR, 'dataset/VOC2007_Test')
DETECTION_EVAL_DIR = os.path.join(SCRIPT_DIR, 'detection_eval')
DETECTION_MODEL_DIR = os.path.join(SCRIPT_DIR, 'detection_model')

# VOC类别
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 类别名称到ID的映射
CLASS_NAME_TO_ID = {name: i for i, name in enumerate(VOC_CLASSES)}


def yolo_to_coco(images_dir, labels_dir, output_dir, dataset_name='train'):
    """
    将YOLO格式转换为COCO格式
    
    Args:
        images_dir: 图像目录
        labels_dir: YOLO标注目录
        output_dir: 输出目录（COCO格式数据集将保存在这里）
        dataset_name: 数据集名称（用于创建子目录）
    
    Returns:
        coco_annotation_file: COCO格式标注文件路径
        images_output_dir: 图像输出目录
    """
    print(f"\n{'='*60}")
    print(f"正在转换虚拟数据集（YOLO -> COCO）...")
    print(f"{'='*60}")
    
    # 创建输出目录结构
    images_output_dir = os.path.join(output_dir, dataset_name, 'images')
    os.makedirs(images_output_dir, exist_ok=True)
    
    coco_data = {
        'info': {
            'description': f'{dataset_name} dataset converted from YOLO format',
            'version': '1.0',
            'year': 2026,
            'contributor': 'MMDetection',
            'date_created': datetime.now().strftime('%Y/%m/%d')
        },
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    # 添加类别
    for i, name in enumerate(VOC_CLASSES):
        coco_data['categories'].append({
            'id': i,
            'name': name,
            'supercategory': 'none'
        })
    
    # 读取图像和标注
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    ann_id = 0
    
    for img_id, img_file in enumerate(image_files):
        img_path = os.path.join(images_dir, img_file)
        
        # 从图像文件名中提取数字ID（例如：从"1-PRED.jpg"提取"1"）
        # 支持格式：1-PRED.jpg, 1-GEN.jpg, 1.jpg等
        match = re.search(r'(\d+)', os.path.splitext(img_file)[0])
        if match:
            image_id = match.group(1)
            # 查找对应的标注文件（1-GEN.txt格式）
            label_file = os.path.join(labels_dir, f'{image_id}-GEN.txt')
        else:
            # 如果无法提取ID，尝试直接替换扩展名
            label_file = os.path.join(labels_dir, os.path.splitext(img_file)[0].replace('-PRED', '-GEN') + '.txt')
        
        # 读取图像尺寸
        try:
            img = Image.open(img_path)
            width, height = img.size
        except Exception as e:
            print(f"警告: 无法读取图像 {img_file}: {e}")
            continue
        
        # 复制图像到输出目录
        img_output_path = os.path.join(images_output_dir, img_file)
        shutil.copy2(img_path, img_output_path)
        
        coco_data['images'].append({
            'id': img_id,
            'file_name': img_file,
            'width': width,
            'height': height
        })
        
        # 读取YOLO标注
        ann_count = 0
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        w_norm = float(parts[3])
                        h_norm = float(parts[4])
                        
                        # YOLO格式转换为COCO格式（左上角坐标+宽高）
                        x = (x_center - w_norm / 2) * width
                        y = (y_center - h_norm / 2) * height
                        w = w_norm * width
                        h = h_norm * height
                        
                        # 确保坐标有效
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, width - x)
                        h = min(h, height - y)
                        
                        if w > 0 and h > 0:
                            coco_data['annotations'].append({
                                'id': ann_id,
                                'image_id': img_id,
                                'category_id': cls_id,
                                'bbox': [x, y, w, h],
                                'area': w * h,
                                'iscrowd': 0
                            })
                            ann_id += 1
                            ann_count += 1
            if ann_count > 0:
                print(f"  ✓ {img_file}: {ann_count} 个标注")
        else:
            print(f"  ✗ 警告: 标注文件不存在: {os.path.basename(label_file)} (图像: {img_file})")
    
    # 保存COCO格式标注文件
    coco_annotation_file = os.path.join(output_dir, dataset_name, 'annotations.json')
    with open(coco_annotation_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\n✓ 转换完成:")
    print(f"  - 图像数量: {len(coco_data['images'])} 张")
    print(f"  - 标注数量: {len(coco_data['annotations'])} 个")
    print(f"  - 保存位置: {output_dir}/{dataset_name}")
    
    return coco_annotation_file, images_output_dir


def voc_xml_to_coco(voc_dir, output_dir, dataset_name='test', max_samples=30):
    """
    将VOC XML格式转换为COCO格式
    
    Args:
        voc_dir: VOC数据集根目录（包含Annotations, JPEGImages, ImageSets）
        output_dir: 输出目录
        dataset_name: 数据集名称
        max_samples: 最大样本数量（随机抽取），如果为None则使用全部样本
    
    Returns:
        coco_annotation_file: COCO格式标注文件路径
        images_output_dir: 图像输出目录
    """
    print(f"\n{'='*60}")
    print(f"正在转换VOC2007_Test（VOC XML -> COCO）...")
    if max_samples is not None:
        print(f"将随机抽取 {max_samples} 个样本")
    print(f"{'='*60}")
    
    # 创建输出目录
    images_output_dir = os.path.join(output_dir, dataset_name, 'images')
    os.makedirs(images_output_dir, exist_ok=True)
    
    # 读取测试集图像ID列表
    test_txt = os.path.join(voc_dir, 'ImageSets', 'Main', 'test.txt')
    if not os.path.exists(test_txt):
        raise FileNotFoundError(f"测试集列表文件不存在: {test_txt}")
    
    with open(test_txt, 'r') as f:
        all_image_ids = [line.strip() for line in f.readlines()]
    
    # 随机抽取指定数量的样本
    if max_samples is not None and len(all_image_ids) > max_samples:
        random.seed(42)  # 设置随机种子以确保可重复性
        image_ids = random.sample(all_image_ids, max_samples)
        print(f"从 {len(all_image_ids)} 个样本中随机抽取了 {len(image_ids)} 个")
    else:
        image_ids = all_image_ids
        print(f"使用全部 {len(image_ids)} 个样本")
    
    coco_data = {
        'info': {
            'description': f'{dataset_name} dataset converted from VOC XML format',
            'version': '1.0',
            'year': 2026,
            'contributor': 'MMDetection',
            'date_created': datetime.now().strftime('%Y/%m/%d')
        },
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    # 添加类别
    for i, name in enumerate(VOC_CLASSES):
        coco_data['categories'].append({
            'id': i,
            'name': name,
            'supercategory': 'none'
        })
    
    ann_id = 0
    annotations_dir = os.path.join(voc_dir, 'Annotations')
    images_dir = os.path.join(voc_dir, 'JPEGImages')
    
    for img_id, image_id in enumerate(image_ids):
        xml_file = os.path.join(annotations_dir, f'{image_id}.xml')
        img_file = f'{image_id}.jpg'
        img_path = os.path.join(images_dir, img_file)
        
        if not os.path.exists(xml_file):
            print(f"警告: XML文件不存在: {xml_file}")
            continue
        
        if not os.path.exists(img_path):
            print(f"警告: 图像文件不存在: {img_path}")
            continue
        
        # 解析XML
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 获取图像尺寸
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # 复制图像
        img_output_path = os.path.join(images_output_dir, img_file)
        shutil.copy2(img_path, img_output_path)
        
        coco_data['images'].append({
            'id': img_id,
            'file_name': img_file,
            'width': width,
            'height': height
        })
        
        # 解析标注
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in CLASS_NAME_TO_ID:
                continue
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # VOC格式（左上角+右下角）转换为COCO格式（左上角+宽高）
            x = xmin
            y = ymin
            w = xmax - xmin
            h = ymax - ymin
            
            if w > 0 and h > 0:
                coco_data['annotations'].append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': CLASS_NAME_TO_ID[name],
                    'bbox': [x, y, w, h],
                    'area': w * h,
                    'iscrowd': 0
                })
                ann_id += 1
    
    # 保存COCO格式标注文件
    coco_annotation_file = os.path.join(output_dir, dataset_name, 'annotations.json')
    with open(coco_annotation_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\n✓ 转换完成:")
    print(f"  - 图像数量: {len(coco_data['images'])} 张")
    print(f"  - 标注数量: {len(coco_data['annotations'])} 个")
    print(f"  - 保存位置: {output_dir}/{dataset_name}")
    
    return coco_annotation_file, images_output_dir


# 全局变量用于存储评估结果
eval_metrics_history = []


class MetricsSaverHook(Hook):
    """自定义Hook，用于捕获评估结果并保存到CSV"""
    def __init__(self, eval_dir, timestamp):
        super().__init__()
        self.eval_dir = eval_dir
        self.timestamp = timestamp
        os.makedirs(eval_dir, exist_ok=True)
        self.csv_file = os.path.join(eval_dir, f'eval_metrics_{timestamp}.csv')
    
    def after_val_epoch(self, runner, metrics=None):
        """在每个验证epoch后调用"""
        # 尝试从多个来源获取评估结果
        eval_results = None
        
        # 方法1: 从metrics参数获取
        if metrics is not None:
            eval_results = metrics
        # 方法2: 从message_hub获取（MMDetection常用方式）
        elif hasattr(runner, 'message_hub'):
            # 尝试从message_hub获取最新的评估结果
            if runner.message_hub.get_info('val') is not None:
                eval_results = runner.message_hub.get_info('val')
        
        if eval_results is None:
            # 如果都没有，尝试从日志中提取（最后的手段）
            print("警告: 无法直接获取评估结果，将尝试从日志中提取")
            return
        
        # 提取关键指标
        metric_dict = {
            'epoch': runner.epoch + 1,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 从eval_results中提取指标（可能是字典或列表）
        if isinstance(eval_results, dict):
            for key, value in eval_results.items():
                if 'mAP' in key or 'AP' in key:
                    # 简化键名
                    clean_key = key.replace('coco/bbox_', '').replace('coco/', '')
                    metric_dict[clean_key] = value
        elif isinstance(eval_results, list) and len(eval_results) > 0:
            # 如果是列表，取第一个元素
            if isinstance(eval_results[0], dict):
                for key, value in eval_results[0].items():
                    if 'mAP' in key or 'AP' in key:
                        clean_key = key.replace('coco/bbox_', '').replace('coco/', '')
                        metric_dict[clean_key] = value
        
        # 保存到历史记录
        eval_metrics_history.append(metric_dict)
        
        # 保存到CSV
        df = pd.DataFrame(eval_metrics_history)
        df.to_csv(self.csv_file, index=False)
        print(f"✓ 评估指标已保存到: {self.csv_file}")


def evaluate_and_save_metrics(eval_results, eval_dir, timestamp, epoch=None):
    """
    评估模型并保存指标到CSV
    
    Args:
        eval_results: 评估结果字典
        eval_dir: 评估结果保存目录
        timestamp: 时间戳（用于文件命名）
        epoch: 当前epoch（可选）
    """
    os.makedirs(eval_dir, exist_ok=True)
    
    # 提取关键指标
    metrics = {
        'epoch': epoch if epoch is not None else 'final',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 从eval_results中提取指标
    if isinstance(eval_results, dict):
        for key, value in eval_results.items():
            if 'mAP' in key or 'AP' in key:
                clean_key = key.replace('coco/bbox_', '').replace('coco/', '')
                metrics[clean_key] = value
    elif isinstance(eval_results, list) and len(eval_results) > 0:
        if isinstance(eval_results[0], dict):
            for key, value in eval_results[0].items():
                if 'mAP' in key or 'AP' in key:
                    clean_key = key.replace('coco/bbox_', '').replace('coco/', '')
                    metrics[clean_key] = value
    
    # 读取或创建CSV文件
    csv_file = os.path.join(eval_dir, f'eval_metrics_{timestamp}.csv')
    
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
    else:
        df = pd.DataFrame([metrics])
    
    df.to_csv(csv_file, index=False)
    print(f"✓ 评估指标已保存到: {csv_file}")
    
    return csv_file


def train():
    """主训练函数"""
    print("=" * 60)
    print("开始训练流程")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(DATASET_TOMMD_DIR, exist_ok=True)
    os.makedirs(DETECTION_EVAL_DIR, exist_ok=True)
    os.makedirs(DETECTION_MODEL_DIR, exist_ok=True)
    
    # 生成时间戳（用于文件命名）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # ========== 步骤1: 转换虚拟数据集（YOLO -> COCO） ==========
    train_coco_file, train_images_dir = yolo_to_coco(
        VIRTUAL_IMAGES_DIR,
        VIRTUAL_LABELS_DIR,
        DATASET_TOMMD_DIR,
        dataset_name='train'
    )
    
    # ========== 步骤2: 转换VOC2007_Test（VOC XML -> COCO） ==========
    test_coco_file, test_images_dir = voc_xml_to_coco(
        VOC_TEST_DIR,
        DATASET_TOMMD_DIR,
        dataset_name='test'
    )
    
    # ========== 步骤3: 配置MMDetection训练 ==========
    print(f"\n{'='*60}")
    print("配置MMDetection训练...")
    print(f"{'='*60}")
    
    base_config = './configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712.py'
    if not os.path.exists(base_config):
        raise FileNotFoundError(f"配置文件不存在: {base_config}")
    
    cfg = Config.fromfile(base_config)
    
    # 确保模型类别数量正确（VOC有20个类别）
    # 预训练模型是80类（COCO），但我们会用VOC的20类覆盖
    if hasattr(cfg.model, 'roi_head') and 'bbox_head' in cfg.model.roi_head:
        cfg.model.roi_head.bbox_head.num_classes = 20
        print(f"✓ 模型类别数已设置为: 20 (VOC类别)")
    
    # 训练集配置（使用虚拟数据集）
    train_data_root = os.path.dirname(os.path.dirname(train_coco_file))  # dataset/ToMMD
    cfg.train_dataloader.dataset = dict(
        type='CocoDataset',
        data_root=train_data_root,
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/images/'),
        metainfo=dict(classes=VOC_CLASSES),
        pipeline=cfg.train_pipeline,
        filter_cfg=dict(filter_empty_gt=True, min_size=32)
    )
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    
    # 测试集配置（使用VOC2007_Test）
    test_data_root = os.path.dirname(os.path.dirname(test_coco_file))  # dataset/ToMMD
    cfg.val_dataloader.dataset = dict(
        type='CocoDataset',
        data_root=test_data_root,
        ann_file='test/annotations.json',
        data_prefix=dict(img='test/images/'),
        metainfo=dict(classes=VOC_CLASSES),
        test_mode=True,
        pipeline=cfg.test_pipeline
    )
    cfg.val_dataloader.num_workers = 0
    cfg.val_dataloader.persistent_workers = False
    cfg.test_dataloader = cfg.val_dataloader.copy()
    
    # 评估器配置
    cfg.val_evaluator = dict(
        type='CocoMetric',
        ann_file=test_coco_file,
        metric='bbox',
        classwise=True
    )
    cfg.test_evaluator = cfg.val_evaluator.copy()
    
    # 降低检测阈值（对于小数据集，降低阈值可以看到更多检测结果）
    if hasattr(cfg.model, 'test_cfg') and 'rcnn' in cfg.model.test_cfg:
        cfg.model.test_cfg.rcnn.score_thr = 0.001  # 进一步降低到0.001
        print(f"   检测阈值: 0.001 (已降低以便看到更多检测结果)")
    
    # 训练配置
    # 确保train_cfg正确设置
    # 对于小数据集，增加训练轮数有助于模型学习
    max_epochs = 24  # 增加到24个epoch
    if not hasattr(cfg, 'train_cfg') or cfg.train_cfg is None:
        cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=3)
    else:
        cfg.train_cfg.max_epochs = max_epochs
        # 每3个epoch评估一次（通过val_interval控制）
        if hasattr(cfg.train_cfg, 'val_interval'):
            cfg.train_cfg.val_interval = 3
        elif isinstance(cfg.train_cfg, dict):
            cfg.train_cfg['val_interval'] = 3
        else:
            # 如果是对象，尝试设置属性
            try:
                cfg.train_cfg.val_interval = 3
            except:
                pass
    print(f"   训练轮数: {max_epochs} epochs")
    
    # 确保val_cfg和test_cfg存在
    if not hasattr(cfg, 'val_cfg') or cfg.val_cfg is None:
        cfg.val_cfg = dict(type='ValLoop')
    if not hasattr(cfg, 'test_cfg') or cfg.test_cfg is None:
        cfg.test_cfg = dict(type='TestLoop')
    
    cfg.work_dir = os.path.join(SCRIPT_DIR, 'work_dirs', f'train_{timestamp}')
    cfg.device = 'cpu'
    
    # ========== 使用预训练模型（重要！） ==========
    # MMDetection提供了在COCO上预训练的Faster R-CNN模型
    # 对于小数据集，使用预训练模型可以显著提升性能
    # 模型会自动从MMDetection的model zoo下载
    pretrained_model_url = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    cfg.load_from = pretrained_model_url
    print(f"\n⚠️  注意：将使用预训练的Faster R-CNN模型（COCO预训练）")
    print(f"   模型URL: {pretrained_model_url}")
    print(f"   如果下载失败，可以手动下载并设置cfg.load_from为本地路径")
    
    # 调整学习率（使用预训练模型时，应该使用较小的学习率进行微调）
    if hasattr(cfg, 'optim_wrapper') and cfg.optim_wrapper is not None:
        if isinstance(cfg.optim_wrapper, dict):
            if 'optimizer' in cfg.optim_wrapper:
                if 'lr' in cfg.optim_wrapper['optimizer']:
                    original_lr = cfg.optim_wrapper['optimizer']['lr']
                    # 使用更小的学习率进行微调（原来的1/10）
                    cfg.optim_wrapper['optimizer']['lr'] = original_lr * 0.1
                    print(f"   学习率已从 {original_lr} 调整为 {cfg.optim_wrapper['optimizer']['lr']} (微调模式)")
    
    # 日志配置
    cfg.default_hooks = dict(
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=1),
        param_scheduler=dict(type='ParamSchedulerHook'),
        checkpoint=dict(
            type='CheckpointHook',
            interval=1,
            save_best='coco/bbox_mAP',  # 保存最佳模型
            rule='greater'
        ),
        sampler_seed=dict(type='DistSamplerSeedHook'),
        visualization=dict(type='DetVisualizationHook')
    )
    
    # 添加自定义Hook用于保存评估指标
    metrics_saver = MetricsSaverHook(DETECTION_EVAL_DIR, timestamp)
    
    cfg.log_processor = dict(type='LogProcessor', window_size=1, by_epoch=True)
    cfg.log_level = 'INFO'
    
    # 强制CPU
    torch.set_default_device('cpu')
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # ========== 步骤4: 开始训练 ==========
    print(f"\n{'='*60}")
    print("开始训练...")
    print(f"训练集: {train_data_root}/train")
    print(f"测试集: {test_data_root}/test")
    print(f"工作目录: {cfg.work_dir}")
    print(f"训练轮数: {cfg.train_cfg.max_epochs}")
    print(f"每3个epoch评估一次")
    print(f"{'='*60}\n")
    
    runner = Runner.from_cfg(cfg)
    runner.model = runner.model.to('cpu')
    
    # 注册自定义Hook
    runner.register_hook(metrics_saver, priority='LOW')
    
    print(f"训练数据集大小: {len(runner.train_dataloader.dataset)} 张图像")
    print(f"测试数据集大小: {len(runner.val_dataloader.dataset)} 张图像")
    print(f"每个epoch的iteration数: {len(runner.train_dataloader)}\n")
    
    # 训练
    runner.train()
    
    # ========== 步骤5: 最终评估 ==========
    print(f"\n{'='*60}")
    print("进行最终评估...")
    print(f"{'='*60}")
    
    final_eval_results = runner.test()
    
    # 保存最终评估指标
    if final_eval_results:
        evaluate_and_save_metrics(
            final_eval_results[0] if isinstance(final_eval_results, list) else final_eval_results,
            DETECTION_EVAL_DIR,
            timestamp,
            epoch=cfg.train_cfg.max_epochs
        )
    
    # ========== 步骤6: 保存最终模型 ==========
    print(f"\n{'='*60}")
    print("保存最终模型...")
    print(f"{'='*60}")
    
    final_model_path = os.path.join(DETECTION_MODEL_DIR, f'model_{timestamp}.pth')
    os.makedirs(DETECTION_MODEL_DIR, exist_ok=True)
    
    # 保存模型权重
    checkpoint_path = os.path.join(cfg.work_dir, 'epoch_12.pth')
    if os.path.exists(checkpoint_path):
        shutil.copy2(checkpoint_path, final_model_path)
        print(f"✓ 最终模型已保存到: {final_model_path}")
    else:
        # 如果没有找到epoch_12.pth，尝试找最新的checkpoint
        checkpoints = [f for f in os.listdir(cfg.work_dir) if f.endswith('.pth')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: os.path.getmtime(os.path.join(cfg.work_dir, x)))
            shutil.copy2(os.path.join(cfg.work_dir, latest_checkpoint), final_model_path)
            print(f"✓ 最终模型已保存到: {final_model_path} (使用最新checkpoint: {latest_checkpoint})")
        else:
            print("警告: 未找到模型checkpoint文件")
    
    print(f"\n{'='*60}")
    print("训练完成！")
    print(f"评估指标: {DETECTION_EVAL_DIR}/eval_metrics_{timestamp}.csv")
    print(f"最终模型: {final_model_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    train()
