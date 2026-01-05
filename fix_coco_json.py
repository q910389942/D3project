"""
修复 COCO 格式 JSON 文件，添加缺失的 'info' 字段
"""
import json
import os
from datetime import datetime

def fix_coco_json(json_file, dataset_name=None):
    """
    修复 COCO JSON 文件，添加缺失的 'info' 字段
    
    Args:
        json_file: JSON 文件路径
        dataset_name: 数据集名称（用于生成描述信息）
    """
    if not os.path.exists(json_file):
        print(f"文件不存在: {json_file}")
        return False
    
    # 读取现有 JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 检查是否已有 'info' 字段
    if 'info' in coco_data:
        print(f"✓ {json_file} 已包含 'info' 字段，跳过")
        return True
    
    # 确定数据集名称
    if dataset_name is None:
        # 从文件路径推断
        if 'train' in json_file.lower():
            dataset_name = 'train'
        elif 'test' in json_file.lower():
            dataset_name = 'test'
        elif 'val' in json_file.lower():
            dataset_name = 'val'
        else:
            dataset_name = 'dataset'
    
    # 添加 'info' 字段
    coco_data['info'] = {
        'description': f'{dataset_name} dataset',
        'version': '1.0',
        'year': 2026,
        'contributor': 'MMDetection',
        'date_created': datetime.now().strftime('%Y/%m/%d')
    }
    
    # 确保字段顺序正确（info 应该在第一位）
    fixed_data = {
        'info': coco_data['info'],
        'images': coco_data.get('images', []),
        'annotations': coco_data.get('annotations', []),
        'categories': coco_data.get('categories', [])
    }
    
    # 保存修复后的 JSON
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 已修复: {json_file}")
    return True


if __name__ == '__main__':
    # 修复所有找到的 annotations.json 文件
    json_files = [
        'dataset/ToMMD/test/annotations.json',
        'dataset/ToMMD/train/annotations.json',
    ]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 60)
    print("修复 COCO JSON 文件，添加缺失的 'info' 字段")
    print("=" * 60)
    
    for json_file in json_files:
        full_path = os.path.join(script_dir, json_file)
        if os.path.exists(full_path):
            fix_coco_json(full_path)
        else:
            print(f"⚠ 文件不存在: {full_path}")
    
    print("\n" + "=" * 60)
    print("修复完成！")
    print("=" * 60)


