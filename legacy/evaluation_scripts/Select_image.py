import os
import random
import shutil

def select_images_for_experiment(a_dir, b_dir, output_dir, num_per_phase=50):
    """
    从前时相(A)和后时相(B)中各选指定数量的图片，复制到输出目录并标记来源
    
    参数:
        a_dir: 前时相图片目录
        b_dir: 后时相图片目录
        output_dir: 输出目录
        num_per_phase: 从每个时相选择的图片数量(默认为50)
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取A和B目录中的图片文件列表
    a_images = [f for f in os.listdir(a_dir) if f.endswith(('.tif', '.png', '.jpg'))]
    b_images = [f for f in os.listdir(b_dir) if f.endswith(('.tif', '.png', '.jpg'))]
    
    # 随机选择指定数量的图片
    selected_a = random.sample(a_images, num_per_phase)
    selected_b = random.sample(b_images, num_per_phase)
    
    # 复制并重命名A时相的图片
    for img in selected_a:
        src = os.path.join(a_dir, img)
        dst = os.path.join(output_dir, f"A_{img}")
        shutil.copy2(src, dst)
    
    # 复制并重命名B时相的图片
    for img in selected_b:
        src = os.path.join(b_dir, img)
        dst = os.path.join(output_dir, f"B_{img}")
        shutil.copy2(src, dst)
    
    print(f"已从A时相选择{len(selected_a)}张图片，从B时相选择{len(selected_b)}张图片")
    print(f"总计{len(selected_a) + len(selected_b)}张图片已保存到{output_dir}")

# 使用示例
if __name__ == "__main__":
    # 设置路径
    a_dir = "D:/Deeplearning/Change_Agent_main/sampled/LEVIR256_sample_100/A"      # 前时相图片目录
    b_dir = "D:/Deeplearning/Change_Agent_main/sampled/LEVIR256_sample_100/B"      # 后时相图片目录
    output_dir = "D:/Deeplearning/Change_Agent_main/sampled/LEVIR256_sample_100/output"  # 输出目录
    
    # 执行选择
    select_images_for_experiment(a_dir, b_dir, output_dir)