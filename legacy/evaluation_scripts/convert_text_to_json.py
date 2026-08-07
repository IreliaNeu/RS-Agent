import os
import json

def convert_txt_to_jsonl(input_folder, output_file):
    """
    将指定文件夹中的所有txt文件内容转换为JSON Lines格式
    
    参数:
        input_folder: 包含txt文件的文件夹路径
        output_file: 输出的jsonl文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        # 遍历文件夹中的所有文件
        for filename in os.listdir(input_folder):
            if filename.endswith('.txt'):
                file_path = os.path.join(input_folder, filename)
                
                # 读取txt文件内容
                with open(file_path, 'r', encoding='utf-8') as txt_file:
                    content = txt_file.read().strip()
                
                # 创建JSON对象并写入文件
                json_obj = {"Original Caption": content}
                jsonl_file.write(json.dumps(json_obj) + '\n')

# 使用示例
input_folder = 'D:/Deeplearning/Change_Agent_main/sampled_result/WHU256_sample_100'  # 替换为你的txt文件所在文件夹路径
output_file = 'D:/Deeplearning/Change_Agent_main/sampled_result/input_WHU256.jsonl'              # 输出的jsonl文件名

convert_txt_to_jsonl(input_folder, output_file)
print(f"转换完成，结果已保存到 {output_file}")