import os
import csv
import json


def process_files(root_dir, output_file, output_format='csv'):
    data = []

    # 遍历根目录及其子目录
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)

        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(class_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        file_content = file.read()
                        data.append({
                            'file_path': file_path,
                            'caption': file_content,
                            'class_name': class_name,
                            'file_name': filename[:-4]  # 去掉文件扩展名.txt
                        })

    # 将数据保存到csv或json文件中
    if output_format == 'csv':
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['file_path', 'caption', 'class_name', 'file_name']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    elif output_format == 'json':
        with open(output_file, 'w', encoding='utf-8') as jsonfile:
            json.dump(data, jsonfile, ensure_ascii=False, indent=4)
    else:
        raise KeyError


if __name__ == '__main__':
    process_files(root_dir='/discobox/wjpeng/dataset/S-MiT/transcriptions',
                  output_file='S-MiT_source_captions.csv',
                  output_format='csv')

    process_files(root_dir='/discobox/wjpeng/dataset/S-MiT/transcriptions',
                  output_file='S-MiT_source_captions.json',
                  output_format='json')
