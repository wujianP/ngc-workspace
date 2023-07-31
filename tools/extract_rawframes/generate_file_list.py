import os


def collect_image_paths(folder_path):
    image_paths = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.startswith("img_") and filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            image_paths.append(image_path)
        else:
            raise FileNotFoundError
    return image_paths


def process_path_file(path_file, output_file):
    with open(path_file, 'r') as f:
        folder_paths = f.readlines()

    with open(output_file, 'w') as output:
        for folder_path in folder_paths:
            folder_path = folder_path[36:-5]  # discard post-fix '.jpg' and pre-fix '/dev/shm/k400/'
            path_root = '/discobox/wjpeng/dataset/k400/raw_frames'
            folder_path = path_root + folder_path
            if os.path.isdir(folder_path):
                image_paths = collect_image_paths(folder_path)
                for image_path in image_paths:
                    output.write(image_path + '\n')
            else:
                raise FileExistsError


# Example usage:
if __name__ == '__main__':
    path_file_path = '/discobox/wjpeng/code/202306/ngc-workspace/tools/extract_rawframes/build_report_bk.txt'  # Replace with the actual path to your path.txt file
    output_file_path = '/discobox/wjpeng/dataset/k400/ann/rawframe_list.txt'  # Replace with the desired output file path
    process_path_file(path_file_path, output_file_path)
