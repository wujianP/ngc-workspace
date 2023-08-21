import zipfile
import os
import argparse


def zip_directory(directory_path, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, directory_path)
                zipf.write(file_path, arcname=arcname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--out', type=str)
    args = parser.parse_args()

    zip_directory(directory_path=args.dir, output_zip_path=args.out)
