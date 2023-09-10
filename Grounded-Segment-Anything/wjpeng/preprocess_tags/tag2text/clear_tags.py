import numpy as np


with open('cleared_tag_list.txt', 'r') as file:
    lines = file.readlines()
    excluded_tag_indices = []
    for idx, line in enumerate(lines):
        if line.startswith('#'):
            print(f'idx: {idx} line: {line}')
            excluded_tag_indices.append(idx)
    np.save('excluded_tag_indices', excluded_tag_indices)
