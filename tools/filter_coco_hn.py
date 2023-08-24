
if __name__ == '__main__':
    path = 'coco_hard_mannal.CSV'
    with open(path, 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        input = line.strip().split('[sep]')[0].strip()
        output = line.strip().split('[sep]')[1].strip()
        if len(output) > 5:
            data.append({
                'input': input[:-1],
                'output': output[1:]
            })
    import torch
    torch.save(data, 'coco_hard_human_filtered.pth')
