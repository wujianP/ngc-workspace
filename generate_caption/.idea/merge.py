all_file = []
for file in files:
    file = torch.load(file)
    all_file.extend(file)
