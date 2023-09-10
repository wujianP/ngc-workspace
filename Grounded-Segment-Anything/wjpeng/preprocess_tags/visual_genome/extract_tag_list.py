from datasets import load_dataset


if __name__ == '__main__':
    dataset = load_dataset(path="visual_genome", name="objects_v1.2.0",
                           split="train", cache_dir='/DDN_ROOT/wjpeng/dataset/visual-genome')
    from IPython import embed
    embed()
    # find all tags of Visual Genome
    tags = set()
    for i in range(len(dataset)):
        objects = dataset[i]['objects']
        for obj in objects:
            names = obj['names']
            for name in names:
                tags.add(name)
        print(f'{(i+1)/len(dataset)*100:.2f} (%)')
    # save to tag_list.txt
    with open('/discobox/wjpeng/code/202306/ngc-workspace/Grounded-Segment-Anything/wjpeng/preprocess_tags/visual_genome/tag_list.txt', 'w', encoding='utf-8') as file:
        for tag in tags:
            file.write(tag + '\n')
    file.close()
