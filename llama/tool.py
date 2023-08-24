from dataset import CocoDataset


if __name__ == '__main__':
    coco = CocoDataset(image_root='/discobox/wjpeng/dataset/coco2014/images/train2014',
                       json='/discobox/wjpeng/dataset/coco2014/annotations/captions_train2014.json')
    output = '/discobox/wjpeng/dataset/coco2014/captions_demo.txt'
    for i in range(len(coco)):
        if i % 500 == 0:
            cap, _ = coco[i]
            with open(output, mode='a') as f:
                f.write(cap + '\n')
            print(f'{100*i/len(coco):.2f}%')
    print(len(coco))