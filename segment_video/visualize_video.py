import numpy as np
import wandb
import os
from PIL import Image

wandb.login()


def load_masks_frames(seg_ret, video_name):
    import os
    from PIL import Image
    import numpy as np
    masks = []
    frames = []
    for k, v in seg_ret.items():
        if k.startswith(video_name):
            masks.append(v['masks'])
            frame_path = os.path.join('/discobox/wjpeng/dataset/k400/raw_frames', k)
            frame = Image.open(frame_path).convert('RGB')
            frames.append(np.array(frame))
    return masks, frames


if __name__ == '__main__':
    run = wandb.init('visualize-video')
    seg_ret_path = '/discobox/wjpeng/dataset/k400/ann/groundedSAM/DINO-SwinT-imgSize800_SAMHQ-ViTB-imgSize1024_stride8.npy'
    seg_ret = np.load(seg_ret_path, allow_pickle=True).item()
    video_name = ['train/dancing_ballet/8jS_PCZUO3A_000062_000072',
                  'train/shaking_hands/iQVa2URAuAU_000006_000016',
                  'train/trimming_trees/ExQg288lBrI_000000_000010',
                  'train/brushing_teeth/fmecM7Fx1tQ_000045_000055',
                  'train/petting_cat/rF6WHdvfhzU_000044_000054',
                  'train/parkour/FESQKVYD92Q_000003_000013',
                  'train/train/yoga/UX3k9_185Ns_000069_000079',
                  'train/washing_hands/Hwx_8FhcqY4_000074_000084']
    for i in range(len(video_name)):
        video_name = video_name[i]
        masks, frames = load_masks_frames(seg_ret, video_name)
        mask_frames = []
        for mask, frame in zip(masks, frames):
            ret = np.zeros_like(frame)
            ret[mask] = frame[mask]
            mask_frames.append(ret)
        # TCHW
        video = np.array(mask_frames).transpose((0, 3, 1, 2))
        run.log({'video': wandb.Video(video, fps=4)})
