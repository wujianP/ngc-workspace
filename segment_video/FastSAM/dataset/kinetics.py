import os
import numpy as np
import pandas as pd
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset


class KineticsDataset(Dataset):
    """Load your own video classification dataset."""

    def __init__(self,
                 anno_path,
                 data_path,
                 clip_len=8,
                 frame_sample_rate=8,
                 num_segment=1,
                 args=None):
        self.anno_path = anno_path  # path to .csv
        self.data_path = data_path  # path to video root
        self.clip_len = clip_len    # num_frames per clip
        self.frame_sample_rate = frame_sample_rate  # stride between frames
        self.num_segment = num_segment
        self.args = args
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=' ')
        self.dataset_samples = list(cleaned.values[:, 0])   # video path list
        self.label_array = list(cleaned.values[:, 1])       # video label list

    def __getitem__(self, index):
        sample = self.dataset_samples[index]    # video path
        buffer = self.loadvideo_decord(sample)
        if len(buffer) == 0:
            while len(buffer) == 0:
                warnings.warn("video {} not correctly loaded".format(sample))
                index = np.random.randint(self.__len__())
                sample = self.dataset_samples[index]
                buffer = self.loadvideo_decord(sample)
        return buffer, self.label_array[index]

    def loadvideo_decord(self, sample, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = sample

        if not (os.path.exists(fname)):
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        # handle temporal segments
        converted_len = int(self.clip_len * self.frame_sample_rate)
        seg_len = len(vr) // self.num_segment

        all_index = []
        if seg_len <= converted_len:
            index = np.linspace(0, seg_len, num=seg_len // self.frame_sample_rate)
            index = np.concatenate((index, np.ones(self.clip_len - seg_len // self.frame_sample_rate) * seg_len))
            index = np.clip(index, 0, seg_len - 1).astype(np.int64)
        else:
            end_idx = np.random.randint(converted_len, seg_len)
            str_idx = end_idx - converted_len
            index = np.linspace(str_idx, end_idx, num=self.clip_len)
            index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
        all_index.extend(list(index))

        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer

    def __len__(self):
        return len(self.dataset_samples)


