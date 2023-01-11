import math
import numbers
from typing import Optional
import numpy as np


class SpecAugment():
    """SpecAugment (https://arxiv.org/abs/1904.08779)"""
    def __init__(self, args):
        self.time_warp_w, self.freq_mask_n, self.freq_mask_f = 0, 0, 0
        self.time_mask_n, self.time_mask_t = 0, 0
        self.time_mask_p, self.mask_value = 0.0, None
        if "specaugment" in args:
            self.time_warp_w = args.specaugment["time_warp_W"]
            self.freq_mask_n = args.specaugment["freq_mask_N"]
            self.freq_mask_f = args.specaugment["freq_mask_F"]
            self.time_mask_n = args.specaugment["time_mask_N"]
            self.time_mask_t = args.specaugment["time_mask_T"]
            self.time_mask_p = args.specaugment["time_mask_p"]
            if "mask_value" in args.specaugment:
                self.mask_value = args.specaugment["mask_value"]

        if self.freq_mask_n > 0:
            assert self.freq_mask_f > 0, (
                f"freq_mask_F ({self.freq_mask_f}) must be larger than 0 when doing freq masking."
            )
        if self.time_mask_n > 0:
            assert self.time_mask_t > 0, (
                f"time_mask_T ({self.time_mask_t}) must be larger than 0 when doing time masking."
            )

    def __call__(self, spectrogram):
        assert len(spectrogram.shape) == 2, "spectrogram must be a 2-D tensor."
        spectrogram = spectrogram.cpu().numpy()
        distorted = spectrogram.copy()  # make a copy of input spectrogram.
        num_frames = spectrogram.shape[0]  # or 'tau' in the paper.
        num_freqs = spectrogram.shape[1]  # or 'miu' in the paper.
        mask_value = self.mask_value

        if mask_value is None:  # if no value was specified, use local mean.
            mask_value = spectrogram.mean()

        if num_frames == 0:
            return spectrogram

        if num_freqs < self.freq_mask_f:
            return spectrogram

        if self.time_warp_w > 0:
            if 2 * self.time_warp_w < num_frames:
                import cv2

                w0 = np.random.randint(self.time_warp_w, num_frames - self.time_warp_w)
                w = np.random.randint(-self.time_warp_w + 1, self.time_warp_w)
                upper, lower = distorted[:w0, :], distorted[w0:, :]
                upper = cv2.resize(
                    upper, dsize=(num_freqs, w0 + w), interpolation=cv2.INTER_LINEAR
                )
                lower = cv2.resize(
                    lower,
                    dsize=(num_freqs, num_frames - w0 - w),
                    interpolation=cv2.INTER_LINEAR,
                )
                distorted = np.concatenate((upper, lower), axis=0)

        for _i in range(self.freq_mask_n):
            f = np.random.randint(0, self.freq_mask_f)
            f0 = np.random.randint(0, num_freqs - f)
            if f != 0:
                distorted[:, f0 : f0 + f] = mask_value

        max_time_mask_t = min(
            self.time_mask_t, math.floor(num_frames * self.time_mask_p)
        )
        if max_time_mask_t < 1:
            return distorted

        for _i in range(self.time_mask_n):
            t = np.random.randint(0, max_time_mask_t)
            t0 = np.random.randint(0, num_frames - t)
            if t != 0:
                distorted[t0 : t0 + t, :] = mask_value

        return distorted
