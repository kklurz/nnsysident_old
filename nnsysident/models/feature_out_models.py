import numpy as np
import warnings
from torch.nn import functional as F
from neuralpredictors.layers.encoders import FiringRateEncoder
from neuralpredictors.layers.readouts.gaussian import FullGaussian2d
import torch.nn as nn

class FiringRateEncoder_(FiringRateEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
            self,
            inputs,
            *args,
            targets=None,
            data_key=None,
            behavior=None,
            pupil_center=None,
            trial_idx=None,
            shift=None,
            detach_core=False,
            feature_out=False,
            **kwargs
    ):
        x = self.core(inputs)
        if detach_core:
            x = x.detach()

        if self.shifter:
            if pupil_center is None:
                raise ValueError("pupil_center is not given")
            shift = self.shifter[data_key](pupil_center, trial_idx)

        x = self.readout(x, data_key=data_key, shift=shift, feature_out=feature_out, **kwargs)

        if self.modulator:
            assert ~feature_out, "feature_out can not be used together with modulator"
            if behavior is None:
                raise ValueError("behavior is not given")
            x = self.modulator[data_key](x, behavior=behavior)

        if not feature_out:
            x = nn.functional.elu(x + self.offset) + 1
        return x


class FullGaussian2d_(FullGaussian2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, sample=None, shift=None, out_idx=None, feature_out=False, **kwargs):
        """
        Propagates the input forwards through the readout
        Args:
            x: input data
            sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
                            or use the mean, mu, of the Gaussian distribution without sampling.
                           if sample is None (default), samples from the N(mu,sigma) during training phase and
                             fixes to the mean, mu, during evaluation phase.
                           if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
            shift (bool): shifts the location of the grid (from eye-tracking data)
            out_idx (bool): index of neurons to be predicted
            feature_out (bool): whether to output the feature vector
        Returns:
            y: neuronal activity
        """
        N, c, w, h = x.size()
        c_in, w_in, h_in = self.in_shape
        if (c_in, w_in, h_in) != (c, w, h):
            warnings.warn("the specified feature map dimension is not the readout's expected input dimension")
        feat = self.features.view(1, c, self.outdims)
        bias = self.bias
        outdims = self.outdims

        if self.batch_sample:
            # sample the grid_locations separately per image per batch
            grid = self.sample_grid(batch_size=N, sample=sample)  # sample determines sampling from Gaussian
        else:
            # use one sampled grid_locations for all images in the batch
            grid = self.sample_grid(batch_size=1, sample=sample).expand(N, outdims, 1, 2)

        if out_idx is not None:
            if isinstance(out_idx, np.ndarray):
                if out_idx.dtype == bool:
                    out_idx = np.where(out_idx)[0]
            feat = feat[:, :, out_idx]
            grid = grid[:, out_idx]
            if bias is not None:
                bias = bias[out_idx]
            outdims = len(out_idx)

        if shift is not None:
            grid = grid + shift[:, None, None, :]

        y = F.grid_sample(x, grid, align_corners=self.align_corners)
        if feature_out:
            return y
        y = (y.squeeze(-1) * feat).sum(1).view(N, outdims)

        if self.bias is not None:
            y = y + bias
        return y
