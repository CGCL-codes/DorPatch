import math
import torch
import numpy as np


class MaskWindow(object):
    # save information for patch cleanser defense
    def __init__(self, img_size, patch_ratio=0.03, n_patch=1):
        self.img_size = img_size

        self.mask_size = math.floor(
            math.sqrt(img_size**2 * patch_ratio / n_patch))  # 9
        self.num_mask_per_axis = 6
        self.stride = int(
            np.ceil((img_size - self.mask_size + 1) / self.num_mask_per_axis))
        self.window_size = self.mask_size + self.stride - 1
        self.n_patch = n_patch

        basic_mask = self.generate_mask_set()
        n_mask = len(basic_mask)
        mask_shape = list(basic_mask.shape[1:])
        # prepare double masks that can serve as 1. verifiable set, 2. multiple patch possible combination
        combined_mask = (basic_mask[None, :] *
                         basic_mask[:, None]).view([-1] + mask_shape)
        # symmetric matrix: only keep upper triangular non-diagonal part to remove repeated combinations
        selected = torch.triu(torch.ones(
            (n_mask, n_mask), dtype=torch.bool), diagonal=1)
        # remove repeated redundant
        combined_mask = combined_mask[selected.view(-1)]

        if n_patch == 1:
            self.mask_set = basic_mask
            self.double_mask_set = combined_mask
        elif n_patch == 2:
            self.mask_set = combined_mask
            self.double_mask_set = (
                combined_mask[None, :] * basic_mask[:, None]).view([-1] + mask_shape)
        else:
            raise NotImplementedError
        self.reverse_mask_set = ~self.mask_set
        print("mask size: %d, window size: %d, stride: %d" %
              (self.mask_size, self.window_size, self.stride))

    def generate_mask_set(self):
        num_mask_per_axis = self.num_mask_per_axis
        window_size = self.window_size
        stride = self.stride
        img_size = self.img_size
        mask_set = torch.ones(
            (num_mask_per_axis ** 2, 1, img_size, img_size), dtype=torch.bool).cuda()
        for i in range(num_mask_per_axis):
            for j in range(num_mask_per_axis):
                x_start = stride * i
                x_end = min(img_size, x_start + window_size)
                y_start = stride * j
                y_end = min(img_size, y_start + window_size)
                mask_set[i * num_mask_per_axis + j, :,
                         x_start:x_end, y_start:y_end] = False
        return mask_set


class PatchCleanser(object):
    def __init__(self, mask_window, model, result=None):
        self.mask_window = mask_window
        self.model = model
        self.result = result  # record the PC result

    def robust_predict(self, img, certify=False):
        # first-round masking
        mask_set = self.mask_window.mask_set
        masked_imgs = self.mask(img, mask_set)
        preds_1 = self.model(masked_imgs).argmax(1)  # one-masked predictions
        preds_2 = None  # two-masked predictions
        labels, counts = preds_1.unique(sorted=False, return_counts=True)
        label_majority = labels[counts.argmax()].item()

        pred = label_majority
        if len(labels) == 1:  # consistent in first round
            certifiable, preds_2 = self.robustness_certificate(img, pred)
        else:
            certifiable = False
            for label, count in zip(labels, counts):
                if label == label_majority:
                    continue
                # second-round masking for all inconsistent one-masked images
                for masked_img in masked_imgs[preds_1 == label]:
                    preds_1_2 = self.model(
                        self.mask(masked_img, mask_set)).argmax(axis=-1)
                    if (preds_1_2 == label).all():
                        pred = label.item()   # corrected masked-image produces consistent prediction

        preds_1 = preds_1.detach().cpu().numpy()
        if certify and preds_2 is None:
            preds_2 = self.robustness_certificate(img, label_majority)[1]
        if preds_2 is not None:
            preds_2 = preds_2.detach().cpu().numpy()
        return PatchCleanserRecord(pred, certifiable, preds_1, preds_2)

    def mask(self, img, msk):
        return img * msk + 0.5 * ~msk

    def robustness_certificate(self, img, label, batch_size=64):
        # certified robustness for this image only if all two mask predictions are consistent
        double_mask_set = self.mask_window.double_mask_set

        preds = []
        for i in range(math.ceil(len(double_mask_set) / batch_size)):
            preds.append(self.model(self.mask(
                img, double_mask_set[i * batch_size: (i + 1) * batch_size])).argmax(1))
        preds = torch.cat(preds)
        consistent = (preds == label)
        return consistent.all().item(), consistent

    def reset(self):
        self.result = None

    def collect(self, records):
        self.result = PatchCleanserResult(records)


class PatchCleanserRecord(object):
    def __init__(self, pred, certifiable, preds_1, preds_2):
        self.prediction = pred  # robust prediction
        self.certification = certifiable
        self.preds_1 = preds_1  # one-masked predictions
        self.preds_2 = preds_2  # two-masked predictions


class PatchCleanserResult(object):
    def __init__(self, records):
        self.predictions = np.stack([r.prediction for r in records])
        self.certifications = np.stack([r.certification for r in records])
        self.predictions_1 = np.stack([r.preds_1 for r in records])
        self.predictions_2 = [r.preds_2 for r in records]
