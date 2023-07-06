import numpy as np


class MixMetrics(object):
    def __init__(self, num_class=2):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def tp(self):
        return self.confusion_matrix[1, 1]

    def fn(self):
        return self.confusion_matrix[1, 0]

    def fp(self):
        return self.confusion_matrix[0, 1]

    def tn(self):
        return self.confusion_matrix[0, 0]

    def precision(self):
        return self.tp() / (self.tp() + self.fp())

    def recall(self):
        return self.tp() / (self.tp() + self.fn())

    def f1score(self):
        tp = self.tp()
        fp = self.fp()
        fn = self.fn()
        return 2 * tp / (2 * tp + fp + fn)

    def pixel_accuracy(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def pixel_accuracy_class(self):
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        acc = np.nanmean(acc)
        return acc

    def intersection_over_union(self):
        tp = self.tp()
        fn = self.fn()
        fp = self.fp()
        return tp / (tp + fn + fp)

    def mean_intersection_over_union(self):
        mIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        mIoU = np.nanmean(mIoU)
        return mIoU

    def frequency_weighted_intersection_over_union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        fwIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return fwIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def mix_metrics(num_class=2):
    return MixMetrics(num_class)
