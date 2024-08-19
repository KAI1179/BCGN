# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from maskrcnn_benchmark.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        sample = transposed_batch[0]
        idx = transposed_batch[1]
        rot = transposed_batch[2]
        zoom_factor = transposed_batch[3]


        # return images, targets, img_ids
        return sample, idx, rot, zoom_factor


class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))

