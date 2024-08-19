# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class GT_List(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, GT_perImage):
        # device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        # bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)


        # self.GT_perImage = GT_perImage
        (self.pos, self.cos, self.sin, self.width) = GT_perImage
        # self.ind = ind
        self.extra_fields = {}
        self.triplet_extra_fields = []  # e.g. relation field, which is not the same size as object bboxes and should not respond to __getitem__ slicing v[item]

    def add_field(self, field, field_data, is_triplet=False):
        # if field in self.extra_fields:
        #     print('{} is already in extra_fields. Try to replace with new data. '.format(field))
        self.extra_fields[field] = field_data
        if is_triplet:
            self.triplet_extra_fields.append(field)

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())


    # Tensor-like methods


    # Tensor-like methods

    def to(self, device):
        bbox = GT_List((self.pos.to(device), self.cos.to(device), self.sin.to(device), self.width.to(device)))
        # for k, v in self.extra_fields.items():
        #     if hasattr(v, "to"):
        #         v = v.to(device)
        #     if k in self.triplet_extra_fields:
        #         bbox.add_field(k, v, is_triplet=True)
        #     else:
        #         bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = GT_List((self.pos, self.cos, self.sin, self.width)[item])
        for k, v in self.extra_fields.items():
            if k in self.triplet_extra_fields:
                bbox.add_field(k, v[item][:,item], is_triplet=True)
            else:
                bbox.add_field(k, v[item])
        return bbox


    def __len__(self):
        return len(self.pos.shape[0])
        # pass

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1
        if remove_empty:
            box = (self.pos, self.cos, self.sin, self.width)
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        return self

    def area(self):
        box = (self.pos, self.cos, self.sin, self.width)
        # if self.mode == "xyxy":
        #     TO_REMOVE = 1
        #     area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        # elif self.mode == "xywh":
        #     area = box[:, 2] * box[:, 3]
        # else:
        #     raise RuntimeError("Should not be here")
        pass

        # return area

    def copy(self):
        return GT_List(self.pos)

    def copy_with_fields(self, fields, skip_missing=False):
        GT = GT_List((self.pos, self.cos, self.sin, self.width))
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                if field in self.triplet_extra_fields:
                    bbox.add_field(field, self.get_field(field), is_triplet=True)
                else:
                    bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return GT

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_objects={}, ".format(len(self.pos.shape[0]))
        # s += "image_width={}, ".format(self.size[0])
        # s += "image_height={}, ".format(self.size[1])
        # s += "mode={})".format(self.mode)
        return s



if __name__ == "__main__":
    bbox = BoxList([[0, 0, 10, 10], [0, 0, 5, 5]], (10, 10))
    s_bbox = bbox.resize((5, 5))
    print(s_bbox)
    print(s_bbox.bbox)

    t_bbox = bbox.transpose(0)
    print(t_bbox)
    print(t_bbox.bbox)
