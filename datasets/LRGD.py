import glob
import os
import re

import pickle
import torch
import numpy as np
import random
from datasets.dataset_processing import grasp_anything, image


class LRGDDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for the Grasp-Anything dataset.
    """

    def __init__(self, file_path, output_size=224, ds_rotate=0, include_depth=False, include_rgb=True, random_rotate=False,
                 random_zoom=False, input_only=False, trainval_test='trainval', **kwargs):  ## trainval // text // test_none_related
        """
        :param file_path: Grasp-Anything Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(LRGDDataset, self).__init__(**kwargs)

        # self.grasp_files = glob.glob(os.path.join(file_path, 'Grasps', '*.txt'))
        # self.rgb_files = glob.glob(os.path.join(file_path, 'JPEGImages', '*.jpg'))


        self._data_path = file_path
        self._image_set = trainval_test  ## train or test
        self._image_index = self._load_image_set_index_ours()


        self.length = len(self._image_index)

        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_rgb = include_rgb

        # clip_path ='/data1/xukai_1/code/grasping/reason_grasp/checkpoint/ViT-B-16.pt'
        # _, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained=clip_path)

        # self.grasp_files = []

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        # if ds_rotate:
        #     self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
        #                                                                          :int(self.length * ds_rotate)]

    def _load_image_set_index_ours(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main', '11_ours',  ## 改动
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_crop_attrs(self, idx):
        gtbbs = grasp_anything.GraspRectangles.load_from_vmrd_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 1008 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 756 - self.output_size))
        return center, left, top

    def _get_crop_attrs_gai(self, file_idx, obj_idx):
        # gtbbs = grasp_anything.GraspRectangles.load_from_vmrd_file(self.grasp_files[idx])
        filename = os.path.join(self._data_path, 'Grasps', file_idx + '.txt')
        gtbbs = grasp_anything.GraspRectangles.load_from_vmrd_file_gai(filename, obj_idx)
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 1008 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 756 - self.output_size))
        return center, left, top

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        # Jacquard try
        gtbbs = grasp_anything.GraspRectangles.load_from_vmrd_file(self.grasp_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        gtbbs.rotate(rot, center)
        gtbbs.offset((-top, -left))
        gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return gtbbs

    def get_gtbb_gai(self, file_idx, obj_idx, rot=0, zoom=1.0):
        # Jacquard try

        filename = os.path.join(self._data_path, 'Grasps', file_idx + '.txt')

        gtbbs = grasp_anything.GraspRectangles.load_from_vmrd_file_gai(filename, obj_idx)
        center, left, top = self._get_crop_attrs_gai(file_idx, obj_idx)
        gtbbs.rotate(rot, center)
        gtbbs.offset((-top, -left))
        gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_file = self.grasp_files[idx].replace("Grasps", "JPEGImages").replace("txt", "jpg")
        rgb_img = image.Image.from_file(rgb_file)

        # Cornell try
        center, left, top = self._get_crop_attrs(idx)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left), (min(756, top + self.output_size), min(1008, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))

        return rgb_img.img

    # def get_rgb_clip(self, idx, rot=0, zoom=1.0, normalise=True):
    #
    #     rgb_file = self.grasp_files[idx].replace("Grasps", "JPEGImages").replace("txt", "jpg")
    #     # rgb_img_clip = self.preprocess(Image.open(rgb_file)).unsqueeze(0)
    #
    #     data_transform = transforms.Compose(
    #         [
    #             transforms.Resize(
    #                 (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
    #             ),
    #             # transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize(
    #                 mean=(0.48145466, 0.4578275, 0.40821073),
    #                 std=(0.26862954, 0.26130258, 0.27577711),
    #             ),
    #         ])
    #     image_1 = data_transform(_image_1).unsqueeze(0)
    #     return image_1

    def get_rgb_gai(self, file_idx, obj_idx, rot=0, zoom=1.0, normalise=True):
        # rgb_file = self.grasp_files[idx].replace("Grasps", "JPEGImages").replace("txt", "jpg")
        file_name = str(file_idx) + '.jpg'
        rgb_file = os.path.join(self._data_path, 'JPEGImages', file_name)

        rgb_img = image.Image.from_file(rgb_file)

        # Cornell try
        center, left, top = self._get_crop_attrs_gai(file_idx, obj_idx)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left), (min(756, top + self.output_size), min(1008, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))

        return rgb_img.img

    def __getitem__(self, idx):
        if self.random_rotate:
            rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0

        file_index, obj_idx, ins_idx = self._image_index[idx].split('_')
        # print(self._image_index[idx])

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(file_index, rot, zoom_factor)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb_gai(file_index, obj_idx, rot, zoom_factor)

        # Load the grasps
        bbs = self.get_gtbb_gai(file_index, obj_idx, rot, zoom_factor)

        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        width_img = np.clip(width_img, 0.0, self.output_size / 2) / (self.output_size / 2)

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        instruction_path = os.path.join(self._data_path, 'Instructions', self._image_index[idx] + '.txt')
        with open(instruction_path, 'r') as file:
            x_instru = file.readlines()


        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2 * ang_img))
        sin = self.numpy_to_torch(np.sin(2 * ang_img))
        width = self.numpy_to_torch(width_img)

        # return x, (pos, cos, sin, width), idx, rot, zoom_factor
        return x, x_instru, (pos, cos, sin, width), idx, rot, zoom_factor

    def __len__(self):
        return len(self._image_index)