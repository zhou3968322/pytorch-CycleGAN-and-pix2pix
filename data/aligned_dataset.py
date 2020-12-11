import os, json
from data.base_dataset import BaseDataset, get_params, get_transform, default_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import random
from generator.img_util import convert_poly_to_rect


def _handler_mask_data(mask_data):
    # 从mask中随机抠出一个64x64的框
    mask_data = np.array(mask_data)
    crop_box = []
    for mask_polys in mask_data:
        mask_rect = convert_poly_to_rect(mask_polys)
        mask_height = mask_rect[3] - mask_rect[1]
        mask_width = mask_rect[2] - mask_rect[0]
        if mask_width >= 64:
            xs = random.randint(0,mask_width - 64)
            xe = xs + 64
        elif mask_rect[2] >= 64:
            xs = mask_rect[2] - 64
            xe = mask_rect[2]
        else:
            xs = mask_rect[0]
            xe = mask_rect[0] + 64
        if mask_height >= 64:
            ys = random.randint(0,mask_height - 64)
            ye = ys + 64
        elif mask_rect[3] >= 64:
            ys = mask_rect[3] - 64
            ye = mask_rect[3]
        else:
            ys = mask_rect[1]
            ye = mask_rect[1] + 64
        crop_box.append([xs, ys, xe, ye])
    return np.array(crop_box)


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        mask_dir = os.path.join(opt.dataroot, "mask")
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        self.dataset_len = len(self.AB_paths)
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        mask_paths = []
        for ab_path in self.AB_paths:
            ab_name = os.path.basename(ab_path)
            mask_name = "{}_{}.json".format(opt.phase, ab_name.rsplit('.', 1)[0])
            mask_path = os.path.join(mask_dir, mask_name)
            mask_paths.append(mask_path)
            assert os.path.isfile(mask_path)
        self.mask_paths = mask_paths
        random.seed(30)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index % self.dataset_len]
        try:
            AB = Image.open(AB_path).convert('RGB')
        except Exception as e:
            return self.__getitem__(index + 1)
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        # transform_params = get_params(self.opt, A.size)
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        # B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        #
        # A = A_transform(A)
        # B = B_transform(B)
        current_transform = default_transform()
        A = current_transform(A)
        B = current_transform(B)
        mask_path = self.mask_paths[index % self.dataset_len]
        with open(mask_path, "r") as fr:
            mask_data = json.loads(fr.read())
        crop_box = _handler_mask_data(mask_data)
        return {'A': A, 'B': B, "crop_box": crop_box,
                'A_paths': AB_path,'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
