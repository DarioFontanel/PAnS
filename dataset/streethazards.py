import os
import torch.utils.data as data
import json

from PIL import Image


class StreetHazardsSegmentation(data.Dataset):

    def __init__(self, root, split='train', transform=None, basic_transform=None, multiple_resizes_test=False):

        root = os.path.expanduser(root)
        self.base_dir = "StreetHazards"
        root = os.path.join(root, self.base_dir)
        main_folder = os.path.join(root, split)

        self.images = []

        with open(os.path.join(root, split, split + ".odgt"), 'r') as f:
            for handler in f:
                fnames = json.loads(handler.rstrip())

                self.images = [(os.path.join(main_folder, x['fpath_img']), os.path.join(main_folder, x['fpath_segm']))
                               for x in fnames]

        self.transform = transform
        self.basic_transform = basic_transform
        self.multiple_resizes_test = multiple_resizes_test

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])
        assert (target.mode == "L")

        if self.transform is not None:
            if self.multiple_resizes_test:
                imgs = []
                for t in self.transform:
                    imgs.append(t(img, None))
                _, target = self.basic_transform(img, target)
                img = imgs
            else:
                img, target = self.transform(img, target)

            target = target - 1

        return img, target

    def __len__(self):
        return len(self.images)
