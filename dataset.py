from os import listdir
from os.path import join

import torch.utils.data as data
import torchvision.transforms as transforms

from util import is_image_file, load_img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, imageSize):
        super(DatasetFromFolder, self).__init__()
        self.photo_path = image_dir
        self.image_filenames = [x for x in listdir(
            self.photo_path) if is_image_file(x)]

        transform_list = [transforms.Grayscale(num_output_channels=1),
                          transforms.Resize(imageSize),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # Load Image
        input = load_img(join(self.photo_path, self.image_filenames[index]))
        input = self.transform(input)

        return input

    def __len__(self):
        return len(self.image_filenames)
