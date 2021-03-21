import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __init__(self, data_dir, format, barcode_to_label):
        super(ImageFolderWithPaths, self).__init__(data_dir, format)
        self.barcode_to_label = barcode_to_label

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        # tuple_with_path = (original_tuple + (path,))
        barcode = int(path.split('/')[-1].split('_')[1])
        label = self.barcode_to_label[barcode]
        new_tuple = (original_tuple[0], label)
        return new_tuple


# Utility function for image display
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = np.clip(std * inp + mean, 0,1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()
