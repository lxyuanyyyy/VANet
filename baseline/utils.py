from typing import List

from matplotlib import pyplot as plt
import torch
from torch import nn


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


def plot_image(img, label, name):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0] * 0.3081 + 0.1307, cmap='gray', interpolation='none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == '__main__':
    def twoSum(nums: List[int], target: int) -> List[int]:
        d = {}
        for idx, i in enumerate(nums):
            d[target - i] = idx
            if d.get(i) != None and len(d) > 1:
                return d.get(i), idx


    a = [3, 2, 4]
    r = twoSum(a, 6)
    print(r)
