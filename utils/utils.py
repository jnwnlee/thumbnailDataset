import numpy as np
import matplotlib.pyplot as plt
import torchvision

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    if title is not None:
        plt.title(title)
    plt.imshow(inp)
    plt.pause(0.001)  # pause a bit so that plots are updated


#https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
# w/ loader
