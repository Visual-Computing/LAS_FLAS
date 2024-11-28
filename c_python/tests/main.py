#!/usr/bin/env python3

from PIL import Image
import numpy as np
from flas import flas, apply_sorting


HEIGHT, WIDTH = 64, 32
DIM = 3


def main():
    features = np.random.random((HEIGHT, WIDTH, DIM)).astype(np.float32)
    # print(features)
    frozen = np.zeros((HEIGHT, WIDTH)).astype(np.bool)
    frozen[:10, :10] = True
    # frozen = np.logical_not(frozen)
    print(frozen)

    sorting = flas(features, frozen)

    sorted_features = apply_sorting(features, sorting)
    # print(sorted_features, sorted_features.dtype, sorted_features.shape)

    image = Image.fromarray((sorted_features[:, :, :3] * 255).astype(np.uint8))
    image.save('images/image1.png', 'PNG')


if __name__ == '__main__':
    main()

