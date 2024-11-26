#!/usr/bin/env python3

from PIL import Image
import numpy as np
import PIL
import flas_c_py


def main():
    height, width = 128, 64
    dim = 3
    features = np.random.random((height, width, dim)).astype(np.float32)
    # print(features)
    in_use = np.zeros((height, width)).astype(np.bool)
    # print(in_use)

    success, result = flas_c_py.flas(features, in_use, True, 0.5, 0.93, 1, 1.0, 1.0, 100.0, 0.01, 1.0, 9)
    # print(result)

    sorted_features = features.reshape(height * width, -1)[result.flatten()].reshape(height, width, -1)
    # print(sorted_features, sorted_features.dtype, sorted_features.shape)

    image = Image.fromarray((sorted_features[:, :, :3] * 255).astype(np.uint8))
    image.save('images/image1.png', 'PNG')


if __name__ == '__main__':
    main()

