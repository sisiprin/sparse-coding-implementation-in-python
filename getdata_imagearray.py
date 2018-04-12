import numpy as np
import scipy.io as sio

def loadmat(filename):
    f = sio.loadmat(filename)
    return f[filename[:-4]]

def getdata_imagearray(IMAGES, winsize, num_patches):
    IMAGES = IMAGES['IMAGES']
    num_images = IMAGES.shape[2]
    image_size = IMAGES.shape[0]

    sz = winsize
    BUFF = 4

    totalsamples = 0
    X = np.zeros((sz ** 2, num_patches))
    for i in range(num_images):
        print('%d/%d' % (i+1, num_images))
        this_image = IMAGES[:,:,i]

        getsample = np.floor( num_patches / num_images)
        if i == num_images - 1:
            getsample = num_patches - totalsamples

        for j  in range(int(getsample)):
            r = BUFF + np.ceil((image_size - sz - 2 * BUFF) * np.random.rand())
            c = BUFF + np.ceil((image_size - sz - 2 * BUFF) * np.random.rand())

            totalsamples = totalsamples + 1
            temp = np.reshape(this_image[r-1:r+sz-1, c-1:c+sz-1], sz ** 2, 1)

            X[:, totalsamples-1] = temp - np.mean(temp)
    return X
