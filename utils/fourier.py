import cv2
import numpy as np


def distance(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


def get_circle_mask(h, w, r):
    center = (int(h / 2), int(w / 2))
    mask = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if distance((i, j), center) < r:
                mask[i, j] = 1.0
    return mask


def get_gaussian_mask(h, w, r):
    center = (int(h / 2), int(w / 2))
    mask = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            mask[i, j] = np.exp(-(distance((i, j), center)**2)/(2*(r**2)))
    return mask


def fourier_transformation(img, mask, mode='both'):
    h, w, c = img.shape

    # spectrum = np.zeros_like(img)
    lfc = np.zeros_like(img)
    hfc = np.zeros_like(img)

    for i in range(c):
        origin = img[:, :, i]
        f = np.fft.fft2(origin)
        f_shift = np.fft.fftshift(f)
        # spectrum[:, :, i] = 20 * np.log(np.abs(f_shift))

        low_masked = np.multiply(f_shift, mask)
        low_f_ishift = np.fft.ifftshift(low_masked)
        lfc[:, :, i] = np.abs(np.fft.ifft2(low_f_ishift))

        high_masked = f_shift * (1 - mask)
        high_f_ishift = np.fft.ifftshift(high_masked)
        hfc[:, :, i] = np.abs(np.fft.ifft2(high_f_ishift))

    if mode == 'lfc':
        return lfc
    elif mode == 'hfc':
        return hfc
    elif mode == 'both':
        return lfc, hfc
    else:
        raise NotImplementedError(f'Fourier transformation return mode({mode}) is not implemented')


if __name__ == '__main__':
    img = cv2.imread('money.jpg')
    img = cv2.resize(img, (224, 224))
    l, h = fourier_transformation(img, 12)