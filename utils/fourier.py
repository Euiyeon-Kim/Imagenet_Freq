import cv2
import numpy as np

DISTINCT = ['n02129604', 'n04086273', 'n04254680', 'n07745940', 'n02690373', 'n03796401', 'n12620546', 'n11879895',
            'n02676566', 'n01806143', 'n02007558', 'n01695060', 'n03532672', 'n03065424', 'n03837869', 'n07711569',
            'n07734744', 'n03676483', 'n09229709', 'n07831146']
SIMILAR = ['n02100735', 'n02110185', 'n02096294', 'n02417914', 'n02110063', 'n02089867', 'n02102177', 'n02092339',
           'n02098105', 'n02105641', 'n02096051', 'n02110341', 'n02086910', 'n02113712', 'n02113186', 'n02091467',
           'n02106550', 'n02091831', 'n02104365', 'n02086079']


def distance(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


def get_threshold_mask(h, w, r):
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


def fourier_transformation(img, r, mask_shape='gaussian', mode='both'):
    h, w, c = img.shape

    if mask_shape == 'gaussian':
        mask = get_gaussian_mask (h, w, r)
    elif mask_shape == 'circle':
        mask = get_threshold_mask(h, w, r)
    else:
        raise NotImplementedError(f'Fourier transformation {mask_shape} mask is not implemented')

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