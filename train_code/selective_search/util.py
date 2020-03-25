import numpy as np
from itertools import product

from skimage.segmentation import felzenszwalb
from skimage.color import rgb2hsv, rgb2lab, rgb2grey


def oversegmentation(img, k):
    """
        Generating various starting regions using the method of
        Felzenszwalb.
        k effectively sets a scale of observation, in that
        a larger k causes a preference for larger components.
        sigma = 0.8 which was used in the original paper.
        min_size = 100 refer to Keon's Matlab implementation.
    """
    img_seg = felzenszwalb(img, scale=k, sigma=0.8, min_size=100)

    return img_seg


def switch_color_space(img, target):
    """
        RGB to target color space conversion.
        I: the intensity (grey scale), Lab, rgI: the rg channels of
        normalized RGB plus intensity, HSV, H: the Hue channel H from HSV
    """

    if target == 'HSV':
        return rgb2hsv(img)

    elif target == 'Lab':
        return rgb2lab(img)

    elif target == 'I':
        return rgb2grey(img)

    elif target == 'rgb':
        img = img / np.sum(img, axis=0)
        return img

    elif target == 'rgI':
        img = img / np.sum(img, axis=0)
        img[:,:,2] = rgb2grey(img)
        return img

    elif target == 'H':
        return rgb2hsv(img)[:,:,0]

    else:
        raise "{} is not suported.".format(target)

def load_strategy(mode):
    # TODO: Add mode sanity check

    cfg = {
        "single": {
            "ks": [100],
            "colors": ["HSV"],
            "sims": ["CTSF"]
        },
        "lab": {
            "ks": [100],
            "colors": ["Lab"],
            "sims": ["CTSF"]
        },
        "fast": {
            "ks": [50, 100],
            "colors": ["HSV", "Lab"],
            "sims": ["CTSF", "TSF"]
        },
        "quality": {
            "ks": [50, 100, 150, 300],
            "colors": ["HSV", "Lab", "I", "rgI", "H"],
            "sims": ["CTSF", "TSF", "F", "S"]
        }
    }

    if isinstance(mode, dict):
        cfg['manual'] = mode
        mode = 'manual'

    colors, ks, sims = cfg[mode]['colors'], cfg[mode]['ks'], cfg[mode]['sims']

    return product(colors, ks, sims)
