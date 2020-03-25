import numpy as np
from skimage.feature import local_binary_pattern

def _calculate_color_sim(ri, rj):
    """
        Calculate color similarity using histogram intersection
    """
    return sum([min(a, b) for a, b in zip(ri["color_hist"], rj["color_hist"])])


def _calculate_texture_sim(ri, rj):
    """
        Calculate texture similarity using histogram intersection
    """
    return sum([min(a, b) for a, b in zip(ri["texture_hist"], rj["texture_hist"])])


def _calculate_size_sim(ri, rj, imsize):
    """
        Size similarity boosts joint between small regions, which prevents
        a single region from engulfing other blobs one by one.

        size (ri, rj) = 1 − [size(ri) + size(rj)] / size(image)
    """
    return 1.0 - (ri['size'] + rj['size']) / imsize


def _calculate_fill_sim(ri, rj, imsize):
    """
        Fill similarity measures how well ri and rj fit into each other.
        BBij is the bounding box around ri and rj.

        fill(ri, rj) = 1 − [size(BBij) − size(ri) − size(ri)] / size(image)
    """

    bbsize = (max(ri['box'][2], rj['box'][2]) - min(ri['box'][0], rj['box'][0])) * (max(ri['box'][3], rj['box'][3]) - min(ri['box'][1], rj['box'][1]))

    return 1.0 - (bbsize - ri['size'] - rj['size']) / imsize


def calculate_color_hist(mask, img):
    """
        Calculate colour histogram for the region.
        The output will be an array with n_BINS * n_color_channels.
        The number of channel is varied because of different
        colour spaces.
    """

    BINS = 25
    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)

    channel_nums = img.shape[2]
    hist = np.array([])

    for channel in range(channel_nums):
        layer = img[:, :, channel][mask]
        hist = np.concatenate([hist] + [np.histogram(layer, BINS)[0]])

    # L1 normalize
    hist = hist / np.sum(hist)

    return hist


def generate_lbp_image(img):

    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
    channel_nums = img.shape[2]

    lbp_img = np.zeros(img.shape)
    for channel in range(channel_nums):
        layer = img[:, :, channel]
        lbp_img[:, :,channel] = local_binary_pattern(layer, 8, 1)

    return lbp_img


def calculate_texture_hist(mask, lbp_img):
    """
        Use LBP for now, enlightened by AlpacaDB's implementation.
        Plan to switch to Gaussian derivatives as the paper in future
        version.
    """

    BINS = 10
    channel_nums = lbp_img.shape[2]
    hist = np.array([])

    for channel in range(channel_nums):
        layer = lbp_img[:, :, channel][mask]
        hist = np.concatenate([hist] + [np.histogram(layer, BINS)[0]])

    # L1 normalize
    hist = hist / np.sum(hist)

    return hist


def calculate_sim(ri, rj, imsize, sim_strategy):
    """
        Calculate similarity between region ri and rj using diverse
        combinations of similarity measures.
        C: color, T: texture, S: size, F: fill.
    """
    sim = 0

    if 'C' in sim_strategy:
        sim += _calculate_color_sim(ri, rj)
    if 'T' in sim_strategy:
        sim += _calculate_texture_sim(ri, rj)
    if 'S' in sim_strategy:
        sim += _calculate_size_sim(ri, rj, imsize)
    if 'F' in sim_strategy:
        sim += _calculate_fill_sim(ri, rj, imsize)

    return sim
