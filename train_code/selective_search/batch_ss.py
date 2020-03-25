'''
CVPR 2020 submission, Paper ID 6791
Source code for 'Learning to Cartoonize Using White-Box Cartoon Representations'
'''


import numpy as np
from adaptive_color import label2rgb
from joblib import Parallel, delayed
from skimage.segmentation import felzenszwalb
from util import switch_color_space
from structure import HierarchicalGrouping


def color_ss_map(image, color_space='Lab', k=10, 
                 sim_strategy='CTSF', seg_num=200, power=1):
    
    img_seg = felzenszwalb(image, scale=k, sigma=0.8, min_size=100)
    img_cvtcolor = label2rgb(img_seg, image, kind='mix')
    img_cvtcolor = switch_color_space(img_cvtcolor, color_space)
    S = HierarchicalGrouping(img_cvtcolor, img_seg, sim_strategy)
    S.build_regions()
    S.build_region_pairs()

    # Start hierarchical grouping
    
    while S.num_regions() > seg_num:
        
        i,j = S.get_highest_similarity()
        S.merge_region(i,j)
        S.remove_similarities(i,j)
        S.calculate_similarity_for_new_region()
    
    image = label2rgb(S.img_seg, image, kind='mix')
    image = (image+1)/2
    image = image**power
    image = image/np.max(image)
    image = image*2 - 1
    
    return image


def selective_adacolor(batch_image, seg_num=200, power=1):
    num_job = np.shape(batch_image)[0]
    batch_out = Parallel(n_jobs=num_job)(delayed(color_ss_map)\
                         (image, seg_num, power) for image in batch_image)
    return np.array(batch_out)


if __name__ == '__main__':
    pass