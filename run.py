import os
from whiteboxcartoonization.cartoonize import cartoonize

model_path  = './whiteboxcartoonization/saved_models'
load_folder = './whiteboxcartoonization/test_images'
save_folder = './whiteboxcartoonization/cartoonized_images'

if not os.path.exists(save_folder):
    os.mkdir(save_folder)
cartoonize(load_folder, save_folder, model_path)
    