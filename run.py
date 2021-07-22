import os
from cartoonbox.cartoonize import cartoonize

model_path  = './cartoonbox/models'
load_folder = './cartoonbox/input'
save_folder = './cartoonbox/output'

if not os.path.exists(save_folder):
    os.mkdir(save_folder)
cartoonize(load_folder, save_folder, model_path)