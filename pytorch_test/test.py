import torch
import cv2
import numpy as np
from model import Whitebox


net = Whitebox()
net.load_state_dict(torch.load("state_dict.pth"))
net.eval()
if torch.cuda.is_available():
    net.cuda()


def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]
    return image


def process(img):
    """
    inputs:
        - img: np.array from cv2.imread
    """
    img = resize_crop(img)
    img = img.astype(np.float32)/127.5 - 1
    img_tensor = torch.FloatTensor(img.transpose(2, 0, 1)).unsqueeze(0)
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    with torch.no_grad():
        output = net(img_tensor, r=1, eps=5e-3)
    output = 127.5*(output.cpu().numpy().squeeze().transpose(1, 2, 0)+1)
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output


if __name__ == "__main__":
    import sys
    import os
    im_path = sys.argv[1]
    out = process(cv2.imread(im_path))
    fn, ext = os.path.splitext(im_path)
    cv2.imwrite("res.jpg", out)
