import torch
import cv2
import numpy as np
from model import Whitebox


net = Whitebox()
net.load_state_dict(torch.load("state_dict.pth"))
net.eval()
if torch.cuda.is_available():
    net.cuda()


def process(img):
    """
    inputs:
        - img: np.array from cv2.imread
    """
    img = img.astype(np.float32)/127.5 - 1
    h, w = img.shape[:2]
    h = (h//4)*4
    w = (w//4)*4
    img = img[:h, :w, :]
    img_tensor = torch.FloatTensor(img.transpose(2, 0, 1)).unsqueeze(0)
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    with torch.no_grad():
        output = net(img_tensor)
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
    # cv2.imwrite(fn+"_cartoonized"+ext, out)