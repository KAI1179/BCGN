import numpy as np
import cv2
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model.utils.config import cfg
# from model.utils.net_utils import create_mrt
import time
import datetime

def draw_single_grasp(self, img, grasp, test_str=None, text_bg_color=(255, 0, 0)):
    gr_c = (int((grasp[0] + grasp[4]) / 2), int((grasp[1] + grasp[5]) / 2))
    for j in range(4):
        if j % 2 == 0:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        p1 = (int(grasp[2 * j]), int(grasp[2 * j + 1]))
        p2 = (int(grasp[(2 * j + 2) % 8]), int(grasp[(2 * j + 3) % 8]))
        cv2.line(img, p1, p2, color, 2)

    # put text
    if test_str is not None:
        text_len = len(test_str)
        text_w = 17 * text_len
        gtextpos = (gr_c[0] - text_w / 2, gr_c[1] + 20)
        gtext_lu = (gr_c[0] - text_w / 2, gr_c[1])
        gtext_rd = (gr_c[0] + text_w / 2, gr_c[1] + 25)
        cv2.rectangle(img, gtext_lu, gtext_rd, text_bg_color, -1)
        cv2.putText(img, test_str, gtextpos,
                    cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 255, 255), thickness=2)
    return img

