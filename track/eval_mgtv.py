import os
import json
from typing import *

import numpy as np

def parse_file(file_name: str) -> List[List[Tuple]]:
    ret = [ ]
    with open(file_name, 'r') as f:
        for line in f.readlines():
            cur = [ ]
            line = line.strip()
            frame_id, *points = line.split(' ')
            assert len(points) == 4, 'frame {} in file {} missing points'.format(frame_id, file_name)
            for point in points:
                x, y = map(float, point.strip().slpit(','))
                cur.append((x, y))
            ret.append(cur)
    return ret

def compute_one_frame(pred: List[Tuple], gt: List[Tuple]):
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    difference = pred - gt
    ret = np.sum(np.pow(difference, 2))
    return ret.item() / 4.0

def compute_one_video(pred_list, gt_list):
    N = len(pred_list) # num_frames
    pred = np.asarray(pred_list)
    gt = np.asarray(gt_list)
    difference = pred - gt
    ret = np.sum(np.pow(difference, 2))

    return ret.item() / (4 * N)


def eval_mse(result_root_path, gt_root_path):
    ret = 0.0
    count = 0
    for txt_name in os.lsdir(result_root_path):
        absolute_txt_name = os.path.join(result_root_path, txt_name)
        # txt_base_name is just video name(id)
        txt_base_name, extension = os.path.splitext(txt_name)
        absolute_gt_name = os.path.join(gt_root_path, txt_base_name, 'points.txt')

        pred_list = parse_file(absolute_txt_name)
        gt_list = parse_file(absolute_gt_name)

        assert len(pred_list) == len(gt_list), 'pred differs from gt in #frames'
        ret += compute_one_video(pred_list, gt_list)
        count += 1
    return ret / float(count)
