import os
from os.path import join, isdir
from os import makedirs
import argparse
import json
import numpy as np
import torch

import cv2
import time

from .util import crop_chw, gaussian_shaped_labels, cxy_wh_2_rect1, rect1_2_cxy_wh, cxy_wh_2_bbox
from .net import DCFNetCollection
from .eval_mgtv import eval_mse


class TrackerConfig(object):
    # These are the default hyper-params for DCFNet
    # OTB2013 / AUC(0.665)
    # feature_path = '/home/xqwang/projects/tracking/UDT/snapshots/crop_125_2.0/checkpoint.pth.tar'
    crop_sz = 125

    lambda0 = 1e-4
    padding = 2
    output_sigma_factor = 0.1
    interp_factor = 0.01
    # num_scale = 3
    num_scale = 1
    scale_step = 1.0275
    # scale_factor = scale_step ** (np.arange(num_scale) - num_scale / 2)
    scale_factor = [scale_step]
    min_scale_factor = 0.2
    max_scale_factor = 5
    scale_penalty = 0.9925
    # scale_penalties = torch.FloatTensor(scale_penalty ** (np.abs((np.arange(num_scale) - num_scale / 2)))).cuda()
    scale_penalties = torch.FloatTensor([scale_penalty]).cuda()

    net_input_size = [crop_sz, crop_sz]
    net_average_image = np.array([104, 117, 123]).reshape(-1, 1, 1).astype(np.float32)
    output_sigma = crop_sz / (1 + padding) * output_sigma_factor
    y = gaussian_shaped_labels(output_sigma, net_input_size)
    yf = torch.rfft(torch.Tensor(y).view(1, 1, crop_sz, crop_sz).cuda(), signal_ndim=2)
    cos_window = torch.Tensor(np.outer(np.hanning(crop_sz), np.hanning(crop_sz))).cuda()

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

class DCFNetCollectionTracker:
    def __init__(self, model_path):
        self.model_path = model_path

        # default parameter and load feature extractor network
        self.config = TrackerConfig()
        self.net = DCFNetCollection(self.config)
        self.net.load_param(self.model_path)
        self.net.eval().cuda()
    
    # dataset is just dataset_key
    def validate(self, dataset, experiment, data_root_path, gt_root_path, result_output_path):
        json_path = join(data_root_path, dataset + '.json')
        annos = json.load(open(json_path, 'r'))
        videos = sorted(annos.keys())

        # loop videos
        for video_id, video in enumerate(videos):  # run without resetting
            video_path_name = annos[video]['name']
            points = np.array(annos[video]['initial_points']).astype(np.float64)
            target_pos, target_sz = rect1_2_cxy_wh(np.array(annos[video]['initial_circum_rectangle']))
            
            image_files = [join(data_root_path, 'videos', video_path_name, frame_id) for frame_id in annos[video]['image_files']]
            n_images = len(image_files)

            im = cv2.imread(image_files[0])  # HxWxC

            # confine results
            min_sz = np.maximum(self.config.min_scale_factor * target_sz, 4)
            max_sz = np.minimum(im.shape[:2], self.config.max_scale_factor * target_sz)

            # crop template
            window_sz = target_sz * (1 + self.config.padding)
            bbox = cxy_wh_2_bbox(target_pos, window_sz)
            patch = crop_chw(im, bbox, self.config.crop_sz)

            target = patch - self.config.net_average_image
            self.net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda())

            res = [ points.tolist() ]  # save in .txt == list[list[list]]
            patch_crop = np.zeros((self.config.num_scale, patch.shape[0], patch.shape[1], patch.shape[2]), np.float32)
            for f in range(1, n_images):  # track
                im = cv2.imread(image_files[f])

                for i in range(self.config.num_scale):  # crop multi-scale search region
                    window_sz = target_sz * (self.config.scale_factor[i] * (1 + self.config.padding))
                    bbox = cxy_wh_2_bbox(target_pos, window_sz)
                    patch_crop[i, :] = crop_chw(im, bbox, self.config.crop_sz)

                search = patch_crop - self.config.net_average_image
                response = self.net(torch.Tensor(search).cuda())

                for pi in range(4):
                    value, index = torch.max(response[pi].view(self.config.num_scale, -1), dim = 1) # S
                    value *= self.config.scale_penalties
                    best_scale_id = torch.argmax(value)
                    r_max, c_max = unravel_index(index[best_scale_id], self.config.net_input_size)
                    if r_max > self.config.net_input_size[1] / 2:
                        r_max = r_max - self.config.net_input_size[1]
                    if c_max > self.config.net_input_size[0] / 2:
                        c_max = c_max - self.config.net_input_size[0]
                    window_sz = target_sz * (self.config.scale_factor[best_scale_id] * (1 + self.config.padding))
                    points[pi] += np.array([c_max.item(), r_max.item()]) * window_sz / self.config.net_input_size
                target_pos, target_sz = get_new_target(points)

                # model update
                window_sz = target_sz * (1 + self.config.padding)
                bbox = cxy_wh_2_bbox(target_pos, window_sz)
                patch = crop_chw(im, bbox, self.config.crop_sz)
                target = patch - self.config.net_average_image
                self.net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda(), lr=self.config.interp_factor)

                res.append(points.tolist())  # 1-index

            # save result
            txt_base_path = join(result_output_path, dataset, experiment)
            os.makedirs(txt_base_path, exist_ok = True)
            result_path = join(txt_base_path, video + '.txt')
            with open(result_path, 'w') as f:
                for fi, ps in enumerate(res):
                    f.write('{} '.format(fi))
                    for pi, p in enumerate(ps):
                        f.write('{:.2f},{:.2f}'.format(p[0], p[1]))
                        f.write('\n' if pi == len(ps) - 1 else ' ')
        mse = eval_mse(txt_base_path, gt_root_path)
        return mse
 
def get_new_target(points):
    maxxy = np.max(points, axis = 0)
    minxy = np.min(points, axis = 0)
    rect = np.array([ minxy[0], minxy[1], maxxy[0] - minxy[0], maxxy[1] - minxy[1] ])
    return rect1_2_cxy_wh(rect)
    
def test_main():
    dataset = 'mgtv_val'
    experiment = 'onebyone'
    model_path = join('/home/xqwang/projects/tracking/UDT/snapshots', experiment, 'crop_125_2.0/checkpoint.pth.tar')
    gt_root_path = '/home/xqwang/projects/tracking/datasets/mgtv/val'
    data_root_path = '/home/xqwang/projects/tracking/datasets/mgtv/val_preprocessed'
    result_output_path = '/home/xqwang/projects/tracking/UDT/result'     
    
    tracker = DCFNetCollectionTracker(model_path)
    mse = tracker.validate(dataset, experiment, data_root_path, gt_root_path, result_output_path)
    print('MSE on Val set: {:.4f}'.format(mse))
