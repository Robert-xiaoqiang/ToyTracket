import os
from os.path import join, isdir
from os import mkdir
import argparse
import numpy as np
import json
import cv2
import time
import pdb
# from __future__ import print_function

parse = argparse.ArgumentParser(description='Generate training data (cropped) for DCFNet_pytorch')
parse.add_argument('-v', '--visual', dest='visual', action='store_true', help='whether visualise crop')
parse.add_argument('-o', '--output_size', dest='output_size', default=125, type=int, help='crop output size')
parse.add_argument('-p', '--padding', dest='padding', default=2, type=float, help='crop padding size')

args = parse.parse_args()

print(args)


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    bbox = [float(x) for x in bbox]
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def cxy_wh_2_bbox(cxy, wh):
    return np.array([cxy[0] - wh[0] / 2, cxy[1] - wh[1] / 2, cxy[0] + wh[0] / 2, cxy[1] + wh[1] / 2])  # 0-index

def main1():
    vid = json.load(open('vid.json', 'r'))

    num_all_frame = 405014
    num_val = 3000
    # crop image
    lmdb = dict()
    lmdb['down_index'] = np.zeros(num_all_frame, np.int)  # buff
    lmdb['up_index'] = np.zeros(num_all_frame, np.int)

    crop_base_path = 'crop_{:d}_{:1.1f}'.format(args.output_size, args.padding)
    if not isdir(crop_base_path):
        mkdir(crop_base_path)

    count = 0
    begin_time = time.time()
    for subset in vid:
        for video in subset:
            frames = video['frame']
            n_frames = len(frames)
            for f, frame in enumerate(frames):
                img_path = join(video['base_path'], frame['img_path'])
                im = cv2.imread(img_path)
                avg_chans = np.mean(im, axis=(0, 1))
                img_sz = frame['frame_sz']
        
                target_pos = [img_sz[0]/2, img_sz[1]/2]
                target_sz = [img_sz[0]/6, img_sz[1]/6]
                window_sz = np.array(target_sz) * (1 + args.padding)
                crop_bbox = cxy_wh_2_bbox(target_pos, window_sz)
                patch = crop_hwc(im, crop_bbox, args.output_size)
                cv2.imwrite(join(crop_base_path, '{:08d}.jpg'.format(count)), patch)
                # cv2.imwrite('crop.jpg'.format(count), patch)

                lmdb['down_index'][count] = f
                lmdb['up_index'][count] = n_frames - f
                count += 1
                if count % 100 == 0:
                    elapsed = time.time() - begin_time
                    print("Processed {} images in {:.2f} seconds. "
                            "{:.2f} images/second.".format(count, elapsed, count / elapsed))

    template_id = np.where(lmdb['up_index'] > 1)[0]  # NEVER use the last frame as template! I do not like bidirectional.
    rand_split = np.random.choice(len(template_id), len(template_id), replace = False)
    lmdb['train_set'] = template_id[rand_split[:(len(template_id)-num_val)]]
    lmdb['val_set'] = template_id[rand_split[(len(template_id)-num_val):]]
    print(len(lmdb['train_set']))
    print(len(lmdb['val_set']))

    # to list for json
    lmdb['train_set'] = lmdb['train_set'].tolist()
    lmdb['val_set'] = lmdb['val_set'].tolist()
    lmdb['down_index'] = lmdb['down_index'].tolist()
    lmdb['up_index'] = lmdb['up_index'].tolist()

    print('lmdb json, please wait 5 seconds~')
    json.dump(lmdb, open('dataset.json', 'w'), indent=2)
    print('done!')


def main2(train_root_path):
    vid = json.load(open(os.path.join(train_root_path, 'vid.json'), 'r'))

    num_all_frame = vid['total_frame']
    num_val = 3000
    # crop image
    lmdb = dict()
    lmdb['down_index'] = np.zeros(num_all_frame, np.int)  # buff
    lmdb['up_index'] = np.zeros(num_all_frame, np.int)

    crop_base_path = os.path.join(train_root_path, 'crop_{:d}_{:1.1f}'.format(args.output_size, args.padding))
    os.makedirs(crop_base_path, exist_ok = True)

    count = 0
    begin_time = time.time()
    for video in vid['videos']:
        frames = video['frame']
        n_frames = len(frames)
        for f, frame in enumerate(frames):
            img_path = join(video['base_path'], frame['img_path'])
            im = cv2.imread(img_path)

            # img_sz = frame['frame_sz']
            # target_pos = [img_sz[0]/2, img_sz[1]/2]
            # target_sz = [img_sz[0]/6, img_sz[1]/6]
            # target_pos, target_sz = rect1_2_cxy_wh(video['initial_circum_rectangle'])

            # window_sz = np.array(target_sz) * (1 + args.padding)
            # crop_bbox = cxy_wh_2_bbox(target_pos, window_sz)
            # patch = crop_hwc(im, crop_bbox, args.output_size)

            patch = cv2.resize(im, (args.output_size, args.output_size))
            cv2.imwrite(join(crop_base_path, '{:05d}.jpg'.format(count)), patch)

            lmdb['down_index'][count] = f
            lmdb['up_index'][count] = n_frames - f
            count += 1
            if count % 100 == 0:
                elapsed = time.time() - begin_time
                print("Processed {} images in {:.2f} seconds. "
                        "{:.2f} images/second.".format(count, elapsed, count / elapsed))

    template_id = np.where(lmdb['up_index'] > 1)[0]  # NEVER use the last frame as template! I do not like bidirectional.
    # rand_split = np.random.choice(len(template_id), len(template_id), replace = False)
    lmdb['train_set'] = template_id
    lmdb['val_set'] = None
    print(len(lmdb['train_set']))
    # print(len(lmdb['val_set']))

    # to list for json
    lmdb['train_set'] = lmdb['train_set'].tolist()
    lmdb['val_set'] = None
    lmdb['down_index'] = lmdb['down_index'].tolist()
    lmdb['up_index'] = lmdb['up_index'].tolist()

    print('lmdb json, please wait 5 seconds~')
    json_file_name = os.path.join(train_root_path, 'dataset.json')
    with open(json_file_name, 'w') as f:
        json.dump(lmdb, f, indent=2)
    print('done!')

if __name__ == '__main__':
    train_preprocessed_path = '/home/xqwang/projects/tracking/datasets/mgtv/train_preprocessed'
    main2(train_preprocessed_path)
