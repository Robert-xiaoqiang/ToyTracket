import cv2
from PIL import Image
import os
import json
from typing import *
import numpy as np

mgtv_root_path = '/home/xqwang/projects/tracking/datasets/mgtv/val'
prepross_output_path = '/home/xqwang/projects/tracking/datasets/mgtv/val_preprocessed'
# TO-DO
# counter-clockwise ordering
def ccworder(points: list) -> list:
    pass

def parse_txt(file_name: str) -> List[tuple]:
    l = [ ]
    minx = miny = 65535
    maxx = maxy = -1
    with open(file_name, 'r') as f:
        content = f.readline().strip()
        frame_id, *ps = content.split(' ')
        for p in ps:
            x, y = map(float, p.split(','))
            minx, maxx = min(x, minx), max(x, maxx)
            miny, maxy = min(y, miny), max(y, maxy)
            l.append((x, y))
    return l, [ minx, miny, maxx, maxy ]

def main():
    testset = { }

    for video_path in sorted(os.listdir(mgtv_root_path)):
        initial_points, circum_rectangle = parse_txt(os.path.join(mgtv_root_path, video_path, 'target_points.txt'))
        video_dict = {
            'name': video_path,
            'initial_points': initial_points,
            'initial_circum_rectangle': circum_rectangle,
            'image_files': [ ]
        }
        video_file_name = os.path.join(mgtv_root_path, video_path, 'raw.mp4')
        frame = [ ]
        vidcap = cv2.VideoCapture(video_file_name)
        success = True
        count = 0
        while True:
            success, image = vidcap.read()
            if not success: break
            f_name = '{:05d}.JPEG'.format(count)
            f_dir_name = os.path.join(prepross_output_path, 'videos', video_path)
            os.makedirs(f_dir_name, exist_ok = True)
            absolute_f_name = os.path.join(f_dir_name, f_name)
            cv2.imwrite(absolute_f_name, image)

            frame.append(f_name)
            count += 1

        video_dict['image_files'] = frame
        testset[video_path] = video_dict
        print('testing video: {} has {} frames, cirrumrectangle: {}'.format(video_path, count, circum_rectangle))

    json_file_name = os.path.join(prepross_output_path, 'mgtv_val.json')
    with open(json_file_name, 'w') as f:
        json.dump(testset, f, indent=2)
    print('done!')

if __name__ == '__main__':
    main()