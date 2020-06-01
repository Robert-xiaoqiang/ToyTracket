import cv2
from PIL import Image
import os
import json
from typing import *
import numpy as np

mgtv_root_path = '/home/xqwang/projects/tracking/datasets/mgtv/train'
prepross_output_path = '/home/xqwang/projects/tracking/datasets/mgtv/train_preprocessed'
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
    # top-left(x, y), width, height
    return l, [ minx, miny, maxx - minx, maxy - miny ]
def main():
    vid = {
        'total_frame': 0,
        'videos': [ ]
    }
    total_frame = 0
    for video_path in sorted(os.listdir(mgtv_root_path)):
        initial_points, circum_rectangle = parse_txt(os.path.join(mgtv_root_path, video_path, 'points.txt'))
        video_dict = {
            'base_path': os.path.join(mgtv_root_path, video_path),
            'initial_points': initial_points,
            'initial_circum_rectangle': circum_rectangle,
            'frame': [ ]
        }
        video_file_name = os.path.join(mgtv_root_path, video_path, 'video.mp4')
        frame = [ ]
        vidcap = cv2.VideoCapture(video_file_name)
        success = True
        count = 0
        while True:
            success, image = vidcap.read()
            if not success: break
            f_name = '{:05d}.JPEG'.format(count)
            absolute_f_name = os.path.join(prepross_output_path, 'videos', video_path, f_name)
            os.makedirs(absolute_f_name, exist_ok = True)
            cv2.imwrite(absolute_f_name, image)
            f = {
                'frame_sz': [ image.shape[1], image.shape[0] ],
                'img_path': f_name
            }
            frame.append(f)
            count += 1
            total_frame += 1
        video_dict['frame'] = frame
        vid['videos'].append(video_dict)
        print('video: {} has {} frames, cirrumrectangle: {}'.format(video_path, count, circum_rectangle))
    vid['total_frame'] = total_frame
    print('save json (raw vid info), please wait 1 min~')
    print('total frame number: {}'.format(total_frame)) # change crop_image.py according to it
    json_file_name = os.path.join(prepross_output_path, 'vid.json')
    with open(json_file_name, 'w') as f:
        json.dump(vid, f, indent=2)
    print('done!')

if __name__ == '__main__':
    main()