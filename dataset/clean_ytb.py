'''
    remove redundant jpeg images without annotations
'''
import os

root = '/mnt/YouTube-VOS/valid'
annotations = os.path.join(root, 'Annotations')
images = os.path.join(root, 'JPEGImages')
video_list = sorted(os.listdir(images))

images_24fps = '/mnt/YouTube-VOS/valid_all_frames/JPEGImages'

def clean_valid_5fps():
    for video in video_list:
        imgs_path = os.path.join(images, video)
        annotations_path = os.path.join(annotations, video)

        imgs_list = sorted(os.listdir(imgs_path))
        annotations_list = sorted(os.listdir(annotations_path))

        first_annotation = annotations_list[0][:-4]
        for img in imgs_list:
            if img[:-4] != first_annotation:
                os.remove(os.path.join(imgs_path, img))
                print(video, "img: ", img, "first annotation: ", first_annotation)
            else:
                break

def clean_valid_24fps():
    for video in video_list:
        imgs_path = os.path.join(images, video)
        imgs_list = sorted(os.listdir(imgs_path))

        imgs_24fps_path = os.path.join(images_24fps, video)
        imgs_24fps_list = sorted(os.listdir(imgs_24fps_path))

        first_img = int(imgs_list[0][:-4])
        last_img = int(imgs_list[-1][:-4])
        for img_24fps in imgs_24fps_list:
            frame_idx = int(img_24fps[:-4])
            if frame_idx < first_img or frame_idx > last_img:
                os.remove(os.path.join(imgs_24fps_path, img_24fps))

        imgs_24fps_list = sorted(os.listdir(imgs_24fps_path))
        if imgs_24fps_list[0] != imgs_list[0] or imgs_24fps_list[-1] != imgs_list[-1]:
            print(video)
    print("success")



