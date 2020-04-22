import argparse
import os
import argparse
import os
import time

import torch
import torch.nn as nn
import numpy as np

import dataset
import modeling

from lib.utils import AverageMeter, save_prediction, idx2onehot
from lib.predict import predict, prepare_first_frame

parser = argparse.ArgumentParser()
parser.add_argument('--ref_num', '-n', type=int, default=9,
                    help='number of reference frames for inference')
parser.add_argument('--data', type=str,
                    help='path to inference dataset')
parser.add_argument('--resume', '-r', type=str,
                    help='path to the resumed checkpoint')
parser.add_argument('--model', type=str, default='resnet50',
                    help='network architecture, resnet18, resnet50 or resnet101')
parser.add_argument('--start_idx', type=int, default=0,
                    help='video index to start from')
parser.add_argument('--temperature', '-t', type=float, default=1.0,
                    help='temperature parameter')
parser.add_argument('--range', type=int, default=40,
                    help='range of frames for inference')
parser.add_argument('--sigma1', type=float, default=15.0,
                    help='smaller sigma in the motion model for dense spatial weight')
parser.add_argument('--sigma2', type=float, default=21.0,
                    help='bigger sigma in the motion model for sparse spatial weight')
parser.add_argument('--all_frames', action='store_true',
                    help='use all valid frames to inference')
parser.add_argument('--save', '-s', type=str,
                    help='path to save predictions')

device = torch.device("cuda")


def main():
    global args
    args = parser.parse_args()

    model = modeling.VOSNet(model=args.model)
    model = nn.DataParallel(model)
    model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    model.eval()

    if args.all_frames:
        data_dir = os.path.join(args.data, 'valid_all_frames/JPEGImages')
    else:
        data_dir = os.path.join(args.data, 'valid/JPEGImages')
    inference_dataset = dataset.YouTubeInference(data_dir)
    inference_loader = torch.utils.data.DataLoader(inference_dataset,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=8)
    inference(inference_loader, model, args)


def inference(inference_loader, model, args):
    global pred_visualize, palette, d, feats_history, label_history, weight_dense, weight_sparse
    batch_time = AverageMeter()
    annotation_dir = os.path.join(args.data, 'valid/Annotations/')
    annotation_list = sorted(os.listdir(annotation_dir))

    last_video = args.start_idx
    frame_idx = 0
    additional_idx = 0
    with torch.no_grad():
        for i, (input, curr_video, img_original, img_number) in enumerate(inference_loader):
            if curr_video < args.start_idx:
                continue
            if curr_video != last_video:
                # save prediction
                pred_visualize = pred_visualize.cpu().numpy()
                for f in range(1, frame_idx):
                    save_path = args.save
                    save_name = str(img_numbers[f-1]).zfill(5)
                    video_name = annotation_list[last_video]
                    save_prediction(np.asarray(pred_visualize[f - 1], dtype=np.int32),
                                    palette, save_path, save_name, video_name, original_shape)

                frame_idx = 0
                additional_idx = 0
                print("End of video %d. Processing a new annotation..." % (last_video + 1))
            if frame_idx == 0:
                input = input.to(device)
                with torch.no_grad():
                    feats_history = model(input)
                img_numbers = []
                label_history, d, palette, \
                weight_dense, weight_sparse, \
                additional_labels, original_shape = prepare_first_frame(curr_video,
                                                                     args.save,
                                                                     annotation_dir,
                                                                     args.sigma1,
                                                                     args.sigma2)
                frame_sentry = -1
                additional_label = None
                if additional_idx < len(additional_labels):
                    frame_sentry, additional_label = additional_labels[additional_idx]
                    additional_idx += 1
                frame_idx += 1
                last_video = curr_video
                continue
            img_numbers.append(img_number.item())
            (batch_size, num_channels, H, W) = input.shape
            input = input.to(device)

            start = time.time()
            features = model(input)
            (_, feature_dim, H_d, W_d) = features.shape
            prediction = predict(feats_history,
                                 features[0],
                                 label_history,
                                 weight_dense,
                                 weight_sparse,
                                 frame_idx,
                                 d,
                                 args
                                 )
            new_label = torch.argmax(prediction, 0, keepdim=True)  # (1, H*W)

            # if there are additional objects
            if img_number.item() == frame_sentry:
                new_obj_idx = int(np.max(additional_label))
                d = max(d, new_obj_idx + 1)
                # merge additional label into current prediction
                additional_label_mask = additional_label > 0
                additional_label_mask = torch.Tensor(additional_label_mask).bool().cuda().view(1, -1)
                new_label[additional_label_mask] = new_obj_idx
                additional_idx += 1
                if additional_idx < len(additional_labels):
                    frame_sentry, additional_label = additional_labels[additional_idx]
                else:
                    frame_sentry = -1
                    additional_label = None

            # Store all frames' features and ref labels
            label_history = torch.cat((label_history, new_label), 0)
            feats_history = torch.cat((feats_history, features), 0)

            # 1. upsample, 2. argmax
            prediction = torch.nn.functional.interpolate(prediction.view(1, -1, H_d, W_d),
                                                         size=(H, W),
                                                         mode='bilinear',
                                                         align_corners=False)
            prediction = torch.argmax(prediction, 1)  # (1, H, W)

            last_video = curr_video
            frame_idx += 1

            if frame_idx == 2:
                pred_visualize = prediction
            else:
                pred_visualize = torch.cat((pred_visualize, prediction), 0)

            batch_time.update(time.time() - start)

            if i % 10 == 0:
                print('Validate: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                    i, len(inference_loader), batch_time=batch_time))
        # save last video's prediction
        pred_visualize = pred_visualize.cpu().numpy()
        for f in range(1, frame_idx):
            save_path = args.save
            save_name = str(img_numbers[f-1]).zfill(5)
            video_name = annotation_list[last_video]
            save_prediction(np.asarray(pred_visualize[f - 1], dtype=np.int32),
                            palette, save_path, save_name, video_name)
    print('Finished inference.')


if __name__ == '__main__':
    main()
