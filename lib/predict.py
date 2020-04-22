import os
import torch
import numpy as np

import torchvision.transforms.functional as f
from PIL import Image

from .utils import idx2onehot


def predict(ref,
            target,
            ref_label,
            weight_dense,
            weight_sparse,
            frame_idx,
            d,
            args):
    """
    The Predict Function.
    :param ref: (N, feature_dim, H, W)
    :param target: (feature_dim, H, W)
    :param ref_label: (N, H*W)
    :param weight_dense: (H*W, H*W)
    :param weight_sparse: (H*W, H*W)
    :param frame_idx:
    :param d: number of objects
    :param args:
    :return: (d, H*W)
    """
    # sample frames from history features
    # d = ref_label.shape[0]
    sample_idx = sample_frames(frame_idx, args.range, args.ref_num)
    ref_selected = ref.index_select(0, sample_idx)
    ref_label_selected = ref_label.index_select(0, sample_idx).view(-1) # (n*H*W)
    ref_label_selected = idx2onehot(ref_label_selected, d) # (d, n*H*W)

    # get similarity matrix
    (num_ref, feature_dim, H, W) = ref_selected.shape
    ref_selected = ref_selected.permute(0, 2, 3, 1).reshape(-1, feature_dim)
    target = target.reshape(feature_dim, -1)
    global_similarity = ref_selected.mm(target)

    # temperature step
    global_similarity *= args.temperature

    # softmax
    global_similarity = global_similarity.softmax(dim=0)

    # spatial weight and motion model
    global_similarity = global_similarity.contiguous().view(num_ref, H * W, H * W)
    global_similarity = global_similarity.mul(weight_dense)
    global_similarity = global_similarity.view(-1, H * W)

    # get prediction
    prediction = ref_label_selected.mm(global_similarity)
    return prediction


def sample_frames(frame_idx,
                  take_range,
                  num_refs):
    if frame_idx <= num_refs:
        sample_idx = list(range(frame_idx))
    else:
        dense_num = 4 - 1
        sparse_num = num_refs - dense_num
        target_idx = frame_idx
        ref_end = target_idx - dense_num - 1
        ref_start = max(ref_end - take_range, 0)
        sample_idx = np.linspace(ref_start, ref_end, sparse_num).astype(np.int).tolist()
        for j in range(dense_num):
            sample_idx.append(target_idx - dense_num + j)

    return torch.Tensor(sample_idx).long().cuda()


def prepare_first_frame(curr_video,
                        save_prediction,
                        annotation_dir,
                        sigma1=8,
                        sigma2=21):
    annotation_list = sorted(os.listdir(annotation_dir))
    annotation_dir = os.path.join(annotation_dir, annotation_list[curr_video])
    label_list = sorted(os.listdir(annotation_dir))
    print("Found %d label files in No.%d video." % (len(label_list), curr_video))
    first_annotation = Image.open(os.path.join(annotation_dir, label_list[0]))
    original_shape = np.asarray(first_annotation).shape
    # resize to 480p
    first_annotation = f.resize(first_annotation, 480)
    (H, W) = np.asarray(first_annotation).shape
    H_d = int(np.ceil(H / 8))
    W_d = int(np.ceil(W / 8))
    palette = first_annotation.getpalette()
    first_annotation = f.resize(first_annotation, (H_d, W_d))
    label = np.asarray(first_annotation)
    d = np.max(label) + 1
    label = torch.from_numpy(label).long().cuda().view(1, -1)
    weight_dense = get_spatial_weight((H_d, W_d), sigma1)
    weight_sparse = get_spatial_weight((H_d, W_d), sigma2)

    if save_prediction is not None:
        if not os.path.exists(save_prediction):
            os.makedirs(save_prediction)
        save_path = os.path.join(save_prediction, annotation_list[curr_video])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        first_annotation.save(os.path.join(save_path, label_list[0]))

    additional_labels = []
    for i in range(len(label_list)-1):
        additional_frame_idx = int(label_list[i+1][:-4])
        additional_annotation = Image.open(os.path.join(annotation_dir, label_list[i+1]))
        additional_annotation = f.resize(additional_annotation, (H_d, W_d))
        additional_annotation = np.asarray(additional_annotation)
        additional_labels.append([additional_frame_idx, additional_annotation])

    return label, d, palette, weight_dense, weight_sparse, additional_labels, original_shape


def get_spatial_weight(shape, sigma):
    """
    Get soft spatial weights for similarity matrix.
    :param shape: (H, W)
    :param sigma:
    :return: (H*W, H*W)
    """
    (H, W) = shape

    index_matrix = torch.arange(H * W, dtype=torch.long).reshape(H * W, 1).cuda()
    index_matrix = torch.cat((index_matrix / W, index_matrix % W), -1)  # (H*W, 2)
    d = index_matrix - index_matrix.unsqueeze(1)  # (H*W, H*W, 2)
    d = d.float().pow(2).sum(-1)  # (H*W, H*W)
    w = (- d / sigma ** 2).exp()

    return w
