# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# multi-gpu test for epochs
# ------------------------------------------------------------------------------

import os
import time
import argparse
from mpi4py import MPI


parser = argparse.ArgumentParser(description='multi-gpu test all epochs')
parser.add_argument('--start_epoch', default=160, type=int,  help='start epoch')
parser.add_argument('--end_epoch', default=240, type=int,  help='end epoch')
parser.add_argument('--gpu_nums', default=8, type=int,  help='test start epoch')
parser.add_argument('--threads', default=16, type=int)
args = parser.parse_args()

# init gpu and epochs
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
GPU_ID = rank % args.gpu_nums
node_name = MPI.Get_processor_name()  # get the name of the node
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
print("node name: {}, GPU_ID: {}".format(node_name, GPU_ID))
time.sleep(rank * 5)

# run test scripts -- two epoch for each thread
for i in range(2):
    try:
        epoch_ID += args.threads // 2 * 5   # for 16 queue
    except:
        epoch_ID = 5 * rank % (args.end_epoch - args.start_epoch + 1) + args.start_epoch

    if epoch_ID > args.end_epoch:
        continue

    resume = 'checkpoints/checkpoint-epoch-{}.pth.tar'.format(epoch_ID)
    print('==> test {}th epoch'.format(epoch_ID))
    
    save_path = os.path.join('results/ck{}'.format(epoch_ID))
    os.system('python inference.py -r {} -s {}'.format(resume, save_path))
