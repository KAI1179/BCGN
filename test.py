import torch
from torch.utils.data.sampler import Sampler
import os
os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'
import argparse
from BCGN.bcgn_full import grasp_net
from BCGN.post_process import post_process_output
from datasets.LRGD import LRGDDataset
from datasets.collate_batch import BatchCollator
from datasets.dataset_processing import evaluation_anything
import numpy as np
import tqdm


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.008, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--batch-size', default=64, type=int, help='aspect of erasing area')
parser.add_argument('--input-size', default=224, type=int, help='image size')
parser.add_argument('--dataset-path', default='./LRGD/', help='dataset path')
parser.add_argument('--ds-rotate', type=float, default=0.0,
                    help='Shift the start point of the dataset to use a different test/train split')
parser.add_argument('--num-workers', type=int, default=4,  ###8
                    help='Dataset workers')
parser.add_argument('--use-depth', type=int, default=0,
                    help='Use Depth image for training (1/0)')
parser.add_argument('--use-rgb', type=int, default=1,
                    help='Use RGB image for training (1/0)')
parser.add_argument('--iou-threshold', type=float, default=0.25, ##0.25
                    help='Threshold for IOU matching')
parser.add_argument('--avai-threshold', type=float, default=1.19,
                    help='Threshold for available grasping')

parser.add_argument('--warmup', default=5, type=int, help='warm up epoch')
args = parser.parse_args()

device = "cuda:3" if torch.cuda.is_available() else "cpu"

best_iou = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

Dataset = LRGDDataset ## VMRD dataset


test_dataset = Dataset(args.dataset_path,
                      output_size=args.input_size,
                      ds_rotate=args.ds_rotate,
                      # random_rotate=False,
                      # random_zoom=False,
                      # include_depth=args.use_depth,
                      # include_rgb=args.use_rgb,
                      trainval_test='test',
                       )

test_none_dataset = Dataset(args.dataset_path,
                      output_size=args.input_size,
                      ds_rotate=args.ds_rotate,
                      # random_rotate=False,
                      # random_zoom=False,
                      # include_depth=args.use_depth,
                      # include_rgb=args.use_rgb,
                      trainval_test='test_none_related',
                       )
collator = BatchCollator()

test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        # collate_fn=collator,
    )

test_none_dataloader = torch.utils.data.DataLoader(
        test_none_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        # collate_fn=collator,
    )
print('Loading dataset Done')

# Model
print('==> Building model..')

model = grasp_net(input_size=224, dropout=True, prob=0.1, device=device)
model.to(device)

checkpoint = torch.load('./checkpoint/BCGN_full/epoch_41_iou_0.63')  ## YOUR CHRCKPOINT
model.load_state_dict(checkpoint['net'])

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


def test(epoch):
    global best_iou
    model.eval()
    total = 0

    results = {
        'correct': 0,
        'failed': 0,
        'avai_corr': 0,
        'loss': 0,
        'losses': {

        }
    }
    with torch.no_grad():
        for x_img, x_instru, y, index, rot, zoom_factor in tqdm.tqdm(test_dataloader):
            x_img = x_img.to(device)
            pos_pred, cos_pred, sin_pred, width_pred, is_avai = model(x_img, x_instru[0])
            is_avai = is_avai.detach().cpu().squeeze().numpy()
            is_avai = np.argmax(is_avai)

            if is_avai == 1:
                results['avai_corr'] += 1
                file_index, obj_idx, ins_idx = test_dataset._image_index[index].split('_')

                gtbbs = test_dataset.get_gtbb_gai(file_index, obj_idx, rot, zoom_factor)
                q_out, ang_out, w_out = post_process_output(pos_pred, cos_pred, sin_pred, width_pred)
                s = evaluation_anything.calculate_iou_match(q_out,
                                                   ang_out,
                                                   gtbbs,
                                                   no_grasps=1,
                                                   grasp_width=w_out,
                                                   threshold=args.iou_threshold
                                                   )
                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1
            else:
                results['failed'] += 1

            total += 1
        print("correct: {}; failed: {}; total: {}".format(results['correct'], results['failed'], results['correct'] + results['failed']))
        print('acc = %d/%d = %f' % (results['correct'], results['correct'] + results['failed'], results['correct'] / (results['correct'] + results['failed'])))
        return results['correct'], results['failed'], results['avai_corr']


def test_none(epoch):
    global best_iou
    model.eval()
    total = 0

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }
    with torch.no_grad():
        for x_img, x_instru, y, index, rot, zoom_factor in tqdm.tqdm(test_none_dataloader):

            x_img = x_img.to(device)
            pos_pred, cos_pred, sin_pred, width_pred, is_avai = model(x_img, x_instru[0])
            is_avai = is_avai.detach().cpu().squeeze().numpy()
            is_avai = np.argmax(is_avai)
            if is_avai == 1:  ##
                results['failed'] += 1
            else:
                results['correct'] += 1
            total += 1
        print("correct: {}; failed: {}; total: {}".format(results['correct'], results['failed'], results['correct'] + results['failed']))
        print('acc = %d/%d = %f' % (results['correct'], results['correct'] + results['failed'], results['correct'] / (results['correct'] + results['failed'])))
        return results['correct'], results['failed']


if __name__ == '__main__':

    norm_corr, norm_fail, avai_corr = test(1)
    none_corr, none_fail = test_none(1)
    acc_all = (norm_corr + none_corr) / (norm_fail + none_fail + norm_corr + none_corr)
    print('acc all = %f' % acc_all)
    acc_avai = (avai_corr + none_corr) / (norm_fail + none_fail + norm_corr + none_corr)
    print('acc avai = %f' % acc_avai)
    print('norm avai_corr = %d' % avai_corr)



