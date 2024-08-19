import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
import os
os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'
import argparse
from BCGN.Abla_bcgn_noResnet import grasp_net
from BCGN.post_process import post_process_output
from datasets.LRGD import LRGDDataset
from datasets.collate_batch import BatchCollator
from datasets.dataset_processing import evaluation_anything
import random
import tqdm

save_path = './checkpoint/BCGN_abla_noResNet'

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
parser.add_argument('--warmup', default=5, type=int, help='warm up epoch')
args = parser.parse_args()

device = "cuda:3" if torch.cuda.is_available() else "cpu"

best_iou = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

Dataset = LRGDDataset ## VMRD dataset

train_dataset = Dataset(args.dataset_path,
                      output_size=args.input_size,
                      ds_rotate=args.ds_rotate,
                      # random_rotate=False,
                      # random_zoom=False,
                      # include_depth=args.use_depth,
                      # include_rgb=args.use_rgb,
                      trainval_test='trainval',
                        )
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

train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        # collate_fn=collator,
        drop_last=True
    )
test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        # collate_fn=collator,
    )

test_none_dataloader = torch.utils.data.DataLoader(
        test_dataset,
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

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

def select_none_for_train(imgs, instrus):  ##

    none_imgs = []
    none_intrus = []
    # none_GT = []
    sample_num = imgs.shape[0]
    none_sample_num = sample_num // 3
    img_index = list(range(0, none_sample_num))
    for idx in img_index:
        none_imgs.append(imgs[idx, :].unsqueeze(0).contiguous())
        tmp_instru_idx = random.choice([x for x in img_index if x != idx])
        none_intrus.append(instrus[tmp_instru_idx])
    none_imgs = torch.cat(none_imgs, dim=0)
    none_intrus = tuple(none_intrus)
    return none_imgs, none_intrus


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    total = 0

    for x_img, x_instru, y, _, _, _ in tqdm.tqdm(train_dataloader):

        none_imgs, none_tintrus = select_none_for_train(x_img, x_instru[0])
        full_imgs = torch.cat([x_img, none_imgs])
        full_instrus = x_instru[0] + none_tintrus


        avai_gt = torch.zeros((full_imgs.shape[0]), dtype=torch.int64)
        avai_gt[0:x_img.shape[0]] = 1

        # x_img = x_img.to(device)
        full_imgs = full_imgs.to(device)
        yc = [yy.to(device) for yy in y]
        avai_gt = avai_gt.to(device)

        optimizer.zero_grad()

        pos_pred, cos_pred, sin_pred, width_pred, is_avai = model(full_imgs, full_instrus)
        y_pos, y_cos, y_sin, y_width = yc

        p_loss = F.smooth_l1_loss(pos_pred[0:x_img.shape[0], :], y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred[0:x_img.shape[0], :], y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred[0:x_img.shape[0], :], y_sin)
        width_loss = F.smooth_l1_loss(width_pred[0:x_img.shape[0], :], y_width)

        loss_grasp = p_loss + cos_loss + sin_loss + width_loss
        loss_avai = criterion(is_avai, avai_gt)
        loss = loss_grasp + loss_avai

        loss = min((epoch + 1) / args.warmup, 1.0) * loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += x_img.shape[0]

    epoch_loss = train_loss / (total + 1)
    # epoch_acc = correct / total
    print('Epoch Training Loss: {:.4f}, Total samples: {:.1f}'.format(epoch_loss, total))


def test(epoch):
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
        for x_img, x_instru, y, index, rot, zoom_factor in tqdm.tqdm(test_dataloader):


            x_img = x_img.to(device)


            optimizer.zero_grad()
            pos_pred, cos_pred, sin_pred, width_pred, is_avai = model(x_img, x_instru[0])

            file_index, obj_idx, ins_idx = test_dataset._image_index[index].split('_')

            gtbbs = test_dataset.get_gtbb_gai(file_index, obj_idx, rot, zoom_factor)  ## bbs 抓取框 list； bbs_ind 抓取框list 对应的物体index

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

            total += 1
            if total == 300 and epoch < 47:
                print("test 80 batch, Done!")
                break

        print('%d/%d = %f' % (results['correct'], results['correct'] + results['failed'],
                                     results['correct'] / (results['correct'] + results['failed'])))

        # Save best performing network
        iou = results['correct'] / (results['correct'] + results['failed'])
        if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
            save_path_pth = os.path.join(save_path, 'epoch_%02d_iou_%0.2f' % (epoch, iou))
            state = {
                    'net': model.state_dict(),
                }
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            torch.save(state, save_path_pth)

            best_iou = iou


if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch+50):
        train(epoch)
        test(epoch)



