import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
import importlib
from tool import imutils
import argparse
import cv2
import os.path
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

def _crf_with_alpha(pred_prob, ori_img):
    crf_score = imutils.crf_inference_inf(ori_img, pred_prob, labels=21)
    return crf_score

def str2bool(v):
    if v.lower() in ('yes','true','t','y','1','True'):
        return True
    elif v.lower() in ('no','false','f','n','0','False'):
        return False


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU_id')
    parser.add_argument("--weights", default="", type=str)
    parser.add_argument("--network", default="AuxSegNet", type=str)
    parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument("--LISTpath", default="./voc12/val_id.txt", type=str)
    parser.add_argument("--IMpath", default="", type=str)
    parser.add_argument("--num_classes", default=21, type=int)
    parser.add_argument("--use_crf", default=False, type=str2bool)
    args = parser.parse_args()

    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    model = getattr(importlib.import_module('network.' + args.network), 'SegNet')(num_classes=args.num_classes)

    model.load_state_dict(torch.load(args.weights))
    model.eval()
    model.cuda()
    im_path = args.IMpath
    img_list = open(args.LISTpath).readlines()

    with torch.no_grad():
        for idx in tqdm(range(len(img_list))):
            i = img_list[idx]

            img_temp = cv2.imread(os.path.join(im_path, i.strip() + '.jpg'))
            img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB).astype(np.float)
            img_original = img_temp.astype(np.uint8)

            img_temp[:, :, 0] = (img_temp[:, :, 0] / 255. - 0.485) / 0.229
            img_temp[:, :, 1] = (img_temp[:, :, 1] / 255. - 0.456) / 0.224
            img_temp[:, :, 2] = (img_temp[:, :, 2] / 255. - 0.406) / 0.225

            input = torch.from_numpy(img_temp[np.newaxis, :].transpose(0, 3, 1, 2)).float().cuda()

            N, C, H, W = input.size()
            init_prob, prob = model(x=input, require_cls=False, require_sal=False)

            prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=False)
            prob = F.softmax(prob, dim=1)
            output = prob.cpu().data[0].numpy()

            if args.use_crf:
                crf_output = _crf_with_alpha(output, img_original)
                pred = np.argmax(crf_output, 0)
            else:
                pred = np.argmax(output, axis=0)

            save_path = os.path.join(args.save_path, i.strip() + '.png')
            cv2.imwrite(save_path, pred.astype(np.uint8))