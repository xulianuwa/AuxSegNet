import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
import importlib
from tool import imutils
import cv2
import os.path
import torch.nn.functional as F
from tqdm import tqdm

def _crf_with_alpha(pred_prob, ori_img):
    crf_score = imutils.crf_inference_inf(ori_img, pred_prob, labels=21)
    return crf_score


def run(args, step_index):

    seg_output_path = os.path.join(args.seg_output_path, str(step_index))
    os.makedirs(seg_output_path)
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    model = getattr(importlib.import_module('network.' + args.network), 'SegNet')(num_classes=args.num_classes)

    model_path = os.path.join(args.model_path, args.session_name + 's' + str(step_index) + '.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.cuda()
    im_path = args.image_path
    img_list = open(args.test_list).readlines()

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

            save_path = os.path.join(seg_output_path, i.strip() + '.png')
            cv2.imwrite(save_path, pred.astype(np.uint8))