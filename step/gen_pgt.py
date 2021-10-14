import torch
import torch.nn.functional as F
from torch.backends import cudnn
cudnn.enabled = True
import importlib
import os.path
from tqdm import tqdm
from tool.imutils import fusion_cam, put_palette, crf_for_sal, save_as_png
import cv2
import numpy as np
import sys
sys.path.append("..")

def run(args, step_index):

    sal_pgt_path = os.path.join(args.sal_pgt_path, str(step_index))
    os.makedirs(sal_pgt_path, exist_ok=True)

    seg_pgt_path = os.path.join(args.seg_pgt_path, str(step_index))
    os.makedirs(seg_pgt_path, exist_ok=True)
    model_path = os.path.join(args.model_path, args.session_name + 's' + str(step_index) + '.pth')

    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    cls_labels_dict = np.load('voc12/cls_labels.npy', allow_pickle=True).item()

    model = getattr(importlib.import_module('network.' + args.network), 'SegNet')(num_classes=args.num_classes)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.cuda()

    im_path = args.image_path
    img_list = open(args.train_list).readlines()

    with torch.no_grad():
        for idx in tqdm(range(len(img_list))):
            i = img_list[idx]
            img_temp = cv2.imread(os.path.join(im_path, i.strip() + '.jpg'))
            img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB).astype(np.float)
            img_original = img_temp.astype(np.uint8)
            img_temp[:, :, 0] = (img_temp[:, :, 0] / 255. - 0.485) / 0.229
            img_temp[:, :, 1] = (img_temp[:, :, 1] / 255. - 0.456) / 0.224
            img_temp[:, :, 2] = (img_temp[:, :, 2] / 255. - 0.406) / 0.225

            sal_temp_np = cv2.imread(os.path.join(args.init_salpgt_path, i.strip() + '.png'), 0) / 255.
            label_i = cls_labels_dict[i.strip()]
            label_i = torch.from_numpy(label_i)
            label_i = label_i.unsqueeze(0).cuda()
            input = torch.from_numpy(img_temp[np.newaxis, :].transpose(0, 3, 1, 2)).float().cuda()
            N, C, H, W = input.size()

            _, cam, b_mask, ref_bmask, init_seg, seg, ct_aff = model(x=input.cuda(), require_cam=True)

            cam = F.relu(cam.clone())
            cam = cam / (torch.max(torch.max(cam, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0] + 1e-5)
            refined_cam = cam.view(cam.size(0), cam.size(1), -1).permute(0, 2, 1).contiguous()
            refined_cam = torch.matmul(ct_aff, refined_cam)
            refined_cam = refined_cam.permute(0, 2, 1).contiguous().view(*cam.size())
            refined_cam = F.interpolate(refined_cam, size=(H,W), mode='bilinear', align_corners=False)

            ref_bmask = F.sigmoid(ref_bmask)
            ref_sal = F.interpolate(ref_bmask, size=(H, W), mode='bilinear', align_corners=False)
            ref_sal_np = ref_sal.detach().cpu().numpy().squeeze()
            fused_sal = (ref_sal_np + sal_temp_np) / 2
            fused_sal = crf_for_sal(img_original, fused_sal)
            save_as_png(fused_sal.squeeze(), i.strip(), sal_pgt_path)
            updated_sal = torch.from_numpy(fused_sal[np.newaxis, :]).cuda()

            pgt = fusion_cam(refined_cam, updated_sal, label_i, args, [i.strip()])
            pgt_np = pgt.detach().cpu().numpy().squeeze()
            save_name = os.path.join(seg_pgt_path, i.strip() + '.png')
            put_palette(pgt_np, save_name)