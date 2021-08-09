import numpy as np
import pydensecrf.densecrf as dcrf
import torch
import PIL.Image as Image
import os
import cv2


palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
           64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128,
           0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128,
           64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0]

def put_palette(seg_label, out_name):
    out = seg_label.astype(np.uint8)
    out = Image.fromarray(out, mode='P')
    out.putpalette(palette)
    out.save(out_name)

def show_cam_on_image(img, mask, img_name, save_path):
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cv2.imwrite(os.path.join(save_path, img_name + ".jpg"), cam)

def save_as_png(mask, save_name, save_path):
    out_name = save_name + '.png'
    out = mask * 255
    out = Image.fromarray(out.astype(np.uint8), mode='P')
    out.save(os.path.join(save_path, out_name))


def fusion_cam(cam, sal_images, label, args, name='toy'):
    if len(sal_images.size()) == 4:
        sal_images = sal_images.squeeze(1)

    cam = cam * label.unsqueeze(-1).unsqueeze(-1).cuda()
    cam_max = torch.max(cam, dim=1)[0]

    cam_object_region = cam_max > 0.2
    sal_object_region = sal_images > 0.06
    object_region = sal_object_region * cam_object_region
    not_sure_region1 = sal_object_region * (~cam_object_region)
    not_sure_region2 = (~sal_object_region) * (cam_object_region)
    not_sure_region = not_sure_region1 | not_sure_region2

    gt = torch.argmax(cam, dim=1) + 1
    gt = gt.float() * object_region.float()

    for b in range(gt.size(0)):
        gt[b, not_sure_region[b]] = 255
        seg_name = os.path.join(args.seg_pgt_path, name[b] + '.png')
        output_i = gt[b].detach().cpu().numpy()
        put_palette(output_i, seg_name)
    return gt.long()

def crf_inference_inf(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    img_c = np.ascontiguousarray(img)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=4/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=83/scale_factor, srgb=5, rgbim=np.copy(img_c), compat=3)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def crf_for_sal(img, anno, EPSILON=1e-8, tau=1.05):
    img = np.array(img)
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], 2)
    n_energy = -np.log((1.0 - anno + EPSILON)) / (tau * sigmoid(1 - anno))
    p_energy = -np.log(anno + EPSILON) / (tau * sigmoid(anno))
    U = np.zeros((2, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = np.expand_dims(infer[1, :].reshape(img.shape[:2]), 0)

    return res