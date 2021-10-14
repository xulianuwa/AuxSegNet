from torch.backends import cudnn
cudnn.enabled = True
import argparse
import os
import step.train_AuxAff, step.gen_pgt, step.infer_AuxAff


def str2bool(v):
    if v.lower() in ('yes','true','t','y','1','True'):
        return True
    elif v.lower() in ('no','false','f','n','0','False'):
        return False

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU_id')

    # Dataset parameters
    parser.add_argument("--train_list", default="voc12/train_aug_id.txt", type=str)
    parser.add_argument("--test_list", default="voc12/val_id.txt", type=str)
    parser.add_argument("--image_path", default="", type=str)
    parser.add_argument("--init_salpgt_path", default="", type=str, help='off-the-shelf saliency maps')
    parser.add_argument("--init_segpgt_path", default="", type=str, help='initial semantic segmentation pseudo label maps')
    parser.add_argument("--num_classes", default=21, type=int)

    # Parameters for training AuxSegNet
    parser.add_argument("--network", default='AuxSegNet', type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_epochs", default=15, type=int)
    parser.add_argument("--lr", default=0.0007, type=float)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--wt_dec", default=1e-5, type=float)
    parser.add_argument("--init_weights", default='', type=str)

    parser.add_argument("--session_name", default="AuxSegNet_", type=str)
    parser.add_argument("--crop_size", default=321, type=int)
    parser.add_argument('--print_intervals', type=int, default=50)

    parser.add_argument('--sal_loss_weight', type=float, default=1.0)
    parser.add_argument('--cls_loss_weight', type=float, default=1.0)
    parser.add_argument('--seg_loss_weight', type=float, default=1.0)

    parser.add_argument("--num_steps", default=4, type=int, help='number of steps for the iterative affinity learning')

    # Parameters for testing AuxSegNet
    parser.add_argument("--use_crf", default=False, type=str2bool)

    # Output paths
    parser.add_argument("--model_path", default=None, type=str, help='path to save trained AuxSegNet models')
    parser.add_argument("--sal_pgt_path", default=None, type=str, help='path to save updated saliency pgt')
    parser.add_argument("--seg_pgt_path", default=None, type=str, help='path to save updated semantic segmentation pgt')
    parser.add_argument("--seg_output_path", default=None, type=str, help='path to save semantic segmentation output results')

    args = parser.parse_args()

    for s in range(args.num_steps):
        if s > 0:
            args.init_weights = os.path.join(args.model_path, args.session_name + 's' + str(s-1) + '.pth')

        print(f'Training the AuxSegNet at the {s}-th step')
        step.train_AuxAff.run(args, step_index=s)
        print(f'Pseudo label updating at the {s}-th step')
        step.gen_pgt.run(args, step_index=s)

        if s == args.num_steps - 1:
            print('Testing the AuxSegNet')
            step.infer_AuxAff.run(args, step_index=s)
