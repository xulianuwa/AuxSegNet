import torch
import torch.nn as nn
import torch.nn.functional as F
import network.resnet38d
from network.non_local import NLBlockND


class Net(network.resnet38d.Net):
    def __init__(self, num_classes=21):
        super().__init__()

        self.dropout7 = torch.nn.Dropout2d(0.5)

        self.fc8 = nn.Conv2d(4096, num_classes - 1, (1, 1), bias=False)
        torch.nn.init.xavier_uniform_(self.fc8.weight)

        self.seg_conv1 = nn.Conv2d(4096, 128, (3, 3), padding=6, dilation=(6, 6), bias=True)
        torch.nn.init.xavier_uniform_(self.seg_conv1.weight)

        self.seg_conv2 = nn.Conv2d(128, 128, (3, 3), padding=12, dilation=(12, 12), bias=True)
        torch.nn.init.xavier_uniform_(self.seg_conv2.weight)

        self.seg_conv3 = nn.Conv2d(128, 128, (3, 3), padding=18, dilation=(18, 18), bias=True)
        torch.nn.init.xavier_uniform_(self.seg_conv3.weight)

        self.seg_conv4 = nn.Conv2d(128, num_classes, (1, 1), bias=False)
        torch.nn.init.xavier_uniform_(self.seg_conv4.weight)

        self.sal_conv1 = nn.Conv2d(4096, 128, (3, 3), padding=6, dilation=(6, 6), bias=True)
        torch.nn.init.xavier_uniform_(self.sal_conv1.weight)

        self.sal_conv2 = nn.Conv2d(128, 128, (3, 3), padding=12, dilation=(12, 12), bias=True)
        torch.nn.init.xavier_uniform_(self.sal_conv2.weight)

        self.sal_conv3 = nn.Conv2d(128, 128, (3, 3), padding=18, dilation=(18, 18), bias=True)
        torch.nn.init.xavier_uniform_(self.sal_conv3.weight)

        self.sal_conv4 = nn.Conv2d(128, 1, (1, 1), bias=False)
        torch.nn.init.xavier_uniform_(self.sal_conv4.weight)

        self.seg_att = NLBlockND(in_channels=128, inter_channels=128, dimension=2)
        self.sal_att = NLBlockND(in_channels=128, inter_channels=128, dimension=2)

        self.seg_g = nn.Conv2d(128, 128, (1, 1))
        self.seg_W = nn.Conv2d(128, 128, (1, 1))
        self.sal_g = nn.Conv2d(128, 128, (1, 1))
        self.sal_W = nn.Conv2d(128, 128, (1, 1))
        self.cross_att = nn.Sequential(nn.Conv2d(2, 2, (3, 3), padding=1, bias=False),
                                       nn.ReLU(),
                                       nn.Conv2d(2, 2, (1, 1)),
                                       nn.Softmax(dim=1))

    def forward(self, x, return_aff = False):
        batch_size = x.size(0)

        x_seg = super().forward(x)
        x_sal = x_seg.clone()
        x_cls = x_seg.clone()
        x_cam = x_seg.clone()

        x_cls = self.dropout7(x_cls)
        x_cls = F.avg_pool2d(
            x_cls, kernel_size=(x_cls.size(2), x_cls.size(3)), padding=0)
        x_cls = self.fc8(x_cls)
        x_cls = x_cls.view(x_cls.size(0), -1)
        cam = F.conv2d(x_cam, self.fc8.weight)

        x_seg = F.relu(self.seg_conv1(x_seg))
        x_seg = F.relu(self.seg_conv2(x_seg))
        seg_feats = x_seg

        x_sal = F.relu(self.sal_conv1(x_sal))
        x_sal = F.relu(self.sal_conv2(x_sal))
        sal_feats = x_sal

        seg_aff = self.seg_att(seg_feats)
        sal_aff = self.sal_att(sal_feats)

        g_seg_feats = self.seg_g(seg_feats)
        g_seg_feats = g_seg_feats.view(batch_size, seg_feats.size(1), -1)
        g_seg_feats = g_seg_feats.permute(0, 2, 1).contiguous()
        refined_seg_feats = torch.matmul(seg_aff, g_seg_feats)
        refined_seg_feats = refined_seg_feats.permute(0, 2, 1).contiguous()
        refined_seg_feats = refined_seg_feats.view(*seg_feats.size())
        W_seg_feats = self.seg_W(refined_seg_feats)
        seg_feats = seg_feats + W_seg_feats

        seg_feats = F.relu(self.seg_conv3(seg_feats))
        init_seg_mask = self.seg_conv4(seg_feats)

        g_sal_feats = self.sal_g(sal_feats)
        g_sal_feats = g_sal_feats.view(batch_size, sal_feats.size(1), -1).permute(0, 2, 1).contiguous()
        refined_sal_feats = torch.matmul(sal_aff, g_sal_feats)
        refined_sal_feats = refined_sal_feats.permute(0, 2, 1).contiguous().view(*sal_feats.size())
        sal_feats = sal_feats + self.sal_W(refined_sal_feats)

        sal_feats = F.relu(self.sal_conv3(sal_feats))
        init_sal_mask = self.sal_conv4(sal_feats)

        if len(seg_aff.size()) != 4:
            seg_aff = seg_aff.unsqueeze(1)
            sal_aff = sal_aff.unsqueeze(1)

        cross_aff = torch.cat((seg_aff, sal_aff), dim=1)
        cross_w = self.cross_att(cross_aff)
        cross_aff = cross_aff[:, 0] * cross_w[:, 0] + cross_aff[:, 1] * cross_w[:, 1]

        sal_mask = init_sal_mask.view(batch_size, init_sal_mask.size(1), -1)
        sal_mask = sal_mask.permute(0, 2, 1).contiguous()
        refined_sal_mask = torch.matmul(cross_aff, sal_mask)
        refined_sal_mask = refined_sal_mask.permute(0, 2, 1).contiguous().view(*init_sal_mask.size())

        seg_mask = init_seg_mask.view(batch_size, init_seg_mask.size(1), -1)
        seg_mask = seg_mask.permute(0, 2, 1).contiguous()
        refined_seg_mask = torch.matmul(cross_aff, seg_mask)
        refined_seg_mask = refined_seg_mask.permute(0, 2, 1).contiguous().view(*init_seg_mask.size())

        if return_aff:
            return x_cls, cam, init_sal_mask, refined_sal_mask, init_seg_mask, refined_seg_mask, cross_aff
        else:
            return x_cls, cam, init_sal_mask, refined_sal_mask, init_seg_mask, refined_seg_mask


class SegNet(Net):
    def __init__(self, num_classes=21):
        super().__init__(num_classes=num_classes)

    def forward(self, x, require_seg=True, require_cls=True, require_sal=True, require_cam=False):

        if require_seg == True and require_cls == True and require_sal == True and require_cam == False:
            x_cls, cam, init_sal, refined_sal, init_seg, refined_seg = super().forward(x)
            return x_cls, init_sal, refined_sal, init_seg, refined_seg

        if require_seg == True and require_cls == True and require_sal == True and require_cam == True:
            x_cls, cam, init_sal, refined_sal, init_seg, refined_seg, aff = super().forward(x, return_aff=True)
            return x_cls, cam, init_sal, refined_sal, init_seg, refined_seg, aff

        if require_cls == False and require_seg == True and require_sal == False:
            x_cls, cam, init_sal, refined_sal, init_seg, refined_seg = super().forward(x)
            return init_seg, refined_seg

    def get_10x_lr_params(self):
        for name, param in self.named_parameters():
            if 'seg' in name or 'sal' in name or 'fc8' in name:
                yield param

    def get_1x_lr_params(self):
        for name, param in self.named_parameters():
            if 'seg' not in name and 'sal' not in name and 'fc8' not in name:
                yield param