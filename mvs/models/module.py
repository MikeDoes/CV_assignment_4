import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, (3,3), stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(True)
        
        self.convBnReLu1 = nn.intrinsic.ConvBnReLU2d(self.conv1, self.bn1, self.relu)

        self.conv2 = nn.Conv2d(8, 8, (3,3), stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(8)

        self.convBnReLu2 = nn.intrinsic.ConvBnReLU2d(self.conv2, self.bn2, self.relu)
        
        # W/2 x H/2 x 16
        self.conv3 = nn.Conv2d(8, 16, (5,5), stride=2)
        self.bn3 = nn.BatchNorm2d(16)

        self.convBnReLu3 = nn.intrinsic.ConvBnReLU2d(self.conv3, self.bn3, self.relu)

        # W/2 x H/2 x 16
        self.conv4 = nn.Conv2d(16, 16, (3, 3), stride=1, padding='same')
        self.bn4 = nn.BatchNorm2d(16)

        self.convBnReLu4 = nn.intrinsic.ConvBnReLU2d(self.conv4, self.bn4, self.relu)

        # W/2 x H/2 x 16
        self.conv5 = nn.Conv2d(16, 16, (3, 3), stride=1, padding='same')
        self.bn5 = nn.BatchNorm2d(16)

        self.convBnReLu5 = nn.intrinsic.ConvBnReLU2d(self.conv5, self.bn5, self.relu)

        # W/4, H/4 x 32
        self.conv6 = nn.Conv2d(16, 32, (5, 5), stride=2)
        self.bn6 = nn.BatchNorm2d(32)

        self.convBnReLu6 = nn.intrinsic.ConvBnReLU2d(self.conv6, self.bn6, self.relu)

        # W/4, H/4 x 32
        self.conv6 = nn.Conv2d(32, 32, (3, 3), stride=1, padding='same')
        self.bn6 = nn.BatchNorm2d(32)

        self.convBnReLu6 = nn.intrinsic.ConvBnReLU2d(self.conv6, self.bn6, self.relu)

        # W/4, H/4 x 32
        self.conv7 = nn.Conv2d(32, 32, (3, 3), stride=1, padding='same')
        self.bn7 = nn.BatchNorm2d(32)

        self.convBnReLu7 = nn.intrinsic.ConvBnReLU2d(self.conv7, self.bn7, self.relu)

        # W/4, H/4 x 32
        self.conv7 = nn.Conv2d(32, 32, (3, 3), stride=1, padding='same')

    def forward(self, x):
        # x: [B,3,H,W]
        ### Do the ConvBnReLU structure suggested in the assignment guidelines 
        # TODO DONE 
        x = self.convBnReLu1(x)
        x = self.convBnReLu2(x)
        x = self.convBnReLu3(x)
        x = self.convBnReLu4(x)
        x = self.convBnReLu5(x)
        x = self.convBnReLu6(x)
        x = self.conv7(x)
        
        return x


class SimlarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimlarityRegNet, self).__init__()
        # TODO

    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        # TODO


def warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # Batch, Color, Height, Width
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, D]
    # out: [B, C, D, H, W]
    # Batch, Colour, Depth, Height, Width

    B,C,H,W = src_fea.size()
    D = depth_values.size(1)
    # compute the warped positions with depth values

    with torch.no_grad():
        # relative transformation from reference to source view
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))

        # Splitting the projection matrix into rotation and transformation
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, W, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)
        # TODO
        

        # X is reference, y is source? There is no ref_features. They are in the group_wise_correlation function
        # Use this rotation matrix, 'rot' and transformation, 'trans' matrix on each src_feat
        # What's this x, y business about? It is actually the transforming src_feature in the device paramater
        # I believe it is also flattening them with torch.view(x* y)

    # TODO
    # get warped_src_fea with bilinear interpolation (use 'grid_sample' function from pytorch)
    
    # warped_src_fea: [B,C,D,H,W]
    return warped_src_fea

def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]


    # TODO


def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    # TODO

def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    # TODO
