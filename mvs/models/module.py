import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()


        self.conv1 = nn.Conv2d(3, 8, (3,3), stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(True)

        self.conv2 = nn.Conv2d(8, 8, (3,3), stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(8)
        
        # W/2 x H/2 x 16 Need to double check the magic formula
        self.conv3 = nn.Conv2d(8, 16, (5,5), stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(16)

        # W/2 x H/2 x 16
        self.conv4 = nn.Conv2d(16, 16, (3, 3), stride=1, padding='same')
        self.bn4 = nn.BatchNorm2d(16)

        # W/2 x H/2 x 16
        self.conv5 = nn.Conv2d(16, 16, (3, 3), stride=1, padding='same')
        self.bn5 = nn.BatchNorm2d(16)

        # W/4, H/4 x 32 Need to double check the magic formula
        self.conv6 = nn.Conv2d(16, 32, (5, 5), stride=2, padding=2)
        self.bn6 = nn.BatchNorm2d(32)

        # W/4, H/4 x 32
        self.conv7 = nn.Conv2d(32, 32, (3, 3), stride=1, padding='same')
        self.bn7 = nn.BatchNorm2d(32)

        # W/4, H/4 x 32
        self.conv8 = nn.Conv2d(32, 32, (3, 3), stride=1, padding='same')
        self.bn8 = nn.BatchNorm2d(32)

        # W/4, H/4 x 32
        self.conv9 = nn.Conv2d(32, 32, (3, 3), stride=1, padding='same')

    def forward(self, x):
        # x: [B,3,H,W]
        ### Do the ConvBnReLU structure suggested in the assignment guidelines 
        # TODO DONE 
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))
        x = self.conv9(x)
        
        return x


class SimlarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimlarityRegNet, self).__init__()
        # TODO
        self.G = G
        self.relu = nn.ReLU(True)
        
        #Not sure about the padding = 'same'
        self.conv1 = nn.Conv2d(G, 8, (3,3), stride=1, padding='same')
        self.conv2 = nn.Conv2d(8, 16, (3,3), stride=2, padding='same')
        self.conv3 = nn.Conv2d(16, 32, (3,3), stride=2, padding='same')
        self.conv_transpose_1 = nn.Conv2d(32, 16, (3,3), stride=2, padding='same')
        self.conv_transpose_2 = nn.Conv2d(16, 8, (3,3), stride=2, padding='same')
        
        self.conv4 = nn.Conv2d(8, 1, (3,3), stride=1, padding='same')
        

    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        # TODO
        B,G,D,H,W = x.size()
        C_0 = self.relu(self.conv1(x))
        C_1 = self.relu(self.conv2(C_0))
        C_3 = self.relu(self.conv3(C_1))
        C_3 = self.conv_transpose_1(C_3)
        C_4 = self.conv_transpose_2(C_3 + C_1)
        S_ = self.relu(self.conv4(C_4 + C_0))

        S_ = S_.view((B,D,H,W))

        return S_

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

        # y = te
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, W, dtype=torch.float32, device=src_fea.device)])

        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)
        
        

        # X is reference, y is source? There is no ref_features. They are in the group_wise_correlation function
        # Use this rotation matrix, 'rot' and transformation, 'trans' matrix on each src_feat
        # What's this x, y business about? It is actually the transforming src_feature in the device paramater
        # I believe it is also flattening them with torch.view(x* y)

        # TODO
        #  get warped_src_fea with bilinear interpolation (use 'grid_sample' function from pytorch)
        
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, D, 1) * depth_values.view(
            B, 1, D, H * W
        )  # [B, 3, D, H*W]
        proj_xyz = rot_depth_xyz + trans.view(B, 3, 1, 1)  # [B, 3, D, H*W]
       
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, D, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((W - 1) / 2) - 1  # [B, D, H*W]
        proj_y_normalized = proj_xy[:, 1, :, :] / ((H - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, D, H*W, 2]
        grid = proj_xy

        warped_src_fea = F.grid_sample(
            src_fea,
            grid.view(B, D * H, W, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        warped_src_fea.view(B, C, D, H, W)

    # warped_src_fea: [B,C,D,H,W]
    return warped_src_fea

def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]

    B, C, D, H, W = warped_src_fea.size()
    output = torch.ones((B,G,D,H,W))

    channel_in_group = C / G

    for g in range(channel_in_group):
        lower = g * channel_in_group
        upper = (g + 1) * channel_in_group
        output[:, g, :, :, :] = torch.sum(ref_fea[:, lower : upper, :, :].unsqueeze(2) * warped_src_fea[:, lower : upper, :, :, :], dim = 1) * G / C
        
    return output


def depth_regression(P, depth_values):
    # P: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    # TODO
    #Softmax on the D dimension, which is 1
    # P is the priobability volume
    # p is the pixel coordinates
    # depth values is the actual depth of the points
    # D is 192. The number of depth values
    
    B, D, _, _ = p.size()
    output = torch.sum(p * depth_values.view (B, D, 1, 1), dim = 1)

    return output

def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    # TODO
    mask = mask > 1e-5
    masked_depth_est = depth_est[mask]
    masked_depth_gt = depth_gt[mask]
    loss = F.l1_loss(masked_depth_est, masked_depth_gt, reduction='mean')
    
    return loss
