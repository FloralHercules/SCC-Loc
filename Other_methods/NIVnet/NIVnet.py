import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import math

# ==========================================
# 1. 基础模块定义 (Encoder & Generator)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, dim, use_ca=True, ca_reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(dim)
        self.use_ca = bool(use_ca)
        if self.use_ca:
            hidden = max(8, int(dim // max(1, int(ca_reduction))))
            self.ca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, hidden, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, dim, kernel_size=1),
                nn.Sigmoid(),
            )
        
    def forward(self, x):
        residual = x
        x = F.relu(self.in1(self.conv1(x)))
        x = self.in2(self.conv2(x))
        if self.use_ca:
            x = x * self.ca(x)
        return F.relu(residual + x)

class ShapeEncoder(nn.Module):
    def __init__(self, base_channels=20, num_res_blocks=2, use_ca=True):
        super().__init__()
        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4
        self.downsample = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=7, stride=1, padding=3), nn.InstanceNorm2d(c1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(c2), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(c3), nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(*[ResBlock(c3, use_ca=use_ca) for _ in range(int(num_res_blocks))])

    def forward(self, x):
        return self.res_blocks(self.downsample(x))

class AttributeEncoder(nn.Module):
    def __init__(self, attr_dim=8, base_channels=20):
        super().__init__()
        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4
        self.convs = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(c3, c3, kernel_size=4, stride=2, padding=1), nn.ReLU(True)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.final_conv = nn.Conv2d(c3, attr_dim, kernel_size=1)

    def forward(self, x):
        x = self.pool(self.convs(x))
        return self.final_conv(x)

class Generator(nn.Module):
    def __init__(self, attr_dim=8, base_channels=20, num_res_blocks=2, use_ca=True):
        super().__init__()
        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4
        self.shape_channels = c3

        self.res_blocks = nn.Sequential(*[ResBlock(c3, use_ca=use_ca) for _ in range(int(num_res_blocks))])
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(c2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(c1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, 3, kernel_size=7, stride=1, padding=3), nn.Tanh() 
        )
        self.attr_proj = nn.Linear(attr_dim, c3)

    def forward(self, shape_code, attr_code):
        B, C, H, W = shape_code.shape
        attr_code = attr_code.view(B, -1)
        attr_feat = self.attr_proj(attr_code).view(B, self.shape_channels, 1, 1).expand(-1, -1, H, W)
        x = shape_code + attr_feat 
        x = self.res_blocks(x)
        return self.upsample(x)

class FeatureExtractionNetwork(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(latent_dim, latent_dim * 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, shape_code):
        return self.pool2(self.relu(self.conv(self.pool1(shape_code))))

class BidirectionalMatchingLayer(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, F_x, F_y):
        B, C, H, W = F_x.shape
        B2, C2, H2, W2 = F_y.shape
        if H != H2 or W != W2:
            F_y = F.interpolate(F_y, size=(H, W), mode='bilinear', align_corners=False)
        F_x_norm = F.normalize(F_x, p=2, dim=1).reshape(B, C, -1)
        F_y_norm = F.normalize(F_y, p=2, dim=1).reshape(B, C, -1)

        temp = max(self.temperature, 1e-3)
        S = torch.bmm(F_x_norm.transpose(1, 2), F_y_norm).view(B, H * W, H, W)
        S_inv = torch.bmm(F_y_norm.transpose(1, 2), F_x_norm).view(B, H * W, H, W)

        S = torch.softmax(S / temp, dim=1)
        S_inv = torch.softmax(S_inv / temp, dim=1)
        return S, S_inv

class RegressionNetwork(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=(192, 96, 48), pool_hw=4, dropout=0.0):
        super().__init__()
        c1, c2, c3 = [int(v) for v in hidden_channels]
        self.conv1 = nn.Conv2d(int(in_channels), c1, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(c3)
        self.pool = nn.AdaptiveAvgPool2d((int(pool_hw), int(pool_hw)))
        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.fc = nn.Linear(c3 * int(pool_hw) * int(pool_hw), 6) 
        
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, S):
        B = S.shape[0]
        x = F.relu(self.bn1(self.conv1(S)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        theta = self.fc(x.view(B, -1))
        theta_matrix = torch.zeros(B, 3, 3, device=S.device)
        theta_matrix[:, :2, :] = theta.view(B, 2, 3)
        theta_matrix[:, 2, 2] = 1.0
        return theta_matrix



class NIVnet(nn.Module):
    def __init__(self, args, for_training=True):
        super().__init__()
        self.args = args
        self.for_training = for_training
        self.base_channels = int(getattr(args, 'niv_base_channels', 24))
        self.num_res_blocks = int(getattr(args, 'niv_res_blocks', 3))
        self.attr_dim = int(getattr(args, 'niv_attr_dim', 16))
        self.match_hw = int(getattr(args, 'niv_match_hw', 16))
        self.match_temperature = float(getattr(args, 'niv_match_temperature', 0.07))
        self.reg_hidden_channels = int(getattr(args, 'niv_reg_hidden_channels', 192))
        self.reg_mid_channels = int(getattr(args, 'niv_reg_mid_channels', 96))
        self.reg_low_channels = int(getattr(args, 'niv_reg_low_channels', 48))
        self.reg_dropout = float(getattr(args, 'niv_reg_dropout', 0.1))
        self.use_channel_attention = bool(getattr(args, 'niv_use_channel_attention', True))
        latent_dim = self.base_channels * 4
        
        self.E_x_s = ShapeEncoder(base_channels=self.base_channels, num_res_blocks=self.num_res_blocks, use_ca=self.use_channel_attention)  
        self.E_y_s = ShapeEncoder(base_channels=self.base_channels, num_res_blocks=self.num_res_blocks, use_ca=self.use_channel_attention)  
        self.E_x_a = AttributeEncoder(attr_dim=self.attr_dim, base_channels=self.base_channels)
        self.E_y_a = AttributeEncoder(attr_dim=self.attr_dim, base_channels=self.base_channels)
        self.G_x = Generator(attr_dim=self.attr_dim, base_channels=self.base_channels, num_res_blocks=self.num_res_blocks, use_ca=self.use_channel_attention)      
        self.G_y = Generator(attr_dim=self.attr_dim, base_channels=self.base_channels, num_res_blocks=self.num_res_blocks, use_ca=self.use_channel_attention)       
        
        self.feature_extractor = FeatureExtractionNetwork(latent_dim=latent_dim)
        self.matching_layer = BidirectionalMatchingLayer(temperature=self.match_temperature)
        self.regression = RegressionNetwork(
            in_channels=self.match_hw * self.match_hw,
            hidden_channels=(self.reg_hidden_channels, self.reg_mid_channels, self.reg_low_channels),
            pool_hw=4,
            dropout=self.reg_dropout,
        )



        self.lambda1 = float(getattr(args, 'lambda_recon', 10.0)) # L_recon
        self.lambda2 = float(getattr(args, 'lambda_trans', 1.0))  # L_trans
        self.lambda3 = float(getattr(args, 'lambda_grid', 10.0))  # L_grid
        self.lambda4 = float(getattr(args, 'lambda_inten', 1.0))  # L_inten
        self.lambda5 = float(getattr(args, 'lambda_iden', 5.0))   # L_iden
        
        self.four_point_org_single = torch.zeros((1, 2, 2, 2))
        self.four_point_org_single[:, :, 0, 0] = torch.tensor([0.0, 0.0])
        self.four_point_org_single[:, :, 0, 1] = torch.tensor([args.resize_width - 1.0, 0.0])
        self.four_point_org_single[:, :, 1, 0] = torch.tensor([0.0, args.resize_width - 1.0])
        self.four_point_org_single[:, :, 1, 1] = torch.tensor([args.resize_width - 1.0, args.resize_width - 1.0])

    def setup(self):
        self.to(self.args.device)
        self.four_point_org_single = self.four_point_org_single.to(self.args.device)
        if self.for_training:
            self.optimizer_G = torch.optim.AdamW(
                self.parameters(),
                lr=self.args.lr,
                weight_decay=float(getattr(self.args, 'wdecay', 1e-5)),
                eps=float(getattr(self.args, 'epsilon', 1e-8)),
            )

            total_steps = max(1, int(self.args.num_steps))
            scheduler_name = str(getattr(self.args, 'lr_scheduler', 'warmup_cosine')).lower()
            min_lr_ratio = float(getattr(self.args, 'min_lr_ratio', 0.05))
            min_lr_ratio = min(1.0, max(0.0, min_lr_ratio))

            if scheduler_name == 'warmup_cosine':
                warmup_steps = max(0, int(getattr(self.args, 'warmup_steps', 100)))

                def lr_lambda(step_idx):
                    if warmup_steps > 0 and step_idx < warmup_steps:
                        return float(step_idx + 1) / float(max(1, warmup_steps))
                    decay_steps = max(1, total_steps - warmup_steps)
                    progress = float(step_idx - warmup_steps) / float(decay_steps)
                    progress = min(1.0, max(0.0, progress))
                    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

                self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lr_lambda)

            elif scheduler_name == 'onecycle':
                pct_start = float(getattr(self.args, 'onecycle_pct_start', 0.1))
                div_factor = float(getattr(self.args, 'onecycle_div_factor', 10.0))
                final_div_factor = float(getattr(self.args, 'onecycle_final_div_factor', 100.0))
                self.scheduler_G = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer=self.optimizer_G,
                    max_lr=self.args.lr,
                    total_steps=total_steps,
                    pct_start=pct_start,
                    anneal_strategy='cos',
                    cycle_momentum=False,
                    div_factor=div_factor,
                    final_div_factor=final_div_factor,
                )

            elif scheduler_name == 'cosine':
                self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer_G,
                    T_max=total_steps,
                    eta_min=float(self.args.lr) * min_lr_ratio,
                )

            else:
                # Backward-compatible fallback.
                self.scheduler_G = torch.optim.lr_scheduler.StepLR(
                    self.optimizer_G,
                    step_size=max(1, int(self.args.num_steps * 0.4)),
                    gamma=0.5,
                )

    def set_input(self, image1, image2, flow_gt):

        self.x_1 = image1.to(self.args.device)
        self.y_1 = image2.to(self.args.device)
        self.flow_gt = flow_gt.to(self.args.device) 
        
        flow_4cor = torch.zeros((self.flow_gt.shape[0], 2, 2, 2), device=self.args.device)
        flow_4cor[:, :, 0, 0] = self.flow_gt[:, :, 0, 0]
        flow_4cor[:, :, 0, 1] = self.flow_gt[:, :, 0, -1]
        flow_4cor[:, :, 1, 0] = self.flow_gt[:, :, -1, 0]
        flow_4cor[:, :, 1, 1] = self.flow_gt[:, :, -1, -1]
        self.flow_4cor = flow_4cor

    def forward(self):
        self.s_x = self.E_x_s(self.x_1)
        self.a_x = self.E_x_a(self.x_1)
        
        self.s_y = self.E_y_s(self.y_1)
        self.a_y = self.E_y_a(self.y_1)

        if self.training:
            self.x_1_hat = self.G_x(self.s_x, self.a_x)
            self.y_1_hat = self.G_y(self.s_y, self.a_y)
            self.y_2 = self.G_y(self.s_x, self.a_y)

        F_x = self.feature_extractor(self.s_x)
        F_y = self.feature_extractor(self.s_y)
        if F_x.shape[-2:] != (self.match_hw, self.match_hw):
            F_x = F.adaptive_avg_pool2d(F_x, (self.match_hw, self.match_hw))
        if F_y.shape[-2:] != (self.match_hw, self.match_hw):
            F_y = F.adaptive_avg_pool2d(F_y, (self.match_hw, self.match_hw))
        
        S, S_inv = self.matching_layer(F_x, F_y)
        
        self.H_pred = self.regression(S)       
        self.H_inv_pred = self.regression(S_inv)

        self.four_pred = self._matrix_to_4cor(self.H_pred)

    def _matrix_to_4cor(self, H):
        B = H.shape[0]
        corners = self.four_point_org_single.view(1, 2, 4).repeat(B, 1, 1)
        corners_homo = torch.cat([corners, torch.ones(B, 1, 4, device=H.device)], dim=1) 
        
        pred_corners_homo = torch.bmm(H, corners_homo)
        
        pred_corners = pred_corners_homo[:, :2, :] / (pred_corners_homo[:, 2:, :] + 1e-8)
        
        offset = pred_corners - corners
        return offset.view(B, 2, 2, 2)

    def _safe_get_perspective_transform(self, pts_src, pts_dst):
        batch_size = pts_src.shape[0]
        eye = torch.eye(3, dtype=pts_src.dtype, device=pts_src.device).unsqueeze(0)
        H_list = []
        valid_list = []

        for idx in range(batch_size):
            src_i = pts_src[idx:idx + 1]
            dst_i = pts_dst[idx:idx + 1]
            is_valid = bool(torch.isfinite(src_i).all() and torch.isfinite(dst_i).all())

            if is_valid:
                try:
                    H_i = kornia.geometry.transform.get_perspective_transform(src_i, dst_i)
                    if not torch.isfinite(H_i).all():
                        raise RuntimeError('non-finite homography')
                except Exception:
                    H_i = eye.clone()
                    is_valid = False
            else:
                H_i = eye.clone()

            H_list.append(H_i)
            valid_list.append(is_valid)

        H_batch = torch.cat(H_list, dim=0)
        valid_mask = torch.tensor(valid_list, device=pts_src.device, dtype=torch.bool)
        return H_batch, valid_mask

    def optimize_parameters(self):
        self.optimizer_G.zero_grad()
        self.forward()

        
        l_recon = F.l1_loss(self.x_1_hat, self.x_1) + F.l1_loss(self.y_1_hat, self.y_1)
        
        pts_src = self.four_point_org_single.flatten(2).permute(0, 2, 1).repeat(self.x_1.shape[0], 1, 1)
        flow_4pts = self.flow_4cor.flatten(2).permute(0, 2, 1).contiguous()
        pts_dst = pts_src + flow_4pts
        H_gt, valid_h_mask = self._safe_get_perspective_transform(pts_src, pts_dst)
        
        grid_y, grid_x = torch.meshgrid(torch.linspace(0, self.args.resize_width-1, 10), 
                                        torch.linspace(0, self.args.resize_width-1, 10), indexing='ij')
        grid_pts = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1).to(self.args.device) # 100x2
        grid_pts = grid_pts.unsqueeze(0).repeat(self.x_1.shape[0], 1, 1) # Bx100x2
        
        grid_homo = torch.cat([grid_pts, torch.ones_like(grid_pts[:, :, :1])], dim=-1).transpose(1, 2) # Bx3x100
        if bool(valid_h_mask.any()):
            H_gt_valid = H_gt[valid_h_mask]
            H_pred_valid = self.H_pred[valid_h_mask]
            grid_homo_valid = grid_homo[valid_h_mask]

            gt_transformed = torch.bmm(H_gt_valid, grid_homo_valid)
            pred_transformed = torch.bmm(H_pred_valid, grid_homo_valid)

            gt_transformed = gt_transformed[:, :2, :] / (gt_transformed[:, 2:, :] + 1e-8)
            pred_transformed = pred_transformed[:, :2, :] / (pred_transformed[:, 2:, :] + 1e-8)

            l_grid = torch.norm(gt_transformed - pred_transformed, p=2, dim=1).mean()
        else:
            l_grid = self.H_pred.sum() * 0.0
        h_valid_ratio = float(valid_h_mask.float().mean().item()) if valid_h_mask.numel() > 0 else 0.0

        identity = torch.eye(3, device=self.args.device).unsqueeze(0).repeat(self.x_1.shape[0], 1, 1)
        matrix_mul = torch.bmm(self.H_pred, self.H_inv_pred)
        l_iden = F.l1_loss(matrix_mul, identity)

        if bool(valid_h_mask.any()):
            y_1_valid = self.y_1[valid_h_mask]
            y_2_valid = self.y_2[valid_h_mask]
            H_gt_valid = H_gt[valid_h_mask]
            H_pred_valid = self.H_pred[valid_h_mask]

            y_1_aligned_gt = kornia.geometry.transform.warp_perspective(
                y_1_valid,
                torch.linalg.pinv(H_gt_valid),
                (self.args.resize_width, self.args.resize_width),
            )
            y_1_aligned_pred = kornia.geometry.transform.warp_perspective(
                y_1_valid,
                torch.linalg.pinv(H_pred_valid),
                (self.args.resize_width, self.args.resize_width),
            )

            crop_size = 50
            start = (self.args.resize_width - crop_size) // 2

            l_trans = F.l1_loss(
                y_1_aligned_gt[:, :, start:start + crop_size, start:start + crop_size],
                y_2_valid[:, :, start:start + crop_size, start:start + crop_size],
            )
            l_inten = F.l1_loss(
                y_1_aligned_pred[:, :, start:start + crop_size, start:start + crop_size],
                y_2_valid[:, :, start:start + crop_size, start:start + crop_size],
            )
        else:
            l_trans = self.H_pred.sum() * 0.0
            l_inten = self.H_pred.sum() * 0.0

        loss = (self.lambda1 * l_recon + 
                self.lambda2 * l_trans + 
                self.lambda3 * l_grid + 
                self.lambda4 * l_inten + 
                self.lambda5 * l_iden)

        loss.backward()
        
        if self.args.clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
            
        self.optimizer_G.step()

        return {
            'G_loss': loss.item(), 
            'mace': l_grid.item(),   
            'ce_loss': l_iden.item(), 
            'recon_loss': l_recon.item(),
            'trans_loss': l_trans.item(),
            'inten_loss': l_inten.item(),
            'grid_loss': l_grid.item(),
            'iden_loss': l_iden.item(),
            'h_valid_ratio': h_valid_ratio,
        }

    def update_learning_rate(self):
        self.scheduler_G.step()