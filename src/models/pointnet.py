"""
Original PointNet classification architecture (Qi et al., 2017), classification branch only
(no segmentation head — not needed for this project).

Matches the project's architecture diagram:
    input (N,3) -> Input T-Net (3x3) -> shared MLP (64) -> Feature T-Net (64x64)
    -> shared MLP (64,128,1024) -> symmetric max-pool -> FC+dropout -> FC+dropout -> softmax

The classifier's dropout layers stay as nn.Dropout throughout — for plain training/eval they
behave normally (active in train(), off in eval()). For MC Dropout at inference time later,
we force them active during eval via `enable_mc_dropout()` without touching BatchNorm's
running-stats behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """
    Learns a k x k transformation matrix to align the input (k=3, raw xyz) or an
    intermediate feature space (k=64) into a canonical orientation, so the network
    doesn't have to learn every possible rotation/pose separately.

    Structure: shared MLP (64,128,1024) -> max-pool -> FC (512,256) -> predict k*k values,
    initialized to output the identity matrix at the start of training.
    """

    def __init__(self, k: int):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        # x: (B, k, N)  -- k channels (xyz or features), N points
        B = x.shape[0]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, dim=2)[0]  # symmetric max-pool over the N points -> (B, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # (B, k*k)

        # Initialize as identity so early training starts from "no transform"
        identity = torch.eye(self.k, device=x.device, dtype=x.dtype).flatten().unsqueeze(0)
        x = x + identity
        return x.view(B, self.k, self.k)


class PointNetBackbone(nn.Module):
    """
    Produces a single 1024-dim global feature vector per point cloud, permutation-invariant
    to point order. Also returns the two learned transform matrices (for the orthogonality
    regularization loss) — needed during training, ignorable at inference.
    """

    def __init__(self, feature_transform: bool = True):
        super().__init__()
        self.feature_transform = feature_transform

        self.input_tnet = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        if feature_transform:
            self.feature_tnet = TNet(k=64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        # x: (B, N, 3) as loaded from the dataset -> transpose to (B, 3, N) for Conv1d
        x = x.transpose(2, 1)

        input_trans = self.input_tnet(x)              # (B, 3, 3)
        x = torch.bmm(input_trans, x)                  # apply spatial alignment

        x = F.relu(self.bn1(self.conv1(x)))             # (B, 64, N)

        feature_trans = None
        if self.feature_transform:
            feature_trans = self.feature_tnet(x)        # (B, 64, 64)
            x = torch.bmm(feature_trans, x)              # align feature space

        x = F.relu(self.bn2(self.conv2(x)))              # (B, 128, N)
        x = self.bn3(self.conv3(x))                       # (B, 1024, N)  (no ReLU before pooling, standard)

        global_feat = torch.max(x, dim=2)[0]               # symmetric max-pool -> (B, 1024)
        return global_feat, input_trans, feature_trans


class PointNetClassifier(nn.Module):
    """
    Full classification model: backbone -> FC head with dropout.
    Dropout is what MC Dropout will exploit later — kept as standard nn.Dropout so normal
    train()/eval() behavior works for baseline training; enable_mc_dropout() (below) is used
    only when we specifically want stochastic passes at inference.
    """

    def __init__(self, num_classes: int, feature_transform: bool = True, dropout_p: float = 0.3):
        super().__init__()
        self.backbone = PointNetBackbone(feature_transform=feature_transform)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)

    def forward(self, x):
        global_feat, input_trans, feature_trans = self.backbone(x)

        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        logits = self.fc3(x)

        return logits, input_trans, feature_trans


def feature_transform_regularizer(trans: torch.Tensor) -> torch.Tensor:
    """
    Orthogonality regularization for the feature T-Net's learned transform matrix:
    encourages A to behave like a rotation (A @ A^T ~= I) rather than collapsing/destroying
    information. Add this (scaled by a small lambda, e.g. 0.001) to the classification loss.
    """
    B, k, _ = trans.shape
    identity = torch.eye(k, device=trans.device, dtype=trans.dtype).unsqueeze(0).repeat(B, 1, 1)
    diff = identity - torch.bmm(trans, trans.transpose(2, 1))
    return torch.mean(torch.norm(diff, dim=(1, 2)))


def enable_mc_dropout(model: nn.Module):
    """
    Puts the model in eval mode (so BatchNorm uses running stats, not batch stats — important,
    since MC Dropout inference is often done one-sample-at-a-time or with small batches) while
    forcing Dropout layers back into training mode so they keep randomly zeroing activations.
    This is the mechanism that makes repeated forward passes on the same input stochastic.
    Call this instead of model.eval() when you're about to run T stochastic forward passes.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


if __name__ == "__main__":
    # sanity check: fake batch matching what the dataset produces
    torch.manual_seed(0)
    batch_size, num_points, num_classes = 8, 1024, 35  # 40 - 5 OOD classes

    model = PointNetClassifier(num_classes=num_classes)
    x = torch.randn(batch_size, num_points, 3)

    logits, input_trans, feature_trans = model(x)
    print(f"input shape: {x.shape}")
    print(f"logits shape: {logits.shape}  (expected: [{batch_size}, {num_classes}])")
    print(f"input_trans shape: {input_trans.shape}  (expected: [{batch_size}, 3, 3])")
    print(f"feature_trans shape: {feature_trans.shape}  (expected: [{batch_size}, 64, 64])")

    reg_loss = feature_transform_regularizer(feature_trans)
    print(f"feature transform reg loss (should be small, near 0, since untrained/random): {reg_loss.item():.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"total trainable params: {total_params:,}")

    # confirm MC dropout mode actually keeps dropout stochastic across repeated passes
    enable_mc_dropout(model)
    with torch.no_grad():
        out1, _, _ = model(x)
        out2, _, _ = model(x)
    identical = torch.allclose(out1, out2)
    print(f"MC dropout mode -> two passes on same input identical? {identical} (expected: False)")
