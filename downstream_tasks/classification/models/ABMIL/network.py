import torch
import torch.nn as nn
import torch.nn.functional as F


class DAttention(nn.Module):
    def __init__(self, n_classes, dropout, act, n_features=1024):
        super(DAttention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1

        # feature extractor: [N, n_features] -> [N, L]
        blocks = [nn.Linear(n_features, self.L)]
        if act.lower() == "gelu":
            blocks += [nn.GELU()]
        else:
            blocks += [nn.ReLU()]
        if dropout:
            blocks += [nn.Dropout(0.25)]
        self.feature = nn.Sequential(*blocks)

        # attention: [N, L] -> [N, K]
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K),
        )

        # classifier expects flattened [1, L*K]
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, n_classes),
        )

    def _normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensure input is [N, D].
        Handles [D], [1, N, D], >2D (flatten leading dims), etc.
        """
        if x.dim() == 3 and x.size(0) == 1:   # [1, N, D] -> [N, D]
            x = x.squeeze(0)
        if x.dim() == 1:                      # [D] -> [1, D]
            x = x.unsqueeze(0)
        elif x.dim() > 2:                     # [..., D] -> [N, D]
            x = x.view(-1, x.shape[-1])
        return x.float()

    def forward(self, x, return_attn=False):
        # normalize to [N, D]
        x = self._normalize_x(x)              # [N, D]

        # features [N, L] (DO NOT squeeze)
        feature = self.feature(x)             # [N, L]
        if feature.dim() == 1:                # safety, though unlikely
            feature = feature.unsqueeze(0)    # [1, L]

        # attention scores [N, K] (may degenerate to [N] if K==1 in some impls)
        A = self.attention(feature)           # [N, K] or [N]
        if A.dim() == 1:                      # [N] -> [N, 1] to keep 2D
            A = A.unsqueeze(-1)
        elif A.dim() == 0:                    # scalar -> [1, 1] (ultra edge)
            A = A.view(1, 1)

        # KxN, softmax over N
        A = torch.transpose(A, -1, -2)        # [K, N]
        A = F.softmax(A, dim=-1)

        # KxL = (KxN) @ (NxL)
        M = torch.matmul(A, feature)          # [K, L]

        # flatten to [1, L*K] (works for K==1 as well)
        M = M.reshape(1, -1)                  # [1, L*K]

        logits = self.classifier(M)           # [1, n_classes]
        if return_attn:
            return logits, A
        else:
            return logits
