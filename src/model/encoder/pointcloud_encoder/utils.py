import torch

def normalize_pointclouds(pointclouds):
    # pointclouds shape: (b, n, 3)

    # Step 1: Center the point clouds (subtract mean)
    means = pointclouds.mean(dim=1, keepdim=True)  # Shape: (b, 1, 3)
    centered_pointclouds = pointclouds - means

    # Step 2: Scale the point clouds (divide by max norm)
    norms = torch.norm(centered_pointclouds, dim=2, keepdim=True).max(dim=1, keepdim=True)[0]  # Shape: (b, 1, 1)
    normalized_pointclouds = centered_pointclouds / norms  # Shape: (b, n, 3)

    return normalized_pointclouds
