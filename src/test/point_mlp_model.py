import torch
import torch.nn as nn

class PointWiseMLP(nn.Module):
    """
    point-wise MLP, no aggregation
    extract features for each point individually
    """
    
    def __init__(self, in_channels, out_channels, hidden_dims=[128, 128]):
        super(PointWiseMLP, self).__init__()
        # in_channels: input channel dimension (c)
        # out_channels: desired output channel dimension (c_out)
        # hidden_dims: list of hidden layer sizes
        
        layers = []
        prev_dim = in_channels
        
        # Build hidden layers
        for dim in hidden_dims:
            layers.append(nn.Conv1d(prev_dim, dim, kernel_size=1))  # 1D conv acts on channel dim
            layers.append(nn.BatchNorm1d(dim))  # Optional: stabilizes training
            layers.append(nn.ReLU(inplace=True))
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Conv1d(prev_dim, out_channels, kernel_size=1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor):
        # Input x: (b, c, n)
        # Output: (b, c_out, n)
        return self.mlp(x)

if __name__ == "__main__":
    
    # Example usage
    batch_size = 2
    in_channels = 256  # relu3_1 vgg feature dim
    num_points = 1024
    out_channels = 83  # gs attributes dim

    # Random point cloud data
    point_cloud = torch.randn(batch_size, in_channels, num_points)

    # Create and apply the MLP
    mlp = PointWiseMLP(in_channels=in_channels, out_channels=out_channels)
    output = mlp(point_cloud)

    print("Input shape:", point_cloud.shape) 
    print("Output shape:", output.shape)   