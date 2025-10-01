"""
adapt from 3D photo stylization codebase 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.misc.nn_module_tools import convert_to_buffer
from .vgg import NormalizedVGG, make_dvgg

class AdaIN(nn.Module):
    """ Adaptive instance normalization (Huang et al., ICCV 17) """

    def __init__(self):
        super(AdaIN, self).__init__()

    def forward(self, c, s):
        mean = s.mean((-2, -1)).unsqueeze(-1).expand_as(c)
        std = s.std((-2, -1)).unsqueeze(-1).expand_as(c)
        cs = F.instance_norm(c) * std + mean
        return cs

class AdaAttN(nn.Module):
    """ Attention-weighted AdaIN (Liu et al., ICCV 21) """

    def __init__(self, qk_dim=None, v_dim=None, project=False):
        """
        Args:
            qk_dim (int): query and key size.
            v_dim (int): value size.
            project (int): apply projection to input features.
        """
        super(AdaAttN, self).__init__()

        if project:
            assert qk_dim is not None and v_dim is not None, \
                '[ERROR] qk_dim and v_dim must be given for feature projection'
            self.q_embed = nn.Conv1d(qk_dim, qk_dim, 1)
            self.k_embed = nn.Conv1d(qk_dim, qk_dim, 1)
            self.s_embed = nn.Conv1d(v_dim, v_dim, 1)
        else:
            self.q_embed = self.k_embed = self.s_embed = nn.Identity()

    def forward(self, q, k, c, s):
        """
        Args:
            q (float tensor, (bs, qk, *)): query (content) features.
            k (float tensor, (bs, qk, *)): key (style) features.
            c (float tensor, (bs, v, *)): content value features.
            s (float tensor, (bs, v, *)): style value features.

        Returns:
            cs (float tensor, (bs, v, *)): stylized content features.
        """
        shape = c.shape
        q, k = q.flatten(2), k.flatten(2)
        c, s = c.flatten(2), s.flatten(2)

        # QKV attention with projected content and style features
        q = self.q_embed(F.instance_norm(q)).transpose(2, 1)    # (bs, n, qk)
        k = self.k_embed(F.instance_norm(k))                    # (bs, qk, m)
        s = self.s_embed(s).transpose(2, 1)                     # (bs, m, v)
        attn = F.softmax(torch.bmm(q, k), -1)                   # (bs, n, m)
        
        # attention-weighted channel-wise statistics
        mean = torch.bmm(attn, s)                               # (bs, n, v)
        var = F.relu(torch.bmm(attn, s ** 2) - mean ** 2)       # (bs, n, v)
        mean = mean.transpose(2, 1)                             # (bs, v, n)
        std = torch.sqrt(var).transpose(2, 1)                   # (bs, v, n)
        
        cs = F.instance_norm(c) * std + mean                    # (bs, v, n)
        cs = cs.reshape(shape)
        return cs

class AdaAttN3DStylizer(nn.Module):
    """ Attention-weighted AdaIN stylizer (Liu et al., ICCV 21) """

    def __init__(self, feats_in_dim, vgg_layer=3, n_zip_layers=2):
        super(AdaAttN3DStylizer, self).__init__()

        vgg_dims = [64, 128, 256, 512, 512]

        v_dim = vgg_dims[vgg_layer - 1]
        
        qk_dim = v_dim
        self.vgg = NormalizedVGG(vgg_layer, pool='max')
        convert_to_buffer(self.vgg, persistent=False)
        
        n_layers = n_zip_layers
        
        self.adaattn = AdaAttN(qk_dim, v_dim, project=True)

        # transform content features to match VGG feature space
        if n_layers > 0:
            q_zipper = [nn.Conv1d(feats_in_dim, qk_dim, 1)]
            v_zipper = [nn.Conv1d(feats_in_dim, v_dim, 1)]
            v_unzipper = [nn.Conv1d(v_dim, feats_in_dim, 1)]
            for i in range(n_layers - 1):
                q_zipper = q_zipper + [
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv1d(qk_dim, qk_dim, 1),
                ]
                v_zipper = v_zipper + [
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv1d(v_dim, v_dim, 1),
                ]
                v_unzipper = [
                    nn.Conv1d(v_dim, v_dim, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                ] + v_unzipper
            self.q_zipper = nn.Sequential(*q_zipper)
            self.v_zipper = nn.Sequential(*v_zipper)
            self.v_unzipper = nn.Sequential(*v_unzipper)
        else:
            self.q_zipper = self.v_zipper = self.v_unzipper = nn.Identity()

    def forward(self, style, feats_in):
        """
        Args:
            style (float tensor, (bs, 3, h, w)): style image.
            feats_in (float tensor, (bs, c, n)): content features

        Returns:
            feats_out (float tensor, (bs, c, n)): transformed content features.
        """
        # pyramidal query features
        k = s = self.vgg(style)
        q = c = feats_in

        q, c = self.q_zipper(q), self.v_zipper(c)
        cs = self.adaattn(q, k, c, s)
        feats_out = self.v_unzipper(cs)

        return feats_out
    
class LST(nn.Module):

    def __init__(self, in_dim, embed_dim=32, n_layers=3):
        super(LST, self).__init__()

        self.embed_dim = embed_dim

        self.c_zipper = nn.Conv1d(in_dim, embed_dim, 1)
        self.c_unzipper = nn.Conv1d(embed_dim, in_dim, 1)

        c_net, s_net = [], []
        for i in range(n_layers - 1):
            out_dim = max(embed_dim, in_dim // 2)
            c_net.append(
                nn.Sequential(
                    nn.Conv1d(in_dim, out_dim, 1),
                    nn.ReLU(inplace=True),
                )
            )
            s_net.append(
                nn.Sequential(
                    nn.Conv1d(in_dim, out_dim, 1),
                    nn.ReLU(inplace=True),
                )
            )
            in_dim = out_dim
        c_net.append(nn.Conv1d(in_dim, embed_dim, 1))
        s_net.append(nn.Conv1d(in_dim, embed_dim, 1))
        self.c_net = nn.Sequential(*c_net)
        self.s_net = nn.Sequential(*s_net)

        self.c_fc = nn.Linear(embed_dim ** 2, embed_dim ** 2)
        self.s_fc = nn.Linear(embed_dim ** 2, embed_dim ** 2)

    def _vectorized_covariance(self, x):
        cov = torch.bmm(x, x.transpose(2, 1)) / x.size(-1)
        cov = cov.flatten(1)
        return cov

    def forward(self, c, s):
        c_shape = c.shape
        c, s = c.flatten(2), s.flatten(2)
        
        c_mean = c.mean(-1, keepdim=True)
        s_mean = s.mean(-1, keepdim=True)
        c = c - c_mean
        s = s - s_mean

        c_embed = self.c_net(c)
        c_cov = self._vectorized_covariance(c_embed)
        c_mat = self.c_fc(c_cov)
        c_mat = c_mat.reshape(-1, self.embed_dim, self.embed_dim)

        s_embed = self.s_net(s)
        s_cov = self._vectorized_covariance(s_embed)
        s_mat = self.s_fc(s_cov)
        s_mat = s_mat.reshape(-1, self.embed_dim, self.embed_dim)

        mat = torch.bmm(s_mat, c_mat)
        c = self.c_zipper(c)
        c = torch.bmm(mat, c)
        c = self.c_unzipper(c)
        cs = c + s_mean

        cs = cs.reshape(*c_shape)
        return cs

class Linear3DStylizer(nn.Module):
    """ Learned affine transform on point features (Li et al., CVPR 19) """

    def __init__(self, vgg_layer=3):
        super(Linear3DStylizer, self).__init__()

        vgg_dims = [64, 128, 256, 512, 512]

        self.vgg = NormalizedVGG(vgg_layer, pool='max')
        convert_to_buffer(self.vgg, persistent=False)
        
        self.lst = LST(
            in_dim=vgg_dims[vgg_layer - 1], 
            embed_dim=32, 
            n_layers=3
        )

    def forward(self, style, feats_in):
        """
        Args:
            style (float tensor, (bs, 3, h, w)): style image.
            xyz_list (float tensor list): point NDC coordinates at all levels.
            feats_list (float tensor list): point features at all levels.

        Returns:
            feats (float tensor, (bs, c, n)): transformed content features.
        """
        style = self.vgg(style)
        feats_out = self.lst(feats_in, style)

        return feats_out

class AdaIN3DStylizer(nn.Module):
    """ Learned AdaIN on point features (Huang et al., ICCV 17) """

    def __init__(self, vgg_layer=3, n_zip_layers=2):
        super(AdaIN3DStylizer, self).__init__()

        vgg_dims = [64, 128, 256, 512, 512]
        self.vgg = NormalizedVGG(vgg_layer, pool='max')
        convert_to_buffer(self.vgg, persistent=False)

        in_dim = vgg_dims[vgg_layer - 1]

        # content feature projection
        n_layers = n_zip_layers
        if n_layers > 0:
            zipper = [nn.Conv1d(in_dim, in_dim, 1)]
            unzipper = [nn.Conv1d(in_dim, in_dim, 1)]
            for i in range(n_layers - 1):
                zipper = zipper + [
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv1d(in_dim, in_dim, 1),
                ]
                unzipper = [
                    nn.Conv1d(in_dim, in_dim, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                ] + unzipper
            self.zipper = nn.Sequential(*zipper)
            self.unzipper = nn.Sequential(*unzipper)
        else:
            self.zipper = self.unzipper = nn.Identity()

        self.adain = AdaIN()

    def forward(self, style, feats_in):
        """
        Args:
            style (float tensor, (bs, 3, h, w)): style image.
            xyz_list (float tensor list): point NDC coordinates at all levels.
            feats_list (float tensor list): point features at all levels.
            up (bool): if True, upsample output features to input resolution.

        Returns:
            feats (float tensor, (bs, c, n)): transformed content features.
        """
        style = self.vgg(style)
        feats = self.zipper(feats_in)
        feats = self.adain(feats, style)
        feats_out = self.unzipper(feats)
        
        return feats_out

class CrossAttentionTransformer(nn.Module):
    def __init__(self, feature_dim, vgg_layer=4, n_zip_layers=2, num_heads=1):
        super(CrossAttentionTransformer, self).__init__()
        
        vgg_dims = [64, 128, 256, 512, 512]
        self.vgg = NormalizedVGG(vgg_layer, pool='max')
        convert_to_buffer(self.vgg, persistent=False)
        
        in_dim = vgg_dims[vgg_layer - 1]
        
        # content feature projection
        n_layers = n_zip_layers
        if n_layers > 0:
            zipper = [nn.Conv1d(feature_dim, in_dim, 1)]
            unzipper = [nn.Conv1d(in_dim, feature_dim, 1)]
            for i in range(n_layers - 1):
                zipper = zipper + [
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv1d(in_dim, in_dim, 1),
                ]
                unzipper = [
                    nn.Conv1d(in_dim, in_dim, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                ] + unzipper
            self.zipper = nn.Sequential(*zipper)
            self.unzipper = nn.Sequential(*unzipper)
        else:
            self.zipper = self.unzipper = nn.Identity()
        
        self.feature_dim = in_dim  # c (channel dimension)
        self.num_heads = num_heads

        # Linear projections for Query (point cloud), Key/Value (image)
        self.query_proj = nn.Linear(in_dim, in_dim)
        self.key_proj = nn.Linear(in_dim, in_dim)
        self.value_proj = nn.Linear(in_dim, in_dim)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, batch_first=True)

        self.point_pos_enc = nn.Linear(3, in_dim)  # For 3D point coordinates
        
        # Feedforward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, in_dim * 4),
            nn.ReLU(),
            nn.Linear(in_dim * 4, in_dim)
        )
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
    
    def get_2d_sinusoidal_pos_enc(self, h, w, feature_dim, device):
        """
        Generate 2D sinusoidal positional encodings for image grid (h, w).
        Adapted from 1D sinusoidal encoding in "Attention is All You Need".
        """
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        grid_y, grid_x = grid_y.float().to(device), grid_x.float().to(device)
        
        # Frequency terms
        div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / feature_dim)).to(device)
        
        # Sine and cosine for x and y
        pos_x = grid_x[..., None] * div_term  # (h, w, feature_dim//2)
        pos_y = grid_y[..., None] * div_term  # (h, w, feature_dim//2)
        
        pe = torch.zeros(h, w, feature_dim).to(device)
        pe[..., 0::2] = torch.sin(pos_x)  # Even indices: sin
        pe[..., 1::2] = torch.cos(pos_y)  # Odd indices: cos
        return pe.view(1, h * w, feature_dim)  # (1, h*w, feature_dim)
    
    def forward(self, image_features, point_features, point_coords):
        """
        Args:
            image_features: (b, c, h, w) - Image features
            point_features: (b, c, n) - Point cloud features
            point_coords: (b, n, c) - Point cloud coordinates
        Returns:
            fused_features: (b, c, n) - Fused point cloud features
        """
        image_features = self.vgg(image_features)
        point_features = self.zipper(point_features)
        
        b, c, n = point_features.shape
        _, _, h, w = image_features.shape

        # Point cloud positional encoding (learned)
        point_pos = self.point_pos_enc(point_coords)  # (b, n, c)
        
        # Image positional encoding (sinusoidal)
        # image_pos = self.get_2d_sinusoidal_pos_enc(h, w, c, point_features.device)  # (1, h*w, c)
        # image_pos = image_pos.expand(b, -1, -1)  # (b, h*w, c)
        
        # Reshape image features: (b, c, h, w) -> (b, h*w, c)
        image_features = image_features.view(b, c, h * w).permute(0, 2, 1)
        # image_features = image_features.view(b, c, h * w).permute(0, 2, 1) + image_pos # (b, h*w, c)

        # Reshape point features: (b, c, n) -> (b, n, c)
        # point_features = point_features.permute(0, 2, 1) # (b, n, c)
        point_features = point_features.permute(0, 2, 1) + point_pos # (b, n, c)

        # Project to attention space
        queries = self.query_proj(point_features)  # (b, n, c)
        keys = self.key_proj(image_features)       # (b, h*w, c)
        values = self.value_proj(image_features)   # (b, h*w, c)

        # Cross-attention: points query the image
        attn_output, _ = self.attention(queries, keys, values)  # (b, n, c)

        # Residual connection and normalization
        point_features = self.norm1(point_features + attn_output)

        # Feedforward network
        ffn_output = self.ffn(point_features)
        fused_features = self.norm2(point_features + ffn_output)  # (b, n, c)

        # Reshape back to (b, c, n)
        fused_features = fused_features.permute(0, 2, 1)  # (b, c, n)

        fused_features = self.unzipper(fused_features)
        
        return fused_features