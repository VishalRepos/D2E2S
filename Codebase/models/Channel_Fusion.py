import torch
from sklearn.decomposition import PCA
import torch.nn as nn
import torch.nn.functional as F

def Orthographic_projection_fusion(feature1, feature2, feature3):
    batch_size, seq_len, feature_dim = feature1.shape
    
    # Combine features
    combined_features = torch.cat([feature1, feature2, feature3], dim=-1)
    
    # Perform PCA
    pca = PCA(n_components=feature_dim)
    combined_features_flat = combined_features.view(-1, combined_features.size(-1))
    fused_feature_vector = torch.from_numpy(pca.fit_transform(combined_features_flat.cpu().numpy())).to(feature1.device)
    
    # Reshape back to original batch and sequence dimensions
    fused_feature_vector = fused_feature_vector.view(batch_size, seq_len, -1)
    
    return fused_feature_vector

class TextCentredSP(nn.Module):
    def __init__(self, input_dims, shared_dims, private_dims):
        super(TextCentredSP, self).__init__()
        self.input_dims = input_dims
        self.shared_dims = shared_dims
        self.private_dims = private_dims

        # Shared Semantic Mask Matrix
        self.shared_mask = nn.Parameter(torch.ones(self.input_dims))
        # Personalized Semantic Mask Matrix
        self.private_mask = nn.Parameter(torch.ones(self.input_dims))

        # Shared Semantic Encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(self.input_dims, self.shared_dims),
            nn.ReLU()
        )

        # Personalized Semantic Encoder
        self.private_encoder = nn.Sequential(
            nn.Linear(self.input_dims, self.private_dims),
            nn.ReLU()
        )

    def forward(self, h_syn_ori, h_syn_feature):
        # Stitching together the features of the three modalities
        features = torch.cat((h_syn_ori, h_syn_feature), dim=2)

        # Calculating the shared semantic mask matrix
        shared_weights = F.softmax(self.shared_mask.view(-1), dim=0).view(self.input_dims)
        shared_mask = shared_weights > 0.2  # threshold
        shared_mask = shared_mask.float()

        # Calculate the personality semantic mask matrix
        private_mask = 1 - shared_mask

        # Masking of the features of the three modalities
        shared_features = features * shared_mask
        private_features = features * private_mask

        # Encoding shared semantic and individual semantic features
        shared_code = self.shared_encoder(shared_features)
        private_code = self.private_encoder(private_features)

        # Shared semantic and individual semantic features after merged encoding
        output = torch.cat((shared_code, private_code), dim=2)

        return output