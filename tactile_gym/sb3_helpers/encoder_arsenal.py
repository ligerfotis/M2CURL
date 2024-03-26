from typing import Tuple, Any, Union, List, Type

import gym
from gym import spaces
from stable_baselines3.common.preprocessing import is_image_space, get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN, create_mlp
from stable_baselines3.common.type_aliases import TensorDict
from torch import nn
import torch as t


def compute_info_nce_loss(features, target_features, temperature=0.1):
    """
    Computes the contrastive loss between the features and the target features. The features have shape (batch_size, dim)
    and the target features have shape (batch_size, dim). The contrastive loss is computed as:
    loss = -log(exp(feat * target_feat / tau) / sum(exp(feat * target_feat / tau)))

    :param features:
    :param target_features:
    :return:
    """
    # normalize the features
    q = nn.functional.normalize(features, dim=1)
    with t.no_grad():
        k = nn.functional.normalize(target_features, dim=1)

    logits = t.mm(q, k.T.detach()) / temperature
    labels = t.arange(logits.shape[0], dtype=t.long).to(q.device)
    return nn.CrossEntropyLoss()(logits, labels)


class MocoFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, momentum=0.99):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        # Online Encoder
        self.online_encoder = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Momentum Encoder
        self.momentum_encoder = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Linear layer for online encoder, momentum encoder, and projection head
        n_flatten = self.online_encoder(t.zeros(1, *observation_space.shape)).shape[1]
        self.linear_online = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        self.linear_momentum = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

        self.online_encoder = nn.Sequential(self.online_encoder, self.linear_online)
        self.momentum_encoder = nn.Sequential(self.momentum_encoder, self.linear_momentum)

        # MLP Projection Head (Code)
        self.projection_head_q = nn.Sequential(
            nn.Linear(features_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, features_dim),
        )

        self.projection_head_k = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
        )

        self.update_momentum_encoder(momentum)

    def forward(self, observations: t.Tensor) -> t.Tensor:
        online_features = self.online_encoder(observations)
        # codes = self.projection_head(online_features)
        return online_features

    def encode(self, observations: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        representations_q = self.online_encoder(observations)
        code_q = self.projection_head_q(representations_q)
        with t.no_grad():
            representations_k = self.momentum_encoder(observations)
            code_k = self.projection_head_k(representations_k)
        return code_q, code_k

    def compute_loss(self, observations: t.Tensor, temperature: float) -> t.Tensor:
        code_q, code_k = self.encode(observations)
        loss = compute_info_nce_loss(code_q, code_k, temperature=temperature)
        return loss

    def update_momentum_encoder(self, momentum=0.99):
        # Update target encoder with momentum
        for online_params, momentum_params in zip(self.online_encoder.parameters(), self.momentum_encoder.parameters()):
            momentum_params.data = momentum * momentum_params.data + (1.0 - momentum) * online_params.data


class MViTacFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, mm_hyperparams=None):
        super(MViTacFeatureExtractor, self).__init__(observation_space, features_dim)

        self.observation_space_shape = observation_space.shape
        # Hyperparameters
        self.intra_dim = mm_hyperparams['intra_dim']
        self.inter_dim = mm_hyperparams['inter_dim']

        self.temperature = mm_hyperparams['temperature']

        # Vision modality encoders
        self.vision_base_q, self.vision_head_intra_q, self.vision_head_inter_q = self.create_encoder(
            mm_hyperparams['n_channels_vision'])
        self.vision_base_k, self.vision_head_intra_k, self.vision_head_inter_k = self.create_encoder(
            mm_hyperparams['n_channels_vision'])

        # Tactile modality encoders
        self.tactile_base_q, self.tactile_head_intra_q, self.tactile_head_inter_q = self.create_encoder(
            mm_hyperparams['n_channels_touch'])
        self.tactile_base_k, self.tactile_head_intra_k, self.tactile_head_inter_k = self.create_encoder(
            mm_hyperparams['n_channels_touch'])

        # Initialize key encoders with query encoder weights
        self.m = 0.99  # Momentum factor for key encoder updates
        self.init_key_encoders()

        self.weight_intra_vision = mm_hyperparams['weight_intra_vision']
        self.weight_intra_tactile = mm_hyperparams['weight_intra_tactile']
        self.weight_inter_tac_vis = mm_hyperparams['weight_inter_tac_vis']
        self.weight_inter_vis_tac = mm_hyperparams['weight_inter_vis_tac']

    def create_encoder(self, n_channels):
        # create a custom 2 layer CNN base encoder
        base = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),

        )
        observation_space_shape = (n_channels, *self.observation_space_shape[1:])
        # Compute shape by doing one forward pass
        with t.no_grad():
            flatten_shape = base(t.zeros(observation_space_shape))
            n_flatten = flatten_shape.shape[0] * flatten_shape.shape[1]
        # Linear layer for online encoder, momentum encoder, and projection head
        # add a linear layer
        base = nn.Sequential(base, nn.Linear(n_flatten, 512), nn.ReLU())

        head_inter = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(2048, self.inter_dim)
        )

        head_intra = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(2048, self.intra_dim)
        )

        return base, head_intra, head_inter

    def init_key_encoders(self):
        self.momentum_update_key_encoder(self.vision_base_q, self.vision_base_k, self.vision_head_intra_q,
                                         self.vision_head_intra_k)
        self.momentum_update_key_encoder(self.tactile_base_q, self.tactile_base_k, self.tactile_head_intra_q,
                                         self.tactile_head_intra_k)

    def momentum_update_key_encoder(self, base_q, base_k, head_intra_q, head_intra_k):
        for param_q, param_k in zip(base_q.parameters(), base_k.parameters()):
            param_k.data = self.m * param_k.data + (1. - self.m) * param_q.data

        for param_q, param_k in zip(head_intra_q.parameters(), head_intra_k.parameters()):
            param_k.data = self.m * param_k.data + (1. - self.m) * param_q.data

    def forward(self, observations: t.Tensor) -> t.Tensor:
        """
        The forward pass of the encoder computes the features for the query for both modalities and returns their concatenation.
        :param observations:
        :return:
        """
        x_vision_q = observations[:, :3, :, :]  # Vision modality. The first 3 channels are RGB
        x_tactile_q = observations[:, 3:, :, :]  # Tactile modality. The last 1 channel is the tactile sensor

        # Vision modality
        vision_q = self.vision_base_q(x_vision_q)

        # Tactile modality
        tactile_q = self.tactile_base_q(x_tactile_q)

        # Concatenate features
        features_q = t.cat((vision_q, tactile_q), dim=1)

        return features_q

    def compute_loss(self, vision_observations: t.Tensor, tactile_observations: t.Tensor) -> t.Tensor:
        """
        The encode function computes the codes for the query and the key for both modalities.
        The base encoders provide the features and the projection heads provide the codes.
        :param x_vision_q:
        :param x_vision_k:
        :param x_tactile_q:
        :param x_tactile_k:
        :return:
        """
        # Vision modality
        vision_base_q = self.vision_base_q(vision_observations)
        vis_queries_intra = self.vision_head_intra_q(vision_base_q)
        vis_queries_inter = self.vision_head_inter_q(vision_base_q)

        # Use no_grad context for the key encoders to prevent gradient updates
        with t.no_grad():
            vision_base_k = self.vision_base_k(vision_observations)
            vis_keys_intra = self.vision_head_intra_k(vision_base_k)
            vis_keys_inter = self.vision_head_inter_k(vision_base_k)

        tactile_base_q = self.tactile_base_q(tactile_observations)
        tac_queries_intra = self.tactile_head_intra_q(tactile_base_q)
        tac_queries_inter = self.tactile_head_inter_q(tactile_base_q)

        with t.no_grad():
            tactile_base_k = self.tactile_base_k(tactile_observations)
            tac_keys_intra = self.tactile_head_intra_k(tactile_base_k)
            tac_keys_inter = self.tactile_head_inter_k(tactile_base_k)

            # Compute the contrastive loss for each pair of queries and keys
            vis_loss_intra, logits_vis_intra, labels_vis_intra = compute_info_nce_loss(vis_queries_intra,
                                                                                       vis_keys_intra,
                                                                                       self.temperature)
            tac_loss_intra, logits_tact_intra, labels_tac_intra = compute_info_nce_loss(tac_queries_intra,
                                                                                        tac_keys_intra,
                                                                                        self.temperature)
            vis_tac_inter, logits_vis_tac_inter, labels_vision_tactile_inter = compute_info_nce_loss(vis_queries_inter,
                                                                                                     tac_keys_inter,
                                                                                                     self.temperature)
            tac_vis_inter, logits_tac_vis_inter, labels_tactile_vision_inter = compute_info_nce_loss(tac_queries_inter,
                                                                                                     vis_keys_inter,
                                                                                                     self.temperature)

            # Combine losses
            combined_loss = (self.weight_intra_vision * vis_loss_intra
                             + self.weight_intra_tactile * tac_loss_intra
                             + self.weight_inter_tac_vis * vis_tac_inter
                             + self.weight_inter_vis_tac * tac_vis_inter)

            # # Perform momentum update during the forward pass
            # self.momentum_update_key_encoder(self.vision_base_q, self.vision_base_k, self.vision_head_intra_q,
            #                                  self.vision_head_intra_k)

            return combined_loss


class MViTacFeatureExtractor2(BaseFeaturesExtractor):
    """
        Combined feature extractor for Dict observation spaces.
        Builds a feature extractor for each key of the space. Input from each space
        is fed through a separate submodule (CNN or MLP, depending on input shape),
        the output features are concatenated and fed through additional MLP network ("combined").

        :param observation_space:
        :param mlp_extractor_net_arch: Architecture for mlp encoding of state features before concatentation to cnn output
        :param mlp_activation_fn: Activation Func for MLP encoding layers
        :param cnn_output_dim: Number of features to output from each CNN submodule(s)
        """

    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            mlp_extractor_net_arch: Union[int, List[int]] = None,
            mlp_activation_fn: Type[nn.Module] = nn.Tanh,
            cnn_output_dim: int = 64,
            cnn_base: Type[BaseFeaturesExtractor] = NatureCNN,
            mm_hyperparams=None
    ):
        super(MViTacFeatureExtractor2, self).__init__(observation_space, features_dim=1)

        cnn_extractors = {}
        cnn_momentum_extractors = {}
        flatten_extractors = {}

        self.inter_dim = mm_hyperparams['inter_dim']
        self.intra_dim = mm_hyperparams['intra_dim']

        cnn_concat_size = 0
        flatten_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                # create online encoder
                cnn_extractors[key] = cnn_base(subspace, features_dim=cnn_output_dim)
                # create momentum encoder
                cnn_momentum_extractors[key] = cnn_base(subspace, features_dim=cnn_output_dim)
                # compute the size of the concatenated features
                cnn_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                flatten_extractors[key] = nn.Flatten()
                flatten_concat_size += get_flattened_obs_dim(subspace)

        total_concat_size = cnn_concat_size + flatten_concat_size

        # default mlp arch to empty list if not specified
        if mlp_extractor_net_arch is None:
            mlp_extractor_net_arch = []

        for layer in mlp_extractor_net_arch:
            assert isinstance(layer, int), "Error: the mlp_extractor_net_arch can only include ints"

        # once vector obs is flattened can pass it through mlp
        if (mlp_extractor_net_arch != []) and (flatten_concat_size > 0):
            mlp_extractor = create_mlp(
                flatten_concat_size,
                mlp_extractor_net_arch[-1],
                mlp_extractor_net_arch[:-1],
                mlp_activation_fn
            )
            self.mlp_extractor = nn.Sequential(*mlp_extractor)
            self.mlp_extractor_momentum = nn.Sequential(*mlp_extractor)
            final_features_dim = mlp_extractor_net_arch[-1] + cnn_concat_size
        else:
            self.mlp_extractor = None
            final_features_dim = total_concat_size

        self.cnn_extractors = nn.ModuleDict(cnn_extractors)
        self.flatten_extractors = nn.ModuleDict(flatten_extractors)
        self.cnn_momentum_extractors = nn.ModuleDict(cnn_momentum_extractors)

        # Update the features dim manually
        self._features_dim = final_features_dim

        # # create heads for intra and inter modalities
        # self.observation_space_shape_visual = observation_space.spaces['visual'].shape
        # self.observation_space_shape_tactile = observation_space.spaces['tactile'].shape

        # vision heads
        self.vision_head_intra_q, self.vision_head_inter_q = self.create_heads()
        self.vision_head_intra_k, self.vision_head_inter_k = self.create_heads()

        # tactile heads
        self.tactile_head_intra_q, self.tactile_head_inter_q = self.create_heads()
        self.tactile_head_intra_k, self.tactile_head_inter_k = self.create_heads()

        # Initialize key encoders with query encoder weights
        self.m = 0.99  # Momentum factor for key encoder updates
        self.momentum_update_key_encoder()

        self.temperature = mm_hyperparams['temperature']
        self.weight_intra_vision = mm_hyperparams['weight_intra_vision']
        self.weight_intra_tactile = mm_hyperparams['weight_intra_tactile']
        self.weight_inter_tac_vis = mm_hyperparams['weight_inter_tac_vis']
        self.weight_inter_vis_tac = mm_hyperparams['weight_inter_vis_tac']

    def forward(self, observations: TensorDict) -> t.Tensor:
        # encode image obs through cnn
        cnn_encoded_tensor_list = []
        for key, extractor in self.cnn_extractors.items():
            # todo: cuda hard-coded here
            x_modality = observations[key].to("cuda")
            cnn_encoded_tensor_list.append(extractor(x_modality))

        # flatten vector obs
        flatten_encoded_tensor_list = []
        for key, extractor in self.flatten_extractors.items():
            flatten_encoded_tensor_list.append(extractor(observations[key]))

        # encode combined flat vector obs through mlp extractor (if set)
        # and combine with cnn outputs
        if self.mlp_extractor is not None:
            extracted_tensor = self.mlp_extractor(t.cat(flatten_encoded_tensor_list, dim=1))
            comb_extracted_tensor = t.cat([*cnn_encoded_tensor_list, extracted_tensor], dim=1)
        else:
            comb_extracted_tensor = t.cat([*cnn_encoded_tensor_list, *flatten_encoded_tensor_list], dim=1)

        return comb_extracted_tensor

    def create_heads(self):
        head_inter = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(2048, self.inter_dim)
        )

        head_intra = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(2048, self.intra_dim)
        )

        return head_intra, head_inter

    def momentum_update_key_encoder(self, ) -> None:
        # Update target encoder with momentum
        for online_params, momentum_params in zip(self.cnn_extractors.parameters(),
                                                  self.cnn_momentum_extractors.parameters()):
            momentum_params.data = self.m * momentum_params.data + (1.0 - self.m) * online_params.data

    def compute_loss(self, vision_observations: t.Tensor, tactile_observations: t.Tensor) -> tuple[
        Any, Any, Any, Any, Any]:
        """
        The encode function computes the codes for the query and the key for both modalities.
        The base encoders provide the features and the projection heads provide the codes.
        :param tactile_observations:
        :param vision_observations:
        :return:
        """
        # Vision modality online encoder and heads
        vision_base_q = self.cnn_extractors['visual'](vision_observations)
        vis_queries_intra = self.vision_head_intra_q(vision_base_q)
        vis_queries_inter = self.vision_head_inter_q(vision_base_q)
        # Tactile modality online encoder and heads
        tactile_base_q = self.cnn_extractors['tactile'](tactile_observations)
        tac_queries_intra = self.tactile_head_intra_q(tactile_base_q)
        tac_queries_inter = self.tactile_head_inter_q(tactile_base_q)

        # Use no_grad context for the key encoders to prevent gradient updates
        with t.no_grad():
            # Vision modality momentum encoder and heads
            vision_base_k = self.cnn_momentum_extractors['visual'](vision_observations)
            vis_keys_intra = self.vision_head_intra_k(vision_base_k)
            vis_keys_inter = self.vision_head_inter_k(vision_base_k)
            # Tactile modality  momentum encoder and heads
            tactile_base_k = self.cnn_momentum_extractors['tactile'](tactile_observations)
            tac_keys_intra = self.tactile_head_intra_k(tactile_base_k)
            tac_keys_inter = self.tactile_head_inter_k(tactile_base_k)

        # with t.no_grad():
        # Compute the contrastive loss for each pair of queries and keys
        vis_loss_intra = compute_info_nce_loss(vis_queries_intra, vis_keys_intra, self.temperature)
        tac_loss_intra = compute_info_nce_loss(tac_queries_intra, tac_keys_intra, self.temperature)
        vis_tac_inter = compute_info_nce_loss(vis_queries_inter, tac_keys_inter, self.temperature)
        tac_vis_inter = compute_info_nce_loss(tac_queries_inter, vis_keys_inter, self.temperature)

        # Combine losses
        combined_loss = (self.weight_intra_vision * vis_loss_intra
                         + self.weight_intra_tactile * tac_loss_intra
                         + self.weight_inter_tac_vis * vis_tac_inter
                         + self.weight_inter_vis_tac * tac_vis_inter)

        return combined_loss, vis_loss_intra, tac_loss_intra, vis_tac_inter, tac_vis_inter


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with t.no_grad():
            n_flatten = self.cnn(
                t.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: t.Tensor) -> t.Tensor:
        return self.linear(self.cnn(observations))
