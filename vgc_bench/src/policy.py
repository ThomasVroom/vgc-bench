"""
Neural network policy module for VGC-Bench.

Implements the actor-critic policy architecture with attention-based feature
extraction for Pokemon VGC battles. Uses action masking to ensure only legal
moves are selected.
"""

from typing import Any

import torch
from gymnasium import Space
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs
from torch import nn

from vgc_bench.src.utils import (
    abilities,
    act_len,
    chunk_obs_len,
    glob_obs_len,
    items,
    moves,
    side_obs_len,
)

action_map = (
    ["pass", "switch 1", "switch 2", "switch 3", "switch 4", "switch 5", "switch 6"]
    + [f"move {i} target {j}" for i in range(1, 5) for j in range(-2, 3)]
    + [f"move {i} target {j} mega" for i in range(1, 5) for j in range(-2, 3)]
    + [f"move {i} target {j} zmove" for i in range(1, 5) for j in range(-2, 3)]
    + [f"move {i} target {j} dynamax" for i in range(1, 5) for j in range(-2, 3)]
    + [f"move {i} target {j} tera" for i in range(1, 5) for j in range(-2, 3)]
)


class MaskedActorCriticPolicy(ActorCriticPolicy):
    """
    Actor-critic policy with action masking for Pokemon VGC.

    Extends SB3's ActorCriticPolicy with action masking to enforce legal
    moves and uses an attention-based feature extractor for processing
    Pokemon battle observations.

    Attributes:
        choose_on_teampreview: Whether policy controls teampreview decisions.
        actor_grad: Whether to compute gradients for actor during evaluation.
        debug: Whether to print debug information during forward pass.
    """

    def __init__(
        self, *args: Any, d_model: int, choose_on_teampreview: bool, progressive: bool, n_columns: int=1, **kwargs: Any
    ):
        """
        Initialize the masked actor-critic policy.

        Args:
            d_model: Hidden size for policy/value networks and attention extractor.
            choose_on_teampreview: Whether policy controls teampreview.
            progressive: Whether policy uses a progressive architecture.
            n_columns: How many columns to use for the progressive architecture.
            *args: Additional arguments for ActorCriticPolicy.
            **kwargs: Additional keyword arguments for ActorCriticPolicy.
        """
        self.choose_on_teampreview = choose_on_teampreview
        self.actor_grad = True
        self.debug = False
        super().__init__(
            *args,
            **kwargs,
            net_arch=[],
            activation_fn=torch.nn.ReLU,
            features_extractor_class=ProgressiveAttentionExtractor if progressive else AttentionExtractor,
            features_extractor_kwargs={
                "d_model": d_model,
                "choose_on_teampreview": choose_on_teampreview,
                "n_columns": n_columns,
            } if progressive else {
                "d_model": d_model,
                "choose_on_teampreview": choose_on_teampreview,
            },
            share_features_extractor=False,
        )
        if progressive: # smaller head inits to prevent large bias
            torch.nn.init.orthogonal_(self.action_net.weight, gain=0.01) # type: ignore
            torch.nn.init.constant_(self.action_net.bias, 0.0)           # type: ignore
            torch.nn.init.orthogonal_(self.value_net.weight, gain=0.01)
            torch.nn.init.constant_(self.value_net.bias, 0.0)

    def forward(self, obs: PyTorchObs, deterministic=False):
        assert isinstance(obs, dict)
        action_logits, value_logits = self.get_logits(obs, actor_grad=True)
        distribution = self.get_dist_from_logits(action_logits, obs["action_mask"])
        actions = distribution.get_actions(deterministic=deterministic)
        distribution2 = self.get_dist_from_logits(
            action_logits, obs["action_mask"], actions[:, :1]
        )
        actions2 = distribution2.get_actions(deterministic=deterministic)
        distribution.distribution[1] = distribution2.distribution[1]
        actions[:, 1] = actions2[:, 1]
        if self.debug:
            print("value:", value_logits[0][0].item())
            action_dist1 = {
                action_map[i]: f"{p.item():.3e}"
                for i, p in enumerate(distribution.distribution[0].probs[0])
                if p > 0
            }
            action_dist1 = dict(
                sorted(action_dist1.items(), key=lambda x: float(x[1]), reverse=True)
            )
            print("action1 dist:", action_dist1)
            action_dist2 = {
                action_map[i]: f"{p.item():.3e}"
                for i, p in enumerate(distribution.distribution[1].probs[0])
                if p > 0
            }
            action_dist2 = dict(
                sorted(action_dist2.items(), key=lambda x: float(x[1]), reverse=True)
            )
            print("action2 dist:", action_dist2)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, value_logits, log_prob

    def evaluate_actions(self, obs, actions):
        assert isinstance(obs, dict)
        action_logits, value_logits = self.get_logits(obs, self.actor_grad)
        distribution = self.get_dist_from_logits(action_logits, obs["action_mask"])
        distribution2 = self.get_dist_from_logits(
            action_logits, obs["action_mask"], actions[:, :1]
        )
        distribution.distribution[1] = distribution2.distribution[1]
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return value_logits, log_prob, entropy

    def get_logits(
        self, obs: dict[str, torch.Tensor], actor_grad: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract features and compute action/value logits."""
        actor_context = torch.enable_grad() if actor_grad else torch.no_grad()
        features = self.extract_features(obs)
        pi_features, vf_features = features
        with actor_context:
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        with actor_context:
            action_logits = self.action_net(latent_pi)
        value_logits = self.value_net(latent_vf)
        return action_logits, value_logits

    def get_dist_from_logits(
        self,
        action_logits: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> MultiCategoricalDistribution:
        """Create masked action distribution from logits."""
        if action is not None:
            mask = self._update_mask(mask, action)
        mask = torch.where(mask == 1, 0, float("-inf"))
        distribution = self.action_dist.proba_distribution(action_logits + mask)
        assert isinstance(distribution, MultiCategoricalDistribution)
        return distribution

    @staticmethod
    def _update_mask(mask: torch.Tensor, ally_actions: torch.Tensor) -> torch.Tensor:
        """
        Update action mask based on ally's already-chosen action.

        Prevents illegal combinations like both Pokemon switching to the same
        slot, both passing when not forced, or both terastallizing.

        Args:
            mask: Current action mask tensor of shape (batch, 2*act_len).
            ally_actions: Ally's chosen actions of shape (batch, 1).

        Returns:
            Updated mask tensor with illegal actions disabled.
        """
        indices = (
            torch.arange(act_len, device=ally_actions.device)
            .unsqueeze(0)
            .expand(len(ally_actions), -1)
        )
        ally_passed = ally_actions == 0
        ally_force_passed = (
            (mask[:, 0] == 1) & (mask[:, :act_len].sum(1) == 1)
        ).unsqueeze(1)
        ally_switched = (1 <= ally_actions) & (ally_actions <= 6)
        ally_terastallized = (86 < ally_actions) & (ally_actions <= 106)
        updated_half = mask[:, act_len:] * ~(
            ((indices == 0) & ally_passed & ~ally_force_passed)
            | ((indices == ally_actions) & ally_switched)
            | ((86 < indices) & (indices <= 106) & ally_terastallized)
        )
        return torch.cat([mask[:, :act_len], updated_half], dim=1)


class AttentionExtractor(BaseFeaturesExtractor):
    """
    Attention-based feature extractor for Pokemon battle observations.

    Processes Pokemon observations using embeddings for abilities, items, and
    moves, then applies transformer attention to produce a fixed-size feature
    vector.

    Class Attributes:
        embed_len: Dimension of embedding vectors for abilities/items/moves.
        num_heads: Number of attention heads in transformer layers.
        embed_layers: Number of transformer encoder layers.
    """

    embed_len: int = 32
    num_heads: int = 4
    embed_layers: int = 3

    def __init__(
        self, observation_space: Space[Any], d_model: int, choose_on_teampreview: bool
    ):
        """
        Initialize the attention-based feature extractor.

        Args:
            observation_space: Gymnasium observation space specification.
            d_model: Hidden size for token projection and transformer layers.
            choose_on_teampreview: Whether policy controls teampreview decisions.
        """
        super().__init__(observation_space, features_dim=d_model)
        self.choose_on_teampreview = choose_on_teampreview
        self.ability_embed = nn.Embedding(
            len(abilities), self.embed_len, max_norm=self.embed_len**0.5
        )
        self.item_embed = nn.Embedding(
            len(items), self.embed_len, max_norm=self.embed_len**0.5
        )
        self.move_embed = nn.Embedding(
            len(moves), self.embed_len, max_norm=self.embed_len**0.5
        )
        self.pokemon_proj = nn.Linear(chunk_obs_len + 6 * (self.embed_len - 1), d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pokemon_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=self.num_heads,
                dim_feedforward=d_model,
                dropout=0,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=self.embed_layers,
            enable_nested_tensor=False,
        )

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract features from battle observation.

        Embeds Pokemon attributes and applies transformer attention across all
        12 Pokemon (6 per side).

        Args:
            x: Dict with an ``"observation"`` key containing a tensor of
                shape (batch, 12 * chunk_obs_len).

        Returns:
            Feature tensor of shape (batch, d_model).
        """
        x = obs_dict["observation"]
        batch_size = x.size(0)
        pokemon_obs = x.view(batch_size, 12, -1)
        # embedding
        start = glob_obs_len + side_obs_len
        pokemon_obs = torch.cat(
            [
                pokemon_obs[:, :, :start],
                self.ability_embed(pokemon_obs[:, :, start].long()),
                self.item_embed(pokemon_obs[:, :, start + 1].long()),
                self.move_embed(pokemon_obs[:, :, start + 2].long()),
                self.move_embed(pokemon_obs[:, :, start + 3].long()),
                self.move_embed(pokemon_obs[:, :, start + 4].long()),
                self.move_embed(pokemon_obs[:, :, start + 5].long()),
                pokemon_obs[:, :, start + 6 :],
            ],
            dim=-1,
        )
        # pokemon encoder
        pokemon_tokens = self.pokemon_proj(pokemon_obs)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, pokemon_tokens], dim=1)
        return self.pokemon_encoder(tokens)[:, 0, :]


class ProgressiveAttentionExtractor(BaseFeaturesExtractor):
    """
    Attention-based feature extractor for Pokemon battle observations.
    Adapted to a progressive architecture for transfer learning.

    Processes Pokemon observations using embeddings for abilities, items, and
    moves, then applies transformer attention to produce a fixed-size feature
    vector.
    """

    def __init__(
        self, observation_space: Space[Any], d_model: int, choose_on_teampreview: bool, n_columns: int=1
    ):
        """
        Initialize the attention-based feature extractor.

        Args:
            observation_space: Gymnasium observation space specification.
            d_model: Hidden size for token projection and transformer layers.
            choose_on_teampreview: Whether policy controls teampreview decisions.
        """
        super().__init__(observation_space, features_dim=d_model)
        self.d_model = d_model
        self.choose_on_teampreview = choose_on_teampreview

        # initialize columns
        assert n_columns > 0
        self.column_head = self.Column(d_model)
        for _ in range(n_columns - 1):
            self.add_column()

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract features from battle observation.

        Embeds Pokemon attributes and applies transformer attention across all
        12 Pokemon (6 per side).

        Args:
            x: Dict with an ``"observation"`` key containing a tensor of
                shape (batch, 12 * chunk_obs_len).

        Returns:
            Feature tensor of shape (batch, d_model).
        """
        return self.column_head(obs_dict)[0]

    def add_column(self):
        """
        Add a new column.
        """
        self.column_head.freeze()
        self.column_head = self.Column(self.d_model, self.column_head)


    class Column(nn.Module):
        """
        A single column of the ProgressiveAttentionExtractor.

        Class Attributes:
            embed_len: Dimension of embedding vectors for abilities/items/moves.
            num_heads: Number of attention heads in transformer layers.
            embed_layers: Number of transformer encoder layers.
            down_size: Size of the hidden layer in the lateral adapters.
        """

        embed_len: int = 32
        num_heads: int = 4
        embed_layers: int = 3
        down_size: int = 128

        def __init__(self, d_model: int, prev_column=None):
            """
            Initialize a new column.

            Args:
                d_model: Hidden size for token projection and transformer layers.
            """
            super().__init__()
            self.prev_column = prev_column

            # embeddings -> column-specific
            self.ability_embed = nn.Embedding(
                len(abilities), self.embed_len, max_norm=self.embed_len**0.5
            )
            nn.init.normal_(self.ability_embed.weight, std=0.02)
            self.item_embed = nn.Embedding(
                len(items), self.embed_len, max_norm=self.embed_len**0.5
            )
            nn.init.normal_(self.item_embed.weight, std=0.02)
            self.move_embed = nn.Embedding(
                len(moves), self.embed_len, max_norm=self.embed_len**0.5
            )
            nn.init.normal_(self.move_embed.weight, std=0.02)

            # linear layer -> transferable
            self.pokemon_proj = nn.Linear(chunk_obs_len + 6 * (self.embed_len - 1), d_model)
            if prev_column:
                self.input_adapter = nn.Sequential(
                    nn.Linear(d_model, self.down_size),
                    nn.ReLU(),
                    nn.Linear(self.down_size, d_model),
                )
                self.input_alpha = nn.Parameter(torch.full((d_model,), 0.05))

            # CLS token -> column-specific
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

            # transformer layers -> transferable
            self.transformer_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=self.num_heads,
                    dim_feedforward=d_model,
                    dropout=0,
                    batch_first=True,
                    norm_first=True,
                ) for _ in range(self.embed_layers)
            ])
            if prev_column:
                self.transformer_adapters = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(d_model, self.down_size),
                        nn.ReLU(),
                        nn.Linear(self.down_size, d_model),
                    ) for _ in range(self.embed_layers)
                ])
                self.transformer_alpha = [0 for _ in range(self.embed_layers)]
            else:
                self.final_norm = nn.LayerNorm(d_model) # not used, kept for backwards compatibility

        def forward(self, obs_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
            """
            Extract features from battle observation.

            Embeds Pokemon attributes and applies transformer attention across all
            12 Pokemon (6 per side).

            Args:
                x: Dict with an ``"observation"`` key containing a tensor of
                    shape (batch, 12 * chunk_obs_len).

            Returns:
                feature_tensor, pokemon_tokens, [transformer_layer1, transformer_layer2, ...]
            """
            obs = obs_dict["observation"]
            batch_size = obs.size(0)
            pokemon_obs = obs.view(batch_size, 12, -1)

            # embedded observation
            start = glob_obs_len + side_obs_len
            pokemon_obs = torch.cat(
                [
                    pokemon_obs[:, :, :start],
                    self.ability_embed(pokemon_obs[:, :, start].long()),
                    self.item_embed(pokemon_obs[:, :, start + 1].long()),
                    self.move_embed(pokemon_obs[:, :, start + 2].long()),
                    self.move_embed(pokemon_obs[:, :, start + 3].long()),
                    self.move_embed(pokemon_obs[:, :, start + 4].long()),
                    self.move_embed(pokemon_obs[:, :, start + 5].long()),
                    pokemon_obs[:, :, start + 6 :],
                ],
                dim=-1,
            )
            pokemon_tokens = self.pokemon_proj(pokemon_obs)

            # lateral connections to previous column(s)
            old_tokens, old_outputs = None, None
            if self.prev_column:
                with torch.no_grad():
                    _, old_tokens, old_outputs = self.prev_column(obs_dict)
                # transfer linear layer
                pokemon_tokens += self.input_alpha * self.input_adapter(old_tokens)

            # add CLS token
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, pokemon_tokens], dim=1)

            # apply transformer layers of current column
            transformer_outputs = []
            for i in range(self.embed_layers):
                x = self.transformer_layers[i](x)
                if old_outputs: # transfer one layer at a time
                    x += self.transformer_alpha[i] * self.transformer_adapters[i](old_outputs[i])
                transformer_outputs.append(torch.clone(x))

            # return feature tensor and vector for transfer
            return x[:, 0, :], pokemon_tokens, transformer_outputs

        def freeze(self):
            """
            Freeze the column, preventing any of its weights from updating.
            """
            # remove max_norm from embeddings
            self.ability_embed.max_norm = None
            self.item_embed.max_norm = None
            self.move_embed.max_norm = None

            # don't update gradients
            self.requires_grad_(False)
            self.eval()
