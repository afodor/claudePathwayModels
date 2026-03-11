"""
dCGF (data-driven Community Genotype Function) model implementations.

Based on: "A data-driven modeling framework for mapping genotypes to
synthetic microbial community functions" by Qian et al. (2025)

Implements two model variants:
  - dCGF-ES (Enzyme Soup): sums GF columns, feeds through NN
  - dCGF-IS (Individual Species): per-species encoder with shared weights (Deep Set)
"""
import torch
import torch.nn as nn


class dCGF_ES(nn.Module):
    """
    dCGF Enzyme Soup model.

    Assumes community function depends on the total abundance of each genetic
    feature in the community (summing across species). A simple "enzyme soup"
    assumption used in some GEMs-FBA models.

    Architecture:
      1. Sum all columns of GF matrix u (weighted by abundance) -> w vector
      2. Encoder: w -> z (community embedding) via fully-connected NN
      3. Decoder: z -> y (community function) via single-hidden-layer NN
    """
    def __init__(self, n_gf, z_dim=30, encoder_hidden=200, decoder_hidden=100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_gf, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, z_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, 1),
        )

    def forward(self, u_matrices):
        """
        Args:
            u_matrices: list of GF matrices, each of shape (n_gf, n_species_i)
                        where columns are GF vectors scaled by initial abundance.
        Returns:
            predictions: tensor of shape (batch_size, 1)
        """
        predictions = []
        for u in u_matrices:
            # Sum across species (columns) to get community GF vector
            w = u.sum(dim=1)  # (n_gf,)
            z = self.encoder(w)
            y = self.decoder(z)
            predictions.append(y)
        return torch.stack(predictions).squeeze(-1)


class dCGF_IS(nn.Module):
    """
    dCGF Individual Species model.

    Uses a biologically-inspired structure where each species' GF vector is
    independently processed by a shared encoder (universal function alpha),
    accounting for intra-cellular regulation and inter-species interactions.

    Architecture (Deep Set):
      1. For each species i: z_i = alpha(u_i, w) where w = sum of all u_i
         alpha is a shared NN applied to each species independently
      2. Community embedding: z = sum(z_i)
      3. Decoder: z -> y via single-hidden-layer NN

    The encoder alpha takes the concatenation [u_i; w] as input, where:
      - u_i is species i's GF vector scaled by its abundance
      - w is the community-level sum of all GF vectors (context)
    """
    def __init__(self, n_gf, z_dim=30, encoder_hidden=200, decoder_hidden=100):
        super().__init__()
        # Encoder takes [u_i; w] concatenated (2 * n_gf dimensional input)
        self.encoder = nn.Sequential(
            nn.Linear(2 * n_gf, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, z_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, 1),
        )

    def forward(self, u_matrices):
        """
        Args:
            u_matrices: list of GF matrices, each of shape (n_gf, n_species_i)
        Returns:
            predictions: tensor of shape (batch_size, 1)
        """
        predictions = []
        for u in u_matrices:
            n_gf, n_species = u.shape
            # Community-level GF sum (context vector)
            w = u.sum(dim=1)  # (n_gf,)

            # Apply shared encoder to each species
            species_embeddings = []
            for j in range(n_species):
                u_j = u[:, j]  # (n_gf,)
                # Concatenate species GF vector with community context
                encoder_input = torch.cat([u_j, w], dim=0)  # (2*n_gf,)
                z_j = self.encoder(encoder_input)  # (z_dim,)
                species_embeddings.append(z_j)

            # Community embedding = sum of species embeddings
            z = torch.stack(species_embeddings).sum(dim=0)  # (z_dim,)
            y = self.decoder(z)
            predictions.append(y)
        return torch.stack(predictions).squeeze(-1)


class dCGF_IS_Batched(nn.Module):
    """
    Optimized version of dCGF-IS that handles variable-size communities
    more efficiently by padding and masking.
    """
    def __init__(self, n_gf, z_dim=30, encoder_hidden=200, decoder_hidden=100):
        super().__init__()
        self.n_gf = n_gf
        self.encoder = nn.Sequential(
            nn.Linear(2 * n_gf, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, z_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, 1),
        )

    def forward(self, u_matrices):
        """Process a batch of variable-size GF matrices."""
        predictions = []
        for u in u_matrices:
            n_gf, n_species = u.shape
            w = u.sum(dim=1)  # (n_gf,)
            # Build all encoder inputs at once
            w_expanded = w.unsqueeze(0).expand(n_species, -1)  # (n_species, n_gf)
            u_t = u.t()  # (n_species, n_gf)
            encoder_inputs = torch.cat([u_t, w_expanded], dim=1)  # (n_species, 2*n_gf)
            z_all = self.encoder(encoder_inputs)  # (n_species, z_dim)
            z = z_all.sum(dim=0)  # (z_dim,)
            y = self.decoder(z)
            predictions.append(y)
        return torch.stack(predictions).squeeze(-1)
