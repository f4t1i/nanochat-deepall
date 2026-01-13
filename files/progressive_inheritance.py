"""
Progressive Model Inheritance for Nanochat Miniseries
=======================================================

Implementation of elastic weight consolidation (EWC) with subspace constraints
for efficient scaling law exploration.

Formula:
    Φ(W_t, ΔW) = 1/2 Σᵢ Ωᵢᵢ(ΔWᵢ)² + γ·Tr((I-P_S)ΔW ΔWᵀ(I-P_S)ᵀ)

Author: Faton
Target: @karpathy's nanochat miniseries optimization
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class InheritanceConfig:
    """Configuration for progressive model inheritance."""
    lambda_ewc: float = 1.0          # Fisher information weight
    gamma_subspace: float = 0.1      # Subspace constraint weight
    fisher_samples: int = 1000       # Samples for Fisher estimation
    use_inheritance: bool = True
    
    # Validation experiment settings
    validate_first: bool = True
    validation_model_pair: Tuple[str, str] = ('d15', 'd16')


class FisherInformation:
    """Compute and store Fisher Information Matrix diagonal."""
    
    @staticmethod
    def compute_diagonal(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        n_samples: int = 1000,
        device: str = 'cuda'
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate diagonal Fisher Information on validation set.
        
        Args:
            model: Trained model from previous scale
            dataloader: Validation data
            n_samples: Number of samples to average over
            device: Computation device
            
        Returns:
            Dictionary mapping parameter names to Fisher diagonal estimates
        """
        model.eval()
        model.to(device)
        
        fisher = {
            name: torch.zeros_like(param, device=device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        
        n_batches = 0
        for batch_idx, batch in enumerate(dataloader):
            if n_batches >= n_samples:
                break
                
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            model.zero_grad()
            logits = model(input_ids)
            loss = F.cross_entropy(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2
            
            n_batches += 1
        
        # Normalize by number of samples
        for name in fisher:
            fisher[name] /= n_batches
            
        return fisher
    
    @staticmethod
    def save(fisher: Dict[str, torch.Tensor], path: str):
        """Save Fisher information to disk."""
        torch.save(fisher, path)
    
    @staticmethod
    def load(path: str, device: str = 'cuda') -> Dict[str, torch.Tensor]:
        """Load Fisher information from disk."""
        return torch.load(path, map_location=device)


class SubspaceProjection:
    """Compute subspace projection matrix from trained model."""
    
    @staticmethod
    def compute_projection(
        model: torch.nn.Module,
        rank: Optional[int] = None,
        variance_threshold: float = 0.95
    ) -> torch.Tensor:
        """
        Compute projection matrix P_S onto important subspace.
        
        Uses SVD on concatenated weight matrices to identify
        the principal subspace of learned representations.
        
        Args:
            model: Trained model from previous scale
            rank: Fixed rank for projection (optional)
            variance_threshold: Cumulative variance to retain
            
        Returns:
            Projection matrix P_S
        """
        # Collect all weight matrices
        weights = []
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                weights.append(param.data.flatten())
        
        # Concatenate into single matrix
        W = torch.cat(weights).unsqueeze(0)
        
        # Compute SVD
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        
        # Determine rank based on variance threshold
        if rank is None:
            cumsum = torch.cumsum(S ** 2, dim=0)
            total_variance = cumsum[-1]
            rank = torch.searchsorted(
                cumsum, 
                variance_threshold * total_variance
            ).item() + 1
        
        # Construct projection matrix
        U_reduced = U[:, :rank]
        P_S = U_reduced @ U_reduced.T
        
        return P_S, rank


class ModelUpscaler:
    """Upscale model from d(t-1) to d(t) with proper initialization."""
    
    @staticmethod
    def upsample_weights(
        small_model: torch.nn.Module,
        large_model: torch.nn.Module,
        interpolation: str = 'linear'
    ) -> Dict[str, torch.Tensor]:
        """
        Upsample weights from smaller to larger model.
        
        Strategy:
        - Copy shared dimensions directly
        - Initialize new dimensions with scaled random noise
        - Preserve attention patterns through interpolation
        
        Args:
            small_model: Trained smaller model (e.g., d15)
            large_model: Uninitialized larger model (e.g., d16)
            interpolation: Method for upsampling ('linear', 'nearest')
            
        Returns:
            State dict for large model with inherited weights
        """
        small_state = small_model.state_dict()
        large_state = large_model.state_dict()
        upsampled = {}
        
        for name, large_param in large_state.items():
            if name not in small_state:
                # New parameter - random init
                upsampled[name] = large_param
                continue
            
            small_param = small_state[name]
            
            if small_param.shape == large_param.shape:
                # Same shape - direct copy
                upsampled[name] = small_param.clone()
            else:
                # Different shape - interpolate/pad
                upsampled[name] = ModelUpscaler._interpolate_param(
                    small_param, large_param.shape, interpolation
                )
        
        return upsampled
    
    @staticmethod
    def _interpolate_param(
        small: torch.Tensor,
        target_shape: torch.Size,
        method: str = 'linear'
    ) -> torch.Tensor:
        """Interpolate parameter to target shape."""
        if method == 'linear':
            # Linear interpolation with padding
            large = torch.zeros(target_shape, dtype=small.dtype, device=small.device)
            
            # Copy existing dimensions
            slices = tuple(slice(0, min(s, t)) for s, t in zip(small.shape, target_shape))
            large[slices] = small[slices]
            
            # Initialize new dimensions with small random noise
            for i, (s_dim, t_dim) in enumerate(zip(small.shape, target_shape)):
                if t_dim > s_dim:
                    new_slice = list(slices)
                    new_slice[i] = slice(s_dim, t_dim)
                    new_slice = tuple(new_slice)
                    large[new_slice] = torch.randn_like(large[new_slice]) * 0.02
            
            return large
        else:
            raise ValueError(f"Unknown interpolation method: {method}")


class ProgressiveTrainer:
    """Train model with progressive inheritance regularization."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: InheritanceConfig,
        prev_weights: Optional[Dict[str, torch.Tensor]] = None,
        fisher: Optional[Dict[str, torch.Tensor]] = None,
        P_S: Optional[torch.Tensor] = None
    ):
        self.model = model
        self.config = config
        self.prev_weights = prev_weights
        self.fisher = fisher
        self.P_S = P_S
        
        # Precompute I - P_S for efficiency
        if P_S is not None:
            self.I_minus_P_S = torch.eye(P_S.shape[0], device=P_S.device) - P_S
        else:
            self.I_minus_P_S = None
    
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute regularized training loss.
        
        Args:
            batch: Input batch with 'input_ids' and 'labels'
            return_components: Whether to return loss components
            
        Returns:
            Total loss and optional component breakdown
        """
        # Standard cross-entropy loss
        logits = self.model(batch['input_ids'])
        ce_loss = F.cross_entropy(logits, batch['labels'])
        
        if not self.config.use_inheritance or self.prev_weights is None:
            return ce_loss, {'ce': ce_loss.item()}
        
        # EWC regularization term
        ewc_loss = self._compute_ewc_loss()
        
        # Subspace constraint term
        subspace_loss = self._compute_subspace_loss()
        
        # Combine losses
        total_loss = (
            ce_loss + 
            self.config.lambda_ewc * ewc_loss + 
            self.config.gamma_subspace * subspace_loss
        )
        
        if return_components:
            components = {
                'ce': ce_loss.item(),
                'ewc': ewc_loss.item(),
                'subspace': subspace_loss.item(),
                'total': total_loss.item()
            }
            return total_loss, components
        
        return total_loss, {'total': total_loss.item()}
    
    def _compute_ewc_loss(self) -> torch.Tensor:
        """Compute Fisher-weighted parameter drift penalty."""
        ewc = 0.0
        
        for name, param in self.model.named_parameters():
            if name in self.prev_weights and name in self.fisher:
                delta = param - self.prev_weights[name]
                ewc += (self.fisher[name] * delta ** 2).sum()
        
        return 0.5 * ewc
    
    def _compute_subspace_loss(self) -> torch.Tensor:
        """Compute subspace constraint on new parameters."""
        if self.P_S is None or self.I_minus_P_S is None:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        subspace = 0.0
        
        for name, param in self.model.named_parameters():
            if name not in self.prev_weights:
                # New parameter - apply full constraint
                delta = param.flatten()
            else:
                prev_shape = self.prev_weights[name].shape
                if param.shape != prev_shape:
                    # Expanded parameter - constrain new dimensions only
                    delta = param.flatten()[prev_shape.numel():]
                else:
                    continue
            
            # Project onto nullspace
            if delta.numel() > 0:
                projected = torch.matmul(self.I_minus_P_S, delta)
                subspace += (projected ** 2).sum()
        
        return subspace


def train_miniseries_with_inheritance(
    base_size: str = 'd10',
    target_size: str = 'd18',
    config: InheritanceConfig = None
):
    """
    Main training pipeline for progressive inheritance.
    
    Example usage:
        config = InheritanceConfig(
            lambda_ewc=1.0,
            gamma_subspace=0.1,
            fisher_samples=1000
        )
        
        models = train_miniseries_with_inheritance(
            base_size='d10',
            target_size='d18',
            config=config
        )
    """
    if config is None:
        config = InheritanceConfig()
    
    # Extract size indices
    base_idx = int(base_size[1:])
    target_idx = int(target_size[1:])
    
    models = {}
    fishers = {}
    projections = {}
    
    print(f"\n{'='*60}")
    print(f"Progressive Inheritance Training: {base_size} → {target_size}")
    print(f"{'='*60}\n")
    
    # Train base model normally
    print(f"[1/{target_idx-base_idx+1}] Training {base_size} from scratch...")
    models[base_size] = train_base_model(size=base_size)
    
    # Compute Fisher and projection for base model
    fishers[base_size] = FisherInformation.compute_diagonal(
        models[base_size],
        get_validation_dataloader(),
        n_samples=config.fisher_samples
    )
    projections[base_size], rank = SubspaceProjection.compute_projection(
        models[base_size]
    )
    print(f"  ✓ Fisher computed, subspace rank: {rank}")
    
    # Progressive upscaling
    for idx in range(base_idx + 1, target_idx + 1):
        prev_size = f'd{idx-1}'
        curr_size = f'd{idx}'
        
        print(f"\n[{idx-base_idx+1}/{target_idx-base_idx+1}] Training {curr_size} from {prev_size}...")
        
        # Create and upsample model
        large_model = create_model(size=curr_size)
        upsampled_weights = ModelUpscaler.upsample_weights(
            models[prev_size],
            large_model
        )
        large_model.load_state_dict(upsampled_weights, strict=False)
        
        # Create progressive trainer
        trainer = ProgressiveTrainer(
            model=large_model,
            config=config,
            prev_weights=models[prev_size].state_dict(),
            fisher=fishers[prev_size],
            P_S=projections[prev_size]
        )
        
        # Train with regularization
        trained_model = train_with_regularization(
            trainer,
            size=curr_size,
            config=config
        )
        
        models[curr_size] = trained_model
        
        # Compute Fisher and projection for next iteration
        fishers[curr_size] = FisherInformation.compute_diagonal(
            trained_model,
            get_validation_dataloader(),
            n_samples=config.fisher_samples
        )
        projections[curr_size], rank = SubspaceProjection.compute_projection(
            trained_model
        )
        print(f"  ✓ Completed, subspace rank: {rank}")
    
    print(f"\n{'='*60}")
    print("Progressive inheritance training complete!")
    print(f"{'='*60}\n")
    
    return models


# Placeholder functions (to be integrated with nanochat codebase)
def create_model(size: str):
    """Create nanochat model of specified size."""
    raise NotImplementedError("Integrate with nanochat model factory")

def train_base_model(size: str):
    """Train base model from scratch."""
    raise NotImplementedError("Integrate with nanochat training loop")

def get_validation_dataloader():
    """Get validation data loader."""
    raise NotImplementedError("Integrate with nanochat data pipeline")

def train_with_regularization(trainer, size: str, config: InheritanceConfig):
    """Train model with progressive regularization."""
    raise NotImplementedError("Integrate with nanochat training loop")


if __name__ == "__main__":
    # Quick validation experiment: d15 → d16
    print("Running validation experiment: d15 → d16")
    print("Expected cost: ~$20, time: ~2 hours on 8×H100")
    
    config = InheritanceConfig(
        lambda_ewc=1.0,
        gamma_subspace=0.1,
        fisher_samples=1000,
        validate_first=True,
        validation_model_pair=('d15', 'd16')
    )
    
    # This would integrate with nanochat's existing infrastructure
    # models = train_miniseries_with_inheritance('d15', 'd16', config)
