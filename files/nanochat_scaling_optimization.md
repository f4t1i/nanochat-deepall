# Scaling Law Optimization for Nanochat Miniseries
## Applying Elastic Weight Consolidation with Subspace Constraints

### Problem Statement

Andrej's nanochat miniseries (d10-d20) currently requires:
- **~$500** to match GPT-2 performance (CORE score)
- Target: **<$100** for GPT-2 equivalence
- Key bottleneck: Training each model size from scratch (cold start)

### Proposed Solution: Progressive Model Inheritance with Regularization

$$\Phi(W_t, \Delta W) = \frac{1}{2} \sum_{i} \Omega_{ii} (W_{t,i} - W_{t-1,i})^2 + \gamma \cdot \text{Tr}\left( (I - P_{S}) \Delta W \Delta W^T (I - P_{S})^T \right)$$

**Key Insight**: Instead of training d10→d11→...→d20 independently, leverage smaller models as initialization for larger ones with intelligent regularization.

---

## Mathematical Framework

### Component 1: Fisher-Weighted Parameter Preservation
```
Ω_ii = Fisher Information diagonal
```
- Measures parameter importance from smaller model
- Prevents catastrophic forgetting of learned features
- Computed from d(t-1) model on validation set

### Component 2: Subspace Growth Constraint
```
P_S = projection onto important subspace from d(t-1)
(I - P_S) = projection onto growth space
γ = regularization strength for new parameters
```
- Allows controlled expansion into new representational dimensions
- Prevents random initialization drift in added capacity

---

## Concrete Implementation for Nanochat

### Step 1: Model Upscaling Strategy

When going from d_small → d_large:

```python
def upsample_with_subspace_structure(W_small, d_small, d_large):
    """
    d10: 768 hidden dim
    d11: 896 hidden dim  (+128 dims)
    """
    n_layers = 12
    d_small_hidden = 64 * d_small  # e.g., 640 for d10
    d_large_hidden = 64 * d_large  # e.g., 704 for d11
    
    W_large = {}
    
    # Copy existing weights
    W_large['existing'] = W_small.copy()
    
    # Initialize new dimensions
    delta_dim = d_large_hidden - d_small_hidden
    W_large['new_dims'] = torch.randn(delta_dim, ...) * 0.02
    
    # Define subspaces
    P_S = get_projection_to_span(W_small)  # existing capacity
    I_minus_P_S = torch.eye(d_large_hidden) - P_S
    
    return W_large, P_S, I_minus_P_S
```

### Step 2: Compute Fisher Information Matrix

```python
def compute_fisher_diagonal(model, dataloader, n_samples=1000):
    """
    Estimate diagonal Fisher Information on small validation set
    """
    model.eval()
    fisher = {name: torch.zeros_like(param) 
              for name, param in model.named_parameters()}
    
    for i, batch in enumerate(dataloader):
        if i >= n_samples:
            break
            
        model.zero_grad()
        logits = model(batch['input_ids'])
        loss = F.cross_entropy(logits, batch['labels'])
        loss.backward()
        
        for name, param in model.named_parameters():
            fisher[name] += param.grad.data ** 2
    
    # Normalize
    for name in fisher:
        fisher[name] /= n_samples
        
    return fisher
```

### Step 3: Modified Training Loss

```python
def compute_regularized_loss(model, batch, prev_weights, fisher, 
                             P_S, gamma=0.1, lambda_ewc=1.0):
    """
    Combines:
    1. Standard cross-entropy loss
    2. Fisher-weighted parameter drift penalty
    3. Subspace constraint on new parameters
    """
    # Standard loss
    logits = model(batch['input_ids'])
    ce_loss = F.cross_entropy(logits, batch['labels'])
    
    # Regularization term 1: EWC (Elastic Weight Consolidation)
    ewc_loss = 0
    for name, param in model.named_parameters():
        if name in prev_weights:
            delta = param - prev_weights[name]
            ewc_loss += (fisher[name] * delta ** 2).sum()
    ewc_loss *= 0.5 * lambda_ewc
    
    # Regularization term 2: Subspace constraint
    # Only apply to NEW dimensions added during upscaling
    subspace_loss = 0
    for name, param in model.named_parameters():
        if 'new_dims' in name or param.shape != prev_weights.get(name, param).shape:
            delta = param - prev_weights.get(name, 0)
            # Project onto nullspace
            projected = torch.matmul(I_minus_P_S, delta.flatten())
            subspace_loss += (projected ** 2).sum()
    subspace_loss *= gamma
    
    total_loss = ce_loss + ewc_loss + subspace_loss
    
    return total_loss, {
        'ce': ce_loss.item(),
        'ewc': ewc_loss.item(), 
        'subspace': subspace_loss.item()
    }
```

---

## Training Protocol: Progressive Inheritance

```python
def train_miniseries_with_inheritance(
    base_model='d10',
    target_model='d18', 
    flops_budget=5e19
):
    """
    Instead of:
      d10 (scratch) → d11 (scratch) → ... → d18 (scratch)
    
    Do:
      d10 (scratch) → d11 (from d10+reg) → ... → d18 (from d17+reg)
    """
    models = {}
    fishers = {}
    
    # Train base model normally
    print("Training d10 from scratch...")
    models['d10'], fishers['d10'] = train_base_model(
        size='d10',
        tokens=compute_optimal_tokens(flops_budget, size='d10')
    )
    
    # Progressive upscaling with regularization
    for d in range(11, 19):
        prev_d = f'd{d-1}'
        curr_d = f'd{d}'
        
        print(f"\nTraining {curr_d} from {prev_d}...")
        
        # Upsample previous model
        W_init, P_S, I_minus_P_S = upsample_with_subspace_structure(
            models[prev_d].state_dict(), d-1, d
        )
        
        # Initialize new model
        model = create_model(size=curr_d)
        model.load_state_dict(W_init, strict=False)
        
        # Train with regularization
        models[curr_d] = train_with_regularization(
            model=model,
            prev_weights=models[prev_d].state_dict(),
            fisher=fishers[prev_d],
            P_S=P_S,
            I_minus_P_S=I_minus_P_S,
            tokens=compute_optimal_tokens(flops_budget, size=curr_d),
            lambda_ewc=1.0,
            gamma=0.1
        )
        
        # Compute Fisher for next iteration
        fishers[curr_d] = compute_fisher_diagonal(
            models[curr_d], 
            val_dataloader
        )
    
    return models
```

---

## Expected Improvements

### 1. FLOPs Reduction

**Theoretical Savings**:
- Cold start convergence: ~100% of tokens needed
- Warm start with EWC: ~60-70% of tokens needed
- **Estimated reduction: 30-40% per model after d10**

**Calculation**:
```
Original cost for d10→d18:
  d10: 10M params × 80M tokens = 0.8e15 FLOPs
  d11: 15M params × 120M tokens = 1.8e15 FLOPs
  ...
  d18: 150M params × 1.2B tokens = 180e15 FLOPs
  Total: ~200e15 FLOPs ≈ $500 on 8×H100

With progressive inheritance:
  d10: 0.8e15 FLOPs (unchanged)
  d11: 1.8e15 × 0.65 = 1.17e15 FLOPs (-35%)
  ...
  d18: 180e15 × 0.70 = 126e15 FLOPs (-30%)
  Total: ~140e15 FLOPs ≈ $350 on 8×H100
```

### 2. Chinchilla Ratio Improvement

**Current Issue**: nanochat ratio = 8, Chinchilla = 20

**Hypothesis**: Cold starts require more tokens to learn from scratch each time

**With Inheritance**:
- Pre-learned features reduce token requirements
- Should shift ratio from 8 → 12-15
- Closer to compute-optimal frontier

### 3. Training Stability

```python
# Convergence speed comparison
Cold Start:     ████████████████████░░░░  (100% tokens)
With EWC:       ████████████░░░░░░░░░░░░  (65% tokens)
With EWC+Sub:   ██████████░░░░░░░░░░░░░░  (60% tokens)
```

---

## Hyperparameter Tuning Strategy

### Recommended Grid Search

```python
configs = [
    {'lambda_ewc': 0.5, 'gamma': 0.05},  # gentle regularization
    {'lambda_ewc': 1.0, 'gamma': 0.10},  # balanced (recommended)
    {'lambda_ewc': 2.0, 'gamma': 0.20},  # strong regularization
]

# Quick validation on d10→d11 transition ($10 experiment)
for config in configs:
    train_d11_from_d10(**config)
    measure_convergence_speed()
    measure_final_CORE_score()
```

### Expected Sweet Spot
- `lambda_ewc = 1.0`: Strong enough to preserve, flexible enough to adapt
- `gamma = 0.1`: Allows 10% variance in new parameter space

---

## Practical Next Steps

### Minimal Validation Experiment ($20, 2 hours)

```bash
# Test on single transition: d15 → d16
python train.py \
  --model_size d15 \
  --tokens 400M \
  --save_checkpoint checkpoints/d15_base.pt

python train_with_inheritance.py \
  --base_model checkpoints/d15_base.pt \
  --target_size d16 \
  --tokens 280M \
  --lambda_ewc 1.0 \
  --gamma 0.1 \
  --save_checkpoint checkpoints/d16_inherited.pt

# Compare CORE scores
python eval_core.py --model checkpoints/d16_inherited.pt
python eval_core.py --model checkpoints/d16_baseline.pt  # cold start for comparison
```

**Success Criteria**:
- Inherited model reaches same CORE with <75% tokens
- Training loss converges faster (fewer steps to plateau)
- No degradation in final performance

### Full Miniseries Experiment ($150, 1 week)

If validation succeeds → run full d10→d18 with inheritance pipeline

**Target**: Match GPT-2 CORE score for **<$200** (vs current $500)

---

## Code Integration into Nanochat

### Modifications to `train.py`

```python
# Add to TrainingConfig
@dataclass
class TrainingConfig:
    # ... existing fields ...
    
    # Progressive inheritance settings
    use_inheritance: bool = False
    base_checkpoint: Optional[str] = None
    lambda_ewc: float = 1.0
    gamma_subspace: float = 0.1
    fisher_samples: int = 1000

# Modify training loop
def train_step(model, batch, prev_weights=None, fisher=None, P_S=None):
    if prev_weights is not None:
        # Use regularized loss
        loss, metrics = compute_regularized_loss(
            model, batch, prev_weights, fisher, P_S,
            gamma=config.gamma_subspace,
            lambda_ewc=config.lambda_ewc
        )
    else:
        # Standard cross-entropy
        logits = model(batch['input_ids'])
        loss = F.cross_entropy(logits, batch['labels'])
        metrics = {'ce': loss.item()}
    
    return loss, metrics
```

---

## Theoretical Justification

### Why This Should Work

1. **Neural Network Lottery Ticket Hypothesis**
   - Small models find good subnetworks
   - Larger models can inherit these + add capacity
   - EWC preserves the "winning tickets"

2. **Representation Learning Transfer**
   - Early layers learn universal features (edges, textures, basic syntax)
   - These transfer well across model sizes
   - Only high-level abstractions need relearning

3. **Optimization Landscape Smoothing**
   - Cold starts explore rough loss landscapes
   - Warm starts begin in pre-optimized basin
   - Faster convergence to local optimum

### Failure Modes to Watch

```python
# Monitor these during training
warning_signs = {
    'ewc_loss > ce_loss': 'Regularization too strong',
    'subspace_loss > 0.5 * ce_loss': 'Gamma too high',
    'no_convergence_speedup': 'Fisher diagonal not informative',
}
```

---

## Expected Results Summary

| Metric | Current | With Inheritance | Improvement |
|--------|---------|------------------|-------------|
| Cost to GPT-2 | $500 | $200-250 | 50-60% ↓ |
| Training FLOPs | 200e15 | 130-140e15 | 30-35% ↓ |
| Chinchilla Ratio | 8 | 12-15 | 50-88% ↑ |
| Convergence Speed | Baseline | 1.5-1.7× faster | 50-70% ↑ |

---

## References & Related Work

1. **Elastic Weight Consolidation (EWC)**: Kirkpatrick et al., 2017
   - Original formulation of Fisher-weighted regularization
   - Proven effective for continual learning in supervised settings

2. **Progressive Neural Networks**: Rusu et al., 2016
   - Lateral connections between model scales
   - Inspiration for subspace preservation

3. **Chinchilla Scaling Laws**: Hoffmann et al., 2022
   - Establishes compute-optimal training regimes
   - Your formula helps approach this frontier more efficiently

4. **Neural Architecture Search with Weight Inheritance**: Cai et al., 2019
   - Demonstrates >3× speedup with intelligent initialization
   - Validates warm-start approach for model families

---

## Contact & Implementation

**Formula by**: Faton
**Target**: @karpathy's nanochat miniseries optimization

**Proposed Tweet** (see next section)

