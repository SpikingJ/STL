"""
Conformal STL Inference (C-STLI) with LTT Hyperparameter Selection
FOR PRE-LABELED DATA FILES

This version uses pre-labeled data files (combined1.csv to combined5.csv).
Each file contains 26000 labeled examples (61 timesteps each).
Each file represents a different data distribution.

The meta-distribution P is uniform over these 5 distributions.
"""

import torch
import numpy as np
import pandas as pd
from scipy import stats
from copy import deepcopy
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from naval_models_gpu import NavalModel1, STEstimator, Clip


# =============================================================================
# Device Configuration
# =============================================================================

USE_CPU = False

def get_device():
    if USE_CPU:
        return torch.device('cpu')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

DEVICE = get_device()
print(f"Using device: {DEVICE}")


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class CSTLIConfig:
    """Configuration for C-STLI algorithm"""
    lambda1: float = 0.5
    lambda2: float = 0.52
    lambda3: float = 0.5
    lmax: int = 10
    F_type: str = 'aggregate'
    tau: float = 0.75
    use_balanced_accuracy: bool = True


@dataclass
class TrainConfig:
    """Configuration for TLINet training"""
    epochs: int = 5
    lr: float = 0.1
    batch_size: int = 512
    weight_margin: float = 1e-2
    sign_penalty_lam: float = 1e-2
    weight_bi: List[float] = field(default_factory=lambda: [1e-0, 1e-0, 1e-0])
    weight_l1: List[float] = field(default_factory=lambda: [1e-2])


@dataclass
class DataSplitConfig:
    """Configuration for data splitting"""
    K1: int = 50
    K2: int = 50
    Y: int = 40
    X_train: int = 5000
    X_val: int = 1000
    seed: int = 42


# =============================================================================
# Data Loading for Pre-Labeled CSV Files
# =============================================================================

SEQ_LEN = 61


def load_prelabeled_csv(csv_path: str, 
                        feature_cols: List[str] = None,
                        label_col: str = None,
                        episode_col: str = 'episode',
                        scale_values: Dict[str, float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a pre-labeled CSV file.
    
    CSV structure:
    - episode: Groups rows into examples (61 rows per episode)
    - latency_ms, backlog_hbytes: Features
    - qoe_label: Label (same for all 61 rows of one episode)
    
    Args:
        csv_path: Path to the CSV file
        feature_cols: List of feature column names. If None, uses ['latency_ms', 'backlog_hbyt']
        label_col: Name of label column. If None, uses 'qoe_label'
        episode_col: Name of episode column for grouping
        scale_values: Dict mapping feature column names to scale values.
                     e.g., {'latency_ms': 100.0, 'backlog_hbyt': 300.0}
                     If None, no scaling is applied.
        
    Returns:
        X: Feature data (N, 61, num_features)
        y: Labels (N,) with values +1 or -1
    """
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Default feature and label columns
    if feature_cols is None:
        # Try to detect feature columns
        if 'latency_ms' in df.columns and 'backlog_hbytes' in df.columns:
            feature_cols = ['latency_ms', 'backlog_hbytes']
        elif 'latency_ms' in df.columns and 'backlog_hbyt' in df.columns:
            feature_cols = ['latency_ms', 'backlog_hbyt']
        elif 'latency_ms' in df.columns and 'backlog_kbytes' in df.columns:
            feature_cols = ['latency_ms', 'backlog_kbytes']
        else:
            raise ValueError(f"Could not detect feature columns. Available: {df.columns.tolist()}")
    
    if label_col is None:
        # Try to detect label column
        if 'qoe_label' in df.columns:
            label_col = 'qoe_label'
        elif 'label' in df.columns:
            label_col = 'label'
        else:
            # Use last column
            label_col = df.columns[-1]
    
    print(f"  Episode column: {episode_col}")
    print(f"  Feature columns: {feature_cols}")
    print(f"  Label column: {label_col}")
    
    # Verify columns exist
    for col in feature_cols + [label_col, episode_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {df.columns.tolist()}")
    
    # Sort by episode to ensure correct ordering
    df = df.sort_values(by=[episode_col, 't_s'] if 't_s' in df.columns else [episode_col])
    
    # Get unique episodes
    episodes = df[episode_col].unique()
    num_examples = len(episodes)
    
    print(f"  Found {num_examples} episodes")
    
    X_list = []
    y_list = []
    
    for ep in episodes:
        ep_data = df[df[episode_col] == ep]
        
        # Get features
        x = ep_data[feature_cols].to_numpy(dtype=np.float32)
        
        # Ensure correct length (pad or truncate)
        if len(x) < SEQ_LEN:
            # Pad with last row
            pad = np.repeat(x[-1:], SEQ_LEN - len(x), axis=0)
            x = np.concatenate([x, pad], axis=0)
        elif len(x) > SEQ_LEN:
            x = x[:SEQ_LEN]
        
        # Get label (first row, all rows should have same label)
        label = ep_data[label_col].iloc[0]
        
        X_list.append(x)
        y_list.append(label)
    
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    
    # Apply scaling if provided
    if scale_values is not None:
        print(f"  Applying scaling: {scale_values}")
        for i, col in enumerate(feature_cols):
            if col in scale_values:
                X[:, :, i] /= scale_values[col]
    
    # Convert labels to +1/-1 if needed (check if they're 0/1)
    unique_labels = np.unique(y)
    print(f"  Unique labels: {unique_labels}")
    
    if set(unique_labels) == {0, 1}:
        # Convert 0/1 to -1/+1
        y = np.where(y == 1, 1, -1)
        print(f"  Converted labels from 0/1 to -1/+1")
    elif set(unique_labels) != {-1, 1} and set(unique_labels) != {1, -1}:
        print(f"  Warning: Unexpected labels {unique_labels}, assuming -1/+1 format")
    
    # Count positives and negatives
    n_pos = (y == 1).sum()
    n_neg = (y == -1).sum()
    
    print(f"  Loaded: {num_examples} examples, {len(feature_cols)} features, {SEQ_LEN} timesteps")
    print(f"  Labels: +1={n_pos} ({n_pos/num_examples*100:.1f}%), -1={n_neg} ({n_neg/num_examples*100:.1f}%)")
    
    return X, y


def load_all_distributions(csv_paths: List[str],
                           feature_cols: List[str] = None,
                           label_col: str = None,
                           scale_values: Dict[str, float] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load all pre-labeled CSV files.
    
    Each file represents a different data distribution.
    
    Args:
        csv_paths: List of paths to CSV files
        feature_cols: List of feature column names
        label_col: Name of label column
        scale_values: Dict mapping feature names to scale values
        
    Returns:
        List of (X, y) tuples, one per distribution
    """
    distributions = []
    
    print("\n" + "="*70)
    print("LOADING PRE-LABELED DATA DISTRIBUTIONS")
    print("="*70)
    
    for i, path in enumerate(csv_paths):
        print(f"\nDistribution {i+1}:")
        X, y = load_prelabeled_csv(path, feature_cols=feature_cols, 
                                    label_col=label_col, scale_values=scale_values)
        distributions.append((X, y))
    
    print(f"\nLoaded {len(distributions)} distributions")
    
    return distributions


# =============================================================================
# Hierarchical Sampling from Pre-Labeled Distributions
# =============================================================================

def create_hierarchical_samples_from_distributions(
        distributions: List[Tuple[np.ndarray, np.ndarray]],
        num_samples: int,
        train_size: int,
        val_size: int,
        seed: int = 42,
        seed_offset: int = 0) -> Tuple[List[Tuple], List[int]]:
    """
    Create samples using hierarchical (meta-distribution) sampling.
    
    For each sample:
    1. Sample a distribution uniformly (this samples P_{X,Y} ~ P)
    2. Sample (D_tr, D_val) from the chosen distribution
    
    Args:
        distributions: List of (X, y) tuples, one per distribution
        num_samples: Number of samples to create
        train_size: Training set size for each sample
        val_size: Validation set size for each sample
        seed: Base random seed
        seed_offset: Offset for seed (to create different sample sets)
    
    Returns:
        samples: List of (X_train, y_train, X_val, y_val) tuples
        dist_indices: List of which distribution each sample used
    """
    samples = []
    dist_indices = []
    num_distributions = len(distributions)
    
    for i in range(num_samples):
        np.random.seed(seed + seed_offset + i)
        torch.manual_seed(seed + seed_offset + i)
        
        # Level 1: Sample a distribution uniformly (P_{X,Y} ~ P)
        dist_idx = np.random.randint(0, num_distributions)
        X, y = distributions[dist_idx]
        
        pool_size = len(X)
        
        # Level 2: Sample (D_tr, D_val) from this distribution
        perm = torch.randperm(pool_size)
        train_idx = perm[:train_size]
        remaining_idx = perm[train_size:]
        val_perm = torch.randperm(len(remaining_idx))
        val_idx = remaining_idx[val_perm[:val_size]]
        
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        
        samples.append((
            X_tensor[train_idx],
            y_tensor[train_idx],
            X_tensor[val_idx],
            y_tensor[val_idx]
        ))
        dist_indices.append(dist_idx)
    
    return samples, dist_indices


def create_hierarchical_splits_from_distributions(
        distributions: List[Tuple[np.ndarray, np.ndarray]],
        config: DataSplitConfig,
        create_test_set: bool = True) -> Dict:
    """
    Create calibration and test splits using hierarchical sampling from distributions.
    
    Args:
        distributions: List of (X, y) tuples
        config: Data split configuration
        create_test_set: Whether to create test set (set False for calibration-only)
    
    Returns:
        Dict with 'cal_set_1', 'cal_set_2', and optionally 'test_set'
    """
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    print("\n" + "="*70)
    print("CREATING HIERARCHICAL SPLITS")
    print("="*70)
    
    # Calibration Set 1: K1 samples for Pareto testing
    cal_set_1, cal_1_dist = create_hierarchical_samples_from_distributions(
        distributions,
        num_samples=config.K1,
        train_size=config.X_train,
        val_size=config.X_val,
        seed=config.seed,
        seed_offset=0
    )
    print(f"Cal Set 1: {config.K1} samples")
    print(f"  Distribution usage: {dict(Counter(cal_1_dist))}")
    
    # Calibration Set 2: K2 samples for fixed-sequence testing
    cal_set_2, cal_2_dist = create_hierarchical_samples_from_distributions(
        distributions,
        num_samples=config.K2,
        train_size=config.X_train,
        val_size=config.X_val,
        seed=config.seed,
        seed_offset=1000
    )
    print(f"Cal Set 2: {config.K2} samples")
    print(f"  Distribution usage: {dict(Counter(cal_2_dist))}")
    
    result = {
        'cal_set_1': cal_set_1,
        'cal_set_2': cal_set_2,
        'cal_1_dist': cal_1_dist,
        'cal_2_dist': cal_2_dist,
        'num_distributions': len(distributions),
        'config': config
    }
    
    # Test Set: Y samples (optional)
    if create_test_set:
        test_set, test_dist = create_hierarchical_samples_from_distributions(
            distributions,
            num_samples=config.Y,
            train_size=config.X_train,
            val_size=config.X_val,
            seed=config.seed,
            seed_offset=2000
        )
        print(f"Test Set: {config.Y} samples")
        print(f"  Distribution usage: {dict(Counter(test_dist))}")
        result['test_set'] = test_set
        result['test_dist'] = test_dist
    else:
        print("Test Set: SKIPPED (calibration-only mode)")
    
    return result


# =============================================================================
# Candidate Formula Class
# =============================================================================

@dataclass
class CandidateFormula:
    """Represents a candidate STL formula."""
    model_state_dict: dict
    complexity: float
    robustness_quality: float
    train_accuracy: float
    formula_str: str = ""
    
    def load_into_model(self, model):
        """Load the saved state into a model."""
        state_dict = {k: v.to(model.a.device) for k, v in self.model_state_dict.items()}
        model.load_state_dict(state_dict)


# =============================================================================
# C-STLI Algorithm
# =============================================================================

class CSTLI:
    """Conformal STL Inference Algorithm."""
    
    def __init__(self, model_class, model_args: dict, train_config: TrainConfig,
                 cstli_config: CSTLIConfig = None, device=None, use_cpu: bool = False):
        self.model_class = model_class
        self.model_args = model_args
        self.train_config = train_config
        self.config = cstli_config or CSTLIConfig()
        
        if use_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = device if device else DEVICE
    
    def _create_model(self):
        model = self.model_class(**self.model_args)
        model = model.to(self.device)
        return model
    
    def _same_sign_regularizer(self, model, lam: float = 1e-2):
        penalty = 0.0
        for k in range(model.nf):
            penalty = penalty + lam * torch.relu(-(model.a[k] * model.b[k]))
        return penalty
    
    def _train_tlinet(self, model, train_data: torch.Tensor, 
                      train_labels: torch.Tensor):
        train_data = train_data.to(self.device)
        train_labels = train_labels.to(self.device)
        
        dataset = TensorDataset(train_data, train_labels)
        dataloader = DataLoader(dataset, batch_size=self.train_config.batch_size, 
                                shuffle=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.train_config.lr)
        relu = torch.nn.ReLU()
        
        model.train()
        for epoch in range(self.train_config.epochs):
            for data_batch, labels_batch in dataloader:
                optimizer.zero_grad()
                r, reg = model(data_batch)
                
                loss = torch.mean(relu(model.eps - labels_batch * r))
                loss = loss - self.train_config.weight_margin * model.eps + reg
                loss = loss + self._same_sign_regularizer(model, self.train_config.sign_penalty_lam)
                
                loss.backward()
                optimizer.step()
        
        model.eval()
        return model
    
    def count_temporal_ops(self, model) -> int:
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        try:
            model.translate_formula()
        except:
            pass
        formula_str = buffer.getvalue().strip()
        sys.stdout = old_stdout
        
        return formula_str.count('F[') + formula_str.count('G[')
    
    def get_formula_string(self, model) -> str:
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        try:
            model.translate_formula()
        except:
            pass
        formula_str = buffer.getvalue().strip()
        sys.stdout = old_stdout
        return formula_str
    
    def compute_robustness_quality(self, model, data: torch.Tensor) -> float:
        with torch.no_grad():
            data = data.to(self.device)
            r, _ = model(data)
            return torch.sigmoid(r.mean()).item()
    
    def compute_distance(self, model1, model2, data: torch.Tensor) -> float:
        with torch.no_grad():
            data = data.to(self.device)
            r1, _ = model1(data)
            r2, _ = model2(data)
            diff = torch.abs(r1 - r2).mean()
            return torch.sigmoid(diff).item()
    
    def is_diverse(self, new_model, candidate_set: List[CandidateFormula], 
                   train_data: torch.Tensor, lambda2: float) -> bool:
        if len(candidate_set) == 0:
            return True
        
        for candidate in candidate_set:
            existing_model = self._create_model()
            candidate.load_into_model(existing_model)
            dist = self.compute_distance(new_model, existing_model, train_data)
            if dist <= lambda2:
                return False
        return True
    
    def compute_F(self, candidate_set: List[CandidateFormula]) -> float:
        if len(candidate_set) == 0:
            return 0.0
        
        if self.config.F_type == 'aggregate':
            return sum(c.robustness_quality for c in candidate_set) / len(candidate_set)
        elif self.config.F_type == 'max':
            return max(c.robustness_quality for c in candidate_set)
        else:
            return sum(c.robustness_quality for c in candidate_set) / len(candidate_set)
    
    def evaluate_formula_balanced(self, model, val_data: torch.Tensor, 
                                   val_labels: torch.Tensor) -> float:
        model.eval()
        
        with torch.no_grad():
            val_data = val_data.to(self.device)
            val_labels = val_labels.to(self.device)
            
            if not self.config.use_balanced_accuracy:
                return model.accuracy_formula(val_data, val_labels)
            
            r, _ = model(val_data)
            predictions = torch.sign(r).squeeze()
            
            pos_mask = (val_labels == 1)
            neg_mask = (val_labels == -1)
            
            n_pos = pos_mask.sum().item()
            n_neg = neg_mask.sum().item()
            
            if n_pos == 0 or n_neg == 0:
                return model.accuracy_formula(val_data, val_labels)
            
            tp = ((predictions == 1) & pos_mask).sum().item()
            tn = ((predictions == -1) & neg_mask).sum().item()
            
            tpr = tp / n_pos
            tnr = tn / n_neg
            
            return (tpr + tnr) / 2
    
    def run(self, train_data: torch.Tensor, train_labels: torch.Tensor,
            verbose: bool = False) -> List[CandidateFormula]:
        candidate_set = []
        nf = self.model_args.get('nf', 6)
        
        for l in range(self.config.lmax):
            model = self._create_model()
            model = self._train_tlinet(model, train_data, train_labels)
            
            N = self.count_temporal_ops(model)
            complexity = N / nf if nf > 0 else 0.0
            
            if N == 0:
                if verbose:
                    print(f"    l={l+1}: Empty formula, skipping")
                continue
            
            # λ1: Complexity check
            if complexity > self.config.lambda1:
                if verbose:
                    print(f"    l={l+1}: Complexity {complexity:.2f} > λ1={self.config.lambda1}, skipping")
                continue
            
            # λ2: Diversity check
            is_diverse = self.is_diverse(model, candidate_set, train_data, self.config.lambda2)
            if not is_diverse:
                if verbose:
                    print(f"    l={l+1}: Not diverse enough (distance ≤ λ2={self.config.lambda2}), skipping")
                continue
            
            robustness_quality = self.compute_robustness_quality(model, train_data)
            formula_str = self.get_formula_string(model)
            
            candidate = CandidateFormula(
                model_state_dict=deepcopy({k: v.cpu() for k, v in model.state_dict().items()}),
                complexity=complexity,
                robustness_quality=robustness_quality,
                train_accuracy=0.0,
                formula_str=formula_str
            )
            candidate_set.append(candidate)
            
            if verbose:
                print(f"    l={l+1}: Added formula (complexity={complexity:.2f}, F={robustness_quality:.3f})")
            
            # λ3: Stopping rule
            F = self.compute_F(candidate_set)
            if F > self.config.lambda3:
                if verbose:
                    print(f"    Stopping: F={F:.3f} > λ3={self.config.lambda3}")
                break
        
        return candidate_set
    
    def validate(self, candidate_set: List[CandidateFormula],
                 val_data: torch.Tensor, val_labels: torch.Tensor) -> Tuple[bool, float]:
        if len(candidate_set) == 0:
            return False, 0.0
        
        best_accuracy = 0.0
        
        for candidate in candidate_set:
            model = self._create_model()
            candidate.load_into_model(model)
            
            acc = self.evaluate_formula_balanced(model, val_data, val_labels)
            best_accuracy = max(best_accuracy, acc)
            
            if acc > self.config.tau:
                return True, acc
        
        return False, best_accuracy


# =============================================================================
# LTT Calibration
# =============================================================================

EPSILON_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def run_pareto_testing_multi_eps(cstli: CSTLI, cal_set_1: List[Tuple],
                                  lambda_grid: List[Tuple[float, float, float]],
                                  delta: float = 0.05,
                                  epsilon_values: List[float] = None,
                                  verbose: bool = True) -> Dict:
    """
    Run Pareto testing on cal_set_1 to determine viable configurations for each epsilon.
    """
    if epsilon_values is None:
        epsilon_values = EPSILON_VALUES
    
    K1 = len(cal_set_1)
    
    print("\n" + "="*70)
    print("PARETO TESTING (Cal Set 1)")
    print("="*70)
    print(f"K1 = {K1} samples")
    print(f"Lambda grid: {len(lambda_grid)} configurations")
    print(f"Epsilon values: {epsilon_values}")
    
    # Evaluate all configurations
    results = {}
    
    for config_idx, (l1, l2, l3) in enumerate(lambda_grid):
        if verbose:
            print(f"\n[{config_idx+1}/{len(lambda_grid)}] λ = ({l1}, {l2}, {l3})")
        
        cstli.config.lambda1 = l1
        cstli.config.lambda2 = l2
        cstli.config.lambda3 = l3
        
        failures = 0
        total_set_size = 0
        
        for k, (train_data, train_labels, val_data, val_labels) in enumerate(cal_set_1):
            candidate_set = cstli.run(train_data, train_labels, verbose=False)
            success, _ = cstli.validate(candidate_set, val_data, val_labels)
            
            if not success:
                failures += 1
            total_set_size += len(candidate_set)
        
        avg_set_size = total_set_size / K1
        risk = failures / K1
        
        # Compute p-values for all epsilon values
        p_values = {}
        for eps in epsilon_values:
            p_values[eps] = stats.binom.cdf(failures, K1, eps)
        
        results[(l1, l2, l3)] = {
            'failures': failures,
            'risk': risk,
            'avg_set_size': avg_set_size,
            'p_values': p_values
        }
        
        if verbose:
            print(f"  Risk: {risk:.3f}, Avg Set Size: {avg_set_size:.2f}")
    
    # Filter Pareto-optimal configurations for each epsilon
    pareto_configs = {eps: [] for eps in epsilon_values}
    
    for eps in epsilon_values:
        adjusted_delta = delta / len(lambda_grid)
        
        for config, data in results.items():
            if data['p_values'][eps] < adjusted_delta:
                pareto_configs[eps].append(config)
        
        print(f"\nε = {eps}: {len(pareto_configs[eps])} configs passed Pareto test")
    
    return {
        'results': results,
        'pareto_configs': pareto_configs,
        'epsilon_values': epsilon_values,
        'delta': delta
    }


def run_fixed_sequence_testing_multi_eps(cstli: CSTLI, cal_set_2: List[Tuple],
                                          pareto_results: Dict,
                                          delta: float = 0.05,
                                          verbose: bool = True) -> Dict:
    """
    Run fixed-sequence testing on cal_set_2 to select best lambda for each epsilon.
    """
    K2 = len(cal_set_2)
    epsilon_values = pareto_results['epsilon_values']
    pareto_configs = pareto_results['pareto_configs']
    pareto_data = pareto_results['results']
    
    print("\n" + "="*70)
    print("FIXED-SEQUENCE TESTING (Cal Set 2)")
    print("="*70)
    print(f"K2 = {K2} samples")
    
    # Evaluate ALL Pareto configs on cal_set_2 (done once)
    all_pareto_configs = set()
    for eps in epsilon_values:
        all_pareto_configs.update(pareto_configs[eps])
    all_pareto_configs = list(all_pareto_configs)
    
    print(f"Total unique Pareto configs to evaluate: {len(all_pareto_configs)}")
    
    cal2_results = {}
    
    for config_idx, (l1, l2, l3) in enumerate(all_pareto_configs):
        cstli.config.lambda1 = l1
        cstli.config.lambda2 = l2
        cstli.config.lambda3 = l3
        
        failures = 0
        total_set_size = 0
        
        for k, (train_data, train_labels, val_data, val_labels) in enumerate(cal_set_2):
            candidate_set = cstli.run(train_data, train_labels, verbose=False)
            success, _ = cstli.validate(candidate_set, val_data, val_labels)
            
            if not success:
                failures += 1
            total_set_size += len(candidate_set)
        
        avg_set_size = total_set_size / K2
        
        # Compute p-values for all epsilon
        p_values = {}
        for eps in epsilon_values:
            p_values[eps] = stats.binom.cdf(failures, K2, eps)
        
        cal2_results[(l1, l2, l3)] = {
            'failures': failures,
            'avg_set_size': avg_set_size,
            'p_values': p_values
        }
    
    # Select best lambda for each epsilon
    best_lambdas = {}
    reliable_sets = {}
    
    for eps in epsilon_values:
        # Order Pareto configs by set size (from Pareto testing)
        configs = pareto_configs[eps]
        if len(configs) == 0:
            print(f"\nε = {eps}: No Pareto configs available")
            best_lambdas[eps] = None
            reliable_sets[eps] = []
            continue
        
        sorted_configs = sorted(configs, key=lambda c: pareto_data[c]['avg_set_size'])
        
        # Fixed-sequence testing
        adjusted_delta = delta / len(configs)
        reliable = []
        
        for config in sorted_configs:
            if config in cal2_results:
                if cal2_results[config]['p_values'][eps] < adjusted_delta:
                    reliable.append(config)
        
        reliable_sets[eps] = reliable
        
        if len(reliable) > 0:
            best_lambdas[eps] = reliable[0]  # Smallest set size among reliable
            print(f"\nε = {eps}: Best λ* = {best_lambdas[eps]}")
            print(f"  Reliable configs: {len(reliable)}/{len(configs)}")
        else:
            best_lambdas[eps] = None
            print(f"\nε = {eps}: No reliable config found")
    
    return {
        'best_lambdas': best_lambdas,
        'reliable_sets': reliable_sets,
        'cal2_results': cal2_results,
        'epsilon_values': epsilon_values
    }


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_on_test_set(cstli: CSTLI, test_set: List[Tuple], 
                         best_lambda: Tuple[float, float, float],
                         verbose: bool = True) -> Dict:
    """Evaluate C-STLI with given lambda on test set."""
    cstli.config.lambda1 = best_lambda[0]
    cstli.config.lambda2 = best_lambda[1]
    cstli.config.lambda3 = best_lambda[2]
    
    Y = len(test_set)
    failures = 0
    total_set_size = 0
    total_complexity = 0
    total_diversity = 0
    
    set_sizes = []
    complexities = []
    diversities = []
    
    for k, (train_data, train_labels, val_data, val_labels) in enumerate(test_set):
        candidate_set = cstli.run(train_data, train_labels, verbose=False)
        success, _ = cstli.validate(candidate_set, val_data, val_labels)
        
        if not success:
            failures += 1
        
        set_size = len(candidate_set)
        total_set_size += set_size
        set_sizes.append(set_size)
        
        # Complexity
        if set_size > 0:
            cpx = sum(c.complexity for c in candidate_set) / set_size
        else:
            cpx = 0.0
        total_complexity += cpx
        complexities.append(cpx)
        
        # Diversity
        if set_size >= 2:
            distances = []
            for i in range(len(candidate_set)):
                for j in range(i + 1, len(candidate_set)):
                    model_i = cstli._create_model()
                    model_j = cstli._create_model()
                    candidate_set[i].load_into_model(model_i)
                    candidate_set[j].load_into_model(model_j)
                    dist = cstli.compute_distance(model_i, model_j, train_data)
                    distances.append(dist)
            div = sum(distances) / len(distances) if distances else 0.0
        else:
            div = 0.0
        total_diversity += div
        diversities.append(div)
    
    risk = failures / Y
    avg_set_size = total_set_size / Y
    avg_complexity = total_complexity / Y
    avg_diversity = total_diversity / Y
    
    return {
        'risk': risk,
        'avg_set_size': avg_set_size,
        'avg_complexity': avg_complexity,
        'avg_diversity': avg_diversity,
        'failures': failures,
        'Y': Y,
        'set_sizes': set_sizes,
        'complexities': complexities,
        'diversities': diversities
    }


# =============================================================================
# Main
# =============================================================================

def main():
    # CSV paths for pre-labeled data
    csv_paths = [
        "data/combined1.csv",
        "data/combined2.csv",
        "data/combined3.csv",
        "data/combined4.csv",
        "data/combined5.csv",
    ]
    
    # Check if files exist
    missing = [p for p in csv_paths if not os.path.exists(p)]
    if missing:
        print(f"Missing files: {missing}")
        return
    
    # =========================================================================
    # Load Pre-Labeled Data
    # =========================================================================
    
    # Scale values for normalization (similar to original threshold-based scaling)
    # Set to None if data is already normalized or you don't want scaling
    scale_values = {
        'latency_ms': 100.0,
        'backlog_hbytes': 300.0
    }
    
    distributions = load_all_distributions(
        csv_paths,
        feature_cols=['latency_ms', 'backlog_hbytes'],
        label_col='qoe_label',
        scale_values=scale_values
    )
    
    # Print distribution statistics
    print("\n" + "="*70)
    print("DISTRIBUTION STATISTICS")
    print("="*70)
    for i, (X, y) in enumerate(distributions):
        n_pos = (y == 1).sum()
        n_neg = (y == -1).sum()
        print(f"Distribution {i+1}: {len(y)} examples, +1={n_pos} ({n_pos/len(y)*100:.1f}%), -1={n_neg}")
    
    # =========================================================================
    # Configuration
    # =========================================================================
    
    config = DataSplitConfig(
        K1=50,
        K2=50,
        Y=40,
        X_train=5000,
        X_val=1000,
        seed=42
    )
    
    # Get model dimensions from data
    first_X, _ = distributions[0]
    dim = first_X.shape[2]  # Number of features
    length = first_X.shape[1]  # Sequence length (61)
    
    print(f"\nData dimensions: {first_X.shape}")
    print(f"  Sequence length: {length}")
    print(f"  Features: {dim}")
    
    train_config = TrainConfig(epochs=5, lr=0.1, batch_size=512)
    
    model_args = {
        'nf': dim * 3,  # 3 predicates per feature
        'nc': 1,
        'length': length,
        'weight_bi': train_config.weight_bi,
        'weight_l1': train_config.weight_l1
    }
    
    # =========================================================================
    # Create Hierarchical Splits (Calibration Only - No Test Set)
    # =========================================================================
    
    splits = create_hierarchical_splits_from_distributions(
        distributions, config, create_test_set=False
    )
    
    cal_set_1 = splits['cal_set_1']
    cal_set_2 = splits['cal_set_2']
    
    # =========================================================================
    # Define Lambda Grid
    # =========================================================================
    
    lambda1_values = [0.33, 0.50, 0.67]
    lambda2_values = [0.50, 0.52, 0.54, 0.56, 0.58]
    lambda3_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    lambda_grid = [
        (l1, l2, l3)
        for l1 in lambda1_values
        for l2 in lambda2_values
        for l3 in lambda3_values
    ]
    
    print(f"\nLambda grid: {len(lambda_grid)} configurations")
    
    # =========================================================================
    # Run LTT Calibration
    # =========================================================================
    
    cstli = CSTLI(NavalModel1, model_args, train_config, use_cpu=USE_CPU)
    
    # Pareto Testing
    pareto_results = run_pareto_testing_multi_eps(
        cstli, cal_set_1, lambda_grid,
        delta=0.05,
        epsilon_values=EPSILON_VALUES,
        verbose=True
    )
    
    # Fixed-Sequence Testing
    fst_results = run_fixed_sequence_testing_multi_eps(
        cstli, cal_set_2, pareto_results,
        delta=0.05,
        verbose=True
    )
    
    best_lambdas = fst_results['best_lambdas']
    
    # =========================================================================
    # Save Results (Calibration Only - No Test Evaluation)
    # =========================================================================
    
    save_data = {
        'best_lambdas': best_lambdas,
        'pareto_results': pareto_results,
        'fst_results': fst_results,
        'config': config,
        'lambda_grid': lambda_grid,
        'epsilon_values': EPSILON_VALUES,
        'num_distributions': len(distributions)
    }
    
    save_path = 'cstli_prelabeled_results.pt'
    torch.save(save_data, save_path)
    print(f"\nResults saved to: {save_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "="*70)
    print("CALIBRATION SUMMARY")
    print("="*70)
    
    print(f"\n{'ε':<8} {'Best λ*':<40}")
    print("-" * 50)
    
    for eps in EPSILON_VALUES:
        bl = best_lambdas.get(eps, None)
        if bl is not None:
            print(f"{eps:<8} {str(bl):<40}")
        else:
            print(f"{eps:<8} {'None (no reliable config found)':<40}")
    
    print("\n" + "="*70)
    print("C-STLI CALIBRATION COMPLETE")
    print("="*70)
    print("\nUse the saved best_lambdas in a separate script for test evaluation.")


if __name__ == "__main__":
    main()