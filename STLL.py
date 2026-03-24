"""
C-STLI Benchmark 5: STLI (Single Formula Baseline)

Original STLI method - produces a single formula without calibration.
- No hyperparameters (no λ1, λ2, λ3)
- No calibration needed
- Set size = 1 (always)
- Diversity = 0 (always)
- Results same for all ε (epsilon-independent)

Input: combined1.csv to combined5.csv
Output: benchmark5_stli_results.pt
"""

import torch
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import os
import matplotlib
matplotlib.use('Agg')

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
class STLIConfig:
    """Configuration for STLI (single formula)"""
    tau: float = 0.75
    use_balanced_accuracy: bool = True


@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 0.1
    batch_size: int = 512
    weight_margin: float = 1e-2
    sign_penalty_lam: float = 1e-2
    weight_bi: List[float] = field(default_factory=lambda: [1e-0, 1e-0, 1e-0])
    weight_l1: List[float] = field(default_factory=lambda: [1e-2])


@dataclass
class DataSplitConfig:
    K: int = 50  # Number of test samples
    X_train: int = 5000
    X_val: int = 1000
    seed: int = 12345  # Different seed for test evaluation
    seed_offset: int = 5000


# =============================================================================
# Data Loading
# =============================================================================

SEQ_LEN = 61


def load_prelabeled_csv(csv_path: str, feature_cols: List[str] = None,
                        label_col: str = None, episode_col: str = 'episode',
                        scale_values: Dict[str, float] = None) -> Tuple[np.ndarray, np.ndarray]:
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if feature_cols is None:
        if 'latency_ms' in df.columns and 'backlog_hbytes' in df.columns:
            feature_cols = ['latency_ms', 'backlog_hbytes']
        else:
            raise ValueError(f"Could not detect feature columns.")
    
    if label_col is None:
        label_col = 'qoe_label' if 'qoe_label' in df.columns else df.columns[-1]
    
    df = df.sort_values(by=[episode_col, 't_s'] if 't_s' in df.columns else [episode_col])
    episodes = df[episode_col].unique()
    
    X_list, y_list = [], []
    for ep in episodes:
        ep_data = df[df[episode_col] == ep]
        x = ep_data[feature_cols].to_numpy(dtype=np.float32)
        if len(x) < SEQ_LEN:
            x = np.concatenate([x, np.repeat(x[-1:], SEQ_LEN - len(x), axis=0)], axis=0)
        elif len(x) > SEQ_LEN:
            x = x[:SEQ_LEN]
        X_list.append(x)
        y_list.append(ep_data[label_col].iloc[0])
    
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    
    if scale_values:
        for i, col in enumerate(feature_cols):
            if col in scale_values:
                X[:, :, i] /= scale_values[col]
    
    if set(np.unique(y)) == {0, 1}:
        y = np.where(y == 1, 1, -1)
    
    print(f"  Loaded: {len(y)} examples, +1={(y==1).sum()}, -1={(y==-1).sum()}")
    return X, y


def load_all_distributions(csv_paths, feature_cols=None, label_col=None, scale_values=None):
    distributions = []
    for i, path in enumerate(csv_paths):
        print(f"\nDistribution {i+1}:")
        X, y = load_prelabeled_csv(path, feature_cols, label_col, scale_values=scale_values)
        distributions.append((X, y))
    return distributions


def create_hierarchical_samples(distributions, num_samples, train_size, val_size, seed=42, seed_offset=0):
    samples, dist_indices = [], []
    for i in range(num_samples):
        np.random.seed(seed + seed_offset + i)
        torch.manual_seed(seed + seed_offset + i)
        dist_idx = np.random.randint(0, len(distributions))
        X, y = distributions[dist_idx]
        perm = torch.randperm(len(X))
        train_idx, val_idx = perm[:train_size], perm[train_size:train_size+val_size]
        X_t, y_t = torch.from_numpy(X), torch.from_numpy(y)
        samples.append((X_t[train_idx], y_t[train_idx], X_t[val_idx], y_t[val_idx]))
        dist_indices.append(dist_idx)
    return samples, dist_indices


# =============================================================================
# STLI (Single Formula)
# =============================================================================

class STLI:
    """
    Original STLI method - produces a single formula.
    No hyperparameters, no calibration.
    """
    
    def __init__(self, model_class, model_args, train_config, stli_config=None, use_cpu=False):
        self.model_class = model_class
        self.model_args = model_args
        self.train_config = train_config
        self.config = stli_config or STLIConfig()
        self.device = torch.device('cpu') if use_cpu else DEVICE
    
    def _create_model(self):
        return self.model_class(**self.model_args).to(self.device)
    
    def _train_tlinet(self, model, train_data, train_labels):
        train_data, train_labels = train_data.to(self.device), train_labels.to(self.device)
        dataset = TensorDataset(train_data, train_labels)
        dataloader = DataLoader(dataset, batch_size=self.train_config.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.train_config.lr)
        relu = torch.nn.ReLU()
        
        model.train()
        for _ in range(self.train_config.epochs):
            for data_batch, labels_batch in dataloader:
                optimizer.zero_grad()
                r, reg = model(data_batch)
                loss = torch.mean(relu(model.eps - labels_batch * r)) - self.train_config.weight_margin * model.eps + reg
                penalty = sum(torch.relu(-(model.a[k] * model.b[k])) for k in range(model.nf))
                loss = loss + 1e-2 * penalty
                loss.backward()
                optimizer.step()
        model.eval()
        return model
    
    def _get_formula_str(self, model):
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        try:
            model.translate_formula()
        except:
            pass
        formula_str = buffer.getvalue().strip()
        sys.stdout = old_stdout
        return formula_str
    
    def _count_ops(self, model):
        formula_str = self._get_formula_str(model)
        return formula_str.count('F[') + formula_str.count('G[')
    
    def _evaluate_balanced_accuracy(self, model, val_data, val_labels):
        """Evaluate using balanced accuracy."""
        val_data, val_labels = val_data.to(self.device), val_labels.to(self.device)
        
        with torch.no_grad():
            r, _ = model(val_data)
            preds = torch.sign(r).squeeze()
            
            pos_mask = (val_labels == 1)
            neg_mask = (val_labels == -1)
            n_pos, n_neg = pos_mask.sum().item(), neg_mask.sum().item()
            
            if n_pos > 0 and n_neg > 0:
                tpr = ((preds == 1) & pos_mask).sum().item() / n_pos
                tnr = ((preds == -1) & neg_mask).sum().item() / n_neg
                return (tpr + tnr) / 2
            else:
                return (preds == val_labels).float().mean().item()
    
    def train_and_evaluate(self, train_data, train_labels, val_data, val_labels):
        """
        Train a single TLINet model and evaluate.
        
        Returns:
            dict with success, complexity, accuracy, formula_str
        """
        # Train single model
        model = self._create_model()
        model = self._train_tlinet(model, train_data, train_labels)
        
        # Get formula info
        formula_str = self._get_formula_str(model)
        N = self._count_ops(model)
        nf = self.model_args.get('nf', 6)
        complexity = N / nf if nf > 0 else 0.0
        
        # Evaluate
        accuracy = self._evaluate_balanced_accuracy(model, val_data, val_labels)
        success = accuracy > self.config.tau
        
        return {
            'success': success,
            'complexity': complexity,
            'accuracy': accuracy,
            'formula_str': formula_str,
            'num_ops': N
        }


# =============================================================================
# Evaluation
# =============================================================================

EPSILON_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def evaluate_stli(stli, test_samples, verbose=True):
    """
    Evaluate STLI on test samples.
    
    Since STLI has no hyperparameters, results are the same for all epsilon.
    """
    K = len(test_samples)
    
    print(f"\n{'='*70}")
    print("EVALUATING STLI")
    print(f"{'='*70}")
    print(f"K = {K} test samples")
    
    failures = 0
    total_complexity = 0.0
    complexities = []
    accuracies = []
    
    for k, (train_data, train_labels, val_data, val_labels) in enumerate(test_samples):
        if verbose and (k + 1) % 10 == 0:
            print(f"  Sample {k+1}/{K}...")
        
        result = stli.train_and_evaluate(train_data, train_labels, val_data, val_labels)
        
        if not result['success']:
            failures += 1
        
        complexities.append(result['complexity'])
        accuracies.append(result['accuracy'])
        total_complexity += result['complexity']
    
    risk = failures / K
    avg_complexity = total_complexity / K
    avg_accuracy = sum(accuracies) / K
    
    # STLI always produces set size = 1, diversity = 0
    avg_set_size = 1.0
    avg_diversity = 0.0
    
    print(f"\nResults:")
    print(f"  Risk: {risk:.3f} ({failures}/{K} failures)")
    print(f"  Avg Complexity: {avg_complexity:.3f}")
    print(f"  Avg Accuracy: {avg_accuracy:.3f}")
    print(f"  Set Size: {avg_set_size} (always 1)")
    print(f"  Diversity: {avg_diversity} (always 0)")
    
    return {
        'risk': risk,
        'avg_set_size': avg_set_size,
        'avg_complexity': avg_complexity,
        'avg_diversity': avg_diversity,
        'avg_accuracy': avg_accuracy,
        'failures': failures,
        'K': K,
        'complexities': complexities,
        'accuracies': accuracies
    }


# =============================================================================
# Main
# =============================================================================

def main():
    csv_paths = [f"data/combined{i}.csv" for i in range(1, 6)]
    if any(not os.path.exists(p) for p in csv_paths):
        print("Missing data files!")
        return
    
    print("="*70)
    print("BENCHMARK 5: STLI (SINGLE FORMULA BASELINE)")
    print("="*70)
    
    # Load data
    scale_values = {'latency_ms': 100.0, 'backlog_hbytes': 300.0}
    distributions = load_all_distributions(
        csv_paths, ['latency_ms', 'backlog_hbytes'], 'qoe_label', scale_values
    )
    
    # Configuration
    config = DataSplitConfig(K=50, X_train=5000, X_val=1000, seed=12345, seed_offset=5000)
    
    # Create test samples (using different seed than calibration)
    test_samples, test_dist = create_hierarchical_samples(
        distributions, config.K, config.X_train, config.X_val, 
        config.seed, config.seed_offset
    )
    print(f"\nTest samples: {len(test_samples)}")
    print(f"  Distribution usage: {dict(Counter(test_dist))}")
    
    # Model setup
    first_X, _ = distributions[0]
    model_args = {
        'nf': first_X.shape[2] * 3, 'nc': 1, 'length': first_X.shape[1],
        'weight_bi': [1e-0, 1e-0, 1e-0], 'weight_l1': [1e-2]
    }
    train_config = TrainConfig()
    stli_config = STLIConfig(tau=0.75)
    
    # Evaluate STLI
    stli = STLI(NavalModel1, model_args, train_config, stli_config, use_cpu=USE_CPU)
    metrics = evaluate_stli(stli, test_samples, verbose=True)
    
    # Create results for all epsilon (same values since STLI is epsilon-independent)
    results_by_eps = {}
    for eps in EPSILON_VALUES:
        results_by_eps[eps] = {
            'risk': metrics['risk'],
            'avg_set_size': metrics['avg_set_size'],
            'avg_complexity': metrics['avg_complexity'],
            'avg_diversity': metrics['avg_diversity']
        }
    
    # Save
    save_data = {
        'STLI': results_by_eps,
        'raw_results': metrics,
        'epsilon_values': EPSILON_VALUES,
        'config': config,
        'test_dist': test_dist
    }
    
    torch.save(save_data, 'benchmark5_stli_results.pt')
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nSTLI Results (same for all ε):")
    print(f"  Risk: {metrics['risk']:.3f}")
    print(f"  Set Size: {metrics['avg_set_size']:.1f}")
    print(f"  Complexity: {metrics['avg_complexity']:.3f}")
    print(f"  Diversity: {metrics['avg_diversity']:.1f}")
    
    print(f"\n{'ε':<8} {'Risk':<10} {'Set Size':<12} {'Complexity':<12} {'Diversity':<10}")
    print("-" * 55)
    for eps in EPSILON_VALUES:
        r = results_by_eps[eps]
        print(f"{eps:<8} {r['risk']:<10.3f} {r['avg_set_size']:<12.1f} {r['avg_complexity']:<12.3f} {r['avg_diversity']:<10.1f}")
    
    print("\nSaved: benchmark5_stli_results.pt")


if __name__ == "__main__":
    main()