"""
Evaluation Script for All C-STLI Schemes (Pre-labeled Data)

This script:
1. Loads saved best_lambda results from all schemes
2. Creates test samples using hierarchical sampling from pre-labeled distributions
3. Runs C-STLI with best_lambda for each epsilon for each scheme
4. Computes: average risk, set size, complexity, diversity
5. Saves results for later plotting

Schemes:
- Full (λ1+λ2+λ3): best_lambdas.pt
- Benchmark 1 (λ3 only): benchmark1_lambda3_only_results.pt
- Benchmark 2 (λ1+λ3): benchmark2_lambda1_lambda3_results.pt
- Benchmark 3 (λ2+λ3): benchmark3_lambda2_lambda3_results.pt
- Benchmark 4 (Bonferroni): benchmark4_bonferroni_results.pt

Note: STLI is evaluated separately in benchmark5_stli.py

Input: combined1.csv to combined5.csv
Output: evaluation_results.pt
"""

import torch
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import time
import os
import sys

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
    lambda1: float = 0.5
    lambda2: float = 0.52
    lambda3: float = 0.5
    lmax: int = 10
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
    seed: int = 12345
    seed_offset: int = 5000


# =============================================================================
# Data Loading (Pre-labeled)
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
# Candidate Formula
# =============================================================================

@dataclass
class CandidateFormula:
    model_state_dict: dict
    complexity: float
    robustness_quality: float
    formula_str: str = ""
    
    def load_into_model(self, model):
        state_dict = {k: v.to(model.a.device) for k, v in self.model_state_dict.items()}
        model.load_state_dict(state_dict)


# =============================================================================
# C-STLI Base Class
# =============================================================================

class CSTLI_Base:
    def __init__(self, model_class, model_args, train_config, cstli_config=None, use_cpu=False):
        self.model_class = model_class
        self.model_args = model_args
        self.train_config = train_config
        self.config = cstli_config or CSTLIConfig()
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
        import io
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
    
    def _compute_robustness(self, model, data):
        with torch.no_grad():
            r, _ = model(data.to(self.device))
            return torch.sigmoid(r.mean()).item()
    
    def _compute_distance(self, model1, model2, data):
        with torch.no_grad():
            data = data.to(self.device)
            r1, _ = model1(data)
            r2, _ = model2(data)
            return torch.sigmoid(torch.abs(r1 - r2).mean()).item()
    
    def _compute_F(self, candidate_set):
        if not candidate_set:
            return 0.0
        return sum(c.robustness_quality for c in candidate_set) / len(candidate_set)
    
    def evaluate_formula(self, candidate, val_data, val_labels):
        model = self._create_model()
        candidate.load_into_model(model)
        model.eval()
        
        val_data, val_labels = val_data.to(self.device), val_labels.to(self.device)
        
        with torch.no_grad():
            r, _ = model(val_data)
            preds = torch.sign(r).squeeze()
            pos_mask, neg_mask = (val_labels == 1), (val_labels == -1)
            n_pos, n_neg = pos_mask.sum().item(), neg_mask.sum().item()
            
            if n_pos > 0 and n_neg > 0:
                tpr = ((preds == 1) & pos_mask).sum().item() / n_pos
                tnr = ((preds == -1) & neg_mask).sum().item() / n_neg
                return (tpr + tnr) / 2
            else:
                return (preds == val_labels).float().mean().item()


# =============================================================================
# C-STLI Variants
# =============================================================================

class CSTLI_Full(CSTLI_Base):
    """Full C-STLI with λ1, λ2, λ3."""
    
    def run(self, train_data, train_labels):
        candidate_set = []
        nf = self.model_args.get('nf', 6)
        
        for _ in range(self.config.lmax):
            model = self._create_model()
            model = self._train_tlinet(model, train_data, train_labels)
            
            N = self._count_ops(model)
            if N == 0:
                continue
            
            complexity = N / nf
            
            # λ1: Complexity check
            if complexity > self.config.lambda1:
                continue
            
            # λ2: Diversity check
            is_diverse = True
            for prev in candidate_set:
                prev_model = self._create_model()
                prev.load_into_model(prev_model)
                if self._compute_distance(model, prev_model, train_data) <= self.config.lambda2:
                    is_diverse = False
                    break
            if not is_diverse:
                continue
            
            candidate = CandidateFormula(
                model_state_dict=deepcopy({k: v.cpu() for k, v in model.state_dict().items()}),
                complexity=complexity,
                robustness_quality=self._compute_robustness(model, train_data),
                formula_str=self._get_formula_str(model)
            )
            candidate_set.append(candidate)
            
            # λ3: Stopping rule
            if self._compute_F(candidate_set) > self.config.lambda3:
                break
        
        return candidate_set


class CSTLI_Lambda3Only(CSTLI_Base):
    """λ3 only (stopping rule)."""
    
    def run(self, train_data, train_labels):
        candidate_set = []
        nf = self.model_args.get('nf', 6)
        
        for _ in range(self.config.lmax):
            model = self._create_model()
            model = self._train_tlinet(model, train_data, train_labels)
            
            N = self._count_ops(model)
            if N == 0:
                continue
            
            candidate = CandidateFormula(
                model_state_dict=deepcopy({k: v.cpu() for k, v in model.state_dict().items()}),
                complexity=N / nf,
                robustness_quality=self._compute_robustness(model, train_data),
                formula_str=self._get_formula_str(model)
            )
            candidate_set.append(candidate)
            
            if self._compute_F(candidate_set) > self.config.lambda3:
                break
        
        return candidate_set


class CSTLI_Lambda1Lambda3(CSTLI_Base):
    """λ1 + λ3 (complexity + stopping)."""
    
    def run(self, train_data, train_labels):
        candidate_set = []
        nf = self.model_args.get('nf', 6)
        
        for _ in range(self.config.lmax):
            model = self._create_model()
            model = self._train_tlinet(model, train_data, train_labels)
            
            N = self._count_ops(model)
            if N == 0:
                continue
            
            complexity = N / nf
            
            # λ1: Complexity check
            if complexity > self.config.lambda1:
                continue
            
            candidate = CandidateFormula(
                model_state_dict=deepcopy({k: v.cpu() for k, v in model.state_dict().items()}),
                complexity=complexity,
                robustness_quality=self._compute_robustness(model, train_data),
                formula_str=self._get_formula_str(model)
            )
            candidate_set.append(candidate)
            
            if self._compute_F(candidate_set) > self.config.lambda3:
                break
        
        return candidate_set


class CSTLI_Lambda2Lambda3(CSTLI_Base):
    """λ2 + λ3 (diversity + stopping)."""
    
    def run(self, train_data, train_labels):
        candidate_set = []
        nf = self.model_args.get('nf', 6)
        
        for _ in range(self.config.lmax):
            model = self._create_model()
            model = self._train_tlinet(model, train_data, train_labels)
            
            N = self._count_ops(model)
            if N == 0:
                continue
            
            complexity = N / nf
            
            # λ2: Diversity check
            is_diverse = True
            for prev in candidate_set:
                prev_model = self._create_model()
                prev.load_into_model(prev_model)
                if self._compute_distance(model, prev_model, train_data) <= self.config.lambda2:
                    is_diverse = False
                    break
            if not is_diverse:
                continue
            
            candidate = CandidateFormula(
                model_state_dict=deepcopy({k: v.cpu() for k, v in model.state_dict().items()}),
                complexity=complexity,
                robustness_quality=self._compute_robustness(model, train_data),
                formula_str=self._get_formula_str(model)
            )
            candidate_set.append(candidate)
            
            if self._compute_F(candidate_set) > self.config.lambda3:
                break
        
        return candidate_set


# =============================================================================
# Evaluation Functions
# =============================================================================

def compute_set_diversity(formula_set, cstli, train_data):
    if len(formula_set) < 2:
        return 0.0
    
    distances = []
    for i in range(len(formula_set)):
        for j in range(i + 1, len(formula_set)):
            model_i = cstli._create_model()
            model_j = cstli._create_model()
            formula_set[i].load_into_model(model_i)
            formula_set[j].load_into_model(model_j)
            distances.append(cstli._compute_distance(model_i, model_j, train_data))
    
    return sum(distances) / len(distances) if distances else 0.0


def evaluate_scheme_on_test(cstli, best_lambda, test_samples, verbose=False):
    """Evaluate a scheme on test samples."""
    if best_lambda is None:
        return {
            'risk': float('nan'), 'avg_set_size': float('nan'),
            'avg_complexity': float('nan'), 'avg_diversity': float('nan')
        }
    
    l1, l2, l3 = best_lambda
    cstli.config.lambda1 = l1
    cstli.config.lambda2 = l2
    cstli.config.lambda3 = l3
    
    K = len(test_samples)
    failures = 0
    set_sizes, complexities, diversities = [], [], []
    
    for k, (train_data, train_labels, val_data, val_labels) in enumerate(test_samples):
        if verbose and (k + 1) % 10 == 0:
            print(f"    Sample {k+1}/{K}...")
        
        formula_set = cstli.run(train_data, train_labels)
        
        # Check if any formula passes tau
        set_is_good = False
        for formula in formula_set:
            if cstli.evaluate_formula(formula, val_data, val_labels) > cstli.config.tau:
                set_is_good = True
                break
        
        if not set_is_good:
            failures += 1
        
        set_sizes.append(len(formula_set))
        complexities.append(sum(f.complexity for f in formula_set) / len(formula_set) if formula_set else 0.0)
        diversities.append(compute_set_diversity(formula_set, cstli, train_data))
    
    return {
        'risk': failures / K,
        'avg_set_size': sum(set_sizes) / K,
        'avg_complexity': sum(complexities) / K,
        'avg_diversity': sum(diversities) / K,
        'set_sizes': set_sizes,
        'complexities': complexities,
        'diversities': diversities
    }


# =============================================================================
# Main
# =============================================================================

def main():
    # =========================================================================
    # Configuration
    # =========================================================================
    
    csv_paths = [f"data/combined{i}.csv" for i in range(1, 6)]
    
    if any(not os.path.exists(p) for p in csv_paths):
        print("Missing data files!")
        return
    
    # Result files for each scheme
    result_files = {
        'Full (λ1+λ2+λ3)': 'best_lambdas.pt',
        'λ3 only': 'benchmark1_lambda3_only_results.pt',
        'λ1+λ3': 'benchmark2_lambda1_lambda3_results.pt',
        'λ2+λ3': 'benchmark3_lambda2_lambda3_results.pt',
        'Bonferroni': 'benchmark4_bonferroni_results.pt',
    }
    
    # C-STLI class for each scheme
    cstli_classes = {
        'Full (λ1+λ2+λ3)': CSTLI_Full,
        'λ3 only': CSTLI_Lambda3Only,
        'λ1+λ3': CSTLI_Lambda1Lambda3,
        'λ2+λ3': CSTLI_Lambda2Lambda3,
        'Bonferroni': CSTLI_Full,
    }
    
    # Trivial safe lambda when no best_lambda found
    TRIVIAL_SAFE_LAMBDA = (1.0, 0.0, 1.0)
    
    # Epsilon values (excluding 0.05)
    epsilon_values = [0.10, 0.15, 0.20, 0.25, 0.30]
    
    # Test configuration
    num_test_samples = 50
    train_size = 5000
    val_size = 1000
    seed = 12345
    seed_offset = 5000
    
    # =========================================================================
    # Load Results
    # =========================================================================
    
    print("="*70)
    print("LOADING SAVED RESULTS")
    print("="*70)
    
    loaded_results = {}
    for scheme_name, result_file in result_files.items():
        if os.path.exists(result_file):
            loaded_results[scheme_name] = torch.load(result_file, weights_only=False)
            print(f"  ✓ Loaded {result_file}")
            best_lambdas = loaded_results[scheme_name].get('best_lambdas', {})
            print(f"    Best lambdas: {best_lambdas}")
        else:
            print(f"  ✗ Not found: {result_file}")
    
    if not loaded_results:
        print("No result files found!")
        return
    
    # =========================================================================
    # Load Data
    # =========================================================================
    
    print("\n" + "="*70)
    print("LOADING PRE-LABELED DATA")
    print("="*70)
    
    scale_values = {'latency_ms': 100.0, 'backlog_hbytes': 300.0}
    distributions = load_all_distributions(
        csv_paths, ['latency_ms', 'backlog_hbytes'], 'qoe_label', scale_values
    )
    
    # =========================================================================
    # Create Test Samples
    # =========================================================================
    
    print("\n" + "="*70)
    print("CREATING TEST SAMPLES")
    print("="*70)
    
    test_samples, test_dist = create_hierarchical_samples(
        distributions, num_test_samples, train_size, val_size, seed, seed_offset
    )
    
    print(f"Created {len(test_samples)} test samples")
    print(f"Distribution usage: {dict(Counter(test_dist))}")
    
    # =========================================================================
    # Setup Model Args
    # =========================================================================
    
    first_X, _ = distributions[0]
    model_args = {
        'nf': first_X.shape[2] * 3, 'nc': 1, 'length': first_X.shape[1],
        'weight_bi': [1e-0, 1e-0, 1e-0], 'weight_l1': [1e-2]
    }
    train_config = TrainConfig()
    
    # =========================================================================
    # Evaluate All Schemes
    # =========================================================================
    
    print("\n" + "="*70)
    print("EVALUATING ALL SCHEMES ON TEST DATA")
    print("="*70)
    
    all_results = {scheme: {} for scheme in loaded_results.keys()}
    
    for scheme_name in loaded_results.keys():
        print(f"\n{'='*60}")
        print(f"Scheme: {scheme_name}")
        print(f"{'='*60}")
        
        best_lambdas = loaded_results[scheme_name].get('best_lambdas', {})
        cstli_class = cstli_classes[scheme_name]
        
        for eps in epsilon_values:
            best_lambda = best_lambdas.get(eps, None)
            
            print(f"\n  ε = {eps}")
            
            using_trivial = False
            if best_lambda is None:
                print(f"    No best_lambda, using TRIVIAL SAFE: {TRIVIAL_SAFE_LAMBDA}")
                best_lambda = TRIVIAL_SAFE_LAMBDA
                using_trivial = True
            else:
                print(f"    Best λ* = {best_lambda}")
            
            # Create C-STLI instance
            cstli_config = CSTLIConfig(
                lmax=10, tau=0.75,
                lambda1=best_lambda[0], lambda2=best_lambda[1], lambda3=best_lambda[2]
            )
            cstli = cstli_class(NavalModel1, model_args, train_config, cstli_config, use_cpu=USE_CPU)
            
            # Evaluate
            start_time = time.time()
            metrics = evaluate_scheme_on_test(cstli, best_lambda, test_samples, verbose=True)
            elapsed = time.time() - start_time
            
            metrics['best_lambda'] = best_lambda
            metrics['using_trivial_safe'] = using_trivial
            all_results[scheme_name][eps] = metrics
            
            print(f"    Risk: {metrics['risk']:.3f}")
            print(f"    Avg Set Size: {metrics['avg_set_size']:.2f}")
            print(f"    Avg Complexity: {metrics['avg_complexity']:.3f}")
            print(f"    Avg Diversity: {metrics['avg_diversity']:.3f}")
            print(f"    Time: {elapsed:.1f}s")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    
    torch.save(all_results, 'evaluation_results.pt')
    print(f"\nResults saved to: evaluation_results.pt")
    
    # =========================================================================
    # Summary Table
    # =========================================================================
    
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    
    for eps in epsilon_values:
        print(f"\nε = {eps}")
        print(f"{'Scheme':<20} {'Risk':<10} {'Set Size':<12} {'Complexity':<12} {'Diversity':<12}")
        print("-" * 66)
        for scheme_name in all_results.keys():
            m = all_results[scheme_name].get(eps, {})
            risk = m.get('risk', float('nan'))
            size = m.get('avg_set_size', float('nan'))
            cpx = m.get('avg_complexity', float('nan'))
            div = m.get('avg_diversity', float('nan'))
            print(f"{scheme_name:<20} {risk:<10.3f} {size:<12.2f} {cpx:<12.3f} {div:<12.3f}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()