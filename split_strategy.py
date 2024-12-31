import numpy as np
from typing import List, Tuple, Any
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
from scipy.optimize import minimize
import random

class SplitOptimizer:
    def __init__(self, contents: List[Tuple[Any, int]], num_splits: int):
        self.contents = contents
        self.num_splits = num_splits
        self.key_groups = self._group_by_key()
        self.features = self._extract_features()
        
    def _group_by_key(self):
        groups = defaultdict(list)
        for item in self.contents:
            groups[item[0]].append(item)
        return groups
    
    def _extract_features(self):
        """Extract features for each key group"""
        features = []
        for key, group in self.key_groups.items():
            size = len(group)
            freq_sum = sum(item[1] for item in group)
            avg_freq = freq_sum / size
            max_freq = max(item[1] for item in group)
            min_freq = min(item[1] for item in group)
            
            features.append({
                'key': key,
                'size': size,
                'avg_freq': avg_freq,
                'max_freq': max_freq,
                'min_freq': min_freq,
                'total_freq': freq_sum,
                'freq_range': max_freq - min_freq
            })
        return features

    def _evaluate_split(self, split_assignment):
        """Evaluate the quality of the split assignment"""
        splits = [[] for _ in range(self.num_splits)]
        for idx, assignment in enumerate(split_assignment):
            splits[int(assignment)].extend(self.key_groups[self.features[idx]['key']])
        
        # Calculate sizes of each split
        split_sizes = [len(split) for split in splits]
        
        # Calculate imbalance (using coefficient of variation)
        mean_size = np.mean(split_sizes)
        std_size = np.std(split_sizes)
        imbalance = std_size / mean_size if mean_size > 0 else float('inf')
        
        return -imbalance  # Negative because we want to maximize objective

class RFSplitOptimizer(SplitOptimizer):
    def optimize(self):
        # Prepare training data
        X = np.array([[f['size'], f['avg_freq'], f['total_freq'], f['freq_range']] 
                      for f in self.features])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use K-means for initial clustering
        kmeans = KMeans(n_clusters=self.num_splits, random_state=42)
        initial_clusters = kmeans.fit_predict(X_scaled)
        
        # Use random forest to predict optimal assignment
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        
        X_train = []
        y_train = []
        
        # Generate more training samples through random perturbation
        n_samples = 1000
        for _ in range(n_samples):
            # Randomly perturb current assignment
            perturbed_assignment = initial_clusters.copy()
            n_changes = np.random.randint(1, len(initial_clusters) // 4)
            indices = np.random.choice(len(initial_clusters), n_changes, replace=False)
            perturbed_assignment[indices] = np.random.randint(0, self.num_splits, n_changes)
            
            # Evaluate perturbed assignment
            score = self._evaluate_split(perturbed_assignment)
            
            # Add to training data
            X_train.append(X_scaled)
            y_train.append(score)
        
        X_train = np.vstack(X_train)
        y_train = np.array(y_train)
        
        rf.fit(X_train, y_train)
        
        # Use random forest to predict final assignment
        predictions = rf.predict(X_scaled)
        final_clusters = np.array([int(p * self.num_splits) % self.num_splits for p in predictions])
        
        # Build final split result
        splits = [[] for _ in range(self.num_splits)]
        for idx, cluster in enumerate(final_clusters):
            splits[cluster].extend(self.key_groups[self.features[idx]['key']])
        
        return splits

class BayesianSplitOptimizer(SplitOptimizer):
    def optimize(self):
        def objective(x):
            # Convert continuous values to discrete assignments
            assignment = np.array([int(val * self.num_splits) % self.num_splits 
                                 for val in x])
            return self._evaluate_split(assignment)
        
        def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
            """Calculate expected improvement"""
            mu, sigma = gpr.predict(X.reshape(-1, 1), return_std=True)
            sigma = sigma.reshape(-1, 1)
            
            mu_sample_opt = np.max(Y_sample)
            
            with np.errstate(divide='warn'):
                imp = mu - mu_sample_opt - xi
                Z = imp / sigma
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0
                
            return ei
        
        # Initial sampling points
        n_initial = min(10, len(self.features))
        X_sample = np.random.rand(n_initial, len(self.features))
        Y_sample = np.array([objective(x) for x in X_sample])
        
        # Bayesian optimization iterations
        n_iterations = 50
        for i in range(n_iterations):
            # Train Gaussian process regressor
            gpr = GaussianProcessRegressor(n_estimators=100, random_state=42)
            gpr.fit(X_sample, Y_sample)
            
            # Find next sampling point
            x_next = None
            ei_max = -float('inf')
            
            # Random search for candidate points
            n_candidates = 1000
            X_candidates = np.random.rand(n_candidates, len(self.features))
            
            for x in X_candidates:
                ei = expected_improvement(x, X_sample, Y_sample, gpr)
                if ei > ei_max:
                    ei_max = ei
                    x_next = x
            
            # Evaluate new point
            y_next = objective(x_next)
            
            # Update sampling points
            X_sample = np.vstack((X_sample, x_next))
            Y_sample = np.append(Y_sample, y_next)
        
        # Find optimal solution
        best_idx = np.argmax(Y_sample)
        best_assignment = np.array([int(val * self.num_splits) % self.num_splits 
                                  for val in X_sample[best_idx]])
        
        # Build final split result
        splits = [[] for _ in range(self.num_splits)]
        for idx, assignment in enumerate(best_assignment):
            splits[assignment].extend(self.key_groups[self.features[idx]['key']])
        
        return splits
    
def split_contents(contents: List[Tuple[Any, int]], num_splits: int, split_strategy: str = 'naive') -> List[List[Tuple[Any, int]]]:
    """
    Split sorted contents into num_splits groups while keeping same keys together.
    
    Args:
        contents: List of tuples (key, frequency)
        num_splits: Number of splits desired
        split_strategy: Strategy to use for splitting ('naive', 'random', 'heuristic', 'ml')
    
    Returns:
        List of lists containing the split contents
    """
    if not contents:
        return [[] for _ in range(num_splits)]

    if split_strategy == 'naive':
        return naive_split(contents, num_splits)
    elif split_strategy == 'random':
        return random_split(contents, num_splits)
    elif split_strategy == 'heuristic':
        return heuristic_split(contents, num_splits)
    elif split_strategy == 'random-forest':
        optimizer = RFSplitOptimizer(contents, num_splits)
        return optimizer.optimize()
    elif split_strategy == 'bayesian':
        optimizer = BayesianSplitOptimizer(contents, num_splits)
        return optimizer.optimize()
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")

def naive_split(contents: List[Tuple[Any, int]], num_splits: int) -> List[List[Tuple[Any, int]]]:
    """Simple round-robin distribution of same-key groups"""
    # Group by key
    key_groups = defaultdict(list)
    for item in contents:
        key_groups[item[0]].append(item)
    
    # Initialize splits
    splits = [[] for _ in range(num_splits)]
    
    # Distribute groups to minimize size difference
    current_sizes = [0] * num_splits
    
    for key, group in key_groups.items():
        # Find split with minimum current size
        min_split = current_sizes.index(min(current_sizes))
        splits[min_split].extend(group)
        current_sizes[min_split] += len(group)
    
    return splits

def random_split(contents: List[Tuple[Any, int]], num_splits: int) -> List[List[Tuple[Any, int]]]:
    """Randomly assign same-key groups to splits"""
    # Group by key
    key_groups = defaultdict(list)
    for item in contents:
        key_groups[item[0]].append(item)
    
    # Initialize splits
    splits = [[] for _ in range(num_splits)]
    
    # Randomly assign groups to splits
    keys = list(key_groups.keys())
    random.shuffle(keys)
    
    current_sizes = [0] * num_splits
    for key in keys:
        group = key_groups[key]
        # Choose split with probability inversely proportional to current size
        weights = [1.0/(size + 1) for size in current_sizes]
        total = sum(weights)
        weights = [w/total for w in weights]
        
        chosen_split = random.choices(range(num_splits), weights=weights)[0]
        splits[chosen_split].extend(group)
        current_sizes[chosen_split] += len(group)
    
    return splits

def heuristic_split(contents: List[Tuple[Any, int]], num_splits: int) -> List[List[Tuple[Any, int]]]:
    """Use a greedy bin-packing approach"""
    # Group by key and calculate group sizes
    key_groups = defaultdict(list)
    group_sizes = {}
    for item in contents:
        key_groups[item[0]].append(item)
        group_sizes[item[0]] = group_sizes.get(item[0], 0) + 1
    
    # Sort groups by size in descending order
    sorted_keys = sorted(group_sizes.keys(), key=lambda k: group_sizes[k], reverse=True)
    
    # Initialize splits
    splits = [[] for _ in range(num_splits)]
    current_sizes = [0] * num_splits
    
    # First Fit Decreasing algorithm
    for key in sorted_keys:
        group = key_groups[key]
        # Find the split with minimum current size
        min_split = current_sizes.index(min(current_sizes))
        splits[min_split].extend(group)
        current_sizes[min_split] += len(group)
    
    return splits