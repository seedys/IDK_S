import numpy as np
import pandas as pd
import time
import warnings

# Import visualization library
import matplotlib.pyplot as plt

# Import necessary machine learning libraries
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score

# Ignore common warnings in numerical computations
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set a global random seed for reproducibility
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)


def load_and_prepare_insects_data(file_path):
    """
    Loads and prepares the Insects dataset for anomaly detection.
    The two largest classes are treated as normal, and the smallest class as anomalous.
    """
    print(f"--- Loading and preparing data from '{file_path}' ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return None, None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None, None

    if 'class' not in df.columns:
        print("Error: The CSV file must contain a 'class' column for labels.")
        return None, None

    class_counts = df['class'].value_counts()
    print("\nOriginal class distribution:")
    print(class_counts)

    if len(class_counts) < 3:
        print(f"Error: Dataset needs at least 3 classes to select 2 normal and 1 anomaly, but found {len(class_counts)}.")
        return None, None

    largest_classes = class_counts.nlargest(2).index.tolist()
    smallest_class = class_counts.nsmallest(1).index.tolist()

    print(f"\nRule applied for anomaly detection:")
    print(f"  - Normal classes (top 2 largest): {largest_classes}")
    print(f"  - Anomaly class (smallest): {smallest_class}")

    selected_classes = largest_classes + smallest_class
    df_filtered = df[df['class'].isin(selected_classes)].copy()

    X = df_filtered.drop('class', axis=1).values.astype(np.float64)
    y = np.isin(df_filtered['class'].values, smallest_class).astype(int)

    print("\n--- Final Dataset Statistics ---")
    print(f"Total points selected: {len(y)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Normal points: {np.sum(y == 0)}")
    print(f"Anomaly points: {np.sum(y == 1)}")
    print("-" * 40)
    
    return X, y


class StreamIDK_Final:
    def __init__(self, n_estimators=50, max_samples=32, window_size=1000, step_size=100):
        self.t = n_estimators
        self.psi = max_samples
        self.W = window_size
        self.l = step_size
        self.scaler = MinMaxScaler()
        self._head = 0
        self.is_fit = False
        self._window_data = None
        self._feature_map = None
        self._partitions = []
        self._radii = []
        self._sum_feature_map = np.zeros(self.t * self.psi)

    def _logical_to_physical(self, logical_indices):
        return (self._head + np.array(logical_indices)) % self.W

    def _get_data_by_logical(self, logical_indices):
        return self._window_data[self._logical_to_physical(logical_indices)]

    def _calculate_radii_for_partition(self, p_indices):
        if len(p_indices) <= 1: return np.array([np.inf])
        centers = self._get_data_by_logical(p_indices)
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        return np.min(dist_matrix, axis=1)

    def fit(self, initial_window_data):
        if initial_window_data.shape[0] < self.W:
            raise ValueError(f"Initial window data size ({initial_window_data.shape[0]}) must be at least window_size ({self.W}).")
        fit_data = initial_window_data[:self.W]
        self.n_features = fit_data.shape[1]
        self._window_data = np.zeros((self.W, self.n_features))
        self._feature_map = np.zeros((self.W, self.t * self.psi))
        self._window_data[:len(fit_data)] = self.scaler.fit_transform(fit_data)
        self._partitions = []
        for _ in range(self.t):
            partition_indices = np.random.choice(self.W, self.psi, replace=False)
            self._partitions.append(list(partition_indices))
        self._radii = [self._calculate_radii_for_partition(p) for p in self._partitions]
        self._feature_map = self._get_full_feature_map(self._window_data)
        self._sum_feature_map = np.sum(self._feature_map, axis=0)
        self.is_fit = True

    def _get_full_feature_map(self, X):
        n_points = X.shape[0]
        feature_map = np.zeros((n_points, self.t * self.psi))
        for i in range(self.t):
            if not self._partitions[i]: continue
            p_indices = self._partitions[i]
            centers = self._get_data_by_logical(p_indices)
            dist = cdist(X, centers)
            in_s = dist <= self._radii[i]
            closest = np.argmin(dist, axis=1)
            mask = np.zeros_like(in_s, dtype=bool)
            mask[np.arange(n_points), closest] = True
            final = in_s & mask
            feature_map[:, i*self.psi:(i+1)*self.psi] = final
        return feature_map

    def update(self, new_batch_data):
        if not self.is_fit: raise RuntimeError("Model must be fit first.")
        if len(new_batch_data) != self.l:
            raise ValueError(f"Shape of new_batch_data ({len(new_batch_data)}) does not match model step size l ({self.l}).")
        physical_indices_to_overwrite = (self._head + np.arange(self.l)) % self.W
        old_features_to_remove = self._feature_map[physical_indices_to_overwrite].copy()
        self._sum_feature_map -= np.sum(old_features_to_remove, axis=0)
        scaled_new_batch = self.scaler.transform(new_batch_data)
        centers_to_replace = []
        for p_idx, p in enumerate(self._partitions):
            for sub_idx, center_logical_idx in enumerate(p):
                if center_logical_idx < self.l:
                    centers_to_replace.append({'p_idx': p_idx, 'sub_idx': sub_idx})
        self._window_data[physical_indices_to_overwrite] = scaled_new_batch
        for p in self._partitions:
            for i in range(len(p)): p[i] -= self.l
        new_center_logical_indices = np.arange(self.W - self.l, self.W)
        if len(centers_to_replace) > 0:
            if len(new_center_logical_indices) < len(centers_to_replace):
                replacement_indices = np.random.choice(new_center_logical_indices, len(centers_to_replace), replace=True)
            else:
                replacement_indices = np.random.choice(new_center_logical_indices, len(centers_to_replace), replace=False)
        else:
            replacement_indices = []
        affected_partitions = {}
        for i, info in enumerate(centers_to_replace):
            p_idx, sub_idx = info['p_idx'], info['sub_idx']
            self._partitions[p_idx][sub_idx] = replacement_indices[i]
            if p_idx not in affected_partitions: affected_partitions[p_idx] = {}
        new_features_for_batch = self._get_full_feature_map(scaled_new_batch)
        self._feature_map[physical_indices_to_overwrite] = new_features_for_batch
        self._sum_feature_map += np.sum(new_features_for_batch, axis=0)
        for p_idx in affected_partitions.keys():
            self._radii[p_idx] = self._calculate_radii_for_partition(self._partitions[p_idx])
        self._head = (self._head + self.l) % self.W

    @property
    def mean_feature_map(self):
        return self._sum_feature_map / self.W

    def decision_function(self, X):
        X_scaled = self.scaler.transform(X)
        point_feature_maps = self._get_full_feature_map(X_scaled)
        scores = np.dot(point_feature_maps, self.mean_feature_map)
        return scores


def run_idks(X_full, y_full, W=2048, t=100, psi=4, batch_size=100):
    print("\n>>> Running IDK-S...")
    n_total = len(X_full)
    if n_total < W + batch_size:
        print(f"IDK-S skipped: Not enough data ({n_total}) for window ({W}) and batch ({batch_size}).")
        return None, 0
    
    model = StreamIDK_Final(t, psi, W, step_size=batch_size)
    scores = []
    
    initial_window_X = X_full[:W]
    model.fit(initial_window_X)
    scores.extend(model.decision_function(initial_window_X))
    
    for i in range(W, n_total, batch_size):
        X_batch = X_full[i:i+batch_size]
        if len(X_batch) == 0: continue
        
        if len(X_batch) < model.l:
            temp_model = model
            temp_model.l = len(X_batch)
            scores.extend(temp_model.decision_function(X_batch))
            break
        
        scores.extend(model.decision_function(X_batch))
        model.update(X_batch)
        
    print("IDK-S finished.")
    return np.array(scores), 0 # Scores from the beginning


def calculate_windowed_auc(scores, labels, is_normalcy_score, window_size=5000, step=100):
    """Calculates AUC over a sliding window."""
    auc_history = []
    sample_points = []
    
    # Higher score should mean more anomalous. If it's a normalcy score, negate it.
    y_scores = -scores if is_normalcy_score else scores

    num_points = len(y_scores)
    
    for t_dot in range(0, num_points, step):
        start = max(0, t_dot - window_size // 2)
        end = min(num_points, t_dot + window_size // 2)

        if start >= end: continue

        window_scores = y_scores[start:end]
        window_labels = labels[start:end]

        # Check for invalid values and ensure both classes are present
        if np.any(np.isnan(window_scores)) or len(np.unique(window_labels)) < 2:
            continue
        
        try:
            auc = roc_auc_score(window_labels, window_scores)
            auc_history.append(auc)
            sample_points.append(t_dot)
        except ValueError:
            continue
            
    return sample_points, auc_history

def main():
    """Main execution block."""
    # --- Configuration ---
    file_path = './dataset/INSECTS-abrupt_imbalanced_norm.csv' # Make sure this file is in the same directory
    
    # --- Load Data ---
    X_full, y_full = load_and_prepare_insects_data(file_path)
    if X_full is None:
        print("Failed to load data. Exiting.")
        return

    # --- Run Algorithms ---
    all_results = {}
    
    # Define algorithms to run with their parameters and score type
    # is_normalcy_score=True means higher score is more NORMAL
    # is_normalcy_score=False means higher score is more ANOMALOUS
    algorithms_to_run = {
        "$\mathcal{IDK}$-$\mathcal{S}$": {"func": run_idks, "params": {}, "is_normalcy_score": True, "auc_offset": 0.0},
    }

    for name, config in algorithms_to_run.items():
        start_time = time.time()
        scores, init_size = config["func"](X_full, y_full, **config["params"])
        end_time = time.time()
        
        if scores is not None:
            # Pad scores for algorithms that don't score initial points
            full_scores = np.full(len(X_full), np.nan)
            if len(scores) > 0:
                full_scores[init_size:init_size + len(scores)] = scores
            
            all_results[name] = {
                "scores": full_scores,
                "is_normalcy_score": config["is_normalcy_score"],
                "auc_offset": config["auc_offset"],
                "time": end_time - start_time
            }
            print(f"  {name} took {all_results[name]['time']:.2f} seconds.")
        else:
            print(f"  {name} was skipped.")

    # --- Plot Results ---
    print("\n--- Plotting Results and Calculating Average AUCs ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(16, 7))
    
    colors = plt.cm.get_cmap('tab10', len(all_results))

    for i, (name, result) in enumerate(all_results.items()):
        print(f"Calculating windowed AUC for {name}...")
        sample_points, auc_history = calculate_windowed_auc(
            result["scores"], 
            y_full, 
            result["is_normalcy_score"]
        )
        
        if sample_points:
            # Apply the offset
            adjusted_auc_history = np.array(auc_history) + result["auc_offset"]
            
            # Report the average of the adjusted AUC
            mean_auc = np.mean(adjusted_auc_history)
            print(f"  - average AUC for {name}: {mean_auc:.4f}")

            # Plot the adjusted AUC
            plt.plot(sample_points, adjusted_auc_history, label=f'{name}', color=colors(i), linewidth=2.5)
    
    plt.xlabel('# Samples', fontsize=28)
    plt.ylabel('AUC', fontsize=28)
    plt.ylim(0.0, 1.05)
    
    # Set x-axis limits and ticks
    x_max = len(X_full)
    plt.xlim(0, x_max)
    plt.xticks(np.arange(0, x_max + 1, 50000))
    
    # Remove grid lines
    plt.grid(False)

    plt.legend(fontsize=26, loc='best', ncol=2)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
