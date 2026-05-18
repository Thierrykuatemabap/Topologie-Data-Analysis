"""
feature_gen.py
==============
Topological & Geometrical Feature Generator for molecular graphs.

Feature families (ALL purely topological/geometrical — no chemical descriptors):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 1. Persistent Entropy       H0, H1                             →  2 features
 2. Amplitude – Wasserstein  H0, H1                             →  2 features
 3. Amplitude – Bottleneck   H0, H1                             →  2 features
 4. Amplitude – Landscape    H0, H1                             →  2 features
 5. Betti Curves             n_bins × 2 dims  (n_bins=20)       → 40 features
 6. Carlsson Coordinates     C1…C4 × H0, H1                     →  8 features
 7. Spectral – L0 (0-Laplacian)
       • 5 smallest non-trivial eigenvalues                      →  5 features
       • algebraic connectivity (λ₂)                            →  1 feature
       • spectral range (λmax − λ₂)                             →  1 feature
 8. Spectral – L1 (1-Laplacian / Hodge)
       • 5 smallest eigenvalues                                  →  5 features
 9. Graph Statistics
       mean_degree, max_degree, min_degree, std_degree,
       transitivity, avg_clustering, diameter,
       avg_path_length, density, cycle_rank                      → 10 features
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL  (n_bins=20, n_spec=5)                                     → 78 features
                                                                   ≤ 120  ✓
"""

import numpy as np
import networkx as nx
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Giotto-TDA (optional – zeros returned if unavailable)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceEntropy, Amplitude, BettiCurve
    GTDA_AVAILABLE = True
except ImportError:
    GTDA_AVAILABLE = False
    print("[feature_gen] WARNING: giotto-tda not found → TDA features will be 0.")


###############################################################################
# 1.  PERSISTENT HOMOLOGY  (giotto-tda wrapper)
###############################################################################
class PersistentHomologyExtractor:
    """
    Compute Vietoris-Rips persistence on the *distance* matrix derived from
    the weighted adjacency matrix  (distance = 1 / weight).

    Extracts:
        • Persistent entropy       (H0, H1)
        • Amplitude – Wasserstein  (H0, H1)
        • Amplitude – Bottleneck   (H0, H1)
        • Amplitude – Landscape    (H0, H1)
        • Betti curves             (n_bins × 2)
        • Raw persistence diagrams (for Carlsson coordinates)
    """

    def __init__(self, homology_dimensions=(0, 1), n_bins=20, n_jobs=1):
        self.homology_dimensions = homology_dimensions
        self.n_bins  = n_bins
        self.n_jobs  = n_jobs
        self._fitted = False   # track whether persistence has been fit

        if GTDA_AVAILABLE:
            self._build_transformers()

    # ── private ──────────────────────────────────────────────────────────────
    def _build_transformers(self):
        self.vr = VietorisRipsPersistence(
            metric='precomputed',
            homology_dimensions=self.homology_dimensions,
            collapse_edges=True,
            n_jobs=self.n_jobs
        )
        self.entropy = PersistenceEntropy(n_jobs=self.n_jobs)
        self.amp_wass = Amplitude(metric='wasserstein',
                                  metric_params={'p': 1}, n_jobs=self.n_jobs)
        self.amp_bot  = Amplitude(metric='bottleneck', n_jobs=self.n_jobs)
        self.amp_land = Amplitude(metric='landscape',
                                  metric_params={'n_layers': 1,
                                                 'n_bins': self.n_bins},
                                  n_jobs=self.n_jobs)
        self.betti    = BettiCurve(n_bins=self.n_bins, n_jobs=self.n_jobs)

    @staticmethod
    def _adj_to_dist(adj):
        """Weighted adjacency  →  distance matrix  (d = 1/w, self-loops = 0)."""
        W = adj.copy().astype(float)
        np.fill_diagonal(W, 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            D = np.where(W > 0, 1.0 / W, 1e6)
        D = np.maximum(D, D.T)
        np.fill_diagonal(D, 0)
        return D

    def _zeros(self, n):
        nd = len(self.homology_dimensions)
        return {
            'diagrams'  : [np.zeros((0, 3))] * n,
            'entropy'   : np.zeros((n, nd)),
            'amp_wass'  : np.zeros((n, nd)),
            'amp_bot'   : np.zeros((n, nd)),
            'amp_land'  : np.zeros((n, nd)),
            'betti'     : np.zeros((n, nd * self.n_bins)),
        }

    # ── public ───────────────────────────────────────────────────────────────
    def fit_transform(self, adjacency_matrices):
        """Batch extraction.  Returns a dict of arrays indexed by sample."""
        n = len(adjacency_matrices)
        out = self._zeros(n)
        if not GTDA_AVAILABLE or n == 0:
            return out

        dist_mats = np.array([self._adj_to_dist(a) for a in adjacency_matrices])
        try:
            diags = self.vr.fit_transform(dist_mats)
            self._fitted = True
            out['diagrams'] = diags                               # raw diagrams
            out['entropy']  = self.entropy.fit_transform(diags)

            for amp, key in [(self.amp_wass, 'amp_wass'),
                             (self.amp_bot,  'amp_bot'),
                             (self.amp_land, 'amp_land')]:
                try:
                    out[key] = amp.fit_transform(diags)
                except Exception:
                    pass   # leave zeros

            bc = self.betti.fit_transform(diags)
            out['betti'] = bc.reshape(n, -1)
        except Exception as e:
            print(f"[PH] extraction error: {e}")
        return out


###############################################################################
# 2.  CARLSSON COORDINATES
###############################################################################
class CarlssonCoordinates:
    """
    Four coordinates per homology dimension derived from a persistence diagram
    D = {(b_i, d_i)}.

    Let  w_i = d_i − b_i  (persistence / lifetime),  T = max finite death.

        C1 = Σ w_i²
        C2 = Σ w_i² · b_i
        C3 = Σ w_i² · (T − d_i)
        C4 = Σ w_i² · b_i · (T − d_i)

    These capture total mass, birth-centredness, death-centredness and
    joint birth-death spread of the diagram.

    Total: 4 × n_dims  features.
    """

    def __init__(self, homology_dimensions=(0, 1)):
        self.homology_dimensions = homology_dimensions

    @property
    def n_features(self):
        return 4 * len(self.homology_dimensions)

    def _four_coords(self, pts):
        """pts : array (k, 2)  columns = [birth, death]  (finite entries only)."""
        if len(pts) == 0:
            return np.zeros(4)
        b, d = pts[:, 0], pts[:, 1]
        w    = np.clip(d - b, 0, None)
        T    = float(np.max(d)) if len(d) > 0 else 1.0
        if T == 0:
            T = 1.0
        w2 = w ** 2
        return np.array([
            np.sum(w2),
            np.sum(w2 * b),
            np.sum(w2 * (T - d)),
            np.sum(w2 * b * (T - d)),
        ], dtype=float)

    def extract(self, diagram):
        """
        diagram : np.ndarray  shape (n_pts, 3)  columns = [birth, death, dim]
                  (giotto-tda format).
        Returns : np.ndarray  shape (n_features,)
        """
        feats = []
        for dim in self.homology_dimensions:
            mask = diagram[:, 2] == dim
            sub  = diagram[mask]
            if len(sub) == 0:
                feats.extend(np.zeros(4))
            else:
                finite = sub[np.isfinite(sub[:, 1])]
                feats.extend(self._four_coords(finite[:, :2]))
        return np.array(feats, dtype=float)

    def feature_names(self):
        return [f'carlsson_C{c}_H{d}'
                for d in self.homology_dimensions
                for c in range(1, 5)]


###############################################################################
# 3.  SPECTRAL FEATURES
###############################################################################
class SpectralFeatureExtractor:
    """
    Spectral features from the weighted 0-Laplacian and 1-Laplacian.

    ── 0-Laplacian  L0 = D − W ──────────────────────────────────────────────
        D = degree matrix (row sums of off-diagonal W)
        W = weighted adjacency (diagonal zeroed)

        Features extracted:
            • n_spec smallest non-trivial eigenvalues  (λ₂ … λ_{n_spec+1})
            • algebraic connectivity  =  λ₂
            • spectral range          =  λmax − λ₂

    ── 1-Laplacian  L1 = B₁ᵀ L0 B₁  (Hodge / edge Laplacian) ──────────────
        B₁ = signed node-edge incidence matrix  (n_nodes × n_edges)
        Each edge (i,j) with i<j:  B₁[i,e] = −1,  B₁[j,e] = +1

        Features extracted:
            • n_spec smallest eigenvalues of L1
              (captures the independent cycle structure of the graph)

    Total: n_spec + 1 + 1 + n_spec  =  2·n_spec + 2  features.
    For n_spec=5 → 12 features.
    """

    def __init__(self, n_spec=5):
        self.n_spec = n_spec

    @property
    def n_features(self):
        return 2 * self.n_spec + 2

    # ── helpers ───────────────────────────────────────────────────────────────
    def _L0(self, adj):
        W = adj.copy().astype(float)
        np.fill_diagonal(W, 0)
        return np.diag(W.sum(axis=1)) - W

    def _B1(self, adj):
        """Signed incidence matrix  B1  (n_nodes × n_edges)."""
        W = adj.copy().astype(float)
        np.fill_diagonal(W, 0)
        rows, cols = np.where(np.triu(W > 0, k=1))
        n = W.shape[0]
        if len(rows) == 0:
            return np.zeros((n, 0))
        B = np.zeros((n, len(rows)))
        for e, (i, j) in enumerate(zip(rows, cols)):
            B[i, e] = -1.0
            B[j, e] =  1.0
        return B

    def _eigvals(self, M, k):
        """k smallest real eigenvalues of symmetric M (padded with 0 if needed)."""
        n = M.shape[0]
        if n == 0 or k == 0:
            return np.zeros(k)
        k_use = min(k, n)
        try:
            # subset_by_index is 0-based inclusive
            vals = eigh(M, eigvals_only=True,
                        subset_by_index=[0, k_use - 1])
            vals = np.sort(np.real(vals))
        except Exception:
            vals = np.zeros(k_use)
        # pad or trim to exactly k
        if len(vals) < k:
            vals = np.pad(vals, (0, k - len(vals)))
        return vals[:k]

    # ── public ────────────────────────────────────────────────────────────────
    def extract(self, adj):
        """Returns np.ndarray of shape (2·n_spec + 2,)."""
        feats = []

        # ── 0-Laplacian ──────────────────────────────────────────────────────
        L0   = self._L0(adj)
        n    = L0.shape[0]

        # Retrieve n_spec + 1 smallest eigs (the first ≈0 will be skipped)
        all_small = self._eigvals(L0, self.n_spec + 1)
        non_trivial = all_small[1:]                          # skip λ1 ≈ 0
        if len(non_trivial) < self.n_spec:
            non_trivial = np.pad(non_trivial,
                                 (0, self.n_spec - len(non_trivial)))
        feats.extend(non_trivial[:self.n_spec])              # n_spec values

        alg_conn = float(non_trivial[0]) if len(non_trivial) > 0 else 0.0
        feats.append(alg_conn)

        # Spectral range: need λmax → retrieve all eigs (small matrix, cheap)
        all_eigs = self._eigvals(L0, n)
        lmax     = float(all_eigs[-1]) if len(all_eigs) > 0 else 0.0
        feats.append(lmax - alg_conn)                        # spectral range

        # ── 1-Laplacian ──────────────────────────────────────────────────────
        B1 = self._B1(adj)
        if B1.shape[1] > 0:
            L1      = B1.T @ L0 @ B1
            eigs_L1 = self._eigvals(L1, self.n_spec)
        else:
            eigs_L1 = np.zeros(self.n_spec)
        feats.extend(eigs_L1)

        return np.array(feats, dtype=float)

    def feature_names(self):
        names  = [f'L0_eig_{i+2}' for i in range(self.n_spec)]   # λ₂…λ_{n+1}
        names += ['algebraic_connectivity', 'spectral_range']
        names += [f'L1_eig_{i+1}' for i in range(self.n_spec)]
        return names


###############################################################################
# 4.  GRAPH STATISTICS
###############################################################################
class GraphStatisticsExtractor:
    """
    Ten classical graph-theoretic statistics (all purely topological /
    geometrical — no chemical information used).

    ┌──────────────────────┬────────────────────────────────────────────────┐
    │ Feature              │ Mathematical definition                        │
    ├──────────────────────┼────────────────────────────────────────────────┤
    │ mean_degree          │ (1/n) Σ deg(v)                                 │
    │ max_degree           │ max_v deg(v)                                   │
    │ min_degree           │ min_v deg(v)                                   │
    │ std_degree           │ std deviation of degree sequence               │
    │ transitivity         │ 3·triangles / triads  (global clustering)      │
    │ avg_clustering       │ (1/n) Σ c(v)  (local clustering avg)          │
    │ diameter             │ max_{u,v} d(u,v)  on largest connected comp.  │
    │ avg_path_length      │ mean shortest-path length (LCC)                │
    │ density              │ 2|E| / n(n−1)                                  │
    │ cycle_rank           │ |E| − |V| + #components  (1st Betti number)   │
    └──────────────────────┴────────────────────────────────────────────────┘
    Total: 10 features.
    """

    @property
    def n_features(self):
        return 10

    def extract(self, adj):
        W = adj.copy().astype(float)
        np.fill_diagonal(W, 0)
        G = nx.from_numpy_array(W)
        # Remove truly absent edges (weight == 0 after zeroing diagonal)
        zero_edges = [(u, v) for u, v, d in G.edges(data=True)
                      if d.get('weight', 0) == 0]
        G.remove_edges_from(zero_edges)

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        # Degree sequence (unweighted)
        degs = np.array([d for _, d in G.degree()], dtype=float)
        if len(degs) == 0:
            degs = np.zeros(1)

        mean_deg = float(np.mean(degs))
        max_deg  = float(np.max(degs))
        min_deg  = float(np.min(degs))
        std_deg  = float(np.std(degs))

        transitivity   = float(nx.transitivity(G))
        avg_clustering = float(nx.average_clustering(G))

        # Diameter & avg path length on the largest connected component
        diameter = avg_path = 0.0
        if n_nodes > 1:
            lcc = G.subgraph(
                max(nx.connected_components(G), key=len)
            ).copy()
            if lcc.number_of_nodes() > 1:
                try:
                    diameter = float(nx.diameter(lcc))
                    avg_path = float(nx.average_shortest_path_length(lcc))
                except Exception:
                    pass

        density    = float(nx.density(G))
        n_comp     = nx.number_connected_components(G)
        cycle_rank = float(max(0, n_edges - n_nodes + n_comp))

        return np.array([mean_deg, max_deg, min_deg, std_deg,
                         transitivity, avg_clustering,
                         diameter, avg_path,
                         density, cycle_rank], dtype=float)

    def feature_names(self):
        return ['mean_degree', 'max_degree', 'min_degree', 'std_degree',
                'transitivity', 'avg_clustering',
                'diameter', 'avg_path_length',
                'density', 'cycle_rank']


###############################################################################
# 5.  MAIN FEATURE GENERATOR
###############################################################################
class DiscriminativeFeatureGenerator:
    """
    Unified feature generator — combines all four topological / geometrical
    families.  A single StandardScaler is fitted on the *training* set and
    applied to subsequent sets via  transform().

    Parameters
    ----------
    homology_dimensions : tuple
        Homology dimensions for TDA (default: (0, 1)).
    n_bins : int
        Number of bins for Betti curves (default: 20).
    n_spec : int
        Number of Laplacian eigenvalues to keep per matrix (default: 5).
    n_jobs : int
        Parallelism for giotto-tda (default: 1).

    Feature count  (defaults: n_bins=20, n_spec=5, dims=(0,1)):
    ┌──────────────────────────────┬──────────┐
    │ Family                       │ # feats  │
    ├──────────────────────────────┼──────────┤
    │ Persistent entropy           │    2     │
    │ Amplitude Wasserstein        │    2     │
    │ Amplitude Bottleneck         │    2     │
    │ Amplitude Landscape          │    2     │
    │ Betti curves  (20×2)         │   40     │
    │ Carlsson coordinates (4×2)   │    8     │
    │ Spectral L0+L1 (2×5+2)       │   12     │
    │ Graph statistics             │   10     │
    ├──────────────────────────────┼──────────┤
    │ TOTAL                        │   78  ✓  │
    └──────────────────────────────┴──────────┘
    """

    def __init__(self, homology_dimensions=(0, 1), n_bins=20,
                 n_spec=5, n_jobs=1,
                 # legacy alias kept for backward-compat
                 n_laplacian_eigenvalues=None):
        self.homology_dimensions = homology_dimensions
        self.n_bins = n_bins
        self.n_spec = n_spec if n_laplacian_eigenvalues is None \
                      else n_laplacian_eigenvalues

        self.scaler     = StandardScaler()
        self.ph         = PersistentHomologyExtractor(homology_dimensions,
                                                      n_bins, n_jobs)
        self.carlsson   = CarlssonCoordinates(homology_dimensions)
        self.spectral   = SpectralFeatureExtractor(self.n_spec)
        self.graph_stat = GraphStatisticsExtractor()

        self.feature_dim   = None
        self.feature_names_list = []

    # ── private ──────────────────────────────────────────────────────────────
    def _vector(self, adj, ph_out, idx):
        """Assemble the feature vector for sample idx."""
        f = []

        # TDA features
        f.extend(ph_out['entropy'][idx])
        f.extend(ph_out['amp_wass'][idx])
        f.extend(ph_out['amp_bot'][idx])
        f.extend(ph_out['amp_land'][idx])
        f.extend(ph_out['betti'][idx])

        # Carlsson coordinates
        diag = ph_out['diagrams'][idx]
        if isinstance(diag, np.ndarray) and diag.ndim == 2 and diag.shape[1] == 3:
            f.extend(self.carlsson.extract(diag))
        else:
            f.extend(np.zeros(self.carlsson.n_features))

        # Spectral features
        f.extend(self.spectral.extract(adj))

        # Graph statistics
        f.extend(self.graph_stat.extract(adj))

        return np.array(f, dtype=float)

    def _build_names(self):
        names = []
        for d in self.homology_dimensions:
            names.append(f'persistent_entropy_H{d}')
        for metric in ['wasserstein', 'bottleneck', 'landscape']:
            for d in self.homology_dimensions:
                names.append(f'amplitude_{metric}_H{d}')
        for d in self.homology_dimensions:
            for b in range(self.n_bins):
                names.append(f'betti_H{d}_bin{b}')
        names.extend(self.carlsson.feature_names())
        names.extend(self.spectral.feature_names())
        names.extend(self.graph_stat.feature_names())
        return names

    def _pad_or_trim(self, fv):
        if len(fv) < self.feature_dim:
            return np.pad(fv, (0, self.feature_dim - len(fv)))
        return fv[:self.feature_dim]

    # ── public ───────────────────────────────────────────────────────────────
    def generate_features(self, adjacency_matrices, labels):
        """
        Raw (unscaled) feature extraction for a list of adjacency matrices.

        Returns
        -------
        X : np.ndarray  (n_samples, n_features)
        y : np.ndarray  (n_samples,)
        feature_info : dict
        """
        n = len(adjacency_matrices)
        print(f"  Extracting TDA features for {n} molecules …")
        ph_out = self.ph.fit_transform(adjacency_matrices)

        X = []
        for i, adj in enumerate(adjacency_matrices):
            if i % 300 == 0:
                print(f"    [{i+1}/{n}]")
            fv = self._vector(adj, ph_out, i)

            if self.feature_dim is None:
                self.feature_dim        = len(fv)
                self.feature_names_list = self._build_names()
                print_feature_breakdown(self.homology_dimensions,
                                        self.n_bins, self.n_spec)

            X.append(self._pad_or_trim(fv))

        X = np.array(X, dtype=float)
        y = np.array(labels)
        info = {'feature_names': self.feature_names_list,
                'n_features'   : self.feature_dim}
        # Replace any remaining NaN/Inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"  Feature matrix: {X.shape}")
        return X, y, info

    def fit_transform(self, adjacency_matrices, labels):
        """
        Extract features AND fit the scaler on this data.
        Call ONLY on the training set.
        """
        X, y, info = self.generate_features(adjacency_matrices, labels)
        if len(X) == 0:
            return None, None, None
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y, info

    def transform(self, adjacency_matrices, labels=None):
        """
        Extract features and apply the already-fitted scaler.
        Call on the test / validation set AFTER fit_transform on train.
        """
        dummy = labels if labels is not None else [0] * len(adjacency_matrices)
        X, y, info = self.generate_features(adjacency_matrices, dummy)
        X_scaled   = self.scaler.transform(X)
        return X_scaled, y, info


###############################################################################
# UTILITY
###############################################################################
def print_feature_breakdown(homology_dimensions=(0, 1), n_bins=20, n_spec=5):
    nd = len(homology_dimensions)
    rows = [
        ('Persistent entropy',       nd),
        ('Amplitude – Wasserstein',  nd),
        ('Amplitude – Bottleneck',   nd),
        ('Amplitude – Landscape',    nd),
        ('Betti curves',             n_bins * nd),
        ('Carlsson coordinates',     4 * nd),
        ('Spectral L0 (non-trivial)',n_spec),
        ('Spectral L0 alg+range',    2),
        ('Spectral L1',              n_spec),
        ('Graph statistics',         10),
    ]
    total = sum(v for _, v in rows)
    bar   = "=" * 48
    print(f"\n{bar}")
    print("  FEATURE BREAKDOWN")
    print(bar)
    for name, cnt in rows:
        print(f"  {name:<32} {cnt:>4}")
    print(bar)
    ok = "✓  ≤ 120" if total <= 120 else "✗  EXCEEDS 120 !"
    print(f"  {'TOTAL':<32} {total:>4}  {ok}")
    print(f"{bar}\n")
    return total
