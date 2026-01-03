import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
import os

from sklearn.metrics.pairwise import haversine_distances


def geographical_distance(x=None, to_rad=True):
    _AVG_EARTH_RADIUS_KM = 6371.0088

    # Extract values of X if it is a DataFrame, else assume it is 2-dim array of lat-lon pairs
    latlon_pairs = x.values if isinstance(x, pd.DataFrame) else x

    # If the input values are in degrees, convert them in radians
    if to_rad:
        latlon_pairs = np.vectorize(np.radians)(latlon_pairs)

    distances = haversine_distances(latlon_pairs) * _AVG_EARTH_RADIUS_KM

    # Cast response
    if isinstance(x, pd.DataFrame):
        res = pd.DataFrame(distances, x.index, x.index)
    else:
        res = distances

    return res


def thresholded_gaussian_kernel(x, theta=None, threshold=None, threshold_on_input=False):
    if theta is None:
        theta = np.std(x)
    weights = np.exp(-np.square(x / theta))
    if threshold is not None:
        mask = x > threshold if threshold_on_input else weights < threshold
        weights[mask] = 0.
    return weights


def get_similarity_AQI(dist, thr=0.1, include_self=False, force_symmetric=False, sparse=False):
    theta = np.std(dist[:36, :36])  # use same theta for both air and air36
    adj = thresholded_gaussian_kernel(dist, theta=theta, threshold=thr)
    if not include_self:
        adj[np.diag_indices_from(adj)] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj                         


def get_adj_AQI36():
    df = pd.read_csv("./data/pm25/SampleData/pm25_latlng.txt")
    df = df[['latitude', 'longitude']]
    res = geographical_distance(df, to_rad=False).values
    adj = get_similarity_AQI(res)
    return adj
    

def get_similarity_metrla(thr=0.1, force_symmetric=False, sparse=False):
    dist = np.load('./data/metr_la/metr_la_dist.npy')
    finite_dist = dist.reshape(-1)
    finite_dist = finite_dist[~np.isinf(finite_dist)]
    sigma = finite_dist.std()
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < thr] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj



def get_similarity_pemsbay(thr=0.1, force_symmetric=False, sparse=False):
    dist = np.load('./data/pems_bay/pems_bay_dist.npy')
    finite_dist = dist.reshape(-1)
    finite_dist = finite_dist[~np.isinf(finite_dist)] 
    sigma = finite_dist.std()
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < thr] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj


def get_similarity_pems08(thr=0.1, force_symmetric=False, sparse=False, save_dist=True):
    """Compute similarity adjacency for PEMS08 dataset.
    """
    dist_path = './data/pems08/pems08_dist.npy'
    # determine desired node count from data (if possible)
    try:
        data_arr = np.load('./data/pems08/PEMS08.npz')
        data_N = int(data_arr['data'].shape[1])
    except Exception:
        data_N = None

    need_compute = True
    if os.path.exists(dist_path):
        dist = np.load(dist_path)
        if data_N is None or dist.shape[0] == data_N:
            need_compute = False
        else:
            # existing dist has wrong size -> recompute and overwrite
            need_compute = True

    if need_compute:
        # read edge list and construct graph
        df = pd.read_csv('./data/pems08/PEMS08.csv')
        if not {'from', 'to', 'cost'}.issubset(df.columns):
            raise ValueError("PEMS08.csv must contain columns 'from','to','cost'")
        nodes = np.unique(np.concatenate([df['from'].values, df['to'].values]))
        max_node = int(nodes.max())
        if data_N is None:
            N = max_node
        else:
            N = max(max_node, data_N)  # ensure we cover all nodes from data and edges

        # initialize adjacency with infinities
        mat = np.full((N, N), np.inf, dtype=float)
        for i in range(N):
            mat[i, i] = 0.0
        for _, row in df.iterrows():
            a = int(row['from']) - 1
            b = int(row['to']) - 1
            cost = float(row['cost'])
            if 0 <= a < N and 0 <= b < N:
                mat[a, b] = cost
                mat[b, a] = cost
        # compute shortest paths
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path
        dist = shortest_path(csgraph=csr_matrix(mat), directed=False, unweighted=False)
        if save_dist:
            os.makedirs(os.path.dirname(dist_path), exist_ok=True)
            np.save(dist_path, dist)

    finite_dist = dist.reshape(-1)
    finite_dist = finite_dist[~np.isinf(finite_dist)]
    sigma = finite_dist.std()
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < thr] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj


def get_similarity_solar(
    dist,
    thr: float = 0.1,
    include_self: bool = False,
    force_symmetric: bool = False,
    sparse: bool = False,
):
    if isinstance(dist, pd.DataFrame):
        dist_np = dist.to_numpy(dtype=float)
    else:
        dist_np = np.asarray(dist, dtype=float)

    finite = dist_np[np.isfinite(dist_np) & (dist_np > 0)]
    theta = float(np.std(finite)) if finite.size else 1.0
    if not np.isfinite(theta) or theta <= 0:
        theta = 1.0

    adj = thresholded_gaussian_kernel(dist_np, theta=theta, threshold=thr)

    if not include_self:
        np.fill_diagonal(adj, 0.0)
    if force_symmetric:
        adj = np.maximum(adj, adj.T)
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)

    return adj.astype(np.float32)
def get_adj_solar(npz_path: str = "./data/solar/data_&_nodeMetaData.npz", thr: float = 0.1):
    data = np.load(npz_path, allow_pickle=True)
    latlon = data["latlon"]      # (N,2) degrees
    node_ids = data["node_ids"]  # (N,)

    latlon_df = pd.DataFrame(latlon, index=node_ids, columns=["lat", "lon"])
    dist_km = geographical_distance(latlon_df, to_rad=True)

    adj = get_similarity_solar(dist_km, thr=thr, include_self=False, force_symmetric=True)

    return adj
# in Graph-wavenet
def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def compute_support_gwn(adj, device=None):
    adj_mx = [asym_adj(adj), asym_adj(np.transpose(adj))]
    support = [torch.tensor(i).to(device) for i in adj_mx]
    return support


