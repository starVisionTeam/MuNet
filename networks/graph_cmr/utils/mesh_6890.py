from __future__ import division
import torch
import numpy as np
import scipy.sparse

from networks.graph_cmr.models import SMPL
from networks.graph_cmr.models.graph_layers import spmm
from scipy.sparse import lil_matrix
def scipy_to_pytorch(A, U, D):
    """Convert scipy sparse matrices to pytorch sparse matrix."""
    ptU = []
    ptD = []
    
    for i in range(len(U)):
        u = scipy.sparse.coo_matrix(U[i])
        i = torch.LongTensor(np.array([u.row, u.col]))
        v = torch.FloatTensor(u.data)
        ptU.append(torch.sparse.FloatTensor(i, v, u.shape))
    
    for i in range(len(D)):
        d = scipy.sparse.coo_matrix(D[i])
        i = torch.LongTensor(np.array([d.row, d.col]))
        v = torch.FloatTensor(d.data)
        ptD.append(torch.sparse.FloatTensor(i, v, d.shape)) 

    return ptU, ptD


def adjmat_sparse(adjmat, nsize=1):
    """Create row-normalized sparse graph adjacency matrix."""
    adjmat = scipy.sparse.csr_matrix(adjmat)
    if nsize > 1:
        orig_adjmat = adjmat.copy()
        for _ in range(1, nsize):
            adjmat = adjmat * orig_adjmat
    adjmat.data = np.ones_like(adjmat.data)
    for i in range(adjmat.shape[0]):
        adjmat[i,i] = 1
    num_neighbors = np.array(1 / adjmat.sum(axis=-1))
    adjmat = adjmat.multiply(num_neighbors)
    adjmat = scipy.sparse.coo_matrix(adjmat)
    row = adjmat.row
    col = adjmat.col
    data = adjmat.data
    i = torch.LongTensor(np.array([row, col]))
    v = torch.from_numpy(data).float()
    adjmat = torch.sparse.FloatTensor(i, v, adjmat.shape)
    return adjmat



class Mesh(object):
    """Mesh object that is used for handling certain graph operations."""
    def __init__(self, filename='/media/star/dataset_SSD/code/Cloth-shift/PaMIR-main/networks/data/mesh_downsampling.npz',
                 num_downsampling=1, nsize=1, device=torch.device('cuda')):
        # self._A, self._U, self._D = get_graph_params(filename=filename, nsize=nsize)
        #
        # self._A = [a.to(device) for a in self._A]
        # self._U = [u.to(device) for u in self._U]
        # self._D = [d.to(device) for d in self._D]
        # print("_D:",self._D)
        # self.num_downsampling = num_downsampling

        # load template vertices from SMPL and normalize them
        self.device=device
        smpl = SMPL()
        ref_vertices = smpl.v_template
        center = 0.5*(ref_vertices.max(dim=0)[0] + ref_vertices.min(dim=0)[0])[None]
        ref_vertices -= center
        ref_vertices /= ref_vertices.abs().max().item()

        self._ref_vertices = ref_vertices
        self.faces = smpl.faces.int()

    @property
    def adjmat(self):
        """Return the graph adjacency matrix at the specified subsampling level."""
        num_vertices = len( self._ref_vertices)

        # 初始化邻接矩阵
        adjacency_matrix = lil_matrix((num_vertices, num_vertices), dtype=int)

        # 遍历每个三角面片
        for face in self.faces:
            # 遍历每个顶点对
            for i in range(3):
                # 获取当前顶点索引及相邻顶点索引
                current_vertex = face[i]
                next_vertex = face[(i + 1) % 3]

                # 更新邻接矩阵
                adjacency_matrix[current_vertex, next_vertex] = 1
                adjacency_matrix[next_vertex, current_vertex] = 1

        # 将邻接矩阵转换为稀疏矩阵格式
        # adjacency_matrix = adjacency_matrix.tocsr()

        # print(torch.tensor(adjacency_matrix.toarray()).shape)
        # adjacency_matrix = adjacency_matrix.tocsr()
        adjacency_matrix=torch.tensor(adjacency_matrix.toarray()).float().to(self.device)


        return  adjacency_matrix

    @property
    def ref_vertices(self):
        """Return the template vertices at the specified subsampling level."""
        ref_vertices = self._ref_vertices
        # print("ref_vertices:",ref_vertices.shape)
        return ref_vertices.to(self.device)

