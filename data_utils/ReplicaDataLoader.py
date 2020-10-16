import numpy as np
import warnings
import os
import open3d as o3d
from torch.utils.data import Dataset
import lib.transformation as tf
warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ReplicaDataLoader(Dataset):
    def __init__(self, root,  npoint=1024, split='train', uniform=False, normal_channel=False, rot_transform=False, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'replica_shape_names.txt')
        self.rot_transform = rot_transform
        self.split = split

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'replica_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'replica_test.txt'))]

        assert (split == 'train' or split == 'test')
        # this one reads the
        shape_names = [os.path.split(file_name)[1].split("_")[1][:-4] for file_name in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i],  shape_ids[split][i]) for i in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split, len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[fn[0]]
            cls = np.array([cls]).astype(np.int32)
            o3d_pcd = o3d.io.read_point_cloud(fn[1])
            # apply random rotation to only train dataset
            if self.rot_transform:
                rot_tfm = tf.random_rotation()
                o3d_pcd.transform(rot_tfm)
            point_set = np.asarray(o3d_pcd.points).astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]

            point_set[:, :3] = pc_normalize(point_set[:, :3])

            if not self.normal_channel:
                point_set = point_set[:, :3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch
    from tqdm import tqdm
    data = ReplicaDataLoader('../data/replica/', split='train', uniform=True, normal_channel=False, rot_transform=True)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True, num_workers=4)
    for batch_id, data in tqdm(enumerate(DataLoader, 0), total=len(DataLoader), smoothing=0.9):
        points, target = data
        print(points.shape)
        print(target.shape)