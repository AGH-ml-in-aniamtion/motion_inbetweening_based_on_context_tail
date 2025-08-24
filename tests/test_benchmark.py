import os
import pickle
import sys
import unittest
import torch
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
package_root = os.path.join(project_root, "packages")
sys.path.append(package_root)

from motion_inbetween import benchmark
    
class TestBenchmark(unittest.TestCase):

    def test_rmi_style_loss(self):
        positions = torch.ones(32, 27, 22, 3)
        rotations = torch.ones(32, 27, 22, 3, 3)
        pos_new = torch.ones(32, 27, 22, 3)
        rot_new = torch.ones(32, 27, 22, 3, 3)
        parents = torch.tensor([-1,  0,  1,  2,  3,  0,  5,  6,  7,  0,  9, 10, 11, 12, 11, 14, 15, 16,
        11, 18, 19, 20])


        mean_rmi = torch.tensor([[-1.3550e-16,  8.2261e+01,  2.9567e-17],
        [ 1.3790e+00,  8.2255e+01, -8.7167e+00],
        [ 1.2952e+01,  4.8039e+01, -1.3106e+01],
        [-1.9123e+00,  1.7409e+01, -1.1137e+01],
        [ 7.7733e+00,  9.0416e+00, -1.3457e+01],
        [ 1.5946e+00,  8.2281e+01,  8.6834e+00],
        [ 1.3283e+01,  4.7551e+01,  1.2050e+01],
        [-2.2272e+00,  1.7239e+01,  1.1537e+01],
        [ 7.5058e+00,  9.2053e+00,  1.4457e+01],
        [-1.7892e+00,  8.8648e+01, -3.6445e-03],
        [ 4.7643e-01,  9.9766e+01, -7.2591e-02],
        [ 4.2435e+00,  1.0988e+02, -1.7152e-01],
        [ 1.4634e+01,  1.2862e+02, -4.5334e-01],
        [ 2.0094e+01,  1.3648e+02, -1.6513e-01],
        [ 1.1147e+01,  1.2492e+02, -5.2614e+00],
        [ 1.0541e+01,  1.2366e+02, -1.4205e+01],
        [ 6.3485e+00,  9.9732e+01, -2.4182e+01],
        [ 1.6915e+01,  8.7572e+01, -2.2820e+01],
        [ 1.1370e+01,  1.2486e+02,  4.5336e+00],
        [ 1.1296e+01,  1.2274e+02,  1.3351e+01],
        [ 6.6718e+00,  9.9588e+01,  2.3196e+01],
        [ 1.6639e+01,  8.7020e+01,  2.1650e+01]])
        std_rmi = torch.tensor([[47.0959, 28.0501, 23.8774],
        [47.0970, 28.2485, 24.1586],
        [49.8518, 23.5715, 28.0785],
        [54.2343, 21.4503, 30.8202],
        [56.3064, 20.8736, 32.9498],
        [47.3081, 28.1009, 24.3212],
        [49.9828, 23.7494, 28.1126],
        [54.3852, 21.2752, 30.6427],
        [56.3387, 20.8057, 32.7585],
        [47.0860, 28.9321, 23.7596],
        [46.9042, 30.7817, 23.9746],
        [46.9091, 32.8452, 24.6356],
        [47.7384, 38.6454, 27.5110],
        [48.4736, 41.0710, 29.4982],
        [47.5560, 37.0727, 26.6091],
        [48.0217, 37.0878, 26.9706],
        [50.4696, 37.1883, 28.2626],
        [51.3727, 39.6318, 31.9564],
        [47.6694, 37.0179, 26.7824],
        [48.2721, 36.9326, 27.3811],
        [51.1598, 37.6890, 28.8818],
        [51.8224, 40.3614, 32.6812]])


        gpos_loss, gquat_loss, npss_loss, npss_weight = benchmark.get_rmi_style_batch_loss(positions, rotations, pos_new, rot_new,
                                            parents, 1, 5, mean_rmi, std_rmi)

        print("gpos_loss:", gpos_loss, gpos_loss.shape)
        print("gquat_loss:", gquat_loss, gquat_loss.shape)
        print("npss_loss:", npss_loss, npss_loss.shape)
        print("npss_weight:", npss_weight, npss_weight.shape)
        
        npss_weight = npss_weight / np.sum(npss_weight)      # (batch, dim)
        print("weighted npss loss", (npss_loss * npss_weight).shape)
        
        npss_loss = np.sum(npss_loss * npss_weight, axis=-1)
        
        print("sum", npss_loss.sum(), npss_loss.shape)
        
        
        

if __name__ == "__main__":
    unittest.main()