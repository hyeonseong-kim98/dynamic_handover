import hashlib
import os
import tempfile
import time
import warnings
import pickle
from pathlib import Path
import numpy as np
from scipy.spatial import distance
from collections import deque

from .dataset import list_motion_files, resolve_motion_directory

EPSILON = 1e-7
CACHE_VERSION = 1

class SparseIAALClass():
    """
        Intention Aware Apprenticeship Learning,
        given receiver real-time trajectory, determine the eef pose according to the receiver's intention
    """
    def __init__(self, 
                 filepath, 
                 hz                 =   30,
                 hz_obs             =   None,
                 kernel_data_path   =   None,
                 save_kernel_data   =   False,
                 sum_sample         =   5,
                 kappa              =   0.1,
                 use_gpr_whole      =   False,
                 forecasting_time   =   0.,
                 max_data_num       =   None,
                 sliding_window     =   None,
                 inducing_rate      =   0.3,
                 use_te             =   False,
                 chunking_size      =   20,
                 param_te           =   0.1,
                 motion_type        =   "all"):
        
        # Dataset
        self.filepath = resolve_motion_directory(filepath)
        self.motion_type = motion_type
        self.motion_files = []
        self.motion_manifest = []
        self.kernel_data_path = kernel_data_path
        self.save_kernel_data = save_kernel_data
        self.use_gpr_whole = use_gpr_whole
        self.max_data_num = max_data_num
        self.N = None # number of trajectory
        self.object_pos = [] # list with N, each elements (3, L_var)
        self.receiver_pos = [] # list with N, each elements (3, L_var)
        self.motion_length = [] # list with N
        self.receiver_vel = [] # list with N, each elements (3, L_var)
        # self.object_quat = [] # TODO: list with N, each elements (4, L_var)
        self.hz = hz
        self.dt = 1./ self.hz
        self.load_data(self.filepath)

        # Hyperparameter
        self.kernel_hyp = {'gain':1, 'len':1.0, 'w' :1e-6}
        self.kernel_hyp['len'] = self.get_length_parameter(self.receiver_pos)
        self.sum_sample = sum_sample
        self.kappa = kappa
        self.forecasting_step = int(forecasting_time * self.hz)
        self.sliding_window = sliding_window
        if not 0 < inducing_rate <= 1:
            raise ValueError("inducing_rate must be in (0, 1]")
        self.inducing_rate = inducing_rate
        self.use_te = use_te
        self.chunking_size = chunking_size
        self.param_te = param_te

        # Observations
        self.inference_time_list = []
        self.traj_obs = []
        self.mu_list_obs = []
        self.var_list_obs = []
        self.d_ascending_list_obs = []
        self.target_idx_list_obs = []
        self.d_list_obs = [] # Cummulation of trajectory distance for each data traj
        if hz_obs is None:
            self.hz_obs = self.hz
        else: 
            self.hz_obs = hz_obs
        self.dt_obs = 1. / self.hz_obs

        # Inference Output
        self.init_pos = self.object_pos[0][0]
        self.eef_target_pos = [self.init_pos] # Init pose
        print('Initial pose: ',self.init_pos)
        # self.eef_target_quat = [self.init_quat] # TODO

        
        # Initialize Motion Flow Model
        self.receiver_pos_inducing = []
        if self.use_gpr_whole:
            self.gpr_matrix_whole = [] # precomputed matrix
            self.init_gpr_kernel_whole()
        else: 
            self.gpr_matrix_list = [] # precomputed matrix
            self.init_gpr_kernel()

        # Temporal Ensemble
        if self.use_te:
            self.reset_action_buffer()
            self.weight_matrix = np.zeros((self.chunking_size,self.chunking_size,3),dtype=np.float32) #[c,c,dim_action]
            for i in range(self.chunking_size):
                self.weight_matrix[-i-1,-i-1,:] = np.exp(-i*self.param_te)


    def get_mu_var_list(self, pos_obs):
        if len(np.array(pos_obs).shape) >= 2:
            test_pos = np.array(pos_obs)
        else:
            test_pos = np.array([pos_obs])

        mu_test_list = []
        var_test_list = []
        models = self.gpr_matrix_whole if self.use_gpr_whole else self.gpr_matrix_list
        for kernel_data in models:
            mu_test, var_test = self.sparse_gaussian_process_regression(test_pos, kernel_data)
            mu_test_list.append(mu_test)
            var_test_list.append(var_test)

        mu_test_list = np.asarray(mu_test_list)
        var_test_list = np.asarray(var_test_list)
        if self.use_gpr_whole:
            # A pooled flow gives the same score to every demonstration, but all
            # demonstrations remain eligible when retrieving giver trajectories.
            mu_test_list = np.repeat(mu_test_list, self.N, axis=0)
            var_test_list = np.repeat(var_test_list, self.N, axis=0)
        # Preserve the model axis even when only one demonstration is loaded.
        if len(test_pos) == 1:
            return mu_test_list[:, 0, :], var_test_list[:, 0, 0]
        return mu_test_list, var_test_list
            
    def estimate_target(self, pos_obs):
        t_s = time.time()
        pos_obs = np.array(pos_obs)
        if len(self.traj_obs) == 0:
            test_vel = np.array([-EPSILON, 0.])
        else:
            test_vel = (pos_obs - self.traj_obs[-1]) / self.dt_obs

        mu_list, var_list = self.get_mu_var_list(pos_obs)

        d = self.compute_traj_dist(self.kappa, mu_list, var_list, test_vel)
        self.d_list_obs.append(d)

        if self.sliding_window is None or len(self.d_list_obs) < self.sliding_window:
            d_avg = np.mean(self.d_list_obs, axis=0)
        else:
            d_avg = np.mean(self.d_list_obs[-self.sliding_window:], axis=0)
            
        idx_d_ascending = np.argsort(d_avg)
        top_n_idx = idx_d_ascending[:self.sum_sample]
        
        # get target pose via similarity wegited sum
        if self.use_te:
            target_pos = self.get_target_pos_te(pos_obs, top_n_idx, d_avg, self.receiver_pos, self.object_pos, self.forecasting_step)
        else:
            target_pos = self.get_target_pos(pos_obs, top_n_idx, d_avg, self.receiver_pos, self.object_pos, self.forecasting_step)

        # For debuging
        self.traj_obs.append(pos_obs)
        self.mu_list_obs.append(mu_list)
        self.var_list_obs.append(var_list)
        self.d_ascending_list_obs.append(idx_d_ascending)
        self.eef_target_pos.append(target_pos)

        t_e = time.time()
        inference_time = t_e - t_s
        self.inference_time_list.append(inference_time)
        print('Inference Time: ', inference_time)
        return target_pos

    def get_target_pos(self, pos_obs, top_n_idx, d_avg, receiver_trajs, object_trajs, forecasting_step):
        target_pos_list = []
        target_idx_list = []
        top_n_d_avg = d_avg[top_n_idx]
        # Subtracting the minimum preserves exp(-distance) weights and avoids underflow.
        e_d_avg = np.exp(-(top_n_d_avg - np.min(top_n_d_avg)))
        sum_e_d_avg = np.sum(e_d_avg)
        prob = e_d_avg / sum_e_d_avg
        for i in range(len(top_n_idx)):
            target_idx = np.argmin(np.linalg.norm(receiver_trajs[top_n_idx[i]] - pos_obs,axis=1)) + forecasting_step
            if target_idx >= len(receiver_trajs[top_n_idx[i]]):
                target_idx = len(receiver_trajs[top_n_idx[i]]) - 1
            target_idx_list.append(target_idx)
            target_pos_list.append(object_trajs[top_n_idx[i]][target_idx])
        target_pos_list = np.vstack(target_pos_list)
        target_pos_final = np.dot(prob, target_pos_list)
        self.target_idx_list_obs.append(target_idx_list)
        return target_pos_final

    def get_target_pos_te(self, pos_obs, top_n_idx, d_avg, receiver_trajs, object_trajs, forecasting_step):
        target_pos_list = []
        target_idx_list = []
        usable_chunk_length = []
        top_n_d_avg = d_avg[top_n_idx]
        e_d_avg = np.exp(-(top_n_d_avg - np.min(top_n_d_avg)))
        sum_e_d_avg = np.sum(e_d_avg)
        prob = e_d_avg / sum_e_d_avg
        for i in range(len(top_n_idx)):
            target_idx = np.argmin(np.linalg.norm(receiver_trajs[top_n_idx[i]] - pos_obs,axis=1)) + forecasting_step
            if target_idx >= len(receiver_trajs[top_n_idx[i]]):
                target_idx = len(receiver_trajs[top_n_idx[i]]) - 1
            # Check usable forecasting length
            usable_chunk_length.append(len(receiver_trajs[top_n_idx[i]]) - target_idx)
            target_idx_list.append(target_idx)
            target_pos_list.append(object_trajs[top_n_idx[i]][target_idx:target_idx+usable_chunk_length[-1]])
        chunk_length = np.min(usable_chunk_length)
        if chunk_length > self.chunking_size:
            chunk_length = self.chunking_size
        target_pos_list = [arr[:chunk_length] * prob[i] for i, arr in enumerate(target_pos_list)]
        target_pos_list = np.stack(target_pos_list)
        target_pos_list_final = np.sum(target_pos_list, axis=0)
        # print(target_pos_list_final.shape)
        # Temporal ensemble
        action_mat = np.array([[np.nan]*3]*self.chunking_size)
        action_mat[:chunk_length] = target_pos_list_final
        self.action_deq.appendleft(action_mat)
        arr_action_deq = np.array(self.action_deq)
        valid_matrix = np.array(~np.isnan(arr_action_deq))
        weight_matrix_valid = self.weight_matrix * valid_matrix
        sum_weight = np.sum(weight_matrix_valid) / 3.
        final_weight_matrix = weight_matrix_valid / sum_weight
        arr_action_deq[np.isnan(arr_action_deq)] = 0
        target_pos_final = np.sum(final_weight_matrix * arr_action_deq, axis=(0, 1))
        self.target_idx_list_obs.append(target_idx_list)
        
        return target_pos_final

    def compute_traj_dist(self, kappa, mu_list, var_list, test_vel):
        d_temporal = self.cosine_distance(mu_list, test_vel)
        d_spatial = var_list
        return kappa * d_temporal + d_spatial

    def cosine_distance(self,traj1, traj2):
        traj1 = np.atleast_2d(traj1)
        # A stopped receiver or a zero predicted flow must not divide by zero.
        norm1 = np.maximum(np.linalg.norm(traj1, axis=1), EPSILON)
        norm2 = max(np.linalg.norm(traj2), EPSILON)
        cosine = np.dot(traj1, traj2) / (norm1 * norm2)
        return 1 - np.clip(cosine, -1., 1.)

    def sparse_gaussian_process_regression(self, test_pos, kernel_data):

        # Load pre-computed matrix
        # inv_Q_M = kernel_data['inv_Q_M'] 
        # K_data_inducing = kernel_data['K_data_inducing'] 
        # inv_Lambda = kernel_data['inv_Lambda']
        K_tilda_mu  = kernel_data['K_tilda_mu']
        K_tilda_var = kernel_data['K_tilda_var']
        data_pos_inducing = kernel_data['inducing_data_pos']

        # Compute test data kernel matrix
        K_test_inducing = self.kernel_se(test_pos, data_pos_inducing, self.kernel_hyp) # [L x M]
        K_test = self.kernel_se(test_pos, test_pos, self.kernel_hyp) # [L x L]

        # Estimate mu & var
        mu_test = K_test_inducing @ K_tilda_mu # [L_test x M] @ [M x M] @ [N x M].T @ [N x N] @ [N x 1] = [L_test x 1]
        var_test = K_test - K_test_inducing @ K_tilda_var @ K_test_inducing.T # [L_test x L_test] - [L_test x M] @ [M x M] @ [M x L_test] = [L_test x L_test]

        return mu_test, var_test


    def load_data(self, filepath):

        # Load data
        self.motion_files = list_motion_files(filepath, self.motion_type, self.max_data_num)
        for filename in self.motion_files:
            contents = (Path(filepath) / filename).read_bytes()
            motion_data = pickle.loads(contents)
            object_pos = np.asarray(motion_data['pose_object'], dtype=float)
            receiver_pos = np.asarray(motion_data['pose_receiver'], dtype=float)
            if (object_pos.ndim != 2 or object_pos.shape[1] != 3
                    or receiver_pos.ndim != 3 or receiver_pos.shape[1] < 1
                    or receiver_pos.shape[2] < 2 or len(object_pos) != len(receiver_pos)
                    or len(object_pos) < 2):
                raise ValueError("Invalid motion shapes in {}".format(filename))
            receiver_xy = receiver_pos[:, 0, :2]
            if not np.isfinite(object_pos).all() or not np.isfinite(receiver_xy).all():
                raise ValueError("Non-finite motion positions in {}".format(filename))
            self.motion_length.append(len(object_pos))
            self.object_pos.append(object_pos.copy())
            self.receiver_pos.append(receiver_xy.copy())
            self.motion_manifest.append((filename, hashlib.sha256(contents).hexdigest()))

        # Get Velocity matrix of receiver
        for i in range(len(self.object_pos)):
            vel = (np.array(self.receiver_pos[i][1:]) - np.array(self.receiver_pos[i][:-1])) * self.hz
            vel = np.vstack([vel[0], vel])
            self.receiver_vel.append(vel)
            
        self.N = len(self.motion_length)
            
        print('Data Loaded, {} trajectories ready.'.format(self.N))
    
    def reset_obs(self):
        self.inference_time_list = []
        self.traj_obs = []
        self.mu_list_obs = []
        self.var_list_obs = []
        self.d_list_obs = []
        self.d_ascending_list_obs = []
        self.target_idx_list_obs = []
        self.eef_target_pos = [self.init_pos]
        if self.use_te:
            self.reset_action_buffer()

    def reset_action_buffer(self):
        self.action_deq = deque(
            (np.full((self.chunking_size, 3), np.nan) for _ in range(self.chunking_size)),
            maxlen=self.chunking_size,
        )


    def print_avg_inf_time(self):
        print("Average_inferece time: ",np.mean(self.inference_time_list))

    def get_length_parameter(self,X):
        """
            Median distance of the distance matrix
        """
        # flatten the list
        X_flatten = np.vstack(X)[::5]
        dists = distance.cdist(X_flatten, X_flatten)
        median_dist = np.median(dists)
        print('Median distance: ', median_dist)
        return max(median_dist, EPSILON)
    
    def kernel_se(self,x1,x2,hyp:dict):
        """
            Squared-exponential kernel function,
            defined as k(x,y) = exp(-|x-y|**2 / (2 * l**2)),
            a function that computes similarity between two points

            Args:
                x1: (M, shape of input) 
                x2: (N, shape of input)

            Returns:
                K: (M, N)
        """
        D = distance.cdist(x1/hyp['len'], x2/hyp['len'],'sqeuclidean')
        K = hyp['gain']*np.exp(-0.5 * D)
        return K
    
    def init_gpr_kernel(self):
        metadata = {
            'version': CACHE_VERSION,
            'motions': self.motion_manifest,
            'hz': self.hz,
            'inducing_rate': self.inducing_rate,
            'kernel_hyp': self.kernel_hyp,
        }
        cache_path = Path(self.kernel_data_path) if self.kernel_data_path is not None else None
        cache = None
        if cache_path is not None and cache_path.is_file():
            try:
                with cache_path.open('rb') as stream:
                    cache = pickle.load(stream)
            except (OSError, EOFError, pickle.UnpicklingError):
                cache = None
            if not self.valid_kernel_cache(cache, metadata):
                warnings.warn("SPGP cache is outdated or does not match the dataset; rebuilding.")
                cache = None
        if cache is not None:
            self.gpr_matrix_list = [cache['models'][name] for name in self.motion_files]
            self.receiver_pos_inducing = [model['inducing_data_pos'] for model in self.gpr_matrix_list]
        else:
            self.make_gpr_kernel()
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {'metadata': metadata, 'models': dict(zip(self.motion_files, self.gpr_matrix_list))}
                temporary_path = None
                try:
                    with tempfile.NamedTemporaryFile(dir=cache_path.parent, delete=False) as stream:
                        temporary_path = stream.name
                        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(temporary_path, cache_path)
                finally:
                    if temporary_path is not None and os.path.exists(temporary_path):
                        os.unlink(temporary_path)
        print('{} GPR kernel matrix pre-computed'.format(len(self.gpr_matrix_list)))

    def valid_kernel_cache(self, cache, metadata):
        if not isinstance(cache, dict) or cache.get('metadata') != metadata:
            return False
        models = cache.get('models')
        if not isinstance(models, dict) or set(models) != set(self.motion_files):
            return False
        for i, filename in enumerate(self.motion_files):
            model = models[filename]
            inducing_n = max(1, int(self.motion_length[i] * self.inducing_rate))
            indices = np.linspace(0, self.motion_length[i] - 1, num=inducing_n, dtype=int)
            if (not isinstance(model, dict)
                    or not np.array_equal(model.get('inducing_data_pos'), self.receiver_pos[i][indices])
                    or np.shape(model.get('K_tilda_mu')) != (inducing_n, 2)
                    or np.shape(model.get('K_tilda_var')) != (inducing_n, inducing_n)):
                return False
        return True
    
    def make_gpr_kernel(self):
        for data_pos, data_vel in zip(self.receiver_pos, self.receiver_vel):
            model = self.precompute_flow_kernel(data_pos, data_vel)
            self.receiver_pos_inducing.append(model['inducing_data_pos'])
            self.gpr_matrix_list.append(model)

    def precompute_flow_kernel(self, data_pos, data_vel):
        """Build a FITC flow model for one trajectory or a pooled dataset."""
        inducing_n = max(1, int(len(data_pos) * self.inducing_rate))
        indices = np.linspace(0, len(data_pos) - 1, num=inducing_n, dtype=int)
        inducing_data_pos = data_pos[indices]

        K_data = self.kernel_se(data_pos, data_pos, self.kernel_hyp)
        K_data_inducing = self.kernel_se(data_pos, inducing_data_pos, self.kernel_hyp)
        K_inducing_inducing = self.kernel_se(inducing_data_pos, inducing_data_pos, self.kernel_hyp)
        regularization = self.kernel_hyp['w'] * np.eye(inducing_n)
        inv_K_inducing_inducing = np.linalg.inv(K_inducing_inducing + regularization)
        K_approx = K_data_inducing @ inv_K_inducing_inducing @ K_data_inducing.T
        Lambda = np.diag(np.diag(K_data - K_approx))
        inv_Lambda = np.linalg.inv(Lambda + self.kernel_hyp['w'] * np.eye(len(data_pos)))
        Q_M = K_inducing_inducing + K_data_inducing.T @ inv_Lambda @ K_data_inducing
        inv_Q_M = np.linalg.inv(Q_M + regularization)

        return {
            'K_tilda_mu': inv_Q_M @ K_data_inducing.T @ inv_Lambda @ data_vel,
            'K_tilda_var': inv_K_inducing_inducing - inv_Q_M,
            'inducing_data_pos': inducing_data_pos.copy(),
        }

    def init_gpr_kernel_whole(self):
        model = self.precompute_flow_kernel(
            np.vstack(self.receiver_pos), np.vstack(self.receiver_vel),
        )
        self.receiver_pos_inducing.append(model['inducing_data_pos'])
        self.gpr_matrix_whole.append(model)
        print('1 pooled GPR kernel matrix pre-computed')
