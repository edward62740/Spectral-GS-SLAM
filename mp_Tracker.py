import os
import torch
import torch.multiprocessing as mp
import torch.multiprocessing
from random import randint
import sys
import threading
import queue
import cv2
import numpy as np
import open3d as o3d
import pygicp
import time
from scipy.spatial.transform import Rotation
import rerun as rr

from orb_prior import ORBFeaturePrior

sys.path.append(os.path.dirname(__file__))
from arguments import SLAMParameters
from utils.traj_utils import TrajManager
from gaussian_renderer import render, render_2, network_gui
from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix, focal2fov
from tqdm import tqdm
import torchvision
import math
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from scene.cameras import MiniCam
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render


# condition number threshold for degeneracy detection (tau_deg)
_T_DEG = 200.0

# choose the gaussians the planarity ratio is calculated on
# obs = observation gaussians from kNN spawn, map = in-view map gaussians
# obs is more responsive and usually better
_PLANARITY_SOURCE = "obs"

# fall back to orb prior when gicp converges but disagrees with it
# debug only; dont enable
_STRESS_PRIOR = False
_STRESS_RDIS = 6.0    # deg
_STRESS_TDIS = 0.10   # m



class Pipe:
    def __init__(self, convert_SHs_python, compute_cov3D_python, debug):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug

class GaussianModelWrapper:
    def __init__(self, xyz, rots, scales, opacities, sh_degree=0):
        self._xyz = torch.from_numpy(xyz).float().cuda()
        self._rotation = torch.from_numpy(rots).float().cuda()
        self._scaling = torch.from_numpy(scales).float().cuda()
        self._opacity = torch.from_numpy(opacities).float().cuda()
        self.active_sh_degree = sh_degree
        self.max_sh_degree = sh_degree
    
    @property
    def get_xyz(self): return self._xyz
    @property
    def get_rotation(self): return self._rotation
    @property
    def get_scaling(self): return self._scaling
    @property
    def get_opacity(self): return self._opacity
    @property
    def get_features(self): return None 


class Tracker(SLAMParameters):
    def __init__(self, slam):
        super().__init__()
        self.dataset_path = slam.dataset_path
        self.output_path = slam.output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = slam.verbose
        self.keyframe_th = slam.keyframe_th
        self.knn_max_distance = slam.knn_max_distance
        self.overlapped_th = slam.overlapped_th
        self.overlapped_th2 = slam.overlapped_th2
        self.downsample_rate = slam.downsample_rate
        self.test = slam.test
        self.rerun_viewer = slam.rerun_viewer
        self.iter_shared = slam.iter_shared
        
        self.camera_parameters = slam.camera_parameters
        self.W = slam.W
        self.H = slam.H
        self.fx = slam.fx
        self.fy = slam.fy
        self.cx = slam.cx
        self.cy = slam.cy
        self.depth_scale = slam.depth_scale
        self.depth_trunc = slam.depth_trunc
        self.cam_intrinsic = np.array([[self.fx, 0., self.cx],
                                       [0., self.fy, self.cy],
                                       [0.,0.,1]])
        
        self.viewer_fps = slam.viewer_fps
        self.keyframe_freq = slam.keyframe_freq
        self.max_correspondence_distance = slam.max_correspondence_distance
        self.reg = None
        self.current_planarity = 0.0
        self.planarity_scores = []
        self.current_prior_used = False
        self.prior_usage_scores = []
        self.current_eig_ratio = 1.0
        self.eig_ratio_scores = []
        
        # Camera poses
        self.trajmanager = TrajManager(self.camera_parameters[8], self.dataset_path)
        self.poses = [self.trajmanager.gt_poses[0]]
        # Keyframes(added to map gaussians)
        self.last_t = time.time()
        self.iteration_images = 0
        self.end_trigger = False
        self.covisible_keyframes = []
        self.new_target_trigger = False
        
        self.cam_t = []
        self.cam_R = []
        self.points_cat = []
        self.colors_cat = []
        self.rots_cat = []
        self.scales_cat = []
        self.trackable_mask = []
        self.from_last_tracking_keyframe = 0
        self.from_last_mapping_keyframe = 0
        self.scene_extent = 2.5
        
        self.downsample_idxs, self.x_pre, self.y_pre = self.set_downsample_filter(self.downsample_rate)

        # Share
        self.train_iter = 0
        self.mapping_losses = []
        self.new_keyframes = []
        self.gaussian_keyframe_idxs = []

        self.shared_cam = slam.shared_cam
        self.shared_new_points = slam.shared_new_points
        self.shared_new_gaussians = slam.shared_new_gaussians
        self.shared_target_gaussians = slam.shared_target_gaussians
        self.end_of_dataset = slam.end_of_dataset
        self.is_tracking_keyframe_shared = slam.is_tracking_keyframe_shared
        self.is_mapping_keyframe_shared = slam.is_mapping_keyframe_shared
        self.target_gaussians_ready = slam.target_gaussians_ready
        self.new_points_ready = slam.new_points_ready
        self.final_pose = slam.final_pose
        self.demo = slam.demo
        self.is_mapping_process_started = slam.is_mapping_process_started
        self.use_fusion = slam.use_fusion
        
        self.degeneracy_streak = 0
        self.prev_rgb = None
        self.prev_depth = None
        self.velocity = np.eye(4) # identity init
        
        self.flow_prior = None
        #self.gaussians = GaussianModel(0) # sh_degree=0
        self.pipe = Pipe(False, False, False) # convert_SHs_python, compute_cov3D_python, debug
        self.bg_color = [0, 0, 0] # Black background for flow rendering
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")
        
        self.local_target_points = None
        self.local_target_rots = None
        self.local_target_scales = None
        # self.local_target_lifespan = None
        self.local_target_stability = None


    def run(self):
        self.tracking()
    
    def tracking(self):
        self.reg = pygicp.FastGICP()
        self.prior = ORBFeaturePrior(self.W, self.H, self.fx, self.fy, self.cx, self.cy, rerun_viewer=self.rerun_viewer)
        self.used_prior_last_frame = False
       

        tt = torch.zeros((1,1)).float().cuda()
        
        if self.rerun_viewer:
            rr.init("3dgsviewer")
            rr.connect_grpc()
        
        self.rgb_images, self.depth_images = self.get_images(f"{self.dataset_path}/images")
        self.num_images = len(self.rgb_images)
        self.reg.set_max_correspondence_distance(self.max_correspondence_distance)
        self.reg.set_max_knn_distance(self.knn_max_distance)
        if_mapping_keyframe = False

        self.total_start_time = time.time()
        pbar = tqdm(total=self.num_images)

        self.traj_gt = []
        self.traj_coarse = []
        self.traj_fine = []

        for ii in range(self.num_images):
            self.iter_shared[0] = ii
            current_image = self.rgb_images.pop(0)
            depth_image = self.depth_images.pop(0)
            current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
            
            # metric depth conversion
            depth_image = depth_image.astype(np.float32) / self.depth_scale

            points, colors, z_values, trackable_filter = self.downsample_and_make_pointcloud2(depth_image, current_image)

            self.last_source_points = points
            
            T_prior = None
            # GICP
            if self.iteration_images == 0:
                current_pose = self.poses[-1]
                
                if self.rerun_viewer:
                    # rr.set_time_sequence("step", self.iteration_images)
                    rr.set_time("log_time", duration=time.time() - self.total_start_time)
                    rr.log(
                        "cam/current",
                        rr.Transform3D(translation=self.poses[-1][:3,3],
                                    rotation=rr.Quaternion(xyzw=(Rotation.from_matrix(self.poses[-1][:3,:3])).as_quat()))
                    )
                    rr.log(
                        "cam/current",
                        rr.Pinhole(
                            resolution=[self.W, self.H],
                            image_from_camera=self.cam_intrinsic,
                            camera_xyz=rr.ViewCoordinates.RDF,
                        )
                    )
                    rr.log(
                        "cam/current",
                        rr.Image(current_image)
                    )
                    
                # Update Camera pose #
                current_pose = np.linalg.inv(current_pose)
                T = current_pose[:3,3]
                R = current_pose[:3,:3].transpose()
                
                # transform current points
                points = np.matmul(R, points.transpose()).transpose() - np.matmul(R, T)
                # Set initial pointcloud to target points
                self.reg.set_input_target(points)
                
                num_trackable_points = trackable_filter.shape[0]
                input_filter = np.zeros(points.shape[0], dtype=np.int32)
                input_filter[(trackable_filter)] = [range(1, num_trackable_points+1)]
                
                self.reg.set_target_filter(num_trackable_points, input_filter)
                self.reg.calculate_target_covariance_with_filter()

                rots = self.reg.get_target_rotationsq()
                scales = self.reg.get_target_scales()
                rots = np.reshape(rots, (-1,4))
                scales = np.reshape(scales, (-1,3))
                
                # Assign first gaussian to shared memory
                self.shared_new_gaussians.input_values(torch.tensor(points), torch.tensor(colors), 
                                                       torch.tensor(rots), torch.tensor(scales), 
                                                       torch.tensor(z_values), torch.tensor(trackable_filter))
                
                # Update local gaussians
                
                # Add first keyframe
                # depth_image is already float32 metric.
                self.shared_cam.setup_cam(R, T, current_image, depth_image)
                self.shared_cam.cam_idx[0] = self.iteration_images
                
                self.is_tracking_keyframe_shared[0] = 1
                
                # Cache for visualization
                self.local_target_points = points
                self.local_target_rots = rots
                self.local_target_scales = scales
                # Initialize attributes (lifespan=-1, stability=0)
                # self.local_target_lifespan = np.full(points.shape[0], -1, dtype=np.int32)
                self.local_target_stability = np.zeros(points.shape[0], dtype=np.float32)
                
                while self.demo[0]:
                    time.sleep(1e-15)
                    self.total_start_time = time.time()
                if self.rerun_viewer:
                    # rr.set_time_sequence("step", self.iteration_images)
                    rr.set_time("log_time", duration=time.time() - self.total_start_time)
                    rr.log(f"pt/trackable/{self.iteration_images}", rr.Points3D(points, colors=colors, radii=0.02))
            else:
                self.reg.set_input_source(points)
                num_trackable_points = trackable_filter.shape[0]
                input_filter = np.zeros(points.shape[0], dtype=np.int32)
                input_filter[(trackable_filter)] = [range(1, num_trackable_points+1)]
                self.reg.set_source_filter(num_trackable_points, input_filter)
                
                # zero motion init, same as GSICP
                T_pred = self.poses[-1]
                initial_pose = T_pred
                
                # CVM logic (ablation 3)
                # if len(self.poses) > 2:
                #     T_prev = self.poses[-1]
                #     T_prev2 = self.poses[-2]
                #     T_vel = np.linalg.inv(T_prev2) @ T_prev
                #     T_pred = T_prev @ T_vel
                # else:
                #     T_pred = self.poses[-1]
                
                # Initialize logic variables
                use_prior = True
                # prior_rel = self.velocity
                prior_rel = np.eye(4) # zero
                
                is_valid_prior = False 
                T_prior = None
                self.used_prior_last_frame = self.reg.did_use_prior() if self.iteration_images > 1 else False # Update from previous iteration output

                # get the prior pose, hessian etc.
                if self.prev_rgb is not None and len(self.poses) > 3:
                     res_prior = self.prior.get_prior(
                         live_image=current_image,
                         prev_image=self.prev_rgb,
                         prev_depth=self.prev_depth,
                         T_prev=self.poses[-1],
                     )

                     if isinstance(res_prior, tuple):
                         T_prior, H_prior = res_prior
                     else:
                         T_prior = res_prior
                         H_prior = None

                     #T_prior = None
                     #print(T_prior)
                     if T_prior is not None:
                         
                         T_rel_prior = np.linalg.inv(self.poses[-1]) @ T_prior
                         trans_mag = np.linalg.norm(T_rel_prior[:3, 3])
                         # rot
                         trace = np.trace(T_rel_prior[:3, :3])
                         rot_mag = np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
                         if trans_mag < 0.1 and rot_mag < 10.0: # heuristic threshold if the estimate is infeasible
                             if self.used_prior_last_frame:
                                 initial_pose = T_prior
                             
                             #print("Using Feature Prior")
                             self.reg.set_prior_pose(T_prior) #
                             
                             if self.use_fusion:
                                 if H_prior is not None:
                                     # this whole block can technically be removed, no-op
                                     # damping
                                     H_pnp = H_prior + np.eye(6) * 1e2
                                     self.reg.set_prior_information(H_pnp)
                                 else:
                                     self.reg.set_prior_information(np.eye(6) * 1e4) # stiffness

                             is_valid_prior = True
                             if self.rerun_viewer:
                                 rr.log("prior/status", rr.TextDocument(f"Using Feature Prior (Dist: {trans_mag:.3f})"))
                         else:
                             if self.rerun_viewer:
                                 rr.log("prior/status", rr.TextDocument(f"Feature Prior Rejected: Huge Motion (Dist: {trans_mag:.3f}, Rot: {rot_mag:.1f})"))
                        
                     if not is_valid_prior:
                         if self.rerun_viewer and T_prior is None:
                             rr.log("prior/status", rr.TextDocument("Feature Prior Failed (Optimization)"))
                            
                         # zero motion if everything fails
                         initial_pose = self.poses[-1]
                         self.reg.set_prior_pose(self.poses[-1])
                         self.reg.disable_prior_information() # zero-motion prior; fill uses ICP stiffness only

                
                # prior threshold/fusion settings
                if self.use_fusion and len(self.poses) > 3:
                    self.reg.set_degeneracy_threshold(_T_DEG) # the cond number th
                    self.reg.enable_degeneracy_correction(True) # master switch for detection + fusion
                    self.reg.set_robust_loss(True, 1.345) # huber
                    # self.reg.set_prior_information(np.eye(6) * 1e4)
                    self.reg.set_debug_print(False) # c++ debug statements for LM step, etc.
                    
                    
                else:
                    self.reg.enable_degeneracy_correction(False)
                    self.reg.disable_prior_information()
                    self.reg.set_robust_loss(False, 1.345) 
                    

                current_pose = self.reg.align(initial_pose) # ICP align

                # eta from the source covariances, i.e. this frame's depth image only.
                # read after align() since set_input_source clears them and they are
                # refilled in computeTransformation. E[nn^T] rather than the centred
                # covariance: the minor axis is axial, so its sign is an artefact of the
                # eigensolver and the sample mean is not a meaningful centre. installed
                # for the next frame; align() computes these covariances anyway.
                if _PLANARITY_SOURCE == "obs":
                    try:
                        src_rots = np.asarray(self.reg.get_source_rotationsq()).reshape(-1, 4)
                        src_scales = np.asarray(self.reg.get_source_scales()).reshape(-1, 3)
                        if len(src_rots) >= 300 and len(src_rots) == len(src_scales):
                            src_R = Rotation.from_quat(src_rots).as_matrix()
                            minor = np.argmin(src_scales, axis=1)
                            obs_normals = src_R[np.arange(len(minor)), :, minor]
                            M_cam = (obs_normals.T @ obs_normals) / len(obs_normals)
                            R_wc = self.poses[-1][:3, :3]
                            self.reg.set_normal_covariance((R_wc @ M_cam @ R_wc.T).astype(np.float32))
                    except Exception:
                        pass

                # debug
                try:
                    eigenvals = self.reg.get_last_eigenvalues() # to vis degen conds
                    eigenvals = np.minimum(eigenvals, 10000)
                    if self.rerun_viewer:
                        for i in range(6):
                            rr.log(f"deg_eigen/val_{i}", rr.Scalars(eigenvals[i]))
                except AttributeError:
                    pass
                
                # set the velocity
                T_prev = self.poses[-1]
                current_velocity = np.linalg.inv(T_prev) @ current_pose
                
                # c++ returns whether LM step is healthy
                is_converged = True
                try:
                    is_converged = self.reg.has_converged() 
                except AttributeError:
                    pass
                
                dist = np.linalg.norm(current_velocity[:3, 3])
                max_step = 0.5 # 0.5m limit per frame
                
                # this section is for the off chance that the LM solver has some numerical error
                # and causes the estimate to jump
                # unhealthy: lm no converge, or gicp disagrees with orb prior
                unhealthy = (not is_converged)
                if _STRESS_PRIOR and is_converged and is_valid_prior and T_prior is not None:
                    T_dis = np.linalg.inv(current_pose) @ T_prior
                    rot_dis = np.degrees(np.arccos(np.clip((np.trace(T_dis[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
                    trans_dis = np.linalg.norm(T_dis[:3, 3])
                    if rot_dis > _STRESS_RDIS or trans_dis > _STRESS_TDIS:
                        unhealthy = True

                if unhealthy:
                    if not is_converged:
                        print("[Tracker] Solver FAILED! Falling back to ORB prior / zero motion.")
                    else:
                        print("[Tracker] GICP disagrees with ORB prior -> ORB fallback")

                    if T_prior is not None:
                        current_pose = T_prior
                    else:
                        current_pose = self.poses[-1]

                    # safety; reset
                    self.velocity = np.eye(4)

                    dist = np.linalg.norm(current_pose[:3, 3] - self.poses[-1][:3, 3])
                else:
                    # this is a VALID update
                    T_prev = self.poses[-1]
                    current_velocity = np.linalg.inv(T_prev) @ current_pose
                    if self.rerun_viewer:
                        # logging stuff
                        if self.trajmanager.gt_poses is not None and ii < len(self.trajmanager.gt_poses):
                            gt_pose = self.trajmanager.gt_poses[ii]
                            self.traj_gt.append(gt_pose[:3, 3])
                            rr.log("traj/gt", rr.LineStrips3D([self.traj_gt], colors=[[255, 0, 0]]))
                        if T_prior is not None:
                            self.traj_coarse.append(T_prior[:3, 3])
                        elif len(self.traj_coarse) > 0:
                            self.traj_coarse.append(self.traj_coarse[-1]) # Hold position if no prior
                            
                        rr.log("traj/coarse", rr.LineStrips3D([self.traj_coarse], colors=[[0, 0, 255]]))
                        self.traj_fine.append(current_pose[:3, 3])
                        rr.log("traj/fine", rr.LineStrips3D([self.traj_fine], colors=[[0, 255, 0]]))


                    # verify again
                    dist = np.linalg.norm(current_velocity[:3, 3])
                    max_step = 0.5
                    
                    if dist > max_step:
                         print(f"[Tracker] Rejecting Velocity Explosion! Dist: {dist:.3f}")
                         self.velocity = np.eye(4)
                         print("  -> Fallback to Coasting.")
                         current_pose = T_prev # i.e zero motion
                         dist = 0.0
                    else:
                        self.velocity = np.eye(4)
                
                if self.rerun_viewer:
                         rr.log("prior/translation_mag", rr.Scalars(dist))

                # if T_prior is not None:
                #     current_pose = T_prior (ablation )
       
                
                # debug for rerun viewer
                try:
                    eig = self.reg.get_last_eigenvalues()
                    #print(eig)
                    min_eig = np.min(eig)
                    max_eig = np.max(eig)
                    ratio = min_eig / (max_eig + 1e-6)
                    self.current_eig_ratio = float(max_eig / (min_eig + 1e-6))

                    if self.reg.did_use_prior():
                        self.current_prior_used = True
                        self.degeneracy_streak += 1
                        pbar.set_description(f"[PRIOR] D:{dist:.3f} R:{ratio:.1e}")
                    else:
                        self.current_prior_used = False
                        self.degeneracy_streak = 0
                        pbar.set_description(f"[GICP] D:{dist:.3f} R:{ratio:.1e}")

                    # eig is a (6,1)
                    for k in range(6):
                        rr.log(f"deg/ev_{k}", rr.Scalars(eig[k]))
                    
                    # Log Lambda
                    lm_lambda = self.reg.get_final_lambda()
                    rr.log("deg/lambda", rr.Scalars(lm_lambda))

                except AttributeError:
                    self.current_eig_ratio = 1.0
                    pass 
                
                self.poses.append(current_pose)

                # framebuf for orb
                self.prev_rgb = current_image.copy()
                self.prev_depth = depth_image.copy()

                if self.rerun_viewer:
                    # rr.set_time_sequence("step", self.iteration_images)
                    rr.set_time("log_time", duration=time.time() - self.total_start_time)
                    rr.log(
                        "cam/current",
                        rr.Transform3D(translation=self.poses[-1][:3,3],
                                    rotation=rr.Quaternion(xyzw=(Rotation.from_matrix(self.poses[-1][:3,:3])).as_quat()))
                    )
                    rr.log(
                        "cam/current",
                        rr.Pinhole(
                            resolution=[self.W, self.H],
                            image_from_camera=self.cam_intrinsic,
                            camera_xyz=rr.ViewCoordinates.RDF,
                        )
                    )
                    
                    rr.log(
                        "cam/current",
                        rr.Image(current_image) # Already RGB
                    )

                # camera pose finalized
                current_pose = np.linalg.inv(current_pose)
                T = current_pose[:3,3]
                R = current_pose[:3,:3].transpose()

                # transform current points
                # downsample_and_make_pointcloud2 expects depth_image used for points is aligned
                points = np.matmul(R, points.transpose()).transpose() - np.matmul(R, T)
                # Use only trackable points when tracking
                target_corres, distances = self.reg.get_source_correspondence() # get associated points source points
                
                if len(distances) == num_trackable_points + 1:
                     distances = distances[1:]
                elif len(distances) > num_trackable_points:
                     # Fallback if size is different but likely includes 0
                     distances = distances[1:num_trackable_points+1]
                     
                len_corres = len(np.where(distances<self.overlapped_th)[0]) # 5e-4 self.overlapped_th
                
                if  (self.iteration_images >= self.num_images-1 \
                    or len_corres/distances.shape[0] < self.keyframe_th):
                    if_tracking_keyframe = True
                    self.from_last_tracking_keyframe = 0
                else:
                    if_tracking_keyframe = False
                    self.from_last_tracking_keyframe += 1
                
                # Mapping keyframe
                if (self.from_last_tracking_keyframe) % self.keyframe_freq == 0:
                    if_mapping_keyframe = True
                else:
                    if_mapping_keyframe = False
                
                if if_tracking_keyframe:
                    while self.is_tracking_keyframe_shared[0] or self.is_mapping_keyframe_shared[0]:
                        time.sleep(1e-15)
                    
                    rots = np.array(self.reg.get_source_rotationsq())
                    rots = np.reshape(rots, (-1,4))

                    R_d = Rotation.from_matrix(R)    # from camera R
                    R_d_q = R_d.as_quat()            # xyzw
                    rots = self.quaternion_multiply(R_d_q, rots)
                    
                    scales = np.array(self.reg.get_source_scales())
                    scales = np.reshape(scales, (-1,3))
                    
                    not_overlapped_indices_of_trackable_points = self.eliminate_overlapped2(distances, self.overlapped_th2) # 5e-5 self.overlapped_th
                    trackable_filter = trackable_filter[not_overlapped_indices_of_trackable_points]
                    
                    # Add new gaussians
                    self.shared_new_gaussians.input_values(torch.tensor(points), torch.tensor(colors), 
                                                       torch.tensor(rots), torch.tensor(scales), 
                                                       torch.tensor(z_values), torch.tensor(trackable_filter))
                    
                    # Update local gaussians
                    # self.gaussians.add_from_pcd2_tensor(torch.tensor(points).float().cuda(), torch.tensor(colors).float().cuda(), 
                    #                                    torch.tensor(rots).float().cuda(), torch.tensor(scales).float().cuda(), 
                    #                                    torch.tensor(z_values).float().cuda(), torch.tensor(trackable_filter).cuda(),
                    #                                    torch.tensor(is_dynamic_mask).float().cuda())


                    # Add new keyframe
                    # depth_image is alread metric float
                    self.shared_cam.setup_cam(R, T, current_image, depth_image)
                    self.shared_cam.cam_idx[0] = self.iteration_images
                    
                    self.is_tracking_keyframe_shared[0] = 1
                    
                    # Get new target point
                    while not self.target_gaussians_ready[0]:
                        time.sleep(1e-15)
                    target_points, target_rots, target_scales, target_colors, target_opacities, target_stability, _ = self.shared_target_gaussians.get_values_np()
                    
                    current_trans = self.poses[-1][:3, 3]
                    dist_sq = np.sum((target_points - current_trans)**2, axis=1)
                    radius_sq = 5.0**2 # 20m radius
                    
                    local_mask = dist_sq < radius_sq
                    
                    if np.sum(local_mask) < 2000:
                        print("Too few points, using all points")
                        # If too few points, use everything (or expand radius)
                        # Fallback to all points to be safe
                        local_target_points = target_points
                        local_target_rots = target_rots
                        local_target_scales = target_scales
                        # local_target_lifespan = target_lifespan
                        local_target_stability = target_stability
                        local_target_opacities = target_opacities
                        local_target_colors = target_colors

                    else:
                        local_target_points = target_points[local_mask]
                        local_target_rots = target_rots[local_mask]
                        local_target_scales = target_scales[local_mask]
                        # local_target_lifespan = target_lifespan[local_mask]
                        local_target_stability = target_stability[local_mask]
                        local_target_opacities = target_opacities[local_mask]
                        local_target_colors = target_colors[local_mask]

                        
                        if self.rerun_viewer:
                             rr.log("tracker/local_map_size", rr.Scalars(len(local_target_points)))

                    # Cache for visualization
                    self.local_target_points = local_target_points
                    self.local_target_rots = local_target_rots
                    self.local_target_scales = local_target_scales
                    self.local_target_stability = local_target_stability
                    self.local_target_opacities = local_target_opacities
                    self.local_target_colors = local_target_colors


                    self.reg.set_input_target(local_target_points)
                    self.reg.set_target_covariances_fromqs(local_target_rots.flatten(), local_target_scales.flatten())
                    
                    self.reg.set_target_weights(local_target_stability)
                    
                    # normals of cov
                    R_mats = Rotation.from_quat(local_target_rots).as_matrix() # (N, 3, 3)
                    # scale
                    min_scale_idx = np.argmin(local_target_scales, axis=1) # (N,)
                    # cols
                    normals = R_mats[np.arange(len(min_scale_idx)), :, min_scale_idx] # (N, 3)
                    
                    # cov of normals
                    mean_n = np.mean(normals, axis=0)
                    n_diff = normals - mean_n
                    Cn = (n_diff.T @ n_diff) / len(normals)
                    
                    # Cn is still built above: the planarity plot reads it
                    if _PLANARITY_SOURCE == "map":
                        self.reg.set_normal_covariance(Cn.astype(np.float32))
                    
                    # planarity score
                    if len(normals) > 0:
                        # principal component
                        eigvals, eigvecs = np.linalg.eigh(Cn)
                        # The eigenvector with the LARGEST eigenvalue (eigvecs[:,2]) represents 
                        # the direction of maximum spread of normals.
                        # The eigenvector with the SMALLEST eigenvalue (eigvecs[:,0]) represents
                        # the direction of minimum spread of normals (the dominant normal if all align).
                        
                        # Match fast_gicp Eigenvalue Ratio Logic (l3 / l2)
                        l2 = eigvals[1]
                        l3 = eigvals[2]
                        planar_ratio = l3 / (l2 + 1e-6)
                        
                        if planar_ratio <= 2.0:
                            planarity = 0.0
                        elif planar_ratio >= 6.0:
                            planarity = 1.0
                        else:
                            planarity = (planar_ratio - 2.0) / (6.0 - 2.0)
                        
                        self.reg.set_depth_map_planarity(float(planarity))
                        self.current_planarity = float(planarity)
                        
                        if self.rerun_viewer:
                            rr.log("deg/planarity", rr.Scalars(planarity))
                    
                    if self.rerun_viewer:
                        rr.log("deg/normal_cov", rr.Tensor(Cn))
                    
                    self.target_gaussians_ready[0] = 0
                    


                    if self.rerun_viewer:
                        # rr.set_time_sequence("step", self.iteration_images)
                        rr.set_time("log_time", duration=time.time() - self.total_start_time)
                        rr.log(f"pt/trackable/{self.iteration_images}", rr.Points3D(points, colors=colors, radii=0.01))

                elif if_mapping_keyframe:
                    
                    while self.is_tracking_keyframe_shared[0] or self.is_mapping_keyframe_shared[0]:
                        time.sleep(1e-15)
                    
                    rots = np.array(self.reg.get_source_rotationsq())
                    rots = np.reshape(rots, (-1,4))

                    R_d = Rotation.from_matrix(R)    # from camera R
                    R_d_q = R_d.as_quat()            # xyzw
                    rots = self.quaternion_multiply(R_d_q, rots)
                                        
                    scales = np.array(self.reg.get_source_scales())
                    scales = np.reshape(scales, (-1,3))

                    self.shared_new_gaussians.input_values(torch.tensor(points), torch.tensor(colors), 
                                                       torch.tensor(rots), torch.tensor(scales), 
                                                       torch.tensor(z_values), torch.tensor(trackable_filter))
                    
                    # Update local gaussians
                    # self.gaussians.add_from_pcd2_tensor(torch.tensor(points).float().cuda(), torch.tensor(colors).float().cuda(), 
                    #                                    torch.tensor(rots).float().cuda(), torch.tensor(scales).float().cuda(), 
                    #                                    torch.tensor(z_values).float().cuda(), torch.tensor(trackable_filter).cuda(),
                    #                                    torch.tensor(is_dynamic_mask).float().cuda())

                    
                    # depth_image already metric
                    self.shared_cam.setup_cam(R, T, current_image, depth_image)
                    self.shared_cam.cam_idx[0] = self.iteration_images
                    
                    self.is_mapping_keyframe_shared[0] = 1
            pbar.update(1)
            
            self.planarity_scores.append(self.current_planarity)
            self.prior_usage_scores.append(self.current_prior_used)
            self.eig_ratio_scores.append(self.current_eig_ratio)
            
            self.prev_rgb = current_image
            self.prev_depth = depth_image
#             self.prev_mask = mask_dynamic  # Store boolean mask for next frame
            
            while 1/((time.time() - self.total_start_time)/(self.iteration_images+1)) > 60.:    #30. float(self.test)
                time.sleep(1e-15)
                
            self.iteration_images += 1
        
        # Tracking end
        pbar.close()
        self.final_pose[:,:,:] = torch.tensor(self.poses).float()
        self.end_of_dataset[0] = 1
        
        print(f"System FPS: {1/((time.time()-self.total_start_time)/self.num_images):.2f}")
        ate, ate_std = self.evaluate_ate(self.trajmanager.gt_poses, self.poses)
        print(f"ATE RMSE: {ate*100.:.2f}")
        print(f"ATE STD: {ate_std*100.:.2f}")
        
        
        # Save TUM Trajectory
        self.save_tum_trajectory_to_file(self.poses, os.path.join(self.output_path, "trajectory.txt"))
        print(f"Trajectory saved to {os.path.join(self.output_path, 'trajectory.txt')}")

        # Save Trajectory Plot
        self.trajmanager.save_traj(self.poses, os.path.join(self.output_path, "trajectory_plot.png"))
        print(f"Trajectory plot saved to {os.path.join(self.output_path, 'trajectory_plot.png')}")

        # Save Validation Plots
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
            fig, axs = plt.subplots(3, 1, figsize=(10, 4.5), sharex=True, gridspec_kw={'height_ratios': [1, 2.5, 2.5]})
            
            modality_states = np.array([[1.0 if p else 0.0 for p in self.prior_usage_scores]])
            cmap = ListedColormap(['red', 'blue'])
            axs[0].imshow(modality_states, aspect='auto', cmap=cmap, interpolation='nearest',
                          extent=[0, len(self.prior_usage_scores), 0, 1])
            axs[0].set_yticks([]) # Hide y-axis ticks of the bar
            axs[0].set_ylabel("Tracking")
            
            # Add custom legend for the bar
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='blue', label='Prior Active'),
                               Patch(facecolor='red', label='ICP Only')]
            axs[0].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.3))

            
            axs[1].plot(self.eig_ratio_scores, color='purple')
            axs[1].axhline(y=_T_DEG, color='r', linestyle='--')
            axs[1].set_ylabel("Condition Number")
            axs[1].set_yscale('log')
            axs[1].grid(True, which="both", ls="-", alpha=0.5)
            
            axs[2].plot(self.planarity_scores, color='green', label="Planarity Score")
            axs[2].set_xlabel("Frame")
            axs[2].set_ylabel("Planarity Score")
            axs[2].set_ylim(-0.05, 1.05)
            axs[2].grid(True)
            axs[2].legend(loc='upper right')
            
            plt.tight_layout()
            planarity_plot_path = os.path.join(self.output_path, "planarity_plot.png")
            plt.savefig(planarity_plot_path)
            plt.close()
            print(f"Validation plot saved to {planarity_plot_path}")
        except Exception as e:
            print(f"Failed to plot validation graphs: {e}")

    def save_tum_trajectory_to_file(self, poses, save_path):
        from scipy.spatial.transform import Rotation
        
        with open(save_path, 'w') as f:
            # f.write("# timestamp tx ty tz qx qy qz qw\n")

            timestamps = []
            if hasattr(self.trajmanager, 'color_paths') and len(self.trajmanager.color_paths) > 0:

                 for p in self.trajmanager.color_paths:
                     dirname, filename = os.path.split(p)
                     name, ext = os.path.splitext(filename)
                     try:
                         t = float(name)
                         timestamps.append(t)
                     except ValueError:
                         # Fallback for Replica or other names (0, 1, 2...)
                         timestamps.append(None)
            
            for i, pose in enumerate(poses):
                # Get timestamp
                timestamp = 0.0
                if i < len(timestamps) and timestamps[i] is not None:
                     timestamp = timestamps[i]
                else:
                     timestamp = float(i) # Fallback index
                
                # Pose is T_wc (Camera to World)
                # TUM format expects position and quaternion of the camera in world
                t = pose[:3, 3]
                r_mat = pose[:3, :3]
                
                r = Rotation.from_matrix(r_mat)
                q = r.as_quat() # xyzw
                
                f.write(f"{timestamp:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

    
    def get_images(self, images_folder):
        rgb_images = []
        depth_images = []
        if self.trajmanager.which_dataset == "replica":
            image_files = os.listdir(images_folder)
            image_files = sorted(image_files.copy())
            for key in tqdm(image_files): 
                image_name = key.split(".")[0]
                depth_image_name = f"depth{image_name[5:]}"
                
                rgb_image = cv2.imread(f"{self.dataset_path}/images/{image_name}.jpg")
                depth_image = np.array(o3d.io.read_image(f"{self.dataset_path}/depth_images/{depth_image_name}.png"))
                
                rgb_images.append(rgb_image)
                depth_images.append(depth_image)
            return rgb_images, depth_images
        elif self.trajmanager.which_dataset == "tum" or self.trajmanager.which_dataset == "bonn":
            for i in tqdm(range(len(self.trajmanager.color_paths))):
                rgb_image = cv2.imread(self.trajmanager.color_paths[i])
                depth_image = np.array(o3d.io.read_image(self.trajmanager.depth_paths[i]))
                rgb_images.append(rgb_image)
                depth_images.append(depth_image)
            return rgb_images, depth_images

    def run_viewer(self, lower_speed=True):
        if network_gui.conn == None:
            network_gui.try_connect()
        if self.rerun_viewer:
             rr.save(f"{self.output_path}/test.rrd")
             
        # while network_gui.conn != None:
        #     if time.time()-self.last_t < 1/self.viewer_fps and lower_speed:
        #         break
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, self.pipe.convert_SHs_python, self.pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    
        #             # net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render_depth"]
        #             # net_image = torch.concat([net_image,net_image,net_image], dim=0)
        #             # net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=7.0) * 50).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    
        #         self.last_t = time.time()
        #         network_gui.send(net_image_bytes, self.dataset_path) 
        #         if do_training and (not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

    def quaternion_multiply(self, q1, Q2):
        # q1*Q2
        x0, y0, z0, w0 = q1
        
        return np.array([w0*Q2[:,0] + x0*Q2[:,3] + y0*Q2[:,2] - z0*Q2[:,1],
                        w0*Q2[:,1] + y0*Q2[:,3] + z0*Q2[:,0] - x0*Q2[:,2],
                        w0*Q2[:,2] + z0*Q2[:,3] + x0*Q2[:,1] - y0*Q2[:,0],
                        w0*Q2[:,3] - x0*Q2[:,0] - y0*Q2[:,1] - z0*Q2[:,2]]).T

    def set_downsample_filter( self, downsample_scale):
        # Get sampling idxs
        sample_interval = downsample_scale
        h_val = sample_interval * torch.arange(0,int(self.H/sample_interval)+1)
        h_val = h_val-1
        h_val[0] = 0
        h_val = h_val*self.W
        a, b = torch.meshgrid(h_val, torch.arange(0,self.W,sample_interval))
        # For tensor indexing, we need tuple
        pick_idxs = ((a+b).flatten(),)
        # Get u, v values
        v, u = torch.meshgrid(torch.arange(0,self.H), torch.arange(0,self.W))
        u = u.flatten()[pick_idxs]
        v = v.flatten()[pick_idxs]
        
        # Calculate xy values, not multiplied with z_values
        x_pre = (u-self.cx)/self.fx # * z_values
        y_pre = (v-self.cy)/self.fy # * z_values
        
        return pick_idxs, x_pre, y_pre

    def downsample_and_make_pointcloud2(self, depth_img, rgb_img):
        colors = torch.from_numpy(rgb_img).reshape(-1,3).float()[self.downsample_idxs] / 255.0
        z_values = torch.from_numpy(depth_img.astype(np.float32)).flatten()[self.downsample_idxs]

        valid_depth = (z_values > 0.01) & (z_values <= self.depth_trunc)
        
        selected_indices = torch.where(valid_depth)[0]
        
        z_sel = z_values[selected_indices]
        x_sel = self.x_pre[selected_indices] * z_sel
        y_sel = self.y_pre[selected_indices] * z_sel
        points = torch.stack([x_sel, y_sel, z_sel], dim=-1)
        colors = colors[selected_indices]
        trackable_filter = torch.arange(len(points))
        
        return points.numpy(), colors.numpy(), points[:,2].numpy(), trackable_filter.numpy()
    
    def eliminate_overlapped2(self, distances, threshold):
        
        # plt.hist(distances, bins=np.arange(0.,0.003,0.00001))
        # plt.show()
        new_p_indices = np.where(distances>threshold)    # 5e-5
        
        return new_p_indices
        
    def align(self, model, data):

        np.set_printoptions(precision=3, suppress=True)
        model_zerocentered = model - model.mean(1).reshape((3,-1))
        data_zerocentered = data - data.mean(1).reshape((3,-1))

        W = np.zeros((3, 3))
        for column in range(model.shape[1]):
            W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
        U, d, Vh = np.linalg.linalg.svd(W.transpose())
        S = np.matrix(np.identity(3))
        if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
            S[2, 2] = -1
        rot = U*S*Vh
        trans = data.mean(1).reshape((3,-1)) - rot * model.mean(1).reshape((3,-1))

        model_aligned = rot * model + trans
        alignment_error = model_aligned - data

        trans_error = np.sqrt(np.sum(np.multiply(
            alignment_error, alignment_error), 0)).A[0]

        return rot, trans, trans_error

    def evaluate_ate(self, gt_traj, est_traj):

        gt_traj_pts = [gt_traj[idx][:3,3] for idx in range(len(gt_traj))]
        gt_traj_pts_arr = np.array(gt_traj_pts)
        gt_traj_pts_tensor = torch.tensor(gt_traj_pts_arr)
        gt_traj_pts = torch.stack(tuple(gt_traj_pts_tensor)).detach().cpu().numpy().T

        est_traj_pts = [est_traj[idx][:3,3] for idx in range(len(est_traj))]
        est_traj_pts_arr = np.array(est_traj_pts)
        est_traj_pts_tensor = torch.tensor(est_traj_pts_arr)
        est_traj_pts = torch.stack(tuple(est_traj_pts_tensor)).detach().cpu().numpy().T

        _, _, trans_error = self.align(gt_traj_pts, est_traj_pts)

        avg_trans_error = trans_error.mean()
        std_trans_error = trans_error.std()

        return avg_trans_error, std_trans_error