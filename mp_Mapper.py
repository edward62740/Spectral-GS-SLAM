import os
import torch
import torch.multiprocessing as mp
import torch.multiprocessing
import copy
import random
import sys
import cv2
import math
import numpy as np
import time
import rerun as rr
sys.path.append(os.path.dirname(__file__))
from arguments import SLAMParameters
from utils.traj_utils import TrajManager
from utils.loss_utils import l1_loss, ssim
from utils.graphics_utils import focal2fov, getProjectionMatrix
from scene import GaussianModel
from gaussian_renderer import render, render_3, network_gui
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import open3d as o3d
import matplotlib.pyplot as plt

class Pipe():
    def __init__(self, convert_SHs_python, compute_cov3D_python, debug):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug
        
class Mapper(SLAMParameters):
    def __init__(self, slam):   
        super().__init__()
        self.dataset_path = slam.dataset_path
        self.output_path = slam.output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = slam.verbose
        self.keyframe_th = float(slam.keyframe_th)
        self.trackable_opacity_th = slam.trackable_opacity_th
        self.save_results = slam.save_results
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
        
        self.downsample_rate = slam.downsample_rate
        self.viewer_fps = slam.viewer_fps
        self.keyframe_freq = slam.keyframe_freq
        
        # Camera poses
        self.trajmanager = TrajManager(self.camera_parameters[8], self.dataset_path)
        self.poses = [self.trajmanager.gt_poses[0]]
        # Keyframes(added to map gaussians)
        self.keyframe_idxs = []
        self.last_t = time.time()
        self.iteration_images = 0
        self.end_trigger = False
        self.covisible_keyframes = []
        self.new_target_trigger = False
        self.start_trigger = False
        self.if_mapping_keyframe = False
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
        if self.trajmanager.which_dataset == "replica":
            self.prune_th = 2.5
        else:
            self.prune_th = 10.0
        
        self.downsample_idxs, self.x_pre, self.y_pre = self.set_downsample_filter(self.downsample_rate)

        self.gaussians = GaussianModel(self.sh_degree)
        self.pipe = Pipe(self.convert_SHs_python, self.compute_cov3D_python, self.debug)
        self.bg_color = [1, 1, 1] if self.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")
        self.train_iter = 0
        self.mapping_cams = []
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
        self.final_pose = slam.final_pose
        self.demo = slam.demo
        self.is_mapping_process_started = slam.is_mapping_process_started
        self.iter_shared = slam.iter_shared

    def run(self):
        self.mapping()
    
    def mapping(self):
        t = torch.zeros((1,1)).float().cuda()
        self.train_iter = 0
        if self.verbose:
            try:
                network_gui.init("127.0.0.1", 6009)
            except OSError as e:
                print(f"[Mapper] Failed to initialize network GUI: {e}")
        
        if self.rerun_viewer:
            rr.init("3dgsviewer")
            rr.connect_grpc()
        
        # Mapping Process is ready to receive first frame
        self.is_mapping_process_started[0] = 1
        
        # Wait for initial gaussians
        while not self.is_tracking_keyframe_shared[0]:
            time.sleep(1e-15)
            
        self.total_start_time_viewer = time.time()
        
        points, colors, rots, scales, z_values, trackable_filter, _, _ = self.shared_new_gaussians.get_values()
        self.gaussians.create_from_pcd2_tensor(points, colors, rots, scales, z_values, trackable_filter)
        self.gaussians.spatial_lr_scale = self.scene_extent
        self.gaussians.training_setup(self)
        self.gaussians.update_learning_rate(1)
        self.gaussians.active_sh_degree = self.gaussians.max_sh_degree
        self.is_tracking_keyframe_shared[0] = 0

        
        if self.demo[0]:
            a = time.time()
            while (time.time()-a)<60.:
                print(30.-(time.time()-a))
                self.run_viewer()
        self.demo[0] = 0
        
        newcam = copy.deepcopy(self.shared_cam)
        newcam.on_cuda()

        self.mapping_cams.append(newcam)
        self.keyframe_idxs.append(newcam.cam_idx[0])
        self.new_keyframes.append(len(self.mapping_cams)-1)

        new_keyframe = False
        while True:
            if self.end_of_dataset[0]:
                break
 
            if self.verbose:
                self.run_viewer()       
            
            if self.is_tracking_keyframe_shared[0]:
                # get shared gaussians
                points, colors, rots, scales, z_values, trackable_filter, _, _ = self.shared_new_gaussians.get_values()
                
                # Add new gaussians to map gaussians
                self.gaussians.add_from_pcd2_tensor(points, colors, rots, scales, z_values, trackable_filter)

                # Allocate new target points to shared memory
                self.gaussians.update_stability()
                
                target_points, target_rots, target_scales, target_colors, target_opacities, target_stability, _ = self.gaussians.get_trackable_gaussians_tensor(self.trackable_opacity_th)
                self.shared_target_gaussians.input_values(target_points, target_rots, target_scales, target_colors, target_opacities, target_stability)
                self.target_gaussians_ready[0] = 1

                # Add new keyframe
                newcam = copy.deepcopy(self.shared_cam)
                newcam.on_cuda()
            
                self.mapping_cams.append(newcam)
                self.keyframe_idxs.append(newcam.cam_idx[0])
                self.new_keyframes.append(len(self.mapping_cams)-1)
                self.is_tracking_keyframe_shared[0] = 0

            elif self.is_mapping_keyframe_shared[0]:
                # get shared gaussians
                points, colors, rots, scales, z_values, _, _, _ = self.shared_new_gaussians.get_values()
                
                # Add new gaussians to map gaussians
                self.gaussians.add_from_pcd2_tensor(points, colors, rots, scales, z_values, [])
                
                # Add new keyframe
                newcam = copy.deepcopy(self.shared_cam)
                newcam.on_cuda()
                self.mapping_cams.append(newcam)
                self.keyframe_idxs.append(newcam.cam_idx[0])
                self.new_keyframes.append(len(self.mapping_cams)-1)
                self.is_mapping_keyframe_shared[0] = 0
        
            if len(self.mapping_cams)>0:
                
                # train once on new keyframe, and random
                if len(self.new_keyframes) > 0:
                    train_idx = self.new_keyframes.pop(0)
                    viewpoint_cam = self.mapping_cams[train_idx]
                    new_keyframe = True
                else:
                    train_idx = random.choice(range(len(self.mapping_cams)))
                    viewpoint_cam = self.mapping_cams[train_idx]
                
                if self.training_stage==0:
                    gt_image = viewpoint_cam.original_image.cuda()
                    gt_depth_image = viewpoint_cam.original_depth_image.cuda()
                elif self.training_stage==1:
                    gt_image = viewpoint_cam.rgb_level_1.cuda()
                    gt_depth_image = viewpoint_cam.depth_level_1.cuda()
                elif self.training_stage==2:
                    gt_image = viewpoint_cam.rgb_level_2.cuda()
                    gt_depth_image = viewpoint_cam.depth_level_2.cuda()
                
                self.training=True

                render_pkg = render_3(viewpoint_cam, self.gaussians, self.pipe, self.background, training_stage=self.training_stage)
                
                depth_image = render_pkg["render_depth"]
                image = render_pkg["render"]
                viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                
                mask = (gt_depth_image>0.)
                mask = mask.detach()

             
                # Loss
                Ll1_map, _ = l1_loss(image, gt_image)
                L_ssim_map, _ = ssim(image, gt_image)

                d_max = 10.
                Ll1_d_map, _ = l1_loss(depth_image/d_max, gt_depth_image/d_max)

                # Apply mask to residuals
                loss_rgb_map = (1.0 - self.lambda_dssim) * Ll1_map + self.lambda_dssim * (1.0 - L_ssim_map)
                loss_d_map = Ll1_d_map
                mask_norm = mask.sum().float() + 1e-6
                loss_rgb = (loss_rgb_map * mask).sum() / mask_norm
                loss_d = (loss_d_map * mask).sum() / mask_norm 
                
                loss = loss_rgb + 0.1*loss_d
                
                loss.backward()

                # update stability
                with torch.no_grad():
                    # alpha (sensitivity)=250.0, beta (decay)=0.2, gamma (threshold)=0.002
                    self.gaussians.update_gradient_stability(alpha=250.0, beta=0.2, threshold=0.002)

                with torch.no_grad():
                    # 1. Get Gaussian Centers
                    xyz = self.gaussians.get_xyz
                    
                    # 2. Transform to Camera Space
                    R = viewpoint_cam.R
                    T = viewpoint_cam.t
                    # R is w2c, T is w2c translation
                    # xyz_cam = R @ xyz + T
                    xyz_cam = (R @ xyz.T).T + T
                    
                    # 3. Project to Image Plane
                    z_cam = xyz_cam[:, 2]
                    valid_z = z_cam > 0.05
                    
                    # Project: u = fx * x/z + cx, v = fy * y/z + cy
                    u = (xyz_cam[:, 0] / (z_cam + 1e-6)) * self.fx + self.cx
                    v = (xyz_cam[:, 1] / (z_cam + 1e-6)) * self.fy + self.cy
                    
                    u_idx = u.long()
                    v_idx = v.long()
                    
                    # Bounds check
                    # gt_depth_image shape is usually (H, W) or (1, H, W)
                    if len(gt_depth_image.shape) == 3:
                        H_map, W_map = gt_depth_image.shape[1], gt_depth_image.shape[2]
                        gt_depth_map = gt_depth_image[0]
                    else:
                        H_map, W_map = gt_depth_image.shape[0], gt_depth_image.shape[1]
                        gt_depth_map = gt_depth_image

                    valid_coords = (u_idx >= 0) & (u_idx < W_map) & (v_idx >= 0) & (v_idx < H_map) & valid_z
                    
                    # Get indices of valid Gaussians to check
                    valid_indices = torch.nonzero(valid_coords).squeeze()
                    
                    if valid_indices.numel() > 0:
                        # Sample GT depths
                        gt_z = gt_depth_map[v_idx[valid_indices], u_idx[valid_indices]]
                        
                        # Get Gaussian depths
                        gauss_z = z_cam[valid_indices]
                        
                        # 5. The Check: Is Gaussian closer than GT Depth?
                        margin = 0.20 # 20cm margin
                        
                        # Floater condition: Valid GT measurement AND Gaussian is significantly closer
                        is_floater = (gt_z > 0) & (gauss_z < (gt_z - margin))
                        
                        # Get indices of floaters
                        floater_indices = valid_indices[is_floater]
                        
                        # 6. Zero out Gradients
                        if floater_indices.numel() > 0:
                            if self.gaussians._xyz.grad is not None:
                                self.gaussians._xyz.grad[floater_indices] = 0.
                            if self.gaussians._features_dc.grad is not None:
                                self.gaussians._features_dc.grad[floater_indices] = 0.
                            if self.gaussians._features_rest.grad is not None:
                                self.gaussians._features_rest.grad[floater_indices] = 0.
                            if self.gaussians._opacity.grad is not None:
                                self.gaussians._opacity.grad[floater_indices] = 0.
                            if self.gaussians._scaling.grad is not None:
                                self.gaussians._scaling.grad[floater_indices] = 0.
                            if self.gaussians._rotation.grad is not None:
                                self.gaussians._rotation.grad[floater_indices] = 0.

                with torch.no_grad():
                    if self.train_iter % 200 == 0:  # 200
                        self.gaussians.prune_large_and_transparent(0.005, self.prune_th/10.)
                        self.gaussians.prune_large_low_opacity(0.05, 0.5)
                        self.gaussians.prune_screen_space_bullies(viewpoint_cam, thresh=0.3)

                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none = True)
                    
                    if new_keyframe:
                        if self.rerun_viewer:
                            current_i = copy.deepcopy(self.iter_shared[0])
                            rgb_np = image.cpu().numpy().transpose(1,2,0)
                            rgb_np = np.clip(rgb_np, 0., 1.0) * 255
                            rr.set_time("log_time", duration=time.time() - self.total_start_time_viewer)
                            rr.log("rendered_rgb", rr.Image(rgb_np))


                        new_keyframe = False
                        
                self.training = False
                self.train_iter += 1
                # torch.cuda.empty_cache()

        # End of data

        if self.save_results and not self.rerun_viewer:
            self.gaussians.save_ply(os.path.join(self.output_path, "scene.ply"))

        
        self.calc_2d_metric()
    
    def run_viewer(self, lower_speed=True):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            if time.time()-self.last_t < 1/self.viewer_fps and lower_speed:
                break
            try:
                net_image_bytes = None
                custom_cam, do_training, self.pipe.convert_SHs_python, self.pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    
                    # net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render_depth"]
                    # net_image = torch.concat([net_image,net_image,net_image], dim=0)
                    # net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=7.0) * 50).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    
                self.last_t = time.time()
                network_gui.send(net_image_bytes, self.dataset_path) 
                if do_training and (not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

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
    
    def get_image_dirs(self, images_folder):
        color_paths = []
        depth_paths = []
        if self.trajmanager.which_dataset == "replica":
            images_folder = os.path.join(images_folder, "images")
            image_files = os.listdir(images_folder)
            image_files = sorted(image_files.copy())
            for key in tqdm(image_files):
                image_name = key.split(".")[0]
                depth_image_name = f"depth{image_name[5:]}"    
                color_paths.append(f"{self.dataset_path}/images/{image_name}.jpg")            
                depth_paths.append(f"{self.dataset_path}/depth_images/{depth_image_name}.png")
                
            return color_paths, depth_paths
        elif self.trajmanager.which_dataset == "tum" or self.trajmanager.which_dataset == "bonn": 
            return self.trajmanager.color_paths, self.trajmanager.depth_paths

    
    def calc_2d_metric(self):
        psnrs = []
        ssims = []
        lpips = []
        
        cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to("cuda")
        original_resolution = True
        image_names, depth_image_names = self.get_image_dirs(self.dataset_path)
        final_poses = self.final_pose
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        # --- Trajectory plot setup ---
        gt_poses_np = np.array(self.trajmanager.gt_poses)  # (N, 4, 4) c2w
        est_poses_np = final_poses.cpu().numpy()           # (M, 4, 4) c2w
        valid_mask_est = np.abs(est_poses_np[:, 3, 3]) > 0.001
        est_poses_np = est_poses_np[valid_mask_est]
        
        # Synchronise lengths, then align via Umeyama
        n_sync = min(len(gt_poses_np), len(est_poses_np))
        gt_sync  = gt_poses_np[:n_sync]
        est_sync = est_poses_np[:n_sync]
        gt_xyz   = gt_sync[:, :3, 3]
        est_xyz  = est_sync[:, :3, 3]
        
        # Umeyama alignment (est → gt)
        def umeyama_align(src, dst):
            """Return aligned src (scale * R @ src.T + t)."""
            n = src.shape[0]
            mu_src = src.mean(axis=0)
            mu_dst = dst.mean(axis=0)
            src_c = src - mu_src
            dst_c = dst - mu_dst
            sigma = np.dot(dst_c.T, src_c) / n
            U, d, Vt = np.linalg.svd(sigma)
            det_sign = np.linalg.det(np.dot(U, Vt))
            S = np.diag([1, 1, det_sign])
            R_align = np.dot(U, np.dot(S, Vt))
            var_src = (src_c ** 2).sum() / n
            c_align = (d * np.diag(S)).sum() / var_src if var_src > 1e-10 else 1.0
            t_align = mu_dst - c_align * np.dot(R_align, mu_src)
            aligned = (c_align * np.dot(R_align, src.T)).T + t_align
            return aligned
        
        try:
            est_aligned = umeyama_align(est_xyz, gt_xyz)
        except Exception:
            est_aligned = est_xyz  # fallback: no alignment
        
        with torch.no_grad():
            for i in tqdm(range(len(image_names))):
                gt_depth_ = []
                cam = self.mapping_cams[0]
                c2w = final_poses[i]
                
                if original_resolution:
                    gt_rgb = cv2.imread(image_names[i])
                    gt_depth = cv2.imread(depth_image_names[i] ,cv2.IMREAD_UNCHANGED).astype(np.float32)
                    
                    gt_rgb = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR)
                    gt_rgb = gt_rgb/255
                    gt_rgb_ = torch.from_numpy(gt_rgb).float().cuda().permute(2,0,1)
                    
                    gt_depth_ = torch.from_numpy(gt_depth).float().cuda().unsqueeze(0)
                else:
                    gt_rgb_ = cam.original_image.cuda()
                    gt_rgb = np.asarray(gt_rgb_.detach().cpu()).squeeze().transpose((1,2,0))
                    gt_depth_ = cam.original_depth_image.cuda()
                    gt_depth = np.asarray(cam.original_depth_image.detach().cpu()).squeeze()
                
                w2c = np.linalg.inv(c2w)
                # rendered
                R = w2c[:3,:3].transpose()
                T = w2c[:3,3]
                
                cam.R = torch.tensor(R)
                cam.t = torch.tensor(T)
                if original_resolution:
                    cam.image_width = gt_rgb_.shape[2]
                    cam.image_height = gt_rgb_.shape[1]
                else:
                    pass
                
                cam.update_matrix()
                # rendered rgb
                ours_rgb_ = render(cam, self.gaussians, self.pipe, self.background)["render"]
                ours_rgb_ = torch.clamp(ours_rgb_, 0., 1.).cuda()
                
                valid_depth_mask_ = (gt_depth_>0)
                
                gt_rgb_ = gt_rgb_ * valid_depth_mask_
                ours_rgb_ = ours_rgb_ * valid_depth_mask_
                
                square_error = (gt_rgb_-ours_rgb_)**2
                mse_error = torch.mean(torch.mean(square_error, axis=2))
                psnr = mse2psnr(mse_error)
                
                psnrs += [psnr.detach().cpu()]
                _, ssim_error = ssim(ours_rgb_, gt_rgb_)
                ssims += [ssim_error.detach().cpu()]
                lpips_value = cal_lpips(gt_rgb_.unsqueeze(0), ours_rgb_.unsqueeze(0))
                lpips += [lpips_value.detach().cpu()]
                
                if self.save_results and ((i+1)%100==0 or i==len(image_names)-1):
                    ours_rgb = np.asarray(ours_rgb_.detach().cpu()).squeeze().transpose((1,2,0))
                    
                    # Save GT and Render separately
                    gt_rgb_uint8 = (gt_rgb * 255).astype(np.uint8)
                    ours_rgb_uint8 = (ours_rgb * 255).astype(np.uint8)
                    
                    gt_bgr = cv2.cvtColor(gt_rgb_uint8, cv2.COLOR_RGB2BGR)
                    ours_bgr = cv2.cvtColor(ours_rgb_uint8, cv2.COLOR_RGB2BGR)
                    
                    cv2.imwrite(f"{self.output_path}/gt_{i}.png", gt_bgr)
                    cv2.imwrite(f"{self.output_path}/render_{i}.png", ours_bgr)
                
                torch.cuda.empty_cache()
            
            psnrs = np.array(psnrs)
            ssims = np.array(ssims)
            lpips = np.array(lpips)
            
            print(f"PSNR: {psnrs.mean():.2f}\nSSIM: {ssims.mean():.3f}\nLPIPS: {lpips.mean():.3f}")
            print(f"PSNR STD: {psnrs.std():.2f}\nSSIM STD: {ssims.std():.3f}\nLPIPS STD: {lpips.std():.3f}")
            
            # --- Save trajectory alignment plot ---
            try:
                for ax, (xi, yi, xlabel, ylabel) in zip(
                    axs,
                    [(0, 2, "x [m]", "z [m]"), (0, 1, "x [m]", "y [m]")]
                ):
                    ax.plot(gt_xyz[:n_sync, xi], gt_xyz[:n_sync, yi],
                            color="black", linewidth=1.5, label="GT")
                    ax.plot(est_aligned[:, xi], est_aligned[:, yi],
                            color="tab:blue", linewidth=1.5, label="Ours")
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
                    ax.grid(True)
                    ax.legend()
                fig.tight_layout()
                traj_out = os.path.join(self.output_path, "final_scene_trajectory.png")
                fig.savefig(traj_out, dpi=150)
                plt.close(fig)
                print(f"Saved trajectory plot to {traj_out}")
            except Exception as e:
                print(f"Error saving trajectory plot: {e}")
                import traceback; traceback.print_exc()
                plt.close(fig)

def mse2psnr(x):
    return -10.*torch.log(x)/torch.log(torch.tensor(10.))