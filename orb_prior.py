import cv2
import numpy as np
import torch
import rerun as rr
import time

# clahe before orb detect, only if prev frame had < N feats
_ORB_CLAHE = True
_ORB_CLAHE_NFEAT = 150

class ORBFeaturePrior:
    def __init__(self, W, H, fx, fy, cx, cy, rerun_viewer=False):
        # generic interface to work with neural matchers as well
        self.W = W
        self.H = H
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.rerun_viewer = rerun_viewer
        
        # intrinsic
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]], dtype=np.float64)
                           
        # the prior (can be any features, but this is fastest)
        self.orb = cv2.ORB_create(nfeatures=2048)

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if _ORB_CLAHE else None
        self.prev_nfeat = None       # prev frame feat count
        self._prev_was_clahe = False

        # low cost hamming match on cpu
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        

        self.kp_prev = None
        self.des_prev = None
        self.prev_image_gray = None

        self.call_count = 0

    def visualize_matches(self, img_live, kp_live, img_prev, kp_prev, matches, valid_mask=None):
        if not self.rerun_viewer:
            return
            
        # Draw matches
        img_matches = cv2.drawMatches(img_prev, kp_prev, img_live, kp_live, matches, None, 
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        rr.log("cam/current/orb_matches", rr.Image(img_matches))

    def get_prior(self, live_image, prev_image, prev_depth, T_prev):
        """
        calculate the pose of the prior
        T_prev: Previous Pose (Camera -> World in our tracker)
        Returns: T_est (Estimated Pose Camera -> World)
        """
        self.call_count += 1
        
        
        curr_gray = cv2.cvtColor(live_image, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)

        # CLAHE for the feature sparse scenes, essentially only fires on notex_far
        use_clahe_now = (self.clahe is not None and self.prev_nfeat is not None
                         and self.prev_nfeat < _ORB_CLAHE_NFEAT)

        def _prep(g):
            return self.clahe.apply(g) if use_clahe_now else g

        # ORB {kps, descs}
        kp_curr, des_curr = self.orb.detectAndCompute(_prep(curr_gray), None)

        # reuse cache iff same preproc
        if (self.kp_prev is not None and self.des_prev is not None
                and self._prev_was_clahe == use_clahe_now):
             kp_prev, des_prev = self.kp_prev, self.des_prev
        else:
             kp_prev, des_prev = self.orb.detectAndCompute(_prep(prev_gray), None)
        if des_curr is None or des_prev is None or len(des_curr) < 10 or len(des_prev) < 10:
             self.kp_prev = None
             self.des_prev = None
             self.prev_nfeat = len(des_curr) if des_curr is not None else 0
             self._prev_was_clahe = use_clahe_now
             return None, None

        # MATCH
        matches = self.bf.match(des_prev, des_curr)
        matches = sorted(matches, key = lambda x:x.distance)
        
        # cap
        matches = matches[:2048] 
        
        # if too few for valid pnp, then exit as a fail condition
        if len(matches) < 8:
            return None

    
        obj_points = []
        img_points = []
        
        good_matches = []
        
        # back proj
        for m in matches:
            # queryIdx is index in des_prev
            # trainIdx is index in des_curr
            
            idx_prev = m.queryIdx
            idx_curr = m.trainIdx
            
            u_prev, v_prev = map(int, kp_prev[idx_prev].pt)
            u_curr, v_curr = map(int, kp_curr[idx_curr].pt)
            
            if u_prev < 0 or u_prev >= self.W or v_prev < 0 or v_prev >= self.H: continue

            d = prev_depth[v_prev, u_prev]
            
            if d > 0.1 and d < 50.0:
                z = d
                x = (u_prev - self.cx) * z / self.fx
                y = (v_prev - self.cy) * z / self.fy
                
                obj_points.append([x, y, z])
                img_points.append([u_curr, v_curr])
                good_matches.append(m)
                
        obj_points = np.array(obj_points, dtype=np.float64)
        img_points = np.array(img_points, dtype=np.float64)
        
        if len(obj_points) < 20:
            return None

        # PnP solver
        rvec_prior = np.zeros((3, 1), dtype=np.float64)
        tvec_prior = np.zeros((3, 1), dtype=np.float64) # unused; no effect

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_points, 
            img_points, 
            self.K, 
            None,
            rvec=rvec_prior,
            tvec=tvec_prior,
            useExtrinsicGuess=True, 
            iterationsCount=150,
            reprojectionError=3,
            flags=cv2.SOLVEPNP_ITERATIVE # more stable
        )
        if success:
            if self.rerun_viewer and self.call_count % 5 == 0:
                 self.visualize_matches(live_image, kp_curr, prev_image, kp_prev, good_matches)
        
            if inliers is not None:
                inlier_idx = inliers.flatten()
                H_pnp = self._compute_hessian(obj_points[inlier_idx], rvec, tvec)
            else:
                H_pnp = self._compute_hessian(obj_points, rvec, tvec)

            R, _ = cv2.Rodrigues(rvec)
            t = tvec.flatten()
            
            T_prev_to_curr = np.eye(4)
            T_prev_to_curr[:3, :3] = R
            T_prev_to_curr[:3, 3] = t
            
            # X_world = T_wc_prev * X_prev
            # X_curr = T_prev_to_curr * X_prev
            # X_prev = T_prev_to_curr^-1 * X_curr
            # X_world = T_wc_prev * T_prev_to_curr^-1 * X_curr
            # T_wc_curr = T_wc_prev * T_prev_to_curr^-1
            
            T_curr_to_prev = np.linalg.inv(T_prev_to_curr)
            
            # Note: T_prev passed in is T_wc (4x4 numpy)
            T_est = T_prev @ T_curr_to_prev
            
            # update the cache
            self.kp_prev = kp_curr
            self.des_prev = des_curr
            self.prev_nfeat = len(kp_curr)
            self._prev_was_clahe = use_clahe_now

            # T_est is 4x4 numpy array C2W
            return T_est.astype(np.float64), H_pnp.astype(np.float64)
            
        else:
            # cache condition if fail
            self.kp_prev = kp_curr
            self.des_prev = des_curr
            self.prev_nfeat = len(kp_curr)
            self._prev_was_clahe = use_clahe_now
            return None, None

    def _compute_hessian(self, obj_points, rvec, tvec):
        """
        Compute the 6x6 Hessian matrix for the PnP problem.
        obj_points: (N, 3) 3D points in the previous camera frame.
        rvec, tvec: map from previous to current camera frame.
        Returns: 6x6 Hessian matrix (rot, trans).
        """
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)
        
        # Transform points to current camera frame
        P_curr = (R @ obj_points.T + t).T # (N, 3)
        
        x = P_curr[:, 0]
        y = P_curr[:, 1]
        z = P_curr[:, 2]
        z2 = z*z
        
        # Projection Jacobian Jp (N, 2, 3)
        # pi(x,y,z) = [fx*x/z + cx, fy*y/z + cy]
        # d_pi / d_x = fx/z, 0, -fx*x/z^2
        # d_pi / d_y = 0, fy/z, -fy*y/z^2
        
        N = len(obj_points)
        Jp = np.zeros((N, 2, 3))
        Jp[:, 0, 0] = self.fx / z
        Jp[:, 0, 2] = -self.fx * x / z2
        Jp[:, 1, 1] = self.fy / z
        Jp[:, 1, 2] = -self.fy * y / z2
        
        # Pose Jacobian J_xi (N, 3, 6)
        # dx / d_xi = [- [x]_x , I ]
        # [x]_x = [ 0  -z   y ]
        #         [ z   0  -x ]
        #         [-y   x   0 ]
        # -[x]_x = [ 0   z  -y ]
        #          [-z   0   x ]
        #          [ y  -x   0 ]
        
        Jxi = np.zeros((N, 3, 6))
        # Rotation part (skew-symmetric)
        Jxi[:, 0, 1] = z
        Jxi[:, 0, 2] = -y
        Jxi[:, 1, 0] = -z
        Jxi[:, 1, 2] = x
        Jxi[:, 2, 0] = y
        Jxi[:, 2, 1] = -x
        # Translation part (identity)
        Jxi[:, 0, 3] = 1
        Jxi[:, 1, 4] = 1
        Jxi[:, 2, 5] = 1
        
        # Total Jacobian J = Jp @ Jxi (N, 2, 6)
        # J_i = Jp_i (2,3) @ Jxi_i (3,6)
        J = np.matmul(Jp, Jxi) # (N, 2, 6)
        
        # Hessian H = sum(J^T @ J)
        # We can use Einstein summation for efficiency
        H = np.einsum('nij,nik->jk', J, J) # (6, 6)

        H = H / 1e4
        return H
