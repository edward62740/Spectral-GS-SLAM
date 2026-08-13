import cv2
import numpy as np
import torch
import rerun as rr
import time

# cheap feature matcher that produces pose estimates, 
#for any real deployment should be configured as needed

# clahe before orb detect, only if prev frame had < N feats
_ORB_CLAHE = True
_ORB_CLAHE_NFEAT = 150

_ORB_NFEATURES = 2048

# fast threshold, normal and the permissive one for sparse scenes
_ORB_FAST_THRESHOLD = 20
_ORB_FAST_THRESHOLD_SPARSE = 8

# lowe ratio for knn matching
_ORB_MATCH_RATIO = 0.80

# opencv ransac draws from a global rng, so fix it or runs are not reproducible
_ORB_RANSAC_SEED = 0


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
        self.orb = cv2.ORB_create(nfeatures=_ORB_NFEATURES, scaleFactor=1.2, nlevels=8,
                                  edgeThreshold=19, firstLevel=0, WTA_K=2,
                                  scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31,
                                  fastThreshold=_ORB_FAST_THRESHOLD)

        # same, but recovers weaker corners in sparse scenes
        self.orb_sparse = cv2.ORB_create(nfeatures=_ORB_NFEATURES, scaleFactor=1.2, nlevels=8,
                                         edgeThreshold=19, firstLevel=0, WTA_K=2,
                                         scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31,
                                         fastThreshold=_ORB_FAST_THRESHOLD_SPARSE)

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if _ORB_CLAHE else None

        self.prev_nfeat = None       # prev frame feat count
        self._prev_was_clahe = False
        self._prev_was_sparse = False

        # low cost hamming match on cpu. no crossCheck, it throws away too much
        # when descriptors are few; the ratio test below does the filtering
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.kp_prev = None
        self.des_prev = None
        self.prev_image_gray = None

        self.call_count = 0

        cv2.setRNGSeed(_ORB_RANSAC_SEED)

    def _detect_orb(self, gray, use_clahe=False, sparse=False):
        """
        detect orb feats
        sparse: lower fast threshold to recover weaker corners
        use_clahe: equalize before detect
        """
        if use_clahe:
            if self.clahe is None:
                return [], None
            gray_input = self.clahe.apply(gray)
        else:
            gray_input = gray

        orb = self.orb_sparse if sparse else self.orb
        kp, des = orb.detectAndCompute(gray_input, None)

        if kp is None:
            kp = []

        return kp, des

    def _detect_with_fallback(self, gray):
        """
        detect progressively: normal -> lower fast th -> lower fast th + clahe
        Returns: kp, des, sparse_mode, clahe_mode
        """
        kp, des = self._detect_orb(gray, use_clahe=False, sparse=False)
        if len(kp) >= _ORB_CLAHE_NFEAT:
            return kp, des, False, False

        kp_sparse, des_sparse = self._detect_orb(gray, use_clahe=False, sparse=True)
        if len(kp_sparse) >= _ORB_CLAHE_NFEAT:
            return kp_sparse, des_sparse, True, False

        if self.clahe is not None:
            kp_clahe, des_clahe = self._detect_orb(gray, use_clahe=True, sparse=True)
            if len(kp_clahe) > len(kp_sparse):
                return kp_clahe, des_clahe, True, True

        # best we have
        return kp_sparse, des_sparse, True, False

    def visualize_matches(self, img_live, kp_live, img_prev, kp_prev, matches, valid_mask=None):
        if not self.rerun_viewer:
            return

        img_matches = cv2.drawMatches(img_prev, kp_prev, img_live, kp_live, matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        rr.log("cam/current/orb_matches", rr.Image(img_matches))

    def get_prior(self, live_image, prev_image, prev_depth, T_prev):
        """
        calculate the pose of the prior
        T_prev: Previous Pose (Camera -> World in our tracker)
        Returns: T_est (Estimated Pose Camera -> World), H_pnp (PnP Hessian)
        """
        self.call_count += 1

        curr_gray = cv2.cvtColor(live_image, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)

        # only try the fallback if the last frame was already feat starved
        use_sparse_mode = (self.prev_nfeat is not None
                           and self.prev_nfeat < _ORB_CLAHE_NFEAT)

        if use_sparse_mode:
            kp_curr, des_curr, sparse_mode, use_clahe_now = self._detect_with_fallback(curr_gray)
        else:
            kp_curr, des_curr = self._detect_orb(curr_gray, use_clahe=False, sparse=False)
            sparse_mode = False
            use_clahe_now = False

        # reuse cache iff same preproc
        if (self.kp_prev is not None and self.des_prev is not None
                and self._prev_was_clahe == use_clahe_now
                and self._prev_was_sparse == sparse_mode):
            kp_prev, des_prev = self.kp_prev, self.des_prev
        else:
            kp_prev, des_prev = self._detect_orb(prev_gray, use_clahe=use_clahe_now,
                                                 sparse=sparse_mode)

        if (des_curr is None or des_prev is None
                or len(des_curr) < 10 or len(des_prev) < 10):
            self.kp_prev = None
            self.des_prev = None
            self.prev_nfeat = len(des_curr) if des_curr is not None else 0
            self._prev_was_clahe = use_clahe_now
            self._prev_was_sparse = sparse_mode
            return None, None

        # knn + lowe ratio
        knn_matches = self.bf.knnMatch(des_prev, des_curr, k=2)

        matches = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < _ORB_MATCH_RATIO * n.distance:
                matches.append(m)

        matches = sorted(matches, key=lambda x: x.distance)[:2048]

        if len(matches) < 8:
            self.kp_prev = kp_curr
            self.des_prev = des_curr
            self.prev_nfeat = len(kp_curr)
            self._prev_was_clahe = use_clahe_now
            self._prev_was_sparse = sparse_mode
            return None, None

        # back project the prev frame matches
        obj_points = []
        img_points = []
        good_matches = []

        for m in matches:
            # queryIdx is index in des_prev, trainIdx is index in des_curr
            idx_prev = m.queryIdx
            idx_curr = m.trainIdx

            u_prev, v_prev = map(int, kp_prev[idx_prev].pt)
            u_curr, v_curr = map(int, kp_curr[idx_curr].pt)

            if u_prev < 0 or u_prev >= self.W or v_prev < 0 or v_prev >= self.H:
                continue

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
            self.kp_prev = kp_curr
            self.des_prev = des_curr
            self.prev_nfeat = len(kp_curr)
            self._prev_was_clahe = use_clahe_now
            self._prev_was_sparse = sparse_mode
            return None, None

        rvec_prior = np.zeros((3, 1), dtype=np.float64)
        tvec_prior = np.zeros((3, 1), dtype=np.float64)

        cv2.setRNGSeed(_ORB_RANSAC_SEED + self.call_count)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(obj_points, img_points, self.K, None,
                                                          rvec=rvec_prior, tvec=tvec_prior,
                                                          useExtrinsicGuess=True,
                                                          iterationsCount=150,
                                                          reprojectionError=3,
                                                          flags=cv2.SOLVEPNP_ITERATIVE)

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
            T_est = T_prev @ T_curr_to_prev

            self.kp_prev = kp_curr
            self.des_prev = des_curr
            self.prev_nfeat = len(kp_curr)
            self._prev_was_clahe = use_clahe_now
            self._prev_was_sparse = sparse_mode

            return T_est.astype(np.float64), H_pnp.astype(np.float64)

        else:
            # cache current frame even when pnp fails
            self.kp_prev = kp_curr
            self.des_prev = des_curr
            self.prev_nfeat = len(kp_curr)
            self._prev_was_clahe = use_clahe_now
            self._prev_was_sparse = sparse_mode

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

        N = len(obj_points)

        # Projection Jacobian Jp (N, 2, 3)
        Jp = np.zeros((N, 2, 3))
        Jp[:, 0, 0] = self.fx / z
        Jp[:, 0, 2] = -self.fx * x / z2
        Jp[:, 1, 1] = self.fy / z
        Jp[:, 1, 2] = -self.fy * y / z2

        # Pose Jacobian Jxi (N, 3, 6)
        Jxi = np.zeros((N, 3, 6))

        # rot
        Jxi[:, 0, 1] = z
        Jxi[:, 0, 2] = -y
        Jxi[:, 1, 0] = -z
        Jxi[:, 1, 2] = x
        Jxi[:, 2, 0] = y
        Jxi[:, 2, 1] = -x

        # trans
        Jxi[:, 0, 3] = 1
        Jxi[:, 1, 4] = 1
        Jxi[:, 2, 5] = 1

        J = np.matmul(Jp, Jxi)

        H = np.einsum("nij,nik->jk", J, J)
        H = H / 1e4

        return H
