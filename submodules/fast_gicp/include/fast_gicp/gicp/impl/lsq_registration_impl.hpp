#include <fast_gicp/gicp/lsq_registration.hpp>

#include <boost/format.hpp>
#include <fast_gicp/so3/so3.hpp>

namespace fast_gicp {

template <typename PointTarget, typename PointSource>
LsqRegistration<PointTarget, PointSource>::LsqRegistration() {
  this->reg_name_ = "LsqRegistration";
  max_iterations_ = 64;
  rotation_epsilon_ = 2e-3;
  transformation_epsilon_ = 5e-4;

  lsq_optimizer_type_ = LSQ_OPTIMIZER_TYPE::LevenbergMarquardt;
  lm_debug_print_ = false;
  lm_max_iterations_ = 10;
  // 1e-3 (Nielsen's tau range): 1e-9 is effectively Gauss-Newton and cannot
  // recover damping within the inner-iteration budget on near-singular Hessians
  lm_init_lambda_factor_ = 1e-3;
  lm_lambda_ = -1.0;

  lm_lambda_ = -1.0;

  final_hessian_.setIdentity();
  
  prior_pose_.setIdentity();
  degeneracy_threshold_ = 10.0;
  degeneracy_correction_enabled_ = false;
  used_prior_ = false;
  last_eigenvalues_.setZero();
  normal_covariance_.setZero();
  depth_map_planarity_ = 1.0;
}

template <typename PointTarget, typename PointSource>
LsqRegistration<PointTarget, PointSource>::~LsqRegistration() {}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setRotationEpsilon(double eps) {
  rotation_epsilon_ = eps;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setInitialLambdaFactor(double init_lambda_factor) {
  lm_init_lambda_factor_ = init_lambda_factor;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setDebugPrint(bool lm_debug_print) {
  lm_debug_print_ = lm_debug_print;
}

template <typename PointTarget, typename PointSource>
const Eigen::Matrix<double, 6, 6>& LsqRegistration<PointTarget, PointSource>::getFinalHessian() const {
  return final_hessian_;
}

template <typename PointTarget, typename PointSource>
double LsqRegistration<PointTarget, PointSource>::evaluateCost(const Eigen::Matrix4f& relative_pose, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) {
  return this->linearize(Eigen::Isometry3f(relative_pose).cast<double>(), H, b);
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  Eigen::Isometry3d x0 = Eigen::Isometry3d(guess.template cast<double>());

  lm_lambda_ = -1.0;
  converged_ = false;
  used_prior_ = false;

  if (lm_debug_print_) {
    std::cout << "********************************************" << std::endl;
    std::cout << "***************** optimize *****************" << std::endl;
    std::cout << "********************************************" << std::endl;
  }

  for (int i = 0; i < max_iterations_ && !converged_; i++) {
    nr_iterations_ = i;
    Eigen::Isometry3d delta;
    if (!step_optimize(x0, delta)) {
      std::cerr << "lm not converged!!" << std::endl;
      break;
    }

    converged_ = is_converged(delta);
  }

  final_transformation_ = x0.cast<float>().matrix();
  pcl::transformPointCloud(*input_, output, final_transformation_);
}

template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::is_converged(const Eigen::Isometry3d& delta) const {
  double accum = 0.0;
  Eigen::Matrix3d R = delta.linear() - Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = delta.translation();

  Eigen::Matrix3d r_delta = 1.0 / rotation_epsilon_ * R.array().abs();
  Eigen::Vector3d t_delta = 1.0 / transformation_epsilon_ * t.array().abs();

  return std::max(r_delta.maxCoeff(), t_delta.maxCoeff()) < 1;
}

template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::step_optimize(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta) {
  switch (lsq_optimizer_type_) {
    case LSQ_OPTIMIZER_TYPE::LevenbergMarquardt:
      return step_lm(x0, delta);
    case LSQ_OPTIMIZER_TYPE::GaussNewton:
      return step_gn(x0, delta);
  }

  return step_lm(x0, delta);
}

template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::step_gn(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta) {
  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;

  double y0 = linearize(x0, &H, &b);

  Eigen::LDLT<Eigen::Matrix<double, 6, 6>> solver(H);
  Eigen::Matrix<double, 6, 1> d = solver.solve(-b);

  delta.setIdentity();
  delta.linear() = so3_exp(d.head<3>()).toRotationMatrix();
  delta.translation() = d.tail<3>();

  x0 = x0 * delta;
  final_hessian_ = H;

  return true;
}
template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::step_lm(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta) {
  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  double y0 = linearize(x0, &H, &b);
  Eigen::Matrix<double, 6, 6> H_fill_combined = Eigen::Matrix<double, 6, 6>::Zero();
  // planarity 
  Eigen::Matrix<double, 3, 3> H_reg_planar = Eigen::Matrix<double, 3, 3>::Zero();
  bool is_planar_scene = false;

  if (lm_lambda_ < 0.0) {
    lm_lambda_ = lm_init_lambda_factor_ * std::max(1e-4, H.diagonal().array().abs().maxCoeff());
  }

  Eigen::Matrix<double, 6, 6> H_prior_effective = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> b_prior_effective = Eigen::Matrix<double, 6, 1>::Zero();
  double cost_prior_effective = 0.0;

  // -norm
  Eigen::Matrix<double, 6, 6> effective_prior_information = prior_information_;
  Eigen::Isometry3d current_prior_pose(prior_pose_);

  if (use_prior_information_) {
      Eigen::Isometry3d T_check = x0.inverse() * Eigen::Isometry3d(prior_pose_);
      double trans_diff = T_check.translation().norm();
      Eigen::AngleAxisd aa_check(T_check.linear());
      double rot_diff = std::abs(aa_check.angle());
      
      if (trans_diff > 1.0 || rot_diff > 1.0) {
          // reject prior
          effective_prior_information = H;
          current_prior_pose = x0; 
      } else {
          // norm
          double icp_trace = H.trace();
          double prior_trace = prior_information_.trace();
          if (prior_trace > 1e-9) {
              double scale = icp_trace / prior_trace;
              effective_prior_information *= scale;
          }
      }
  }

  bool is_degenerate = false;
  double dynamic_target_eval = 0.0;
  Eigen::Matrix<double, 6, 1> evals = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 6> evecs = Eigen::Matrix<double, 6, 6>::Identity();
  double gap_ratio = 1.0; 

  if (degeneracy_correction_enabled_) {
    // spectral analysis
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigensolver(H);
    evals = eigensolver.eigenvalues();
    evecs = eigensolver.eigenvectors();
    last_eigenvalues_ = evals;

    // correction
    //Eigen::Matrix<double, 6, 6> H_fill_combined = Eigen::Matrix<double, 6, 6>::Zero();
    H_fill_combined.setZero();
    // detection threshold: lambda_max / tau_deg (normalized condition-number test)
    double detection_threshold = evals.maxCoeff() / degeneracy_threshold_;


    gap_ratio = depth_map_planarity_; // Default: Planarity-based default
    
    if (normal_covariance_.norm() > 1e-9) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> nc_solver(normal_covariance_);
        Eigen::Vector3d nc_evals = nc_solver.eigenvalues(); // sorted as l1 <= l2 <= l3
        double l2 = nc_evals(1);
        double l3 = nc_evals(2); 

        // ratio l3/l2
        double planar_ratio = l3 / (l2 + 1e-6);
      
        if (planar_ratio <= 2.0) {
            gap_ratio = 0.0;
        } else if (planar_ratio >= 6.0) {
            gap_ratio = 1.0;
        } else {
            gap_ratio = (planar_ratio - 2.0) / (6.0 - 2.0);
        }

        if (lm_debug_print_) {
             std::cout << "[Degeneracy] Planar Ratio: " << planar_ratio << " -> Gap Ratio: " << gap_ratio << std::endl;
        }
    }
    else gap_ratio = 1.0;

    for (int k = 0; k < 6; k++) {
      if (evals(k) < detection_threshold ) {
        is_degenerate = true;
        used_prior_ = true;

        Eigen::Matrix<double, 6, 1> v_k = evecs.col(k);
        Eigen::Vector3d trans_v_k = v_k.tail<3>(); // dir in T



        double prior_stiffness_in_k = 0.0;
        Eigen::Matrix<double, 6, 1> err_vec; 
        if (use_prior_information_) {
             prior_stiffness_in_k = v_k.dot(effective_prior_information * v_k);
             
             // Pre-calculate error vector
             Eigen::Isometry3d T_err = x0.inverse() * current_prior_pose;
             Eigen::Vector3d t_err = T_err.translation();
             Eigen::AngleAxisd aa_err(T_err.linear());
             Eigen::Vector3d r_err = (std::abs(aa_err.angle()) > 1e-6) ? 
                                     Eigen::Vector3d(aa_err.axis() * aa_err.angle()) : 
                                     Eigen::Vector3d::Zero();
             err_vec << r_err, t_err;
        }
        double target_stiff = prior_stiffness_in_k ;
        double current_stiff = evals(k);
        
        //double stiffness_deficit = std::max(0.0, target_stiff - current_stiff);

        if (gap_ratio > 0.5) {
             // soft assignment
             double soft_w = std::exp(-5.0 * (1.0 - gap_ratio));
        
             // apply prior
             Eigen::Matrix<double, 6, 6> H_fill_k = target_stiff * v_k * v_k.transpose() * soft_w;

             H += H_fill_k;
             H_fill_combined += H_fill_k;

             // update grad
             if (use_prior_information_) {
                 b -= H_fill_k * err_vec;
                 
                 //cost accum
                 cost_prior_effective += 0.5 * err_vec.transpose() * H_fill_k * err_vec;
             }
        }
      }
    }
  }

  if (is_degenerate && use_prior_information_ && gap_ratio > 0.5) {
      // full prior support: stabilizes the extreme-degeneracy case beyond the
      // per-eigenvector projection above
      H_prior_effective += effective_prior_information;

      Eigen::Isometry3d T_err = x0.inverse() * current_prior_pose;
      Eigen::Vector3d t_err = T_err.translation();
      Eigen::AngleAxisd aa_err(T_err.linear());
      Eigen::Vector3d r_err = (std::abs(aa_err.angle()) > 1e-6) ? 
                              Eigen::Vector3d(aa_err.axis() * aa_err.angle()) : 
                              Eigen::Vector3d::Zero();

      Eigen::Matrix<double, 6, 1> err_vec;
      err_vec << r_err, t_err;

      b_prior_effective -= effective_prior_information * err_vec;
      cost_prior_effective += 0.5 * err_vec.transpose() * effective_prior_information * err_vec;
  }


  if (is_planar_scene) {
      //H.block<3, 3>(3, 3) += H_reg_planar;
  }

  // main lm loop
  double nu = 2.0;
  for (int i = 0; i < lm_max_iterations_; i++) {
    Eigen::Matrix<double, 6, 6> H_total = H + H_prior_effective;
    Eigen::Matrix<double, 6, 1> b_total = b + b_prior_effective; 
    
    // y0_total includes ICP cost (y0) + prior projection cost (cost_prior_effective)
    double y0_total = y0 + cost_prior_effective;

    Eigen::Matrix<double, 6, 6> A = H_total + lm_lambda_ * Eigen::Matrix<double, 6, 6>::Identity();
    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> solver(A);
    Eigen::Matrix<double, 6, 1> d = solver.solve(-b_total);

    if (d.array().isNaN().any()) {
      lm_lambda_ *= nu; nu *= 2.0;
      continue;
    }
    
    // helps with stability
    double max_rot = 0.1;   // rad (~5.7 degrees)
    double max_trans = 0.03; // meters

    double rot_norm = d.head<3>().norm();
    double trans_norm = d.tail<3>().norm();

    if (rot_norm > max_rot) d.head<3>() *= (max_rot / rot_norm);
    if (trans_norm > max_trans) d.tail<3>() *= (max_trans / trans_norm);

    delta.setIdentity();
    delta.linear() = so3_exp(d.head<3>()).toRotationMatrix();
    delta.translation() = d.tail<3>();
    Eigen::Isometry3d xi = x0 * delta;

    // error 
    double yi = compute_error(xi);
    double yi_prior_cost = 0.0;
        
        
    // ONLY evaluate the prior cost if it was actually used to compute the step
    if (use_prior_information_ && is_degenerate) {
         // recalculate the prior residual at the trial pose xi
         Eigen::Isometry3d T_err_i = xi.inverse() * current_prior_pose;
         Eigen::Vector3d t_err_i = T_err_i.translation();
         Eigen::AngleAxisd aa_err_i(T_err_i.linear());
         Eigen::Vector3d r_err_i = (std::abs(aa_err_i.angle()) > 1e-6) ? 
                                   Eigen::Vector3d(aa_err_i.axis() * aa_err_i.angle()) : 
                                   Eigen::Vector3d::Zero();
         
         Eigen::Matrix<double, 6, 1> err_vec_i;
         err_vec_i << r_err_i, t_err_i;
         
         // same cost; 
         yi_prior_cost = 0.5 * err_vec_i.transpose() * (H_prior_effective + H_fill_combined) * err_vec_i;
    }

    double yi_total = yi + yi_prior_cost;
    double actual_reduction = y0_total - yi_total;
    // account
    double predicted_reduction =
    -(b_total.dot(d) + 0.5 * d.transpose() * A * d);
    
    double rho = (predicted_reduction > 1e-12) ? (actual_reduction / predicted_reduction) : 0.0;

    if (rho > 0.0) { // same exit cond
      x0 = xi;

      lm_lambda_ *= std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1.0, 3.0));
      final_hessian_ = H_total;
      return true; 
    } else {
      lm_lambda_ *= nu; nu *= 2.0;
    }
    if (is_converged(delta)) return true;
  }
  return false;
}
}  // namespace fast_gicp