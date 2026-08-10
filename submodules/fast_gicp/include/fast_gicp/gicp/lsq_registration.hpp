#ifndef FAST_GICP_LSQ_REGISTRATION_HPP
#define FAST_GICP_LSQ_REGISTRATION_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

namespace fast_gicp {

enum class LSQ_OPTIMIZER_TYPE { GaussNewton, LevenbergMarquardt };

template<typename PointSource, typename PointTarget>
class LsqRegistration : public pcl::Registration<PointSource, PointTarget, float> {
public:
  using Scalar = float;
  using Matrix4 = typename pcl::Registration<PointSource, PointTarget, Scalar>::Matrix4;

  using PointCloudSource = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudSource;
  using PointCloudSourcePtr = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

  using PointCloudTarget = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudTarget;
  using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

#if PCL_VERSION >= PCL_VERSION_CALC(1, 10, 0)
  using Ptr = pcl::shared_ptr<LsqRegistration<PointSource, PointTarget>>;
  using ConstPtr = pcl::shared_ptr<const LsqRegistration<PointSource, PointTarget>>;
#else
  using Ptr = boost::shared_ptr<LsqRegistration<PointSource, PointTarget>>;
  using ConstPtr = boost::shared_ptr<const LsqRegistration<PointSource, PointTarget>>;
#endif

protected:
  using pcl::Registration<PointSource, PointTarget, Scalar>::input_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::nr_iterations_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::max_iterations_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::final_transformation_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::transformation_epsilon_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::converged_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LsqRegistration();
  virtual ~LsqRegistration();

  void setRotationEpsilon(double eps);
  void setInitialLambdaFactor(double init_lambda_factor);
  void setDebugPrint(bool lm_debug_print);

  const Eigen::Matrix<double, 6, 6>& getFinalHessian() const;

  double evaluateCost(const Eigen::Matrix4f& relative_pose, Eigen::Matrix<double, 6, 6>* H = nullptr, Eigen::Matrix<double, 6, 1>* b = nullptr);

  virtual void swapSourceAndTarget() {}
  virtual void clearSource() {}
  virtual void clearTarget() {}

  bool didUsePrior() const {
    return used_prior_;
  }

  Eigen::Matrix<double, 6, 1> getLastEigenvalues() const {
    return last_eigenvalues_;
  }

  std::pair<std::vector<double>, std::vector<double>> get_residuals() const;
  
  std::vector<double> get_sprt_ratios() const {
    return sprt_ratios_;
  }

  void reset_sprt() {
    sprt_ratios_.clear();
  }

  double getFinalLambda() const {
    return lm_lambda_;
  }

  void setPriorPose(const Eigen::Matrix4f& prior) {
    prior_pose_ = prior.cast<double>();
  }

  void setPriorInformation(const Eigen::Matrix<double, 6, 6>& info) {
    prior_information_ = info;
    use_prior_information_ = true;
  }

  void disablePriorInformation() {
    use_prior_information_ = false;
  }

  void setDegeneracyThreshold(double threshold) {
    degeneracy_threshold_ = threshold;
  }

  void setNormalCovariance(const Eigen::Matrix3f& cov) {
    normal_covariance_ = cov.cast<double>();
  }

  void setDepthMapPlanarity(double planarity) {
    depth_map_planarity_ = planarity;
  }

  void enableDegeneracyCorrection(bool enable) {
    degeneracy_correction_enabled_ = enable;
  }

  void setRobustLoss(bool enable, double k = 1.345) {
    use_robust_loss_ = enable;
    robust_k_ = k;
  }

  void setInverseDepthWeighting(bool enable) {
    use_inverse_depth_weighting_ = enable;
  }

protected:
  bool use_robust_loss_ = false;
  double robust_k_ = 1.345;
  bool use_inverse_depth_weighting_ = false;

  virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;

  bool is_converged(const Eigen::Isometry3d& delta) const;

  virtual double linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H = nullptr, Eigen::Matrix<double, 6, 1>* b = nullptr) = 0;
  virtual double compute_error(const Eigen::Isometry3d& trans) = 0;

  bool step_optimize(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta);
  bool step_gn(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta);
  bool step_lm(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta);

protected:
  double rotation_epsilon_;

  LSQ_OPTIMIZER_TYPE lsq_optimizer_type_;
  int lm_max_iterations_;
  double lm_init_lambda_factor_;
  double lm_lambda_;
  bool lm_debug_print_;

  Eigen::Matrix<double, 6, 6> final_hessian_;

  // spectral gs slam
  Eigen::Matrix4d prior_pose_;
  Eigen::Matrix<double, 6, 6> prior_information_;
  bool use_prior_information_ = false;
  double degeneracy_threshold_;
  bool degeneracy_correction_enabled_;
  bool used_prior_;
  Eigen::Matrix<double, 6, 1> last_eigenvalues_;

  Eigen::Matrix3d normal_covariance_;

  double depth_map_planarity_;

  std::vector<double> sprt_ratios_;
  std::vector<double> initial_residuals_;
};
}  // namespace fast_gicp

#endif