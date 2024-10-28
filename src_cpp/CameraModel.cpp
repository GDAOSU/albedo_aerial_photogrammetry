#include "CameraModel.h"

#include <iostream>

#if defined(USE_CERES_SOLVER)
#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/local_parameterization.h>
#include <ceres/problem.h>
#include <ceres/solver.h>

struct RadialTangentialCostFunctor {
 public:
  RadialTangentialCostFunctor(const double& sx, const double& sy, const double& tx, const double& ty)
      : sx(sx), sy(sy), tx(tx), ty(ty) {}

  template <typename T>
  bool operator()(const T* parameters, T* residuals) const {
    const T k1 = parameters[RadialTangentialDistortion::K1];
    const T k2 = parameters[RadialTangentialDistortion::K2];
    const T k3 = parameters[RadialTangentialDistortion::K3];
    const T k4 = parameters[RadialTangentialDistortion::K4];
    const T p1 = parameters[RadialTangentialDistortion::P1];
    const T p2 = parameters[RadialTangentialDistortion::P2];

    T mx2_u = T(sx * sx);
    T my2_u = T(sy * sy);
    T mxy_u = T(sx * sy);
    T rho2_u = mx2_u + my2_u;
    T rho4_u = rho2_u * rho2_u;
    T rho6_u = rho4_u * rho2_u;
    T rho8_u = rho4_u * rho4_u;
    T rad_dist_u = k1 * rho2_u + k2 * rho4_u + k3 * rho6_u + k4 * rho8_u;

    T xbar = 2.0 * mxy_u * p2 + p1 * (3.0 * mx2_u + my2_u) + rad_dist_u * T(sx) + T(sx);
    T ybar = 2.0 * mxy_u * p1 + p2 * (mx2_u + 3.0 * my2_u) + rad_dist_u * T(sy) + T(sy);
    residuals[0] = T(tx) - xbar;
    residuals[1] = T(ty) - ybar;
    return true;
  }

  static ceres::CostFunction* create(const double& sx, const double& sy, const double& tx, const double& ty) {
    return new ceres::AutoDiffCostFunction<RadialTangentialCostFunctor, 2, RadialTangentialDistortion::NUM_PARAM>(
        new RadialTangentialCostFunctor(sx, sy, tx, ty));
  }

 private:
  const double sx, sy;
  const double tx, ty;
};
#if defined(FIT_INVERSE_COEFFS)
bool PinholeRadialTangentialCamera::compute_inverse_coeff(const std::vector<int> sspv) {
  if (!isDistortEnabled) return true;
  for (auto& i : sspv) icoeff[i] = 0.;

  std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> undist_dist_pts;

  undist_dist_pts.reserve(w * h);
  const int stride = std::min(std::min(w, h) / 16, 100);
  for (int xi = 0; xi < w; xi += stride)
    for (int yi = 0; yi < h; yi += stride) {
      Eigen::Vector2d undist, dist;
      fromImage(xi, yi, undist[0], undist[1]);
      dist = distort(undist);
      undist_dist_pts.emplace_back(std::make_pair(undist, dist));
    }

  ceres::Problem problem;

  problem.AddParameterBlock(icoeff, NUM_PARAM);

#if (CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1)
  problem.SetManifold(icoeff, new ceres::SubsetManifold(NUM_PARAM, sspv));
#else
  problem.SetParameterization(icoeff, new ceres::SubsetParameterization(NUM_PARAM, sspv));
#endif

  for (int i = 0; i < undist_dist_pts.size(); ++i) {
    auto& pp = undist_dist_pts[i];
    problem.AddResidualBlock(RadialTangentialCostFunctor::create(pp.second[0], pp.second[1], pp.first[0], pp.first[1]),
                             nullptr, icoeff);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  // options.minimizer_progress_to_stdout = true;
  options.minimizer_progress_to_stdout = false;
  options.logging_type = ceres::SILENT;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // std::cout << summary.FullReport() << std::endl;
  inv_computed = true;

  return true;
}
#endif
#endif