
#include <iostream>
#include <vector>

#include "CameraModel.h"

void test_undistortion() {
  double width = 4032;
  double height = 3024;
  double cx = 2024.39579444945;
  double cy = 1465.47302999949;
  double k1 = 0.0625948881228134;
  double k2 = -0.220599711959651;
  double k3 = 0.268581973405246;
  double p1 = 0.00218213181153315;
  double p2 = -0.000499605302428288;
  double focal_mm = 4.65163086160806;
  double SensorSize = 5.64;
  double focal_px = focal_mm * width / SensorSize;

  using CameraT = PinholeRadialTangentialCamera;
  /////
  CameraT cammodel;
  cammodel.w = width;
  cammodel.h = height;
  cammodel.focal = focal_px;
  cammodel.cx = cx;
  cammodel.cy = cy;
  cammodel.skew = 0.;
  cammodel.aspect = 1.;
  cammodel.set_k1(k1);
  cammodel.set_k2(k2);
  cammodel.set_k3(k3);
  cammodel.set_k4(0.);
  cammodel.set_p1(p1);
  cammodel.set_p2(p2);

#if 1
  std::vector<int> cssp;
// cssp.push_back(DistortionModel::K1);
// cssp.push_back(DistortionModel::K2);
// cssp.push_back(DistortionModel::K3);
// cssp.push_back(DistortionModel::K4);
// cssp.push_back(DistortionModel::P1);
// cssp.push_back(DistortionModel::P2);
#if defined(FIT_INVERSE_COEFFS)
  cammodel.compute_inverse_coeff(cssp);

  std::cout << "Fitting" << std::endl;
  for (int i = 0; i < CameraT::NUM_PARAM; ++i) std::cout << cammodel.icoeff[i] << std::endl;
#endif
#else
  cammodel.set_f(3325.42, true);
  cammodel.set_cx(2024.4, true);
  cammodel.set_cy(1465.47, true);
  cammodel.set_k1(-0.197746, true);
  cammodel.set_k2(0.017423, true);
  cammodel.set_k3(-0.000771526, true);
  cammodel.set_k4(1.27384e-05, true);
  cammodel.set_p1(-0.0276889, true);
  cammodel.set_p2(-0.0185319, true);
  cammodel.set_b1(0, true);
  cammodel.set_b2(0, true);
  cammodel.inv_computed = true;
#endif

  for (int xi = 0; xi < cammodel.w; xi += 500) {
    for (int yi = 0; yi < cammodel.h; yi += 500) {
      Eigen::Vector2d undist;
      cammodel.fromImage(xi, yi, undist[0], undist[1]);
      auto dist = cammodel.distort(undist);
      // auto undistbar = cammodel.distort(dist, true);
      // std::cout << undist.transpose() << "<->" << dist.transpose() << " <-> " << undistbar.transpose() << std::endl;
      auto diff0 = (dist - undist).norm();
      auto undistbar2 = cammodel.undistort(dist);
      auto diff2 = (undistbar2 - undist).norm();
      std::cout << undist.transpose() << "<" << diff0 << ">" << dist.transpose() << " <" << diff2 << "> " << undistbar2.transpose() << std::endl;
    }
  }

  std::cout << "Done" << std::endl;
}

int main(int argc, char** argv) {
  test_undistortion();
  return 0;
}