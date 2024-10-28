#ifndef CAMERAMODEL_H
#define CAMERAMODEL_H

#include <omp.h>

#include <Eigen/Eigen>
#include <iostream>

class RadialTangentialDistortion;
/**
 * @brief
 * K = |f,    skew,   cx|
 *     |0,  aspect*f, cy|
 *     |0,       0,   1.|
 */
class PinholeCamera {
 public:
  int w, h;
  double focal;
  double skew;
  double aspect;
  double cx, cy;

  PinholeCamera() : skew(0.), aspect(1.) {}

  void rescale(double s) {
    int scaled_w = w * s;
    int scaled_h = h * s;
    double scale = (double(scaled_w) / double(w) + double(scaled_h) / double(h)) / 2.0;
    double scaled_focal = scale * focal;
    double scaled_skew = scale * skew;
    double scaled_cx = scale * cx;
    double scaled_cy = scale * cy;

    w = scaled_w;
    h = scaled_h;
    focal = scaled_focal;
    skew = scaled_skew;
    cx = scaled_cx;
    cy = scaled_cy;
  }

  void toImage(const double& x, const double& y, double& u, double& v) const {
    u = cx + x * focal + y * skew;
    v = cy + y * aspect * focal;
  }

  Eigen::Vector2d toImage(const Eigen::Vector2d& yconst) const {
    Eigen::Vector2d y;
    toImage(yconst[0], yconst[1], y[0], y[1]);
    return y;
  }

  void fromImage(const double& u, const double& v, double& x, double& y) const {
    y = (v - cy) / (aspect * focal);
    x = (u - y * skew - cx) / focal;
  }

  Eigen::Vector2d fromImage(const Eigen::Vector2d& yconst) const {
    Eigen::Vector2d y;
    fromImage(yconst[0], yconst[1], y[0], y[1]);
    return y;
  }
};

class RadialTangentialDistortion {
 public:
  enum { K1, K2, K3, K4, P1, P2, NUM_PARAM };
  double coeff[NUM_PARAM];
#if defined(FIT_INVERSE_COEFFS)
  double icoeff[NUM_PARAM];
  bool inv_computed;
#endif
  bool isDistortEnabled;
  RadialTangentialDistortion() {
    isDistortEnabled = false;
    std::fill_n(coeff, NUM_PARAM, 0.);
#if defined(FIT_INVERSE_COEFFS)
    std::fill_n(icoeff, NUM_PARAM, 0.);
    inv_computed = false;
#endif
  }

  inline void enable() { isDistortEnabled = true; }
  inline void disable() { isDistortEnabled = false; }

  inline void set_k1(double v, bool inverse = false) {
#if defined(FIT_INVERSE_COEFFS)
    if (inverse)
      icoeff[K1] = v;
    else
#endif
      coeff[K1] = v;
  }
  inline void set_k2(double v, bool inverse = false) {
#if defined(FIT_INVERSE_COEFFS)
    if (inverse)
      icoeff[K2] = v;
    else
#endif
      coeff[K2] = v;
  }
  inline void set_k3(double v, bool inverse = false) {
#if defined(FIT_INVERSE_COEFFS)
    if (inverse)
      icoeff[K3] = v;
    else
#endif
      coeff[K3] = v;
  }
  inline void set_k4(double v, bool inverse = false) {
#if defined(FIT_INVERSE_COEFFS)
    if (inverse)
      icoeff[K4] = v;
    else
#endif
      coeff[K4] = v;
  }
  inline void set_p1(double v, bool inverse = false) {
#if defined(FIT_INVERSE_COEFFS)
    if (inverse)
      icoeff[P1] = v;
    else
#endif
      coeff[P1] = v;
  }
  inline void set_p2(double v, bool inverse = false) {
#if defined(FIT_INVERSE_COEFFS)
    if (inverse)
      icoeff[P2] = v;
    else
#endif
      coeff[P2] = v;
  }

  inline const double& k1(bool inverse = false) const {
#if defined(FIT_INVERSE_COEFFS)
    if (inverse)
      return icoeff[K1];
    else
#endif
      return coeff[K1];
  }
  inline const double& k2(bool inverse = false) const {
#if defined(FIT_INVERSE_COEFFS)
    if (inverse)
      return icoeff[K2];
    else
#endif
      return coeff[K2];
  }
  inline const double& k3(bool inverse = false) const {
#if defined(FIT_INVERSE_COEFFS)
    if (inverse)
      return icoeff[K3];
    else
#endif
      return coeff[K3];
  }
  inline const double& k4(bool inverse = false) const {
#if defined(FIT_INVERSE_COEFFS)
    if (inverse)
      return icoeff[K4];
    else
#endif
      return coeff[K4];
  }
  inline const double& p1(bool inverse = false) const {
#if defined(FIT_INVERSE_COEFFS)
    if (inverse)
      return icoeff[P1];
    else
#endif
      return coeff[P1];
  }
  inline const double& p2(bool inverse = false) const {
#if defined(FIT_INVERSE_COEFFS)
    if (inverse)
      return icoeff[P2];
    else
#endif
      return coeff[P2];
  }

  Eigen::Vector2d distort(const Eigen::Vector2d& yconst, bool inverse = false) const {
#if defined(FIT_INVERSE_COEFFS)
    if (inverse && !inv_computed) return Eigen::Vector2d(0, 0);
#endif
    Eigen::Vector2d y;

    double mx2_u, my2_u, mxy_u, rho2_u, rho4_u, rho6_u, rho8_u, rad_dist_u;

    mx2_u = yconst[0] * yconst[0];
    my2_u = yconst[1] * yconst[1];
    mxy_u = yconst[0] * yconst[1];
    rho2_u = mx2_u + my2_u;
    rho4_u = rho2_u * rho2_u;
    rho6_u = rho4_u * rho2_u;
    rho8_u = rho4_u * rho4_u;
    rad_dist_u = k1(inverse) * rho2_u + k2(inverse) * rho4_u + k3(inverse) * rho6_u + k4(inverse) * rho8_u;

    y[0] = 2.0 * mxy_u * p2(inverse) + p1(inverse) * (3.0 * mx2_u + my2_u) + rad_dist_u * yconst[0] + yconst[0];
    y[1] = 2.0 * mxy_u * p1(inverse) + p2(inverse) * (mx2_u + 3.0 * my2_u) + rad_dist_u * yconst[1] + yconst[1];
    return y;
  }

  Eigen::Vector2d distort(const Eigen::Vector2d& yconst, Eigen::Matrix2d& J, bool inverse = false) const {
    Eigen::Vector2d y;

    double mx2_u, my2_u, mxy_u, rho2_u, rho4_u, rho6_u, rho8_u, rad_dist_u;

    mx2_u = yconst[0] * yconst[0];
    my2_u = yconst[1] * yconst[1];
    mxy_u = yconst[0] * yconst[1];
    rho2_u = mx2_u + my2_u;
    rho4_u = rho2_u * rho2_u;
    rho6_u = rho4_u * rho2_u;
    rho8_u = rho4_u * rho4_u;
    rad_dist_u = k1(inverse) * rho2_u + k2(inverse) * rho4_u + k3(inverse) * rho6_u + k4(inverse) * rho8_u;

    y[0] = 2.0 * mxy_u * p2(inverse) + p1(inverse) * (3.0 * mx2_u + my2_u) + rad_dist_u * yconst[0] + yconst[0];
    y[1] = 2.0 * mxy_u * p1(inverse) + p2(inverse) * (mx2_u + 3.0 * my2_u) + rad_dist_u * yconst[1] + yconst[1];

    double d_rad_dist_u = k1(inverse) + 2 * k2(inverse) * rho2_u + 3 * k3(inverse) * rho4_u + 4 * k4(inverse) * rho6_u;

    J(0, 0) = 2 * mx2_u * d_rad_dist_u + 6.0 * p1(inverse) * yconst[0] + 2.0 * p2(inverse) * yconst[1] + rad_dist_u + 1;
    J(0, 1) = 2 * mxy_u * d_rad_dist_u + 2.0 * p1(inverse) * yconst[1] + 2.0 * p2(inverse) * yconst[0];
    J(1, 0) = J(0, 1);
    J(1, 1) = 2 * my2_u * d_rad_dist_u + 2.0 * p1(inverse) * yconst[0] + 6.0 * p2(inverse) * yconst[1] + rad_dist_u + 1;

    return y;
  }

  Eigen::Vector2d undistort(const Eigen::Vector2d& yconst) const {
#if defined(FIT_INVERSE_COEFFS)
    Eigen::Vector2d ybar = distort(yconst, true);
#else
    Eigen::Vector2d ybar = yconst;
#endif
    const int n = 5;
    Eigen::Matrix2d F;

    Eigen::Vector2d y_tmp;
    int i = 0;
    for (i = 0; i < n; ++i) {
      y_tmp = distort(ybar, F);
      Eigen::Vector2d e(yconst - y_tmp);
      if (e.dot(e) < 1e-15) break;

      Eigen::Vector2d du = (F.transpose() * F).inverse() * F.transpose() * e;
      ybar += du;
    }
    return ybar;
  }
};

class PinholeRadialTangentialCamera : public PinholeCamera, public RadialTangentialDistortion {
 public:
  // TODO: Deep Copy Constructor
#if defined(FIT_INVERSE_COEFFS)
  bool compute_inverse_coeff(const std::vector<int> sspv = std::vector<int>());
#endif

  void initRayMap() {
    rayx.resize(w * h);
    rayy.resize(w * h);

#pragma omp parallel for collapse(2)
    for (int xi = 0; xi < w; ++xi)
      for (int yi = 0; yi < h; ++yi) {
        size_t idx = yi * w + xi;

        Eigen::Vector2d uv(double(xi) + 0.5, double(yi) + 0.5);
        Eigen::Vector2d distxy = fromImage(uv);
        Eigen::Vector2d undistxy;
        if (isDistortEnabled) {
          undistxy = undistort(distxy);
        } else {
          undistxy = distxy;
        }
        rayx[idx] = undistxy[0];
        rayy[idx] = undistxy[1];
      }
  }

  void sample_ray(const int xi, const int yi, bool distorted, float& x, float& y) const {
    if (!distorted) {
      Eigen::Vector2d uv(xi + 0.5, yi + 0.5);
      Eigen::Vector2d undistxy = fromImage(uv);
      x = undistxy[0];
      y = undistxy[1];
    } else {
      if (rayx.size() == (w * h) && rayy.size() == (w * h)) {
        // use cache
        size_t idx = yi * w + xi;
        x = rayx[idx];
        y = rayy[idx];
      } else {
        Eigen::Vector2d uv(xi + 0.5, yi + 0.5);
        Eigen::Vector2d distxy = fromImage(uv);
        Eigen::Vector2d undistxy;
        if (isDistortEnabled) {
          undistxy = undistort(distxy);
        } else {
          undistxy = distxy;
        }
        x = undistxy[0];
        y = undistxy[1];
      }
    }
  }

  std::vector<float> rayx, rayy;
};

#endif  // CAMERAMODEL_H