#ifndef SPHERE_TESSELLATION_H
#define SPHERE_TESSELLATION_H

#include <math.h>

#include <set>
#include <vector>

double gap0(int n);
double gap1(int n);
double gap2(int n);
double gap3(int n);

/**
 * @brief Unit Spherical tessellation with Fabonacci and 2D Delaunay
 * https://www.redblobgames.com/x/1842-delaunay-voronoi-sphere/
 */
class SphericalDelaunay {
 public:
  inline size_t numpt() const { return _angles.size() / 2; }
  inline size_t numtri() const { return _triangles.size() / 3; }
  void clear() {
    _angles.clear();
    _coords.clear();
    _triangles.clear();
  }

  inline void set_angles(const std::vector<double>& v) { _angles = v; }
  inline void set_coords(const std::vector<double>& v) { _coords = v; }
  inline void set_triangles(const std::vector<size_t>& v) { _triangles = v; }
  inline void set_adjtriangles(const std::vector<int>& v) { _adjtriangles = v; }

  inline const std::vector<double>& angles() const { return _angles; }
  inline const std::vector<double>& coords() const { return _coords; }
  inline const std::vector<size_t>& triangles() const { return _triangles; }
  inline const std::vector<int>& adjtriangles() const { return _adjtriangles; }

  inline bool generate(size_t n, double (*gap)(int)) {
    this->_gap = gap;
    this->generate(n);
    return true;
  }
  bool generate(size_t n) { return false; };

 protected:
  const double RAD2DEG = 180. / M_PI;
  const double DEG2RAD = M_PI / 180.;
  double (*_gap)(int) = gap3;  // function of gap
  std::vector<double> _angles;
  std::vector<double> _coords;
  std::vector<size_t> _triangles;
  std::vector<int> _adjtriangles;

  std::vector<std::vector<size_t>> vtx_2_tri;  // adjacent triangles of given vertex
  std::vector<std::set<size_t>> vtx_2_vtx;     // adjacent vertcies of given vertex

  void build_adjtriangles();

 protected:
  static std::vector<double> Polar2Cartesian(std::vector<double> angles, double radius = 1.);
  static std::vector<double> StereographicProjection(std::vector<double> coords, double zoffset = 1.);

 private:
  int find_adjacent_face(const std::vector<size_t>& candidate_face_list, size_t fi /*to skip*/, size_t tagv);
};

class SphereDelaunay : public SphericalDelaunay {
 public:
  SphereDelaunay() {}
  SphereDelaunay(size_t n) { generate(n); }
  SphereDelaunay(size_t n, double (*gap)(int)) { SphericalDelaunay::generate(n, gap); }

  virtual bool generate(size_t n);

 private:
  std::vector<double> fibonacci_sphere_polar(int n, double (*gap)(int)) const;
};

class HemisphereDelaunay : public SphericalDelaunay {
 public:
  HemisphereDelaunay() {}
  HemisphereDelaunay(size_t n) { generate(n); }
  HemisphereDelaunay(size_t n, double (*gap)(int)) { SphericalDelaunay::generate(n, gap); }
  virtual bool generate(size_t n);

 private:
  std::vector<double> fibonacci_hemisphere_polar(int n, double (*gap)(int)) const;
};

#endif  // SPHERE_TESSELLATION_H