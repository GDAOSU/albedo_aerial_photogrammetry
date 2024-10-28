#include "sphere_tessellation.h"

#include <cstdio>
#include "delaunator/delaunator.hpp"

// Martin Roberts’ original function
double gap0(int n) {
  if (n < 80)
    return 2.66;
  else if (n < 1e3)
    return 3.33;
  else if (n < 4e4)
    return 10;
  else
    return 25;
}
// Jacob’s suggestion
double gap1(int n) { return std::pow(2, 0.25 * std::log2(double(n) + 40)); }
// Fil’s tweaking
double gap2(int n) { return 0.5 * std::pow(double(n) + 40, 0.15); }
// Fil’s constant (obtained by optimizing for the size of the last cell)
double gap3(int n) { return 0.7012; }

void SphericalDelaunay::build_adjtriangles() {
  _adjtriangles.assign(_triangles.size(), -1);
  vtx_2_tri.resize(numpt());
  vtx_2_vtx.resize(numpt());
  for (int fi = 0; fi < numtri(); ++fi) {
    auto& v0 = _triangles[3 * fi];
    auto& v1 = _triangles[3 * fi + 1];
    auto& v2 = _triangles[3 * fi + 2];

    vtx_2_tri[v0].push_back(fi);
    vtx_2_tri[v1].push_back(fi);
    vtx_2_tri[v2].push_back(fi);

    vtx_2_vtx[v0].insert(v1);
    vtx_2_vtx[v0].insert(v2);
    vtx_2_vtx[v1].insert(v0);
    vtx_2_vtx[v1].insert(v2);
    vtx_2_vtx[v2].insert(v0);
    vtx_2_vtx[v2].insert(v1);
  }

  // go through each face, find adjacent triangles
  for (int fi = 0; fi < numtri(); ++fi) {
    auto& v0 = _triangles[3 * fi];
    auto& v1 = _triangles[3 * fi + 1];
    auto& v2 = _triangles[3 * fi + 2];

    // find opposite triangle for v0
    _adjtriangles[3 * fi] = find_adjacent_face(vtx_2_tri[v2], fi, v1);
    _adjtriangles[3 * fi + 1] = find_adjacent_face(vtx_2_tri[v0], fi, v2);
    _adjtriangles[3 * fi + 2] = find_adjacent_face(vtx_2_tri[v1], fi, v0);
  }
}

std::vector<double> SphericalDelaunay::Polar2Cartesian(std::vector<double> angles, double radius) {
  std::vector<double> coords(3 * angles.size() / 2);
  for (auto i = 0; i < angles.size() / 2; ++i) {
    double z = radius * std::sin(angles[2 * i + 1]);
    double x = radius * std::cos(angles[2 * i]) * std::cos(angles[2 * i + 1]);
    double y = radius * std::sin(angles[2 * i]) * std::cos(angles[2 * i + 1]);
    coords[3 * i + 0] = x;
    coords[3 * i + 1] = y;
    coords[3 * i + 2] = z;
  }
  return coords;
}

std::vector<double> SphericalDelaunay::StereographicProjection(std::vector<double> coords, double zoffset) {
  // stereographic projection unit sphere
  std::vector<double> coordsproj(coords.size() / 3 * 2);
  for (int i = 0; i < coords.size() / 3; ++i) {
    double offsetz = coords[3 * i + 2] + zoffset;
    coordsproj[2 * i] = coords[3 * i] / offsetz;
    coordsproj[2 * i + 1] = coords[3 * i + 1] / offsetz;
  }
  return coordsproj;
}

int SphericalDelaunay::find_adjacent_face(const std::vector<size_t>& candidate_face_list, size_t fi /*to skip*/, size_t tagv) {
  for (auto cfi : candidate_face_list) {
    if (cfi == fi) continue;
    auto _sit = _triangles.begin() + cfi * 3;
    auto _eit = _sit + 2;
    auto _pos = std::find(_sit, _eit, tagv);
    if (*_pos == tagv) return cfi;
  }
  return -1;
}

bool SphereDelaunay::generate(size_t n) {
  this->_angles = fibonacci_sphere_polar(n, this->_gap);
  this->_coords = Polar2Cartesian(this->_angles);

  bool spoleFound = false;
  std::vector<double> coords_wo_spole;
  if (this->_angles.size() == 0) return n == 0;

  if (this->_angles[2 * n - 2] == 0 && this->_angles[2 * n - 1] == -M_PI_2) {
    spoleFound = true;
    coords_wo_spole.assign(this->_coords.begin(), this->_coords.end() - 2);
  } else {
    spoleFound = false;
    coords_wo_spole.assign(this->_coords.begin(), this->_coords.end());
  }

  auto coordsproj_wo_pole = StereographicProjection(coords_wo_spole);

  delaunator::Delaunator dela(coordsproj_wo_pole);
  this->_triangles.resize(dela.triangles.size());
  this->_triangles.assign(dela.triangles.begin(), dela.triangles.end());

  // fill a fan around south pole
  size_t pv = dela.hull_start;
  std::vector<size_t> outer;
  do {
    outer.push_back(pv);
    pv = dela.hull_prev[pv];
  } while (pv != dela.hull_start);

  outer.push_back(dela.hull_start);
  for (int i = 0; i < outer.size() - 1; ++i) {
    this->_triangles.push_back(n - 1);
    this->_triangles.push_back(outer[i]);
    this->_triangles.push_back(outer[i + 1]);
  }

  build_adjtriangles();

  return true;
}

std::vector<double> SphereDelaunay::fibonacci_sphere_polar(int n, double (*gap)(int)) const {
  std::vector<double> angles(2 * n);

  double lonstep = DEG2RAD * 720. / (std::sqrt(5.) + 1.);
  double polegap = gap(n) * DEG2RAD;
  double latstep = -2. / (double(n) - 1 + 2 * polegap);  // not n-3!
  double latstart = 1. - polegap * latstep;
  if (n < 1) return angles;
  angles[0] = 0;
  angles[1] = M_PI_2;
  if (n == 1) return angles;
  for (int i = 1; i < n - 1; ++i) {
    double lon = lonstep * double(i) - 2. * M_PI * std::round((lonstep * double(i)) / 2. / M_PI);
    double lat = std::asin(latstart + double(i) * latstep);
    angles[2 * i] = lon;
    angles[2 * i + 1] = lat;
  }
  angles[2 * n - 2] = 0;
  angles[2 * n - 1] = -M_PI_2;
  return angles;
}

bool HemisphereDelaunay::generate(size_t n) {
  this->_angles = fibonacci_hemisphere_polar(n, this->_gap);
  this->_coords = Polar2Cartesian(this->_angles);
  auto coordsproj = StereographicProjection(this->_coords);
  delaunator::Delaunator dela(coordsproj);
  this->_triangles.assign(dela.triangles.begin(), dela.triangles.end());

  size_t pv = dela.hull_start;
  std::vector<size_t> outer;
  do {
    outer.push_back(pv);
    pv = dela.hull_prev[pv];
  } while (pv != dela.hull_start);

  outer.push_back(outer[0]);
  outer.push_back(outer[1]);

  for (int i = 0; i < outer.size() - 2; ++i) {
    const size_t& v0 = outer[i];
    const size_t& v1 = outer[i + 1];
    const size_t& v2 = outer[i + 2];
    if ((this->_coords[3 * v1 + 2] > this->_coords[3 * v0 + 2]) && (this->_coords[3 * v1 + 2] > this->_coords[3 * v2 + 2])) {
      this->_triangles.push_back(v0);
      this->_triangles.push_back(v1);
      this->_triangles.push_back(v2);
    }
  }

  build_adjtriangles();
  return true;
}

std::vector<double> HemisphereDelaunay::fibonacci_hemisphere_polar(int n, double (*gap)(int)) const {
  std::vector<double> angles(2 * n);

  double lonstep = DEG2RAD * 720. / (std::sqrt(5.) + 1.);
  double polegap = gap(n) * DEG2RAD;
  double latstep = -1. / (double(n) + 2 * polegap);  // not n-3!
  double latstart = 1. - polegap * latstep;

  if (n < 1) return angles;
  angles[0] = 0;
  angles[1] = M_PI_2;
  for (int i = 1; i < n; ++i) {
    double lon = lonstep * double(i) - 2. * M_PI * std::round((lonstep * double(i)) / 2. / M_PI);
    double lat = std::asin(latstart + double(i) * latstep);
    angles[2 * i] = lon;
    angles[2 * i + 1] = lat;
  }
  return angles;
}