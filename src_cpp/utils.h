#include <vector>

#include "embcommon/math/affinespace.h"
#include "embcommon/math/bbox.h"
#include "embcommon/math/math.h"
#include "embcommon/math/vec2.h"
#include "embcommon/math/vec3.h"
#include "embcommon/math/vec4.h"
#include "math.h"

std::vector<embree::Vec3fa> fibonacci_hemisphere(int num_samples) {
  std::vector<embree::Vec3fa> points;
  double phi = M_PI * (3. - std::sqrt(5.));  // golden angle in radians
  for (size_t i = 0; i < num_samples; ++i) {
    double z = 1 - (i / float(num_samples - 1));  // z goes from 1 to 0
    double radius = std::sqrt(1. - z * z);        // radius at z
    double theta = phi * i;                       // golden angle increment
    double x = std::cos(theta) * radius;
    double y = std::sin(theta) * radius;
    points.emplace_back(x, y, z);
  }
  return points;
}

std::vector<embree::Vec3fa> fibonacci_sphere(int num_samples) {
  std::vector<embree::Vec3fa> points;
  std::vector<embree::Vec2fa> angles;
  double phi = M_PI * (3. - std::sqrt(5.));  // golden angle in radians
  for (size_t i = 0; i < num_samples; ++i) {
    double z = 1. - (i / float(num_samples - 1)) * 2.;  // z goes from 1 to 0
    double radius = std::sqrt(1. - z * z);              // radius at z
    double theta = phi * i;                             // golden angle increment
    double x = std::cos(theta) * radius;
    double y = std::sin(theta) * radius;
    points.emplace_back(x, y, z);
  }
  return points;
}

std::vector<embree::Vec3fa> cosine_hemisphere(int num_samples) {
  std::vector<embree::Vec3fa> points = fibonacci_sphere(num_samples);
  for (auto& p : points) {
    p.z += 1.f + 1e-5;
    p = normalize(p);
  }
  return points;
}

std::vector<embree::Vec3fa> align_to_normal(const std::vector<embree::Vec3fa>& points, const embree::Vec3fa& normal) {
  std::vector<embree::Vec3fa> rotpoints;
  rotpoints.reserve(points.size());
  if (normal.z > (1.f - 1e-5))
    for (const auto& p : points) rotpoints.push_back(p);
  else {
    embree::LinearSpace3fa R;
    R.vz = embree::normalize(normal);
    R.vy = embree::Vec3fa(-normal.y / std::sqrt(1.f - normal.z * normal.z), normal.x / std::sqrt(1.f - normal.z * normal.z), 0);
    R.vx = embree::cross(R.vy, R.vz);
    R = R.transposed();
    for (const auto& p : points) rotpoints.push_back(R * p);
  }
  return rotpoints;
}