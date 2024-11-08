// EnvironmentMap.h

#ifndef ENVIRONMENTMAP_H
#define ENVIRONMENTMAP_H

#include <array>
#include <string>
#include <vector>

class EnvironmentMap {
 public:
  EnvironmentMap();
  EnvironmentMap(const std::string& filename);
  bool load(const std::string& filename);
  std::array<float, 3> sample(float azimuth, float elevation) const;
  std::array<float, 3> read_pixel(const float u, const float v) const;

  inline bool has_data() const { return !pixels.empty(); }

 private:
  int width, height;
  std::vector<float> pixels;
};

#endif  // ENVIRONMENTMAP_H