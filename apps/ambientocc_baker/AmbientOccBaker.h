#ifndef AMBIENTOCCBAKER_H
#define AMBIENTOCCBAKER_H

#include <embree3/rtcore.h>

#include "utils/AOProps.h"
struct aiScene;
class AmbientOcclusionBaker {
 public:
  AmbientOcclusionBaker() : scene(nullptr), epsilon(0.01f) { device = rtcNewDevice(NULL); }
  ~AmbientOcclusionBaker() {
    if (scene) rtcReleaseScene(scene);
    rtcReleaseDevice(device);
  }
  void init_dir_sampler(bool use_sphere, int num_samples);

  void set_scene(const aiScene* aiscene);
  void set_radius(float radius);
  void set_epsilon(float eps);
  bool render(const aiScene* aiscene);

  bool write_dir_samples(std::string outpath) const;
  bool write_buffer(std::string outpath) const;

 public:
  AOProps aoprops;

 private:
  RTCDevice device;
  RTCScene scene;
  float epsilon;
  float radius;
};
#endif  // AMBIENTOCCBAKER_H