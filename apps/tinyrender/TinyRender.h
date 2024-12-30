#ifndef TINYRENDER_H
#define TINYRENDER_H

#include <embree3/rtcore.h>

#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "CameraModel.h"
#include "utils/AOProps.h"

struct aiScene;

class TinyRender {
 public:
  using CameraT = PinholeRadialTangentialCamera;
  enum SUPPORTED_BUFFERS { GEOMID, BARYCENTRIC, DEPTH, NORMAL, PRIMID, COLOR, INDIR, SUNVIS, SKYVIS, NUM_BUFFERS };
  const static std::string BUFFERS_NAME[NUM_BUFFERS];
  std::string bufferName(int bid) { return BUFFERS_NAME[bid]; }

 public:
  TinyRender();

  ~TinyRender();

  bool load_imagedataset(std::string inpath);
  bool load_modeldataset(std::string inpath);
  bool prepare_data();

  inline const RTCBounds& scene_bounds() { return scenebounds; }
  inline void set_downsampler(int v) { downsampler = v; }
  inline void set_clipping(float near_, float far_) {
    znear = near_;
    zfar = far_;
  }
  inline void set_cull(bool cull) { this->cull = cull; }
  bool set_scene(const aiScene* aiscene);
  inline void set_require_buffers(const std::unordered_map<int, bool>& value) { requireBuffers = value; }
  inline void set_output_dirs(const std::unordered_map<int, std::filesystem::path>& value) { outputDirs = value; }
  bool set_cameras(const nlohmann::json& intrinsicJS);
  CameraT get_camera(std::string camid) const;
  AOProps aoprops;
  SphericalDelaunay dirsamples;
  bool render_frame(size_t id);

  inline size_t num_images() const { return extrinsicJS.size(); }

 private:
  std::unordered_map<int, bool> requireBuffers;
  std::unordered_map<int, std::filesystem::path> outputDirs;

  RTCDevice device;
  RTCScene scene;
  RTCBounds scenebounds;
  int downsampler;
  float znear, zfar;
  bool cull;
  std::unordered_map<std::string, CameraT> cams, scaled_cams;
  const float EPSILON = 0.01f;

  std::vector<std::string> imagelist;

  nlohmann::json imagejson;
  nlohmann::json modeljson;
  nlohmann::json intrinsicJS;
  nlohmann::json extrinsicJS;
  nlohmann::json imagemetaJS;
  std::string cachedModelPath;
  std::string AOResultPath;
  std::string dirSamplesPath;

 public:
  std::vector<float> depthBuffer, normalBuffer;
  std::vector<uint32_t> geomidBuffer;
  std::vector<float> barycentricBuffer;
  std::vector<uint8_t> colorBuffer, indirBuffer;
  std::vector<uint8_t> sunVisBuffer, skyVisBuffer;
  std::vector<uint32_t> skyHitBuffer;
};

#endif  // TINYRENDER_H