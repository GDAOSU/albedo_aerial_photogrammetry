#include "TinyRender.h"

// clang-format off
#include <embree3/rtcore.h>
#include <embcommon/algorithms/parallel_for.h>
#include <embcommon/core/ray.h>
#include <embcommon/tasking/taskschedulertbb.h>
// clang-format on

#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include "stb/stb_image_write.h"

#include <assimp/Importer.hpp>
#include <filesystem>
#include <fstream>

#include "utils/writer.h"

namespace fs = std::filesystem;

const std::string TinyRender::BUFFERS_NAME[TinyRender::NUM_BUFFERS] = {
    "geomid", "barycentric", "depth", "normal", "primid", "color", "indir", "sunvis", "skyvis"};

TinyRender::TinyRender() : scene(nullptr) {
  device = rtcNewDevice(NULL);
  znear = embree::zero;
  zfar = embree::inf;
  cull = false;
}

TinyRender::~TinyRender() {
  if (scene) rtcReleaseScene(scene);
  rtcReleaseDevice(device);
}

bool TinyRender::set_cameras(const nlohmann::json& intrinsicJS) {
  for (auto it = intrinsicJS.begin(); it != intrinsicJS.end(); ++it) {
    CameraT cam;

    cam.w = it.value()["Width"].get<int>();
    cam.h = it.value()["Height"].get<int>();
    cam.focal = it.value()["focal"].get<double>();
    cam.cx = it.value()["cx"].get<double>();
    cam.cy = it.value()["cy"].get<double>();
    cam.skew = it.value()["skew"].get<double>();
    cam.aspect = it.value()["aspect"].get<double>();

    std::string distort_model = it.value().value<std::string>("distmodel", "none");
    if ("radial-tangential") {
      cam.set_k1(it.value().value<double>("K1", 0.));
      cam.set_k2(it.value().value<double>("K2", 0.));
      cam.set_k3(it.value().value<double>("K3", 0.));
      cam.set_k4(it.value().value<double>("K4", 0.));
      cam.set_p1(it.value().value<double>("P1", 0.));
      cam.set_p2(it.value().value<double>("P2", 0.));
      cam.enable();
    } else {
      cam.disable();
    }

    cams[it.key()] = cam;
    CameraT scaled_cam = cam;
    scaled_cam.rescale(downsampler);
#if defined(FIT_INVERSE_COEFFS)
    scaled_cam.compute_inverse_coeff();
#endif
    scaled_cam.initRayMap();
    scaled_cams[it.key()] = scaled_cam;
  }
  return true;
}

bool TinyRender::set_scene(const aiScene* aiscene) {
  // Checking list
  for (int meshID = 0; meshID < aiscene->mNumMeshes; ++meshID) {
    const aiMesh* _mesh = aiscene->mMeshes[meshID];
    if (!_mesh->HasNormals()) {
      spdlog::error("No normal found!");
      return false;
    }
    if (!_mesh->HasPositions()) {
      spdlog::critical("No vertex found!");
      return false;
    }
    if (!_mesh->HasFaces()) {
      spdlog::critical("No face found!");
      return false;
    }
    if (requireBuffers[SKYVIS]) {
      size_t aopropsNumVertices = aoprops.buffer[meshID].size() / aoprops.pack_stride;
      if (_mesh->mNumVertices != aopropsNumVertices) {
        spdlog::error("[AOProps][{}] Vertex Number doesn't match: {} vs {}", meshID, aopropsNumVertices,
                      _mesh->mNumVertices);
        return false;
      }
    }
  }
  if (requireBuffers[SKYVIS] && aoprops.buffer.size() != aiscene->mNumMeshes) {
    spdlog::error("[AOProps] Number of Meshes don't match: {} vs {}", aoprops.buffer.size(), aiscene->mNumMeshes);
    return false;
  }
  // Passed checking
  if (scene) rtcReleaseScene(scene);
  scene = rtcNewScene(device);
  for (int meshID = 0; meshID < aiscene->mNumMeshes; ++meshID) {
    const aiMesh* _mesh = aiscene->mMeshes[meshID];

    RTCGeometry _geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
    rtcSetGeometryBuildQuality(_geom, RTC_BUILD_QUALITY_HIGH);
    rtcSetGeometryVertexAttributeCount(_geom, 2);

    float* _vertices = (float*)rtcSetNewGeometryBuffer(_geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                                                       3 * sizeof(float), _mesh->mNumVertices);

    float* normals = (float*)rtcSetNewGeometryBuffer(_geom, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, RTC_FORMAT_FLOAT3,
                                                     3 * sizeof(float), _mesh->mNumVertices);

    // Sxs: WARNING: Memory leak here
    float* pertri_normals = new float[_mesh->mNumFaces * 3];

    float* skyvis = nullptr;
    if (requireBuffers[SKYVIS]) {
      skyvis = (float*)rtcSetNewGeometryBuffer(_geom, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 1, RTC_FORMAT_FLOAT,
                                               sizeof(float), _mesh->mNumVertices);
    }

    uint32_t* _faces = (uint32_t*)rtcSetNewGeometryBuffer(_geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                                                          3 * sizeof(uint32_t), _mesh->mNumFaces);
#pragma omp parallel for
    for (int _i = 0; _i < _mesh->mNumVertices; ++_i) {
      _vertices[_i * 3 + 0] = _mesh->mVertices[_i].x;
      _vertices[_i * 3 + 1] = _mesh->mVertices[_i].y;
      _vertices[_i * 3 + 2] = _mesh->mVertices[_i].z;

      normals[_i * 3 + 0] = _mesh->mNormals[_i].x;
      normals[_i * 3 + 1] = _mesh->mNormals[_i].y;
      normals[_i * 3 + 2] = _mesh->mNormals[_i].z;

      // Note: the hard coded uniform sky
      if (requireBuffers[SKYVIS]) {
        float numer = 0;
        float denom = 0;
        for (int ri = 0; ri < aoprops.num_samples; ++ri) {
// #define CONSTSKY_COSINE
#ifdef CONSTSKY_COSINE
          float cosTerm = normals[_i * 3] * dirsamples.coords()[ri * 3] +
                          normals[_i * 3 + 1] * dirsamples.coords()[ri * 3 + 1] +
                          normals[_i * 3 + 2] * dirsamples.coords()[ri * 3 + 2];
          // float cosTheta Sky Model = dirsamples.coords()[ri * 3 + 2];  // := z = w_i @ [0,0,1]
          if (aoprops.test(meshID, _i, ri)) {
            // @max

            numer += std::max<float>(0.f, cosTerm);
          }
          denom += std::max<float>(0.f, cosTerm);
#else
          float cosTerm = normals[_i * 3] * dirsamples.coords()[ri * 3] +
                          normals[_i * 3 + 1] * dirsamples.coords()[ri * 3 + 1] +
                          normals[_i * 3 + 2] * dirsamples.coords()[ri * 3 + 2];
          if (aoprops.test(meshID, _i, ri)) {
            // @max
            numer += std::max<float>(0.f, cosTerm);
          }
          denom += std::abs(cosTerm);
#endif
        }

        skyvis[_i] = numer / denom;
      }
    }
#pragma omp parallel for
    for (int _i = 0; _i < _mesh->mNumFaces; ++_i) {
      _faces[_i * 3 + 0] = _mesh->mFaces[_i].mIndices[0];
      _faces[_i * 3 + 1] = _mesh->mFaces[_i].mIndices[1];
      _faces[_i * 3 + 2] = _mesh->mFaces[_i].mIndices[2];

      // compute face normal
      embree::Vec3fa v0(_vertices[_faces[_i * 3] * 3], _vertices[_faces[_i * 3] * 3 + 1],
                        _vertices[_faces[_i * 3] * 3 + 2]);
      embree::Vec3fa v1(_vertices[_faces[_i * 3 + 1] * 3], _vertices[_faces[_i * 3 + 1] * 3 + 1],
                        _vertices[_faces[_i * 3 + 1] * 3 + 2]);
      embree::Vec3fa v2(_vertices[_faces[_i * 3 + 2] * 3], _vertices[_faces[_i * 3 + 2] * 3 + 1],
                        _vertices[_faces[_i * 3 + 2] * 3 + 2]);

      embree::Vec3fa e1 = v1 - v0;
      embree::Vec3fa e2 = v2 - v0;
      embree::Vec3fa faceNormal = embree::cross(e1, e2);
      faceNormal = embree::normalize(faceNormal);

      pertri_normals[_i * 3] = faceNormal.x;
      pertri_normals[_i * 3 + 1] = faceNormal.y;
      pertri_normals[_i * 3 + 2] = faceNormal.z;
    }

    rtcSetGeometryUserData(_geom, pertri_normals);

    rtcCommitGeometry(_geom);
    uint32_t _geomID = rtcAttachGeometry(scene, _geom);
    rtcReleaseGeometry(_geom);
  }
  rtcCommitScene(scene);
  rtcGetSceneBounds(scene, &scenebounds);
  return true;
}

TinyRender::CameraT TinyRender::get_camera(std::string camid) const {
  auto camit = cams.find(camid);
  if (camit->first != camid) {
    throw new std::runtime_error(fmt::format("camera {} not found.", camid));
  } else {
    return camit->second;
  }
}

bool TinyRender::load_imagedataset(std::string inpath) {
  std::ifstream ifs(inpath);
  if (!ifs.good()) {
    spdlog::critical("Input imagedataset not found: {}", inpath);
    return false;
  }
  ifs >> imagejson;
  ifs.close();

  intrinsicJS = imagejson["Intrinsic"];
  extrinsicJS = imagejson["Extrinsic"];
  imagemetaJS = imagejson["ImageMeta"];

  spdlog::info("Found {} Cameras", intrinsicJS.size());
  spdlog::info("Found {}/{} Images", imagemetaJS.size(), extrinsicJS.size());

  imagelist.clear();
  for (auto it = extrinsicJS.begin(); it != extrinsicJS.end(); ++it) imagelist.push_back(it.key());
  return true;
}

bool TinyRender::load_modeldataset(std::string inpath) {
  std::ifstream ifs(inpath);
  if (!ifs.good()) {
    spdlog::critical("Input modeldataset not found: {}", inpath);
    return false;
  }
  ifs >> modeljson;
  ifs.close();

  nlohmann::json modelJS = modeljson["models"];
  spdlog::info("Found {} Models", modelJS.size());

  cachedModelPath = modeljson.value<std::string>("cachedmodel", "");
  AOResultPath = modeljson.value<std::string>("aoprops", "");
  dirSamplesPath = modeljson.value<std::string>("dirsamples", "");

  spdlog::debug("Found cachedmodel {}", cachedModelPath);
  spdlog::debug("Found AOProps {}", AOResultPath);
  spdlog::debug("Found dirsample json {}", dirSamplesPath);

  if (!fs::exists(fs::path(cachedModelPath))) {
    spdlog::critical("Cannot open Cached Model in {}", cachedModelPath);
    return false;
  }
  if (requireBuffers[SKYVIS]) {
    if (!fs::exists(fs::path(AOResultPath))) {
      spdlog::critical("Cannot open AO results: {}", AOResultPath);
      return false;
    }

    LoadSphericalDelaunay(&dirsamples, dirSamplesPath);
  }

  return true;
}

bool TinyRender::prepare_data() {
  if (imagejson.contains("metadata")) {
    std::string imagejson_srs = imagejson["metadata"].value<std::string>("SRS", "Unknown");
    std::string modeljson_srs = modeljson["metadata"].value<std::string>("SRS", "Unknown");
    if (imagejson_srs != modeljson_srs) {
      spdlog::critical("SRS not compatible: {} != {}", imagejson_srs, modeljson_srs);
      return false;
    }
  }
  spdlog::stopwatch sw;
  spdlog::info("Loading Cameras");
  set_cameras(intrinsicJS);
  spdlog::info("Load Cameras: {:.3}s", sw);
  spdlog::info("Loading Scene");
  sw.reset();
  Assimp::Importer importer;
  const aiScene* readScene = importer.ReadFile(cachedModelPath, aiProcess_GenNormals | aiProcess_ValidateDataStructure);
  if (requireBuffers[SKYVIS])
    if (!aoprops.read_buffer(AOResultPath)) {
      spdlog::error("AOProps load failed.");
      return false;
    }

  if (!set_scene(readScene)) return false;

  importer.FreeScene();
  spdlog::info("Load Scene: {:.3}s", sw);

  spdlog::info("Uploaded Scene Bounds: ({},{},{}) - ({},{},{})", scenebounds.lower_x, scenebounds.lower_y,
               scenebounds.lower_z, scenebounds.upper_x, scenebounds.upper_y, scenebounds.upper_z);
  return true;
}

bool TinyRender::render_frame(size_t id) {
  bool with_pertri_normals = true;
  std::string imgname = imagelist[id];
  auto extJS = extrinsicJS[imgname];

  spdlog::debug("imgname {}", imgname);
  std::string camid = extJS["Camera"].get<std::string>();
  spdlog::debug("camid {}", camid);
  CameraT camera = get_camera(camid);

  camera.rescale(1. / double(downsampler));
  auto metaJS = imagemetaJS[imgname];

  embree::LinearSpace3fa PoseR =
      embree::LinearSpace3fa(extJS["r11"].get<float>(), extJS["r21"].get<float>(), extJS["r31"].get<float>(),
                             extJS["r12"].get<float>(), extJS["r22"].get<float>(), extJS["r32"].get<float>(),
                             extJS["r13"].get<float>(), extJS["r23"].get<float>(), extJS["r33"].get<float>());
  embree::Vec3fa PoseC = embree::Vec3fa(extJS["X"].get<float>(), extJS["Y"].get<float>(), extJS["Z"].get<float>());
  embree::Vec3fa SunPos;
  if (requireBuffers[SUNVIS]) {
    SunPos[0] = metaJS["Sun:LocalPos_x"].get<float>();
    SunPos[1] = metaJS["Sun:LocalPos_y"].get<float>();
    SunPos[2] = metaJS["Sun:LocalPos_z"].get<float>();
  }

  size_t num_px = camera.h * camera.w;
  if (requireBuffers[GEOMID]) geomidBuffer = std::vector<uint32_t>(num_px * 5, RTC_INVALID_GEOMETRY_ID);
  if (requireBuffers[BARYCENTRIC]) barycentricBuffer = std::vector<float>(num_px * 2, 0.f);
  if (requireBuffers[DEPTH]) depthBuffer = std::vector<float>(num_px, embree::nan);
  if (requireBuffers[NORMAL]) normalBuffer = std::vector<float>(num_px * 3, 0.f);
  if (requireBuffers[COLOR]) colorBuffer = std::vector<uint8_t>(num_px * 3, 0);
  if (requireBuffers[SUNVIS]) sunVisBuffer = std::vector<uint8_t>(num_px, 0);
  if (requireBuffers[SKYVIS]) {
    skyVisBuffer = std::vector<uint8_t>(num_px, 0);
  }

  IntersectContext context;
  InitIntersectionContext(&context);

  spdlog::stopwatch sw;
  embree::parallel_for(size_t(0), num_px, [&](const embree::range<size_t>& r) {
    size_t tilehits = 0, tileindirhits = 0, tilesunhits = 0, tileskyhits = 0;
    for (size_t pxi = r.begin(); pxi < r.end(); ++pxi) {
      size_t yi = pxi / camera.w;
      size_t xi = pxi - yi * camera.w;
      Ray ray;

      Eigen::Vector3f camray(0, 0, 1);

      camera.sample_ray(xi, yi, true, camray[0], camray[1]);

      init_Ray(ray, PoseC, PoseR.vx * camray[0] + PoseR.vy * camray[1] + PoseR.vz * camray[2], znear, zfar);
      rtcIntersect1(scene, &context.context, RTCRayHit_(ray));
      bool hit = (ray.geomID != RTC_INVALID_GEOMETRY_ID) && (ray.primID != RTC_INVALID_GEOMETRY_ID);
      if (hit) {
        RTCGeometry hitgeom = rtcGetGeometry(scene, ray.geomID);
        embree::Vec3fa hitpos = ray.org + (ray.tfar - EPSILON) * ray.dir;

        embree::Vec3fa interpNormal;

        if (with_pertri_normals) {
          float* pertri_normals = (float*)rtcGetGeometryUserData(hitgeom);
          interpNormal.x = pertri_normals[ray.primID * 3];
          interpNormal.y = pertri_normals[ray.primID * 3 + 1];
          interpNormal.z = pertri_normals[ray.primID * 3 + 2];
        } else {
          rtcInterpolate0(hitgeom, ray.primID, ray.u, ray.v, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, &interpNormal.x, 3);
        }
        if (cull && embree::dot(ray.dir, interpNormal) > 0) {
          continue;
        }
        if (requireBuffers[SUNVIS]) {
          Ray sunray;
          init_Ray(sunray, hitpos, SunPos, EPSILON);
          float sun_surface_angle = dot(SunPos, interpNormal);
          rtcOccluded1(scene, &context.context, RTCRay_(sunray));
          if (sunray.tfar >= 0.f && sun_surface_angle > 0.f) {
            sunVisBuffer[pxi] = 255;
            ++tilesunhits;
          } else
            sunVisBuffer[pxi] = 50;
        }
        if (requireBuffers[SKYVIS]) {
          // integrate sky radiance of the point. Using cosine sampling.
          float skyhits;
          rtcInterpolate0(hitgeom, ray.primID, ray.u, ray.v, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 1, &skyhits, 1);
          skyVisBuffer[pxi] = skyhits * 255.f;
        }
        if (requireBuffers[GEOMID]) {
          RTCGeometry _geom = rtcGetGeometry(scene, ray.geomID);
          uint32_t* _face = (uint32_t*)rtcGetGeometryBufferData(_geom, RTC_BUFFER_TYPE_INDEX, 0);

          geomidBuffer[pxi] = ray.geomID;
          geomidBuffer[pxi + num_px] = ray.primID;

          geomidBuffer[pxi + 2 * num_px] = _face[3 * ray.primID];
          geomidBuffer[pxi + 3 * num_px] = _face[3 * ray.primID + 1];
          geomidBuffer[pxi + 4 * num_px] = _face[3 * ray.primID + 2];
        }
        if (requireBuffers[BARYCENTRIC]) {
          barycentricBuffer[pxi] = ray.u;
          barycentricBuffer[pxi + num_px] = ray.v;
        }

        if (requireBuffers[DEPTH]) depthBuffer[pxi] = ray.tfar;
        if (requireBuffers[NORMAL]) {
          normalBuffer[pxi] = interpNormal.x;
          normalBuffer[pxi + num_px] = interpNormal.y;
          normalBuffer[pxi + 2 * num_px] = interpNormal.z;
        }
      } else {
        spdlog::debug("Hit miss {} {}", xi, yi);
      }
    }
  });
  spdlog::info("Render {:.3}s", sw);

  // Output Section
  sw.reset();
  if (requireBuffers[GEOMID]) {
    std::filesystem::path pth = outputDirs[GEOMID] / (imgname + ".exr");
    SaveUIntNEXR(geomidBuffer, camera.w, camera.h, 5, pth.string());
  }

  if (requireBuffers[BARYCENTRIC]) {
    std::filesystem::path pth = outputDirs[BARYCENTRIC] / (imgname + ".exr");
    SaveHalfNEXR(barycentricBuffer, camera.w, camera.h, 2, pth.string());
  }

  if (requireBuffers[DEPTH]) {
    std::filesystem::path pth = outputDirs[DEPTH] / (imgname + ".exr");
    SaveHalf1EXR(depthBuffer, camera.w, camera.h, pth.string());
  }
  if (requireBuffers[NORMAL]) {
    std::filesystem::path pth = outputDirs[NORMAL] / (imgname + ".exr");
    SaveHalf3EXR(normalBuffer, camera.w, camera.h, pth.string());
  }
  // if (requireBuffers[COLOR]) stbi_write_png((outputDirs[COLOR] / (imgname + ".png")).c_str(),camera.w, camera.h,
  // 3, colorBuffer.data(), camera.w * 3); if (requireBuffers[INDIR]) stbi_write_png((outputDirs[INDIR] / (imgname +
  // ".png")).c_str(), camera.w, camera.h, 3, indirBuffer.data(), camera.w * 3);
  if (requireBuffers[SUNVIS]) {
    std::filesystem::path pth = outputDirs[SUNVIS] / (imgname + ".png");
    stbi_write_png(pth.string().c_str(), camera.w, camera.h, 1, sunVisBuffer.data(), camera.w);
  }

  if (requireBuffers[SKYVIS]) {
    std::filesystem::path pth = outputDirs[SKYVIS] / (imgname + ".png");
    stbi_write_png(pth.string().c_str(), camera.w, camera.h, 1, skyVisBuffer.data(), camera.w);
  }
  spdlog::info("Write Buffers: {:.3}s", sw);

  return true;
}