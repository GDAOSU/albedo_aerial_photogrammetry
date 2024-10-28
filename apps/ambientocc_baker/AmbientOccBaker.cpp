#include "AmbientOccBaker.h"

// clang-format off
#include <embcommon/algorithms/parallel_for.h>
#include <embcommon/algorithms/parallel_reduce.h>
#include <embcommon/core/ray.h>
#include <embcommon/tasking/taskschedulertbb.h>
// clang-format on

#include <assimp/scene.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

void AmbientOcclusionBaker::init_dir_sampler(bool use_sphere, int num_samples) {
  aoprops.init_dir_sampler(use_sphere, num_samples);
}

void AmbientOcclusionBaker::set_scene(const aiScene* aiscene) {
  if (scene) rtcReleaseScene(scene);
  scene = rtcNewScene(device);
  std::vector<size_t> bufsz(aiscene->mNumMeshes);
  for (int meshID = 0; meshID < aiscene->mNumMeshes; ++meshID) {
    RTCGeometry _geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
    rtcSetGeometryBuildQuality(_geom, RTC_BUILD_QUALITY_HIGH);

    const aiMesh* _mesh = aiscene->mMeshes[meshID];
    bufsz[meshID] = _mesh->mNumVertices;

    float* _vertices = (float*)rtcSetNewGeometryBuffer(_geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                                                       3 * sizeof(float), _mesh->mNumVertices);
    uint32_t* _faces = (uint32_t*)rtcSetNewGeometryBuffer(_geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                                                          3 * sizeof(uint32_t), _mesh->mNumFaces);

    for (int _i = 0; _i < _mesh->mNumVertices; ++_i) {
      _vertices[_i * 3 + 0] = _mesh->mVertices[_i].x;
      _vertices[_i * 3 + 1] = _mesh->mVertices[_i].y;
      _vertices[_i * 3 + 2] = _mesh->mVertices[_i].z;
    }
    for (int _i = 0; _i < _mesh->mNumFaces; ++_i) {
      _faces[_i * 3 + 0] = _mesh->mFaces[_i].mIndices[0];
      _faces[_i * 3 + 1] = _mesh->mFaces[_i].mIndices[1];
      _faces[_i * 3 + 2] = _mesh->mFaces[_i].mIndices[2];
    }

    rtcCommitGeometry(_geom);
    uint32_t _geomID = rtcAttachGeometry(scene, _geom);
    rtcReleaseGeometry(_geom);
  }
  rtcCommitScene(scene);
  aoprops.resize(bufsz);
}

void AmbientOcclusionBaker::set_radius(float radius) { this->radius = radius; }
void AmbientOcclusionBaker::set_epsilon(float eps) { this->epsilon = eps; }

bool AmbientOcclusionBaker::render(const aiScene* aiscene) {
  if (aoprops.buffer.empty()) return false;
  if (aoprops.dir_sampler->numpt() == 0) return false;
  if (!aiscene->HasMeshes()) return false;
  spdlog::stopwatch sw;
  IntersectContext context;
  InitIntersectionContext(&context);

  std::vector<embree::Vec3fa> sampleDirs(this->aoprops.dir_sampler->numpt());
  for (int i = 0; i < sampleDirs.size(); ++i) {
    sampleDirs[i].x = this->aoprops.dir_sampler->coords()[3 * i];
    sampleDirs[i].y = this->aoprops.dir_sampler->coords()[3 * i + 1];
    sampleDirs[i].z = this->aoprops.dir_sampler->coords()[3 * i + 2];
  }

  spdlog::info("{} meshes, {} mats, {} textures", aiscene->mNumMeshes, aiscene->mNumMaterials, aiscene->mNumTextures);
  aoprops.reset();
  size_t totalhits = 0;
  for (int meshID = 0; meshID < aiscene->mNumMeshes; ++meshID) {
    const aiMesh* _mesh = aiscene->mMeshes[meshID];
    spdlog::debug("{}: v {} f {} @ {}", meshID, _mesh->mNumVertices, _mesh->mNumFaces, _mesh->mMaterialIndex);

    sw.reset();

    size_t mesh_hits = embree::parallel_reduce(
        uint32_t(0), _mesh->mNumVertices, size_t(0),
        [&context, this, &sampleDirs, &meshID, _mesh](const embree::range<uint32_t>& r) {
          size_t th_hits = 0;
          for (uint32_t vid = r.begin(); vid < r.end(); ++vid) {
            std::vector<Ray> rays(this->aoprops.num_samples);
            embree::Vec3fa pos(_mesh->mVertices[vid].x, _mesh->mVertices[vid].y, _mesh->mVertices[vid].z);
            embree::Vec3fa normal(_mesh->mNormals[vid].x, _mesh->mNormals[vid].y, _mesh->mNormals[vid].z);
            embree::Vec3fa offpos = pos + normal * epsilon;

            for (size_t si = 0; si < sampleDirs.size(); ++si)
              init_Ray(rays[si], offpos, sampleDirs[si], epsilon, radius < 0 ? embree::inf : radius);

            rtcIntersect1M(this->scene, &context.context, RTCRayHit_(rays[0]), rays.size(), sizeof(Ray));

            for (size_t si = 0; si < rays.size(); ++si)
              if (rays[si].geomID == RTC_INVALID_GEOMETRY_ID) {  // found occlusion
                this->aoprops.set(meshID, vid, si, true);
              } else {
                this->aoprops.set(meshID, vid, si, false);
                ++th_hits;
              }
          }
          return th_hits;
        },
        std::plus<size_t>());

    spdlog::info("[{}]: {} verts {} hits. {:.3}s", meshID, _mesh->mNumVertices, mesh_hits, sw);
    totalhits += mesh_hits;
  }
  spdlog::debug("Total hits: {}", totalhits);
  return true;
}

bool AmbientOcclusionBaker::write_dir_samples(std::string outpath) const { return aoprops.write_dir_samples(outpath); }

bool AmbientOcclusionBaker::write_buffer(std::string outpath) const { return aoprops.write_buffer(outpath); }
