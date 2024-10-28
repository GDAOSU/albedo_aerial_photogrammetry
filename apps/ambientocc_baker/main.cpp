

#include <assimp/SceneCombiner.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <assimp/Exporter.hpp>
#include <assimp/Importer.hpp>
#include <cxxopts.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>
namespace fs = std::filesystem;

#include "AmbientOccBaker.h"

cxxopts::Options getOptions(std::string exepath) {
  // clang-format off
  cxxopts::Options options(fs::path(exepath).filename().stem().string(), "Tiny Ray Tracing renderer");
  options.positional_help("intrinsic_csv extrinsic_with_sun_csv pointcloud.ply output_dir").show_positional_help();
  options.add_options()
    ("modeldataset", "Point Cloud Ply. Normal and Color are required.", cxxopts::value<std::string>())
    ("output", "Output dir.", cxxopts::value<std::string>())
    ("use_sphere", "Generate spherical samples if true, otherwise use upper hemisphere.", cxxopts::value<bool>()->default_value("false"))
    ("num_samples", "Number of samples for sky visibility.", cxxopts::value<int>()->default_value("32"))
    ("epsilon", "epsilon", cxxopts::value<float>()->default_value("0.01"))
    ("radius", "radius of sphere to check (-1 : infinite)", cxxopts::value<float>()->default_value("-1"))
    ("p, preview", "Preview parameter.", cxxopts::value<bool>()->default_value("false"))
    ("v, verbose", "TRACE 0 DEBUG 1 INFO 2 WARN 3 ERROR 4 CRITICAL 5 OFF 6", cxxopts::value<int>()->default_value("2"))
    ("h, help", "Print help");
  // options.parse_positional({"intrinsic", "extrinsic", "model", "output"});
  // clang-format on
  return options;
}

aiScene* cacheAndMergeModels(const nlohmann::json& modelJS, std::string cachedModelPath) {
  aiScene* mergeScene = nullptr;

  std::vector<std::string> modellist;
  for (auto it = modelJS.begin(); it != modelJS.end(); ++it) modellist.push_back(it.key());

  std::vector<aiScene*> srcs;

  for (int sceneID = 0; sceneID < modellist.size(); ++sceneID) {
    spdlog::info("Merging {}/{}", sceneID + 1, modellist.size());
    std::shared_ptr<Assimp::Importer> importer = std::make_shared<Assimp::Importer>();
    std::string modelname = modellist[sceneID];
    std::string modelpath = modelJS[modelname]["RedirectPath"].get<std::string>();

    if (!fs::exists(modelpath)) {
      spdlog::error("Cannot load {}", modelpath);
      exit(2);
    }

    const aiScene* readScene = importer->ReadFile(modelpath, aiProcess_GenNormals | aiProcess_ValidateDataStructure);

    bool drop_textures = true;
    aiScene* newScene = new aiScene;
    aiCopyScene(readScene, &newScene);

    if (drop_textures) {
      for (int mi = 0; mi < newScene->mNumMaterials; ++mi) newScene->mMaterials[mi]->Clear();
      delete[] newScene->mTextures;
      newScene->mTextures = nullptr;
      newScene->mNumTextures = 0;

      for (int mi = 0; mi < newScene->mNumMeshes; ++mi) {
        for (int zz = 0; zz < AI_MAX_NUMBER_OF_COLOR_SETS; ++zz) {
          if (newScene->mMeshes[mi]->mColors[zz] != nullptr) {
            delete newScene->mMeshes[mi]->mColors[zz];
            newScene->mMeshes[mi]->mColors[zz] = nullptr;
          }
        }
        for (int zz = 0; zz < AI_MAX_NUMBER_OF_TEXTURECOORDS; ++zz) {
          if (newScene->mMeshes[mi]->mTextureCoords[zz] != nullptr) {
            delete newScene->mMeshes[mi]->mTextureCoords[zz];
            newScene->mMeshes[mi]->mTextureCoords[zz] = nullptr;
          }
          newScene->mMeshes[mi]->mNumUVComponents[zz] = 0;
          if (newScene->mMeshes[mi]->mTextureCoordsNames) {
            if (newScene->mMeshes[mi]->mTextureCoordsNames[zz] != nullptr) {
              delete newScene->mMeshes[mi]->mTextureCoordsNames[zz];
              newScene->mMeshes[mi]->mTextureCoordsNames[zz] = nullptr;
            }
          }
        }
      }
    }

    srcs.push_back(newScene);
  }

  Assimp::SceneCombiner::MergeScenes(&mergeScene, srcs,
                                     AI_INT_MERGE_SCENE_GEN_UNIQUE_NAMES | AI_INT_MERGE_SCENE_GEN_UNIQUE_MATNAMES);

  return mergeScene;
}

int main(int argc, char** argv) {
  spdlog::set_pattern("[%H:%M:%S][%^%l%$] %v");
  spdlog::stopwatch sw;

  auto options = getOptions(argv[0]);
  auto args = options.parse(argc, argv);
  spdlog::set_level((spdlog::level::level_enum)std::max(std::min(args["verbose"].as<int>(), SPDLOG_LEVEL_OFF), 0));
  if (args.count("modeldataset") == 0 || args.count("output") == 0 || args.count("help")) {
    std::cout << options.help() << std::endl;
    exit(0);
  }
  // Setting parameters
  std::string modeljson_path = args["modeldataset"].as<std::string>();
  std::string output_path = args["output"].as<std::string>();
  bool use_sphere = args["use_sphere"].as<bool>();
  int num_samples = args["num_samples"].as<int>();
  float radius = args["radius"].as<float>();
  float epsilon = args["epsilon"].as<float>();

  spdlog::info("Use Sphere: {}", use_sphere ? "Yes" : "No");
  spdlog::info("Num Sample: {}", num_samples);
  spdlog::info("epsilon: {}", epsilon);
  spdlog::info("Radius: {}", radius <= 0 ? std::numeric_limits<float>::infinity() : radius);

  // Loading data
  nlohmann::json modeljson;
  std::ifstream ifs;
  ifs.open(modeljson_path);
  if (!ifs.good()) {
    spdlog::critical("Input modeldataset not found: {}", modeljson_path);
    exit(1);
  }
  ifs >> modeljson;
  ifs.close();

  nlohmann::json modelJS = modeljson["models"];
  spdlog::info("Found {} Models", modelJS.size());

  auto outputDir = fs::path(output_path);
  auto outputDirModel = outputDir / "models";
  if (!fs::exists(outputDirModel)) fs::create_directories(outputDirModel);

  // Processing
  auto cachedModelPath = outputDirModel / "model.glb";
  auto outputDirSamplePath = outputDirModel / "dirsamples.json";
  auto outputAOResultPath = outputDirModel / "aoprops.dat";

  /// Load model data to cahcedScene
  aiScene* cachedScene = nullptr;
  if (!fs::exists(cachedModelPath)) {  // Read in cache

    spdlog::info("Creating Cached Model: {}", cachedModelPath.string());
    // Create cache
    if (!fs::exists(outputDirModel)) fs::create_directories(outputDirModel);
    cachedScene = cacheAndMergeModels(modelJS, cachedModelPath.string());
    Assimp::Exporter exporter;
    exporter.Export(cachedScene, "glb2", cachedModelPath.string());
  }

  // Read in again to make sure the consistency
  spdlog::info("Found Existing Cached Model: {}", cachedModelPath.string());
  Assimp::Importer importer;
  const aiScene* readScene =
      importer.ReadFile(cachedModelPath.string(), aiProcess_GenNormals | aiProcess_ValidateDataStructure);
  cachedScene = new aiScene;
  aiCopyScene(readScene, &cachedScene);

  // Rendering
  AmbientOcclusionBaker AOR;
  AOR.init_dir_sampler(use_sphere, num_samples);
  AOR.set_radius(radius);
  AOR.set_epsilon(epsilon);
  AOR.set_scene(cachedScene);
  AOR.render(cachedScene);
  aiFreeScene(cachedScene);

  // Save results
  AOR.write_dir_samples(outputDirSamplePath.string());
  AOR.write_buffer(outputAOResultPath.string());

  modeljson["cachedmodel"] = cachedModelPath.string();
  modeljson["dirsamples"] = outputDirSamplePath.string();
  modeljson["aoprops"] = outputAOResultPath.string();
  modeljson["aoradius"] = radius;
  modeljson["aoepsilon"] = epsilon;
  std::ofstream ofs(modeljson_path);
  ofs << modeljson.dump(1);
  ofs.close();

  spdlog::info("Done {:.3}s", sw);
  return 0;
}