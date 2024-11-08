// clang-format off
#include <embree3/rtcore.h>
#include <embcommon/algorithms/parallel_for.h>
#include <embcommon/core/ray.h>
#include <embcommon/tasking/taskschedulertbb.h>
// clang-format on
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <assimp/Importer.hpp>
#include <cxxopts.hpp>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <vector>

#include "CameraModel.h"
#include "TinyRender.h"
#include "utils/writer.h"

namespace fs = std::filesystem;

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

using namespace std;

using namespace embree;

cxxopts::Options getOptions(std::string exepath) {
  // clang-format off
  cxxopts::Options options(fs::path(exepath).filename().stem().string(), "Tiny Ray Tracing renderer");
  options.positional_help("intrinsic_csv extrinsic_with_sun_csv pointcloud.ply output_dir").show_positional_help();
  options.add_options()
    ("imagedataset", "image dataset json including intrinsic, extrinsic and metadata.", cxxopts::value<std::string>())
    ("modeldataset", "model dataset json. Normal and Color are required.", cxxopts::value<std::string>())
    ("environment", "environment light", cxxopts::value<std::string>())
    ("output", "Output dir.", cxxopts::value<std::string>())
    ("znear", "Clip to znear", cxxopts::value<float>()->default_value("-1."))
    ("zfar", "Clip to zfar", cxxopts::value<float>()->default_value("-1."))
    ("cull", "cull backface", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
    ("l, level", "Level of downsampling.", cxxopts::value<uint8_t>()->default_value("0"))
    ("s, start", "Start photo index.", cxxopts::value<int>()->default_value("0"))
    ("n, num", "Num of index", cxxopts::value<int>()->default_value("0"))
    ("t, end", "End photo index.", cxxopts::value<int>()->default_value("0"))
    ("p, preview", "Preview parameter.", cxxopts::value<bool>()->default_value("false"))
    ("v, verbose", "TRACE 0 DEBUG 1 INFO 2 WARN 3 ERROR 4 CRITICAL 5 OFF 6", cxxopts::value<int>()->default_value("2"))
    ("h, help", "Print help");
  options.add_options("Buffers")
    ("geomid", "Write geometric index buffer [geomID, primID, vertexID0, vertexID1, vertexID2]", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
    ("barycentric", "Write barycentric buffer", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
    ("depth", "Write depth buffer", cxxopts::value<bool>()->default_value("true")->implicit_value("true"))
    ("normal", "Write normal buffer", cxxopts::value<bool>()->default_value("true")->implicit_value("true"))
    ("primid", "Write primid buffer", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
    ("color", "Write color buffer", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
    ("indir", "Write color buffer", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
    ("sunvis", "Write sunvis buffer", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
    ("skyvis", "Write skyvis buffer", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
    ("skycam", "Write skycam buffer", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
    ("all", "Write all available buffers.", cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
    // ("indirsamples", "Number of samples for Indirect light.", cxxopts::value<size_t>()->default_value("32"))
    // ("skysamples", "Number of samples for sky visibility.", cxxopts::value<size_t>()->default_value("32"));
  // options.parse_positional({"intrinsic", "extrinsic", "model", "output"});
  // clang-format on
  return options;
}

int main(int argc, char** argv) {
  spdlog::stopwatch mainsw, sw;
  // Leading section
  spdlog::set_pattern("[%H:%M:%S][%^%l%$] %v");
  auto options = getOptions(argv[0]);
  auto args = options.parse(argc, argv);
  spdlog::set_level((spdlog::level::level_enum)max(min(args["verbose"].as<int>(), SPDLOG_LEVEL_OFF), 0));
  if (args.count("imagedataset") == 0 || args.count("modeldataset") == 0 || args.count("output") == 0 ||
      args.count("help")) {
    cout << options.help({"", "Buffers"}) << endl;
    exit(0);
  }
  // Setting parameters
  std::string imagejson_path = args["imagedataset"].as<std::string>();
  std::string modeljson_path = args["modeldataset"].as<std::string>();
  std::string environment_path = args["environment"].as<std::string>();
  std::string output_path = args["output"].as<std::string>();

  if (!fs::exists(imagejson_path)) {
    spdlog::critical("Input imagedataset not found: {}", imagejson_path);
    exit(1);
  }

  if (!fs::exists(modeljson_path)) {
    spdlog::critical("Input modeldataset not found: {}", modeljson_path);
    exit(1);
  }

  float znear = args["znear"].as<float>();
  float zfar = args["zfar"].as<float>();
  bool cull = args["cull"].as<bool>();
  if (znear < 0) znear = embree::zero;
  if (zfar < 0) zfar = embree::inf;
  uint8_t nDownsample = args["level"].as<uint8_t>();
  int startID = args["start"].as<int>();
  int numID = args["num"].as<int>();
  int endID = args["end"].as<int>();
  if (numID > 0) endID = startID + numID;
  bool preview = args["preview"].as<bool>();

  bool all = args["all"].as<bool>();

  std::unordered_map<int, bool> requireBuffers;
  for (size_t i = 0; i < TinyRender::NUM_BUFFERS; ++i)
    requireBuffers[i] = args[TinyRender::BUFFERS_NAME[i]].as<bool>() || all;

  TinyRender render;
  render.set_require_buffers(requireBuffers);
  render.set_clipping(znear, zfar);
  render.set_downsampler(std::pow(2, nDownsample));
  render.set_cull(cull);
  if (!render.load_imagedataset(imagejson_path)) exit(2);
  if (!render.load_modeldataset(modeljson_path)) exit(2);
  if (!render.load_environment(environment_path)) exit(2);

  size_t num_images = render.num_images();
  startID = min(startID, int(num_images - 1));
  if (endID == 0)
    endID = num_images;
  else if (endID < 0)
    endID = num_images + endID;
  else
    endID = min(endID, int(num_images));

  /// Print Configs
  spdlog::info("**** Tiny Render ****");
  spdlog::info("Load ImageDataset: {}", imagejson_path);
  spdlog::info("Load ModelDataset: {}", modeljson_path);
  spdlog::info("Environment: {}", environment_path);
  spdlog::info("Downsample Rate: {}", nDownsample);
  spdlog::info("Render Range [{}-{}) [{}/{}]", startID, endID, endID - startID, num_images);
  spdlog::info("Output Dir: {}", output_path);
  spdlog::info("Buffers:");
  for (size_t i = 0; i < TinyRender::NUM_BUFFERS; ++i)
    spdlog::info(" - {}: {}", TinyRender::BUFFERS_NAME[i], requireBuffers[i] ? "On" : "Off");
  // spdlog::info("Indir Samples: {}", indirsamples);
  // spdlog::info("Sky Samples: {} : Pack to {} UInt", skysamples, skysamplepackchannels);
  spdlog::info("***********************");
  if (preview) {
    spdlog::info("Preview Done.");
    return 0;
  }

  if (!render.prepare_data()) exit(2);

  // Create Output Folders
  auto outputDir = fs::path(output_path);
  std::unordered_map<int, fs::path> outputDirs;
  for (size_t i = 0; i < TinyRender::NUM_BUFFERS; ++i) outputDirs[i] = outputDir / TinyRender::BUFFERS_NAME[i];
  for (size_t i = 0; i < TinyRender::NUM_BUFFERS; ++i)
    if (requireBuffers[i] && !fs::exists(outputDirs[i])) fs::create_directories(outputDirs[i]);
  render.set_output_dirs(outputDirs);
  // model

  // Rendering
  for (size_t id = startID; id < endID; ++id) {
    sw.reset();
    if (!render.render_frame(id)) {
      spdlog::error("Rendering failed");
      break;
    }
    spdlog::info("Frame [{}/{}] : {:.3}s", id - startID + 1, endID - startID, sw);
  }

  spdlog::info("Done. [{:.3}s]", mainsw);

  return 0;
}