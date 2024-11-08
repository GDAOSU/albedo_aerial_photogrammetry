// test_environment_map.cpp

#include <ImathVec.h>
#include <ImfArray.h>
#include <ImfRgbaFile.h>
#include <spdlog/spdlog.h>

#include <cmath>

#include "EnvironmentMap.h"

int main() {
  EnvironmentMap envMap(
      "/research/GDA/Research/AlbedoDataset_CVPR25/albedo_generation/flights/20241023_11AM_coe/aerial_albedo/"
      "skycam.exr");
  const int outputWidth = 512;
  const int outputHeight = 256;
  Imf::Array2D<Imf::Rgba> image(outputHeight, outputWidth);

  // XY
  // 0,0 ---------------- width,0
  //  |                    |
  //  |                    |
  // 0,height ----------- width,height

  // UV
  // 0,1 ---------------- 1,1
  // |                    |
  // |                    |
  // 0,0 --------------- 1,0

  // azimuth, elevation
  // 0, 90 ---------------- 360, 90
  // |                        |
  // |                        |
  // 0, -90 --------------- 360, -90

  std::array<float, 3> sun_v = {0.36261032744372684, -0.74011675443893, 0.5663399511158805};

  float sun_elevation = asin(sun_v[2]) * 180.f / M_PI;
  float sun_azi = atan2(sun_v[0], sun_v[1]) * 180.f / M_PI;
  spdlog::info("Sun elevation: {}, azimuth: {}", sun_elevation, sun_azi);
  for (int y = 0; y < outputHeight; ++y) {
    for (int x = 0; x < outputWidth; ++x) {
      float u = (static_cast<float>(x) + 0.5) / outputWidth;
      float v = 1.f - (static_cast<float>(y) + 0.5) / outputHeight;

      float elevation = (v - 0.5f) * 180.f;  // -90 deg to 90 deg
      float azimuth = u * 360.f;             // 0 deg to 360 deg, 0: north, 90: east, 180: south, 270: west

      if (std::fabs(elevation - sun_elevation) < 1 && std::fabs(azimuth - sun_azi) < 1) {
        spdlog::info("Sun at {}, {}", x, y);
        image[y][x] = Imf::Rgba(1.0f, 0.0f, 0.0f, 1.0f);
      } else {
        std::array<float, 3> color = envMap.sample(azimuth, elevation);
        image[y][x] = Imf::Rgba(color[0], color[1], color[2], 1.0f);
      }
    }
  }

  Imf::RgbaOutputFile file("output.exr", outputWidth, outputHeight, Imf::WRITE_RGBA);
  file.setFrameBuffer(&image[0][0], 1, outputWidth);
  file.writePixels(outputHeight);

  spdlog::info("Done");
  return 0;
}