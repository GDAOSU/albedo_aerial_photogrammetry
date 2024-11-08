// EnvironmentMap.cpp

#include "EnvironmentMap.h"

#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfImageChannel.h>
#include <ImfInputFile.h>

#include <cmath>
EnvironmentMap::EnvironmentMap() {}
EnvironmentMap::EnvironmentMap(const std::string& filename) { load(filename); }

bool EnvironmentMap::load(const std::string& filename) {
  // Load EnvMap with OpenEXR
  // read .exr file, the data type is float32
  Imf::InputFile file(filename.c_str());
  Imath::Box2i dw = file.header().dataWindow();
  width = dw.max.x - dw.min.x + 1;
  height = dw.max.y - dw.min.y + 1;
  pixels.resize(width * height * 3);

  Imf::FrameBuffer frameBuffer;
  frameBuffer.insert("R", Imf::Slice(Imf::FLOAT, (char*)(&pixels[0]), sizeof(float) * 3, sizeof(float) * 3 * width));
  frameBuffer.insert("G", Imf::Slice(Imf::FLOAT, (char*)(&pixels[1]), sizeof(float) * 3, sizeof(float) * 3 * width));
  frameBuffer.insert("B", Imf::Slice(Imf::FLOAT, (char*)(&pixels[2]), sizeof(float) * 3, sizeof(float) * 3 * width));

  file.setFrameBuffer(frameBuffer);
  file.readPixels(dw.min.y, dw.max.y);
  return true;
}

std::array<float, 3> EnvironmentMap::read_pixel(const float u, const float v) const {
  // bilinear interpolation
  const float xf = u * width;
  const float yf = (1.0f - v) * height;
  const int x00 = static_cast<int>(xf);
  const int y00 = static_cast<int>(yf);
  const int x01 = (x00 + 1) % width;
  const int y01 = std::min(y00 + 1, height - 1);
  const int wx = xf - x00;
  const int wy = yf - y00;

  const float w00 = (1 - wx) * (1 - wy);
  const float w01 = wx * (1 - wy);
  const float w10 = (1 - wx) * wy;
  const float w11 = wx * wy;

  const int pixel_off00 = y00 * width * 3 + x00 * 3;
  const int pixel_off01 = y01 * width * 3 + x00 * 3;
  const int pixel_off10 = y00 * width * 3 + x01 * 3;
  const int pixel_off11 = y01 * width * 3 + x01 * 3;

  return {w00 * pixels[pixel_off00] + w01 * pixels[pixel_off01] + w10 * pixels[pixel_off10] + w11 * pixels[pixel_off11],
          w00 * pixels[pixel_off00 + 1] + w01 * pixels[pixel_off01 + 1] + w10 * pixels[pixel_off10 + 1] +
              w11 * pixels[pixel_off11 + 1],
          w00 * pixels[pixel_off00 + 2] + w01 * pixels[pixel_off01 + 2] + w10 * pixels[pixel_off10 + 2] +
              w11 * pixels[pixel_off11 + 2]};
}
std::array<float, 3> EnvironmentMap::sample(float azimuth, float elevation) const {
  // azimuth: 0 to 360 deg, 0: north, 90: east, 180: south, 270: west
  // elevation: -90 to 90 deg
  azimuth = azimuth - 90.f;  // The pixels is using 0 west, 90 north, 180 east, 270 south

  azimuth = std::fmod(azimuth, 360.f);  // rotation wrap around
  elevation = std::max(-90.f, std::min(90.f, elevation));

  const float u = std::min(std::max(azimuth / 360.f, 0.f), 1.f);
  const float v = std::min(std::max(elevation / 180.f + 0.5f, 0.f), 1.f);

  return read_pixel(u, v);
}