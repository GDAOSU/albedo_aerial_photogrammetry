#include "writer.h"

#include <malloc.h>
#include <stdio.h>
#include <string.h>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#ifdef USE_TINYEXR
#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>
bool SaveFloat1EXR(const std::vector<float> &buffer, int width, int height, std::string outfilename) {
  EXRHeader header;
  InitEXRHeader(&header);
  header.compression_type = TINYEXR_COMPRESSIONTYPE_NONE;

  EXRImage image;
  InitEXRImage(&image);

  image.num_channels = 1;

  const float *image_ptr[1];
  image_ptr[0] = buffer.data();
  image.images = (unsigned char **)image_ptr;
  image.width = width;
  image.height = height;

  header.num_channels = 1;
  EXRChannelInfo channelinfo;
  header.channels = &channelinfo;
  strncpy(channelinfo.name, "value", 255);
  channelinfo.name[strlen("value")] = '\0';

  int pixel_type = TINYEXR_PIXELTYPE_FLOAT;
  int requested_pixel_type = TINYEXR_PIXELTYPE_HALF;
  header.pixel_types = &pixel_type;
  header.requested_pixel_types = &requested_pixel_type;

  const char *err = nullptr;  // or nullptr in C++11 or later.
  int ret = SaveEXRImageToFile(&image, &header, outfilename.c_str(), &err);
  if (ret != TINYEXR_SUCCESS) {
    fprintf(stderr, "Save EXR err: %s\n", err);
    FreeEXRErrorMessage(err);  // free's buffer for an error message
    return ret;
  }
  return ret;
}

bool SaveUInt1EXR(const std::vector<uint32_t> &buffer, int width, int height, std::string outfilename) {
  EXRHeader header;
  InitEXRHeader(&header);
  header.compression_type = TINYEXR_COMPRESSIONTYPE_NONE;

  EXRImage image;
  InitEXRImage(&image);

  image.num_channels = 1;

  const uint32_t *image_ptr[1];
  image_ptr[0] = buffer.data();
  image.images = (unsigned char **)image_ptr;
  image.width = width;
  image.height = height;

  header.num_channels = 1;
  EXRChannelInfo channelinfo;
  header.channels = &channelinfo;
  strncpy(channelinfo.name, "value", 255);
  channelinfo.name[strlen("value")] = '\0';

  int pixel_type = TINYEXR_PIXELTYPE_UINT;
  int requested_pixel_type = TINYEXR_PIXELTYPE_UINT;
  header.pixel_types = &pixel_type;
  header.requested_pixel_types = &requested_pixel_type;

  const char *err = nullptr;  // or nullptr in C++11 or later.
  int ret = SaveEXRImageToFile(&image, &header, outfilename.c_str(), &err);
  if (ret != TINYEXR_SUCCESS) {
    fprintf(stderr, "Save EXR err: %s\n", err);
    FreeEXRErrorMessage(err);  // free's buffer for an error message
    return ret;
  }
  return ret;
}

bool SaveUIntNEXR(const std::vector<uint32_t> &buffer, int width, int height, int N, std::string outfilename) {
  EXRHeader header;
  InitEXRHeader(&header);
  header.compression_type = TINYEXR_COMPRESSIONTYPE_NONE;

  EXRImage image;
  InitEXRImage(&image);

  image.num_channels = N;

  const uint32_t *image_ptr[N];
  for (int i = 0; i < N; ++i) image_ptr[i] = buffer.data() + width * height * i;
  image.images = (unsigned char **)image_ptr;
  image.width = width;
  image.height = height;

  header.num_channels = N;

  header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
  for (int i = 0; i < N; ++i) {
    snprintf(header.channels[i].name, 255, "c%d", i);
  }

  header.pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
  header.requested_pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
  for (int i = 0; i < header.num_channels; i++) {
    header.pixel_types[i] = TINYEXR_PIXELTYPE_UINT;            // pixel type of input image
    header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_UINT;  // pixel type of output image to be stored in .EXR
  }

  const char *err = nullptr;  // or nullptr in C++11 or later.
  int ret = SaveEXRImageToFile(&image, &header, outfilename.c_str(), &err);

  free(header.channels);
  free(header.pixel_types);
  free(header.requested_pixel_types);

  if (ret != TINYEXR_SUCCESS) {
    fprintf(stderr, "Save EXR err: %s\n", err);
    FreeEXRErrorMessage(err);  // free's buffer for an error message
    return ret;
  }
  return ret;
}

bool SaveFloat3EXR(const std::vector<float> &buffer, int width, int height, std::string outfilename) {
  EXRHeader header;
  InitEXRHeader(&header);
  header.compression_type = TINYEXR_COMPRESSIONTYPE_NONE;

  EXRImage image;
  InitEXRImage(&image);

  image.num_channels = 3;

  const float *image_ptr[3];
  image_ptr[0] = buffer.data();
  image_ptr[1] = buffer.data() + width * height;
  image_ptr[2] = buffer.data() + width * height * 2;

  image.images = (unsigned char **)image_ptr;
  image.width = width;
  image.height = height;

  header.num_channels = 3;
  header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
  strncpy(header.channels[0].name, "x", 255);
  header.channels[0].name[strlen("x")] = '\0';
  strncpy(header.channels[1].name, "y", 255);
  header.channels[1].name[strlen("y")] = '\0';
  strncpy(header.channels[2].name, "z", 255);
  header.channels[2].name[strlen("z")] = '\0';

  header.pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
  header.requested_pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
  for (int i = 0; i < header.num_channels; i++) {
    header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;            // pixel type of input image
    header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;  // pixel type of output image to be stored in .EXR
  }

  const char *err;
  int ret = SaveEXRImageToFile(&image, &header, outfilename.c_str(), &err);

  free(header.channels);
  free(header.pixel_types);
  free(header.requested_pixel_types);

  if (ret != TINYEXR_SUCCESS) {
    fprintf(stderr, "Save EXR err: %s\n", err);
    return ret;
  }
  return ret;
}
#else
#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfImageChannel.h>
#include <ImfOutputFile.h>
std::vector<half> convert_float_to_half(const std::vector<float> &buffer) {
  std::vector<half> out(buffer.size());
  for (size_t i = 0; i < out.size(); ++i) out[i] = half(buffer[i]);
  return out;
}

bool SaveFloat1EXR(const std::vector<float> &buffer, int width, int height, std::string outfilename) {
  Imf::Header header(width, height);
  header.channels().insert("I", Imf::Channel(Imf::FLOAT));
  Imf::OutputFile file(outfilename.c_str(), header);

  Imf::FrameBuffer framebuffer;
  framebuffer.insert("I", Imf::Slice(Imf::FLOAT, (char *)buffer.data(), sizeof(float), sizeof(float) * width));
  file.setFrameBuffer(framebuffer);
  file.writePixels(height);
  return true;
}

bool SaveHalf1EXR(const std::vector<float> &buffer, int width, int height, std::string outfilename) {
  Imf::Header header(width, height);
  header.channels().insert("I", Imf::Channel(Imf::HALF));
  Imf::OutputFile file(outfilename.c_str(), header);

  std::vector<half> halfbuffer = convert_float_to_half(buffer);

  Imf::FrameBuffer framebuffer;
  framebuffer.insert("I", Imf::Slice(Imf::HALF, (char *)halfbuffer.data(), sizeof(half), sizeof(half) * width));
  file.setFrameBuffer(framebuffer);
  file.writePixels(height);
  return true;
}

// bool SaveUIntNEXR(const std::vector<uint32_t> &buffer, int width, int height, int N, std::string outfilename) {
bool SaveUInt1EXR(const std::vector<uint32_t> &buffer, int width, int height, std::string outfilename) {
  Imf::Header header(width, height);
  header.channels().insert("I", Imf::Channel(Imf::UINT));
  Imf::OutputFile file(outfilename.c_str(), header);

  Imf::FrameBuffer framebuffer;
  framebuffer.insert("I", Imf::Slice(Imf::UINT, (char *)buffer.data(), sizeof(uint32_t), sizeof(uint32_t) * width));
  file.setFrameBuffer(framebuffer);
  file.writePixels(height);
  return true;
}
bool SaveFloat3EXR(const std::vector<float> &buffer, int width, int height, std::string outfilename) {
  Imf::Header header(width, height);
  header.channels().insert("R", Imf::Channel(Imf::FLOAT));
  header.channels().insert("G", Imf::Channel(Imf::FLOAT));
  header.channels().insert("B", Imf::Channel(Imf::FLOAT));
  Imf::OutputFile file(outfilename.c_str(), header);

  Imf::FrameBuffer framebuffer;
  framebuffer.insert("R", Imf::Slice(Imf::FLOAT, (char *)&buffer[0], sizeof(float), sizeof(float) * width));
  framebuffer.insert("G",
                     Imf::Slice(Imf::FLOAT, (char *)&buffer[width * height], sizeof(float), sizeof(float) * width));
  framebuffer.insert("B",
                     Imf::Slice(Imf::FLOAT, (char *)&buffer[2 * width * height], sizeof(float), sizeof(float) * width));
  file.setFrameBuffer(framebuffer);
  file.writePixels(height);
  return true;
}

bool SaveHalf3EXR(const std::vector<float> &buffer, int width, int height, std::string outfilename) {
  Imf::Header header(width, height);
  header.channels().insert("R", Imf::Channel(Imf::HALF));
  header.channels().insert("G", Imf::Channel(Imf::HALF));
  header.channels().insert("B", Imf::Channel(Imf::HALF));
  Imf::OutputFile file(outfilename.c_str(), header);

  std::vector<half> halfbuffer = convert_float_to_half(buffer);

  Imf::FrameBuffer framebuffer;
  framebuffer.insert("R", Imf::Slice(Imf::HALF, (char *)&halfbuffer[0], sizeof(half), sizeof(half) * width));
  framebuffer.insert("G",
                     Imf::Slice(Imf::HALF, (char *)&halfbuffer[width * height], sizeof(half), sizeof(half) * width));
  framebuffer.insert(
      "B", Imf::Slice(Imf::HALF, (char *)&halfbuffer[2 * width * height], sizeof(half), sizeof(half) * width));
  file.setFrameBuffer(framebuffer);
  file.writePixels(height);
  return true;
}

bool SaveUIntNEXR(const std::vector<uint32_t> &buffer, int width, int height, int N, std::string outfilename) {
  Imf::Header header(width, height);

  for (int i = 0; i < N; ++i) {
    char tmpbuf[32];
    sprintf(tmpbuf, "c%d", i);
    header.channels().insert(tmpbuf, Imf::Channel(Imf::UINT));
  }
  Imf::OutputFile file(outfilename.c_str(), header);

  Imf::FrameBuffer framebuffer;

  for (int i = 0; i < N; ++i) {
    char tmpbuf[32];
    sprintf(tmpbuf, "c%d", i);
    framebuffer.insert(
        tmpbuf, Imf::Slice(Imf::UINT, (char *)&buffer[i * width * height], sizeof(uint32_t), sizeof(uint32_t) * width));
  }

  file.setFrameBuffer(framebuffer);
  file.writePixels(height);
  return true;
}

bool SaveHalfNEXR(const std::vector<float> &buffer, int width, int height, int N, std::string outfilename) {
  Imf::Header header(width, height);

  for (int i = 0; i < N; ++i) {
    char tmpbuf[32];
    sprintf(tmpbuf, "c%d", i);
    header.channels().insert(tmpbuf, Imf::Channel(Imf::HALF));
  }

  Imf::OutputFile file(outfilename.c_str(), header);

  std::vector<half> halfbuffer = convert_float_to_half(buffer);

  Imf::FrameBuffer framebuffer;

  for (int i = 0; i < N; ++i) {
    char tmpbuf[32];
    sprintf(tmpbuf, "c%d", i);
    framebuffer.insert(
        tmpbuf, Imf::Slice(Imf::HALF, (char *)&halfbuffer[i * width * height], sizeof(half), sizeof(half) * width));
  }

  file.setFrameBuffer(framebuffer);
  file.writePixels(height);
  return true;
}
#endif

bool LoadSphericalDelaunay(SphericalDelaunay *dela, std::string infilename) {
  nlohmann::json doc;
  nlohmann::json coords;
  nlohmann::json triangles;
  nlohmann::json adjtriangles;
  nlohmann::json angles;

  std::ifstream ifs(infilename);
  ifs >> doc;
  ifs.close();
  angles = doc["angles"];
  coords = doc["coords"];
  triangles = doc["triangles"];
  adjtriangles = doc["adjtriangles"];
  size_t numpt = doc["numpt"].get<size_t>();
  size_t numtri = doc["numtri"].get<size_t>();

  std::vector<double> vangles(numpt * 2), vcoords(numpt * 3);
  std::vector<size_t> vtri(numtri * 3);
  std::vector<int> vadjtri(numtri * 3);
  for (size_t i = 0; i < numpt; ++i) {
    vangles[2 * i] = angles[i][0].get<double>();
    vangles[2 * i + 1] = angles[i][1].get<double>();

    vcoords[3 * i] = coords[i][0].get<double>();
    vcoords[3 * i + 1] = coords[i][1].get<double>();
    vcoords[3 * i + 2] = coords[i][2].get<double>();
  }

  for (size_t i = 0; i < numtri; ++i) {
    vtri[3 * i] = triangles[i][0].get<size_t>();
    vtri[3 * i + 1] = triangles[i][1].get<size_t>();
    vtri[3 * i + 2] = triangles[i][2].get<size_t>();

    vadjtri[3 * i] = adjtriangles[i][0].get<int>();
    vadjtri[3 * i + 1] = adjtriangles[i][1].get<int>();
    vadjtri[3 * i + 2] = adjtriangles[i][2].get<int>();
  }

  dela->set_angles(vangles);
  dela->set_coords(vcoords);
  dela->set_triangles(vtri);
  dela->set_adjtriangles(vadjtri);

  return true;
}

bool SaveSphericalDelaunay(const SphericalDelaunay *dela, std::string outfilename) {
  nlohmann::json doc;
  nlohmann::json coords;
  nlohmann::json triangles;
  nlohmann::json adjtriangles;
  nlohmann::json angles;
  doc["numpt"] = dela->numpt();
  doc["numtri"] = dela->numtri();
  if (dela->angles().size() != dela->numpt() * 2 || dela->coords().size() != dela->numpt() * 3) return false;
  for (int i = 0; i < dela->numpt(); ++i) {
    nlohmann::json anglerow, coordsrow;
    anglerow.push_back(dela->angles()[2 * i]);
    anglerow.push_back(dela->angles()[2 * i + 1]);

    coordsrow.push_back(dela->coords()[3 * i]);
    coordsrow.push_back(dela->coords()[3 * i + 1]);
    coordsrow.push_back(dela->coords()[3 * i + 2]);

    angles.push_back(anglerow);
    coords.push_back(coordsrow);
  }
  if (dela->triangles().size() != dela->numtri() * 3 || dela->adjtriangles().size() != dela->numtri() * 3) return false;
  for (int i = 0; i < dela->numtri(); ++i) {
    nlohmann::json trirow, adjrow;
    trirow.push_back(dela->triangles()[3 * i]);
    trirow.push_back(dela->triangles()[3 * i + 1]);
    trirow.push_back(dela->triangles()[3 * i + 2]);

    adjrow.push_back(dela->adjtriangles()[3 * i]);
    adjrow.push_back(dela->adjtriangles()[3 * i + 1]);
    adjrow.push_back(dela->adjtriangles()[3 * i + 2]);

    triangles.push_back(trirow);
    adjtriangles.push_back(adjrow);
  }
  doc["angles"] = angles;
  doc["coords"] = coords;
  doc["triangles"] = triangles;
  doc["adjtriangles"] = adjtriangles;

  std::ofstream ofs(outfilename);

  ofs << doc.dump(1) << std::endl;
  return true;
}