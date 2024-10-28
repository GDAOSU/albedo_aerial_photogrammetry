#include "AOProps.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <fstream>
#include <iostream>

#include "writer.h"

void AOProps::init_dir_sampler(bool use_sphere, int num_samples) {
  if (use_sphere) {
    dir_sampler.reset(new SphereDelaunay(num_samples));
  } else {
    dir_sampler.reset(new HemisphereDelaunay(num_samples));
  }
  this->use_sphere = use_sphere;
  this->num_samples = num_samples;
  this->pack_stride = num_samples / PACKNUM + !!(num_samples % PACKNUM);
}

void AOProps::resize(const std::vector<size_t>& szs) {
  buffer.resize(szs.size());
  pts.resize(szs.size());
  for (size_t i = 0; i < szs.size(); ++i) {
    buffer[i].resize(szs[i] * pack_stride);
    pts.resize(szs[i]);
  }
}

AOProps& AOProps::reset() {
  for (size_t i = 0; i < buffer.size(); ++i) memset(buffer[i].data(), 0u, buffer[i].size());
  return *this;
}

AOProps& AOProps::set(int meshID, int vid, int rid, bool state) {
  assert(meshID < buffer.size());
  assert(vid < buffer[meshID].size());
  assert(rid < pack_stride * PACKNUM);
  char* packhits = buffer[meshID].data() + pack_stride * vid;
  int _channel = rid / PACKNUM;
  int _bit = rid % PACKNUM;
  if (state)
    packhits[_channel] |= 0x1u << _bit;
  else
    packhits[_channel] &= ~(0x1u << _bit);
  return *this;
}

bool AOProps::test(int meshID, int vid, int rid) const {
  assert(meshID < buffer.size());
  assert(vid < buffer[meshID].size());
  assert(rid < pack_stride * PACKNUM);
  const char* packhits = buffer[meshID].data() + pack_stride * vid;
  int _channel = rid / PACKNUM;
  int _bit = rid % PACKNUM;
  return (0x1u << _bit) & packhits[_channel];
}

bool AOProps::read_dir_samples(std::string inpath) {
  if (!dir_sampler) dir_sampler.reset(new SphereDelaunay);
  return LoadSphericalDelaunay(dir_sampler.get(), inpath);
}

bool AOProps::read_buffer(std::string inpath) {
  std::ifstream ifs(inpath, std::ios::binary | std::ios::out);
  if (!ifs.good()) return false;
  bool ret = read_buffer(ifs);
  ifs.close();
  return ret;
}

bool AOProps::read_buffer(std::ifstream& ifs) {
  int mNumMesh;
  ifs.read((char*)&num_samples, sizeof(int));
  ifs.read((char*)&pack_stride, sizeof(int));
  ifs.read((char*)&mNumMesh, sizeof(int));

  std::vector<size_t> verts(mNumMesh);
  for (int mi = 0; mi < mNumMesh; ++mi) {
    int nvert;
    ifs.read((char*)&nvert, sizeof(int));
    verts[mi] = nvert;
  }
  resize(verts);
  for (int mi = 0; mi < mNumMesh; ++mi) {
    ifs.read(buffer[mi].data(), buffer[mi].size());
  }
  return true;
}

bool AOProps::write_dir_samples(std::string outpath) const {
  if (!dir_sampler) return false;
  return SaveSphericalDelaunay(dir_sampler.get(), outpath);
}

bool AOProps::write_buffer(std::string outpath) const {
  std::ofstream ofs;
  ofs.open(outpath, std::ios::binary | std::ios::out);
  if (!ofs.good()) return false;
  bool ret = write_buffer(ofs);
  ofs.close();
  return ret;
}

bool AOProps::write_buffer(std::ofstream& ofs) const {
  ofs.write((char*)&num_samples, sizeof(int));
  ofs.write((char*)&pack_stride, sizeof(int));
  int mNumMesh = buffer.size();
  ofs.write((char*)&mNumMesh, sizeof(int));
  for (int mi = 0; mi < mNumMesh; ++mi) {
    int mNumVert = buffer[mi].size() / pack_stride;
    ofs.write((char*)&mNumVert, sizeof(int));
  }

  for (int mi = 0; mi < mNumMesh; ++mi) {
    ofs.write(buffer[mi].data(), buffer[mi].size());
  }
  return true;
}