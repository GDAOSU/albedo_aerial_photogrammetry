#ifndef AOPROPS_H
#define AOPROPS_H

#include <memory>
#include <string>
#include <vector>

#include "sphere_tessellation.h"

struct AOProps {
  const size_t PACKNUM = sizeof(char) * 8;
  bool use_sphere;
  std::shared_ptr<SphericalDelaunay> dir_sampler;
  int num_samples;
  int pack_stride;
  std::vector<std::vector<char>> buffer;
  std::vector<std::vector<float>> pts;

  void init_dir_sampler(bool use_sphere, int num_samples);
  void resize(const std::vector<size_t>& szs);

  AOProps& reset();
  AOProps& set(int meshID, int vid, int rid, bool state = true);
  bool test(int meshID, int vid, int rid) const;

  // AOProps& set_pts(int meshID, int vid, const float& x, const float& y, const float& z);
  // bool pt(int meshID, int vid, float& x, float& y, float& z);

  bool read_dir_samples(std::string inpath);
  bool read_buffer(std::string inpath);
  bool read_buffer(std::ifstream& ifs);
  // bool read_pts(std::string outpath);
  // bool read_pts(std::ifstream& ifs);

  bool write_dir_samples(std::string outpath) const;
  bool write_buffer(std::string outpath) const;
  bool write_buffer(std::ofstream& ofs) const;
  // bool write_pts(std::string outpath) const;
  // bool write_pts(std::ofstream& ofs) const;
};

#endif  // AOPROPS_H