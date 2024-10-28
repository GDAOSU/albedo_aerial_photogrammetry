#include <stdint.h>

#include <string>
#include <vector>

#include "sphere_tessellation.h"

bool SaveFloat1EXR(const std::vector<float> &buffer, int width, int height, std::string outfilename);
bool SaveHalf1EXR(const std::vector<float> &buffer, int width, int height, std::string outfilename);
bool SaveUInt1EXR(const std::vector<uint32_t> &buffer, int width, int height, std::string outfilename);

bool SaveFloat3EXR(const std::vector<float> &buffer, int width, int height, std::string outfilename);
bool SaveHalf3EXR(const std::vector<float> &buffer, int width, int height, std::string outfilename);

bool SaveHalfNEXR(const std::vector<float> &buffer, int width, int height, int N, std::string outfilename);
bool SaveUIntNEXR(const std::vector<uint32_t> &buffer, int width, int height, int N, std::string outfilename);

bool LoadSphericalDelaunay(SphericalDelaunay *dela, std::string infilename);
bool SaveSphericalDelaunay(const SphericalDelaunay *dela, std::string outfilename);