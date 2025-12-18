#include "borunov_v_block_partitioning/seq/include/ops_seq.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

#include "borunov_v_block_partitioning/common/include/common.hpp"
#include "util/include/util.hpp"

namespace borunov_v_block_partitioning {

BorunovVBlockPartitioningSEQ::BorunovVBlockPartitioningSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BorunovVBlockPartitioningSEQ::ValidationImpl() {
  if (GetInput().size() < 2) {
    return false;
  }
  int w = GetInput()[0];
  int h = GetInput()[1];
  return GetInput().size() == static_cast<size_t>(2 + w * h);
}

bool BorunovVBlockPartitioningSEQ::PreProcessingImpl() {
  int w = GetInput()[0];
  int h = GetInput()[1];
  GetOutput().assign(w * h, 0);
  return true;
}

bool BorunovVBlockPartitioningSEQ::RunImpl() {
  int width = GetInput()[0];
  int height = GetInput()[1];
  const int *pixels = GetInput().data() + 2;

  const float kernel[3][3] = {{1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f},
                              {2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f},
                              {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f}};

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      float sum = 0.0f;

      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          int nx = std::clamp(j + kx, 0, width - 1);
          int ny = std::clamp(i + ky, 0, height - 1);

          sum += static_cast<float>(pixels[ny * width + nx]) * kernel[ky + 1][kx + 1];
        }
      }
      GetOutput()[i * width + j] = static_cast<int>(std::round(sum));
    }
  }
  return true;
}

bool BorunovVBlockPartitioningSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace borunov_v_block_partitioning
