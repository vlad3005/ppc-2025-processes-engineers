#include "borunov_v_block_partitioning/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "borunov_v_block_partitioning/common/include/common.hpp"

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
  const std::size_t expected_size =
      static_cast<std::size_t>(2) + static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  return GetInput().size() == expected_size;
}

bool BorunovVBlockPartitioningSEQ::PreProcessingImpl() {
  int w = GetInput()[0];
  int h = GetInput()[1];
  GetOutput().assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
  return true;
}

bool BorunovVBlockPartitioningSEQ::RunImpl() {
  int width = GetInput()[0];
  int height = GetInput()[1];
  const int *pixels = GetInput().data() + 2;

  const std::array<std::array<float, 3>, 3> kernel = {{
      {1.0F / 16.0F, 2.0F / 16.0F, 1.0F / 16.0F},
      {2.0F / 16.0F, 4.0F / 16.0F, 2.0F / 16.0F},
      {1.0F / 16.0F, 2.0F / 16.0F, 1.0F / 16.0F},
  }};

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      float sum = 0.0F;

      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          int nx = std::clamp(j + kx, 0, width - 1);
          int ny = std::clamp(i + ky, 0, height - 1);

          sum += static_cast<float>(pixels[(ny * width) + nx]) *
                 kernel[static_cast<std::size_t>(ky + 1)][static_cast<std::size_t>(kx + 1)];
        }
      }
      GetOutput()[(i * width) + j] = static_cast<int>(std::round(sum));
    }
  }
  return true;
}

bool BorunovVBlockPartitioningSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace borunov_v_block_partitioning
