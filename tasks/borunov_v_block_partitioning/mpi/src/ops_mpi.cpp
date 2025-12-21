#include "borunov_v_block_partitioning/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "borunov_v_block_partitioning/common/include/common.hpp"

namespace borunov_v_block_partitioning {

BorunovVBlockPartitioningMPI::BorunovVBlockPartitioningMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BorunovVBlockPartitioningMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    if (GetInput().size() < 3) {
      return false;
    }
    int w = GetInput()[0];
    int h = GetInput()[1];
    const std::size_t pixels = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    const std::size_t expected_size = static_cast<std::size_t>(2) + pixels;
    return GetInput().size() == expected_size;
  }
  return true;
}

bool BorunovVBlockPartitioningMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    int w = GetInput()[0];
    int h = GetInput()[1];
    GetOutput().assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
  }
  return true;
}

namespace {

void BroadcastDims(int &width, int &height) {
  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void BroadcastPixels(int *pixels, int width, int height) {
  MPI_Bcast(pixels, width * height, MPI_INT, 0, MPI_COMM_WORLD);
}

void ComputeSendCountsDispls(int width, int height, int size, std::vector<int> &send_counts, std::vector<int> &displs) {
  send_counts.assign(static_cast<std::size_t>(size), 0);
  displs.assign(static_cast<std::size_t>(size), 0);

  const int rows_per_proc = height / size;
  const int remainder = height % size;
  int current_displ = 0;
  for (int i = 0; i < size; ++i) {
    const int proc_rows = rows_per_proc + (i < remainder ? 1 : 0);
    send_counts[i] = proc_rows * width;
    displs[i] = current_displ;
    current_displ += send_counts[i];
  }
}

void ApplyKernelToPartition(const int *pixels, int width, int height, int row_start, int row_end,
                            std::vector<int> &local_res) {
  const std::array<std::array<float, 3>, 3> kernel = {{
      {1.0F / 16.0F, 2.0F / 16.0F, 1.0F / 16.0F},
      {2.0F / 16.0F, 4.0F / 16.0F, 2.0F / 16.0F},
      {1.0F / 16.0F, 2.0F / 16.0F, 1.0F / 16.0F},
  }};

  for (int gi = row_start; gi < row_end; ++gi) {
    const int local_i = gi - row_start;
    for (int j = 0; j < width; ++j) {
      const int x0 = std::clamp(j - 1, 0, width - 1);
      const int x1 = j;
      const int x2 = std::clamp(j + 1, 0, width - 1);

      const int y0 = std::clamp(gi - 1, 0, height - 1);
      const int y1 = gi;
      const int y2 = std::clamp(gi + 1, 0, height - 1);

      float sum = 0.0F;

      sum += static_cast<float>(pixels[(y0 * width) + x0]) * kernel[0][0];
      sum += static_cast<float>(pixels[(y0 * width) + x1]) * kernel[0][1];
      sum += static_cast<float>(pixels[(y0 * width) + x2]) * kernel[0][2];

      sum += static_cast<float>(pixels[(y1 * width) + x0]) * kernel[1][0];
      sum += static_cast<float>(pixels[(y1 * width) + x1]) * kernel[1][1];
      sum += static_cast<float>(pixels[(y1 * width) + x2]) * kernel[1][2];

      sum += static_cast<float>(pixels[(y2 * width) + x0]) * kernel[2][0];
      sum += static_cast<float>(pixels[(y2 * width) + x1]) * kernel[2][1];
      sum += static_cast<float>(pixels[(y2 * width) + x2]) * kernel[2][2];

      local_res[(local_i * width) + j] = static_cast<int>(std::round(sum));
    }
  }
}

}  // namespace

bool BorunovVBlockPartitioningMPI::RunImpl() {
  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int width = 0;
  int height = 0;
  if (rank == 0) {
    width = GetInput()[0];
    height = GetInput()[1];
  }
  BroadcastDims(width, height);

  std::vector<int> pixels_storage;
  int *pixels = nullptr;
  if (rank == 0) {
    pixels = GetInput().data() + 2;
  } else {
    pixels_storage.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    pixels = pixels_storage.data();
  }
  BroadcastPixels(pixels, width, height);

  std::vector<int> send_counts;
  std::vector<int> displs;
  ComputeSendCountsDispls(width, height, size, send_counts, displs);

  const int my_rows = send_counts[rank] / width;
  const int row_start = displs[rank] / width;
  const int row_end = row_start + my_rows;

  std::vector<int> local_res(static_cast<std::size_t>(my_rows) * static_cast<std::size_t>(width));
  ApplyKernelToPartition(pixels, width, height, row_start, row_end, local_res);

  const int local_count = static_cast<int>(local_res.size());  // assumes data size fits into int
  MPI_Gatherv(local_res.data(), local_count, MPI_INT, rank == 0 ? GetOutput().data() : nullptr, send_counts.data(),
              displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool BorunovVBlockPartitioningMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int width = 0;
  int height = 0;

  if (rank == 0) {
    width = GetInput()[0];
    height = GetInput()[1];
  }

  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int total_pixels = width * height;
  if (rank != 0) {
    GetOutput().assign(static_cast<std::size_t>(total_pixels), 0);
  }

  MPI_Bcast(GetOutput().data(), total_pixels, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

}  // namespace borunov_v_block_partitioning
