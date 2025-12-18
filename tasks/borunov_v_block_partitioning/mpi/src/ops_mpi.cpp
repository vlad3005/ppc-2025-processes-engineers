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

bool BorunovVBlockPartitioningMPI::RunImpl() {  // NOLINT(readability-function-cognitive-complexity)
  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Получаем размеры изображения
  int width = 0;
  int height = 0;
  if (rank == 0) {
    width = GetInput()[0];
    height = GetInput()[1];
  }
  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Рассылаем всем процессам полный набор пикселей
  std::vector<int> pixels_storage;
  int *pixels = nullptr;
  if (rank == 0) {
    pixels = GetInput().data() + 2;
  } else {
    pixels_storage.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    pixels = pixels_storage.data();
  }
  MPI_Bcast(pixels, width * height, MPI_INT, 0, MPI_COMM_WORLD);

  // Распределяем строки по процессам (как и раньше, но только для вычислений)
  std::vector<int> send_counts(size);
  std::vector<int> displs(size);
  const int rows_per_proc = height / size;
  const int remainder = height % size;
  int current_displ = 0;
  for (int i = 0; i < size; ++i) {
    const int proc_rows = rows_per_proc + (i < remainder ? 1 : 0);
    send_counts[i] = proc_rows * width;
    displs[i] = current_displ;
    current_displ += send_counts[i];
  }

  const int my_rows = send_counts[rank] / width;
  int row_start = 0;
  for (int i = 0; i < rank; ++i) {
    row_start += send_counts[i] / width;
  }
  const int row_end = row_start + my_rows;

  std::vector<int> local_res(static_cast<std::size_t>(my_rows) * static_cast<std::size_t>(width));
  const std::array<std::array<float, 3>, 3> kernel = {{
      {1.0F / 16.0F, 2.0F / 16.0F, 1.0F / 16.0F},
      {2.0F / 16.0F, 4.0F / 16.0F, 2.0F / 16.0F},
      {1.0F / 16.0F, 2.0F / 16.0F, 1.0F / 16.0F},
  }};

  // Локальная фильтрация на подмножестве строк
  for (int gi = row_start; gi < row_end; ++gi) {
    const int local_i = gi - row_start;
    for (int j = 0; j < width; ++j) {
      float sum = 0.0F;

      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          const int nx = std::clamp(j + kx, 0, width - 1);
          const int gy = std::clamp(gi + ky, 0, height - 1);
          const int val = pixels[(gy * width) + nx];

          sum += static_cast<float>(val) *
                 kernel[static_cast<std::size_t>(ky + 1)][static_cast<std::size_t>(
                     kx +
                     1)];  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index,bugprone-misplaced-widening-cast)
        }
      }

      local_res[(local_i * width) + j] = static_cast<int>(std::round(sum));
    }
  }

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
