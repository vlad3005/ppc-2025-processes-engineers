#include "borunov_v_block_partitioning/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "borunov_v_block_partitioning/common/include/common.hpp"

namespace borunov_v_block_partitioning {

BorunovVBlockPartitioningMPI::BorunovVBlockPartitioningMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BorunovVBlockPartitioningMPI::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    if (GetInput().size() < 3) {
      return false;
    }
    int w = GetInput()[0];
    int h = GetInput()[1];
    return GetInput().size() == static_cast<size_t>(2 + w * h);
  }
  return true;
}

bool BorunovVBlockPartitioningMPI::PreProcessingImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    int w = GetInput()[0];
    int h = GetInput()[1];
    GetOutput().assign(w * h, 0);
  }
  return true;
}

bool BorunovVBlockPartitioningMPI::RunImpl() {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int width = 0, height = 0;
  if (rank == 0) {
    width = GetInput()[0];
    height = GetInput()[1];
  }
  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> send_counts(size);
  std::vector<int> displs(size);
  int rows_per_proc = height / size;
  int remainder = height % size;
  int current_displ = 0;
  for (int i = 0; i < size; ++i) {
    int proc_rows = rows_per_proc + (i < remainder ? 1 : 0);
    send_counts[i] = proc_rows * width;
    displs[i] = current_displ;
    current_displ += send_counts[i];
  }

  int my_rows = send_counts[rank] / width;
  std::vector<int> local_input(send_counts[rank]);

  MPI_Scatterv(rank == 0 ? GetInput().data() + 2 : nullptr, send_counts.data(), displs.data(), MPI_INT,
               local_input.data(), send_counts[rank], MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> up_row(width, 0);
  std::vector<int> down_row(width, 0);
  int up_neighbor = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
  int down_neighbor = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

  MPI_Sendrecv(local_input.data(), width, MPI_INT, up_neighbor, 0, down_row.data(), width, MPI_INT, down_neighbor, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(local_input.data() + (my_rows - 1) * width, width, MPI_INT, down_neighbor, 1, up_row.data(), width,
               MPI_INT, up_neighbor, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::vector<int> local_res(my_rows * width);
  const float kernel[3][3] = {
      {1.0f / 16, 2.0f / 16, 1.0f / 16}, {2.0f / 16, 4.0f / 16, 2.0f / 16}, {1.0f / 16, 2.0f / 16, 1.0f / 16}};

  for (int i = 0; i < my_rows; ++i) {
    for (int j = 0; j < width; ++j) {
      float sum = 0.0f;
      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          int nx = std::clamp(j + kx, 0, width - 1);
          int val = 0;
          int target_row = i + ky;

          if (target_row < 0) {
            val = (rank == 0) ? local_input[i * width + nx] : up_row[nx];
          } else if (target_row >= my_rows) {
            val = (rank == size - 1) ? local_input[i * width + nx] : down_row[nx];
          } else {
            val = local_input[target_row * width + nx];
          }
          sum += static_cast<float>(val) * kernel[ky + 1][kx + 1];
        }
      }
      local_res[i * width + j] = static_cast<int>(std::round(sum));
    }
  }

  MPI_Gatherv(local_res.data(), local_res.size(), MPI_INT, rank == 0 ? GetOutput().data() : nullptr, send_counts.data(),
              displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool BorunovVBlockPartitioningMPI::PostProcessingImpl() {
  int rank;
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
    GetOutput().assign(total_pixels, 0);
  }

  MPI_Bcast(GetOutput().data(), total_pixels, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

}  // namespace borunov_v_block_partitioning
