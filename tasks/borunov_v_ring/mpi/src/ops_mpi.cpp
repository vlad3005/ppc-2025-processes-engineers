#include "borunov_v_ring/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "borunov_v_ring/common/include/common.hpp"

namespace borunov_v_ring {

BorunovVRingMPI::BorunovVRingMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  // GetOutput() инициализируется пустым вектором по умолчанию
}

// Валидация: проверяем корректность ранков источника и назначения
bool BorunovVRingMPI::ValidationImpl() {
  // Avoid calling MPI functions before MPI_Init. If MPI is not
  // initialized yet, only perform simple non-negativity checks so that
  // constructing task parameters or inspecting tasks doesn't require MPI.
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized == 0) {
    return (GetInput().source_rank >= 0 && GetInput().target_rank >= 0);
  }

  // Basic non-negativity check. RunImpl will normalize ranks when MPI is up.
  return (GetInput().source_rank >= 0 && GetInput().target_rank >= 0);
}

bool BorunovVRingMPI::PreProcessingImpl() {
  return true;
}

namespace {
// Helper: determine participation on ring for given ranks
bool ComputeIsParticipant(int ring_rank, int source, int target) {
  if (source == target) {
    return ring_rank == source;
  }
  if (source < target) {
    return (ring_rank >= source && ring_rank <= target);
  }
  return (ring_rank >= source || ring_rank <= target);
}

// Handle the case when current rank is the source: send to next
void HandleSource(BorunovVRingMPI *self, MPI_Comm ring_comm, int ring_rank, int next_rank, int target, int data) {
  std::vector<int> path_history;
  path_history.push_back(ring_rank);
  if (ring_rank == target) {
    std::vector<int> output;
    output = path_history;
    self->GetOutput() = output;
    return;
  }
  int path_size = static_cast<int>(path_history.size());
  MPI_Send(&path_size, 1, MPI_INT, next_rank, 0, ring_comm);
  if (path_size > 0) {
    MPI_Send(path_history.data(), path_size, MPI_INT, next_rank, 1, ring_comm);
  }
  MPI_Send(&data, 1, MPI_INT, next_rank, 2, ring_comm);
}

// Handle participant receiving and forwarding
void HandleParticipant(BorunovVRingMPI *self, MPI_Comm ring_comm, int prev_rank, int next_rank, int ring_rank,
                       int target) {
  int path_size = 0;
  MPI_Status status;
  MPI_Recv(&path_size, 1, MPI_INT, prev_rank, 0, ring_comm, &status);
  // Validate and clamp received path size to avoid invalid allocations
  int comm_size = 0;
  MPI_Comm_size(ring_comm, &comm_size);
  if (path_size < 0) {
    path_size = 0;
  } else if (path_size > comm_size) {
    path_size = comm_size;
  }
  std::vector<int> path_history(static_cast<std::size_t>(path_size));
  if (path_size > 0) {
    MPI_Recv(path_history.data(), path_size, MPI_INT, prev_rank, 1, ring_comm, &status);
  }
  int received_data = 0;
  MPI_Recv(&received_data, 1, MPI_INT, prev_rank, 2, ring_comm, &status);
  path_history.push_back(ring_rank);
  if (ring_rank == target) {
    self->GetOutput() = path_history;
    return;
  }
  path_size = static_cast<int>(path_history.size());
  MPI_Send(&path_size, 1, MPI_INT, next_rank, 0, ring_comm);
  MPI_Send(path_history.data(), path_size, MPI_INT, next_rank, 1, ring_comm);
  MPI_Send(&received_data, 1, MPI_INT, next_rank, 2, ring_comm);
}
}  // namespace

bool BorunovVRingMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &input = GetInput();
  int source = input.source_rank;
  int target = input.target_rank;

  // Normalize ranks modulo world size
  if (world_size > 0) {
    source = source % world_size;
    target = target % world_size;
  }

  // Create ring communicator
  MPI_Group world_group = MPI_GROUP_NULL;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  MPI_Comm ring_comm = MPI_COMM_WORLD;
  MPI_Comm_dup(MPI_COMM_WORLD, &ring_comm);

  int ring_rank = 0;
  int ring_size = 0;
  MPI_Comm_rank(ring_comm, &ring_rank);
  MPI_Comm_size(ring_comm, &ring_size);

  int next_rank = (ring_rank + 1) % ring_size;
  int prev_rank = (ring_rank - 1 + ring_size) % ring_size;

  MPI_Group_free(&world_group);

  bool is_participant = ComputeIsParticipant(ring_rank, source, target);

  if (ring_rank == source) {
    HandleSource(this, ring_comm, ring_rank, next_rank, target, input.data);
  } else if (is_participant) {
    HandleParticipant(this, ring_comm, prev_rank, next_rank, ring_rank, target);
  }

  MPI_Barrier(ring_comm);
  MPI_Comm_free(&ring_comm);
  return true;
}

bool BorunovVRingMPI::PostProcessingImpl() {
  return true;
}

}  // namespace borunov_v_ring
