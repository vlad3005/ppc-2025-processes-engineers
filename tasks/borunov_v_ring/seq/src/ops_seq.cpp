#include "borunov_v_ring/seq/include/ops_seq.hpp"

#include <mpi.h>

#include <array>
#include <vector>

#include "borunov_v_ring/common/include/common.hpp"
#include "util/include/util.hpp"

namespace {
// Helper: determine if a rank participates in the transmission
bool IsParticipant(int rank, int source, int target) {
  if (source == target) {
    return rank == source;
  }
  return (source < target) ? (rank >= source && rank <= target) : (rank >= source || rank <= target);
}

// Helper: send path and data
void SendPath(MPI_Comm comm, int dest, const std::vector<int> &path, int data) {
  int path_size = static_cast<int>(path.size());
  MPI_Send(&path_size, 1, MPI_INT, dest, 0, comm);
  if (path_size > 0) {
    MPI_Send(path.data(), path_size, MPI_INT, dest, 1, comm);
  }
  MPI_Send(&data, 1, MPI_INT, dest, 2, comm);
}

// Helper: receive path and data
std::vector<int> ReceivePath(MPI_Comm comm, int src) {
  int path_size = 0;
  MPI_Status status;
  MPI_Recv(&path_size, 1, MPI_INT, src, 0, comm, &status);

  // Clamp received size to comm size
  int comm_size = 0;
  MPI_Comm_size(comm, &comm_size);
  if (path_size < 0 || path_size > comm_size) {
    path_size = comm_size;
  }

  std::vector<int> path;
  if (path_size > 0) {
    path.resize(path_size);
    MPI_Recv(path.data(), path_size, MPI_INT, src, 1, comm, &status);
  }

  int data = 0;
  MPI_Recv(&data, 1, MPI_INT, src, 2, comm, &status);
  return path;
}

// Sequential fallback
bool RunSequentialFallback(borunov_v_ring::BorunovVRingSEQ *self, int source, int target) {
  int size = ppc::util::GetNumProc();
  if (size <= 0) {
    self->GetOutput().clear();
    return true;
  }
  std::vector<int> path_history;
  int current = source % size;
  int steps = 0;
  while (current != (target % size) && steps < size) {
    path_history.push_back(current);
    current = (current + 1) % size;
    ++steps;
  }
  if (steps < size || current == (target % size)) {
    path_history.push_back(current);
  }
  self->GetOutput() = path_history;
  return true;
}

// Ring topology using Cartesian topology
bool RunMpiBranch(borunov_v_ring::BorunovVRingSEQ *self, int source, int target, int world_size) {
  // Create 1D Cartesian ring topology
  std::array<int, 1> dims = {world_size};
  std::array<int, 1> periods = {1};
  MPI_Comm ring_comm = MPI_COMM_WORLD;
  MPI_Cart_create(MPI_COMM_WORLD, 1, dims.data(), periods.data(), 0, &ring_comm);

  int ring_rank = 0;
  int ring_size = 0;
  MPI_Comm_rank(ring_comm, &ring_rank);
  MPI_Comm_size(ring_comm, &ring_size);

  // Get neighbors using Cart_shift
  int prev_rank = 0;
  int next_rank = 0;
  MPI_Cart_shift(ring_comm, 0, 1, &prev_rank, &next_rank);

  bool is_participant = IsParticipant(ring_rank, source, target);

  if (ring_rank == source) {
    // Source: send path starting with itself
    std::vector<int> path_history;
    path_history.push_back(ring_rank);
    if (ring_rank != target) {
      SendPath(ring_comm, next_rank, path_history, self->GetInput().data);
    } else {
      self->GetOutput() = path_history;
    }
  } else if (is_participant) {
    // Participant: receive, append, and forward
    std::vector<int> path_history = ReceivePath(ring_comm, prev_rank);
    path_history.push_back(ring_rank);
    if (ring_rank == target) {
      self->GetOutput() = path_history;
    } else {
      SendPath(ring_comm, next_rank, path_history, self->GetInput().data);
    }
  }

  MPI_Barrier(ring_comm);
  if (ring_comm != MPI_COMM_WORLD) {
    MPI_Comm_free(&ring_comm);
  }

  return true;
}
}  // namespace

namespace borunov_v_ring {

BorunovVRingSEQ::BorunovVRingSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BorunovVRingSEQ::ValidationImpl() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized == 0) {
    return (GetInput().source_rank >= 0 && GetInput().target_rank >= 0);
  }
  return (GetInput().source_rank >= 0 && GetInput().target_rank >= 0);
}

bool BorunovVRingSEQ::PreProcessingImpl() {
  return true;
}

bool BorunovVRingSEQ::RunImpl() {
  int source = GetInput().source_rank;
  int target = GetInput().target_rank;

  if (!ppc::util::IsUnderMpirun()) {
    return RunSequentialFallback(this, source, target);
  }

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (world_size > 0) {
    source = source % world_size;
    target = target % world_size;
  }

  return RunMpiBranch(this, source, target, world_size);
}

bool BorunovVRingSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace borunov_v_ring
