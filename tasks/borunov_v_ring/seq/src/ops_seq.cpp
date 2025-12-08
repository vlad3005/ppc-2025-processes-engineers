#include "borunov_v_ring/seq/include/ops_seq.hpp"

#include <mpi.h>

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "borunov_v_ring/common/include/common.hpp"

namespace {
// Helper: determine graph neighbors (next, prev) for a given graph_rank.
std::pair<int, int> GetGraphNeighbors(MPI_Comm graph_comm, int cart_size, int graph_rank, int next_rank,
                                      int prev_rank) {
  int nneighbors = 0;
  MPI_Graph_neighbors_count(graph_comm, graph_rank, &nneighbors);
  std::vector<int> neighbors(static_cast<std::size_t>(nneighbors));
  if (nneighbors > 0) {
    MPI_Graph_neighbors(graph_comm, graph_rank, nneighbors, neighbors.data());
  }

  int graph_next = 0;
  int graph_prev = 0;
  if (nneighbors >= 2) {
    int expected_prev = (graph_rank - 1 + cart_size) % cart_size;
    if (neighbors[0] == expected_prev) {
      std::swap(neighbors[0], neighbors[1]);
    }
    graph_next = neighbors[0];
    graph_prev = neighbors[1];
  } else if (nneighbors == 1) {
    graph_next = neighbors[0];
    graph_prev = neighbors[0];
  } else {
    graph_next = next_rank;
    graph_prev = prev_rank;
  }

  return {graph_next, graph_prev};
}

// Helper: send path and data to destination over ring communicator
void SendPath(MPI_Comm comm, int dest, const std::vector<int> &path, int data) {
  int path_size = static_cast<int>(path.size());
  MPI_Send(&path_size, 1, MPI_INT, dest, 0, comm);
  if (path_size > 0) {
    MPI_Send(path.data(), path_size, MPI_INT, dest, 1, comm);
  }
  MPI_Send(&data, 1, MPI_INT, dest, 2, comm);
}

// Helper: receive path and data from source over ring communicator
std::tuple<std::vector<int>, int> ReceivePath(MPI_Comm comm, int src) {
  int path_size = 0;
  MPI_Status status;
  MPI_Recv(&path_size, 1, MPI_INT, src, 0, comm, &status);
  std::vector<int> path(static_cast<std::size_t>(path_size));
  if (path_size > 0) {
    MPI_Recv(path.data(), path_size, MPI_INT, src, 1, comm, &status);
  }
  int data = 0;
  MPI_Recv(&data, 1, MPI_INT, src, 2, comm, &status);
  return {path, data};
}

// Helper: determine if a rank participates in the transmission
bool IsParticipant(int rank, int source, int target) {
  if (source == target) {
    return rank == source;
  }
  return (source < target) ? (rank >= source && rank <= target) : (rank >= source || rank <= target);
}

// Helper: setup and create topologies (cartesian and graph)
struct TopoSetup {
  int cart_rank;
  int cart_size;
  int graph_next;
  int graph_prev;
  MPI_Comm cart_comm;
  MPI_Comm graph_comm;
  int cart_result;
  int graph_result;
};

TopoSetup CreateTopologies(int world_size) {
  TopoSetup topo{};
  int ndims = 1;
  std::array<int, 1> dims = {world_size};
  std::array<int, 1> periods = {1};
  int reorder = 0;

  topo.cart_comm = MPI_COMM_WORLD;
  topo.cart_result = MPI_Cart_create(MPI_COMM_WORLD, ndims, dims.data(), periods.data(), reorder, &topo.cart_comm);
  if (topo.cart_result != MPI_SUCCESS) {
    topo.cart_comm = MPI_COMM_WORLD;
  }

  MPI_Comm_rank(topo.cart_comm, &topo.cart_rank);
  MPI_Comm_size(topo.cart_comm, &topo.cart_size);

  std::array<int, 1> coords{};
  MPI_Cart_coords(topo.cart_comm, topo.cart_rank, ndims, coords.data());

  int next_rank = 0;
  int prev_rank = 0;
  MPI_Cart_shift(topo.cart_comm, 0, 1, &prev_rank, &next_rank);

  std::vector<int> index(static_cast<std::size_t>(topo.cart_size));
  std::vector<int> edges(static_cast<std::size_t>(topo.cart_size) * 2);
  for (int i = 0; i < topo.cart_size; ++i) {
    index[static_cast<std::size_t>(i)] = (i + 1) * 2;
    const std::size_t base = static_cast<std::size_t>(i) * 2;
    edges[base] = (i + 1) % topo.cart_size;
    edges[base + 1] = (i - 1 + topo.cart_size) % topo.cart_size;
  }

  topo.graph_comm = MPI_COMM_WORLD;
  topo.graph_result =
      MPI_Graph_create(topo.cart_comm, topo.cart_size, index.data(), edges.data(), reorder, &topo.graph_comm);
  if (topo.graph_result != MPI_SUCCESS) {
    topo.graph_comm = topo.cart_comm;
  }

  auto [gn, gp] = GetGraphNeighbors(topo.graph_comm, topo.cart_size, topo.cart_rank, next_rank, prev_rank);
  topo.graph_next = gn;
  topo.graph_prev = gp;
  return topo;
}
}  // namespace

namespace borunov_v_ring {

BorunovVRingSEQ::BorunovVRingSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  // GetOutput() инициализируется пустым вектором по умолчанию
}

bool BorunovVRingSEQ::ValidationImpl() {
  // Проверяем валидность входных данных относительно текущего размера коммуникатора
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (GetInput().source_rank < 0 || GetInput().source_rank >= size) {
    return false;
  }
  if (GetInput().target_rank < 0 || GetInput().target_rank >= size) {
    return false;
  }

  return true;
}

bool BorunovVRingSEQ::PreProcessingImpl() {
  return true;
}

bool BorunovVRingSEQ::RunImpl() {
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &input = GetInput();
  int source = input.source_rank;
  int target = input.target_rank;

  auto topo = CreateTopologies(world_size);
  int graph_rank = 0;
  MPI_Comm_rank(topo.graph_comm, &graph_rank);
  MPI_Comm ring_comm = topo.graph_comm;

  std::vector<int> path_history;
  bool is_participant = IsParticipant(graph_rank, source, target);

  if (graph_rank == source) {
    path_history.push_back(graph_rank);
    if (graph_rank != target) {
      SendPath(ring_comm, topo.graph_next, path_history, input.data);
    }
    if (graph_rank == target) {
      GetOutput() = path_history;
    }
  } else if (is_participant) {
    auto recv = ReceivePath(ring_comm, topo.graph_prev);
    path_history = std::get<0>(recv);
    int received_data = std::get<1>(recv);
    path_history.push_back(graph_rank);
    if (graph_rank == target) {
      GetOutput() = path_history;
    } else {
      SendPath(ring_comm, topo.graph_next, path_history, received_data);
    }
  }

  MPI_Barrier(ring_comm);

  if (topo.graph_result == MPI_SUCCESS && topo.graph_comm != MPI_COMM_WORLD) {
    MPI_Comm_free(&topo.graph_comm);
  }
  if (topo.cart_result == MPI_SUCCESS && topo.cart_comm != MPI_COMM_WORLD) {
    MPI_Comm_free(&topo.cart_comm);
  }

  return true;
}

bool BorunovVRingSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace borunov_v_ring
