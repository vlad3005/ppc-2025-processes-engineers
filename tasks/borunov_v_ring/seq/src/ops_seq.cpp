#include "borunov_v_ring/seq/include/ops_seq.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

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
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &input = GetInput();
  int source = input.source_rank;
  int target = input.target_rank;

  // ============================================================
  // СОЗДАНИЕ ВИРТУАЛЬНОЙ ТОПОЛОГИИ КОЛЬЦА С ИСПОЛЬЗОВАНИЕМ MPI_Cart_Create
  // ============================================================

  // Создаем одномерную декартову топологию с периодическими границами (кольцо)
  int ndims = 1;
  int dims[1] = {world_size};  // Размерность: одномерная сетка размером world_size
  int periods[1] = {1};        // Периодические границы (кольцо)
  int reorder = 0;             // Не переупорядочиваем процессы

  MPI_Comm cart_comm;
  int cart_result = MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &cart_comm);

  if (cart_result != MPI_SUCCESS) {
    // Если не удалось создать декартову топологию, используем MPI_COMM_WORLD
    cart_comm = MPI_COMM_WORLD;
  }

  int cart_rank = 0;
  int cart_size = 0;
  MPI_Comm_rank(cart_comm, &cart_rank);
  MPI_Comm_size(cart_comm, &cart_size);

  // Получаем координаты процесса в декартовой топологии
  int coords[1];
  MPI_Cart_coords(cart_comm, cart_rank, ndims, coords);

  // Определяем соседей в кольцевой топологии используя MPI_Cart_shift
  int next_rank = 0;
  int prev_rank = 0;
  MPI_Cart_shift(cart_comm, 0, 1, &prev_rank, &next_rank);  // Сдвиг на 1 в направлении 0

  // ============================================================
  // СОЗДАНИЕ ГРАФОВОЙ ТОПОЛОГИИ С ИСПОЛЬЗОВАНИЕМ MPI_Graph_Create
  // ============================================================

  // Для кольцевой топологии каждый процесс связан с двумя соседями
  // Создаем массивы для графовой топологии
  std::vector<int> index(cart_size);      // Индексы начала списка соседей для каждого процесса
  std::vector<int> edges(cart_size * 2);  // Список всех соседей

  for (int i = 0; i < cart_size; ++i) {
    index[i] = (i + 1) * 2;                              // Каждый процесс имеет 2 соседа
    edges[i * 2] = (i + 1) % cart_size;                  // Следующий процесс
    edges[i * 2 + 1] = (i - 1 + cart_size) % cart_size;  // Предыдущий процесс
  }

  MPI_Comm graph_comm;
  int graph_result = MPI_Graph_create(cart_comm, cart_size, index.data(), edges.data(), reorder, &graph_comm);

  if (graph_result != MPI_SUCCESS) {
    // Если не удалось создать графовую топологию, используем cart_comm
    graph_comm = cart_comm;
  }

  int graph_rank = 0;
  MPI_Comm_rank(graph_comm, &graph_rank);

  // Получаем количество соседей и их список из графовой топологии
  int nneighbors = 0;
  MPI_Graph_neighbors_count(graph_comm, graph_rank, &nneighbors);
  std::vector<int> neighbors(nneighbors);
  if (nneighbors > 0) {
    MPI_Graph_neighbors(graph_comm, graph_rank, nneighbors, neighbors.data());
  }

  // Определяем соседей из графовой топологии
  // В кольце каждый процесс имеет двух соседей: предыдущий и следующий
  int graph_next = 0;
  int graph_prev = 0;
  if (nneighbors >= 2) {
    // Находим следующего и предыдущего соседа
    // Следующий: (rank + 1) % size
    // Предыдущий: (rank - 1 + size) % size
    int expected_next = (graph_rank + 1) % cart_size;
    int expected_prev = (graph_rank - 1 + cart_size) % cart_size;

    // Определяем, какой сосед является следующим, а какой предыдущим
    if (neighbors[0] == expected_next) {
      graph_next = neighbors[0];
      graph_prev = neighbors[1];
    } else if (neighbors[0] == expected_prev) {
      graph_prev = neighbors[0];
      graph_next = neighbors[1];
    } else {
      // Fallback: используем первый и второй сосед
      graph_next = neighbors[0];
      graph_prev = neighbors[1];
    }
  } else if (nneighbors == 1) {
    // Только один сосед (маловероятно для кольца, но на всякий случай)
    graph_next = neighbors[0];
    graph_prev = neighbors[0];
  } else {
    // Нет соседей - используем значения из cart_shift
    graph_next = next_rank;
    graph_prev = prev_rank;
  }

  // Используем графовую топологию для передачи данных
  MPI_Comm ring_comm = graph_comm;

  // ============================================================
  // ПЕРЕДАЧА ДАННЫХ ЧЕРЕЗ КОЛЬЦЕВУЮ ТОПОЛОГИЮ
  // ============================================================

  // Вектор, который будет хранить путь прохождения данных через кольцо
  std::vector<int> path_history;

  // Определяем, участвует ли текущий процесс в передаче данных
  bool is_participant = false;

  if (source == target) {
    is_participant = (graph_rank == source);
  } else {
    if (source < target) {
      is_participant = (graph_rank >= source && graph_rank <= target);
    } else {
      is_participant = (graph_rank >= source || graph_rank <= target);
    }
  }

  if (graph_rank == source) {
    // ========== Я ИСТОЧНИК ==========
    path_history.push_back(graph_rank);

    if (graph_rank == target) {
      GetOutput() = path_history;
    } else {
      // Отправляем данные следующему процессу в кольце
      int path_size = path_history.size();
      MPI_Send(&path_size, 1, MPI_INT, graph_next, 0, ring_comm);
      MPI_Send(path_history.data(), path_size, MPI_INT, graph_next, 1, ring_comm);
      MPI_Send(&input.data, 1, MPI_INT, graph_next, 2, ring_comm);
    }
  } else if (is_participant) {
    // ========== Я ПРОМЕЖУТОЧНЫЙ УЗЕЛ ИЛИ ПОЛУЧАТЕЛЬ ==========

    int path_size = 0;
    MPI_Status status;

    MPI_Recv(&path_size, 1, MPI_INT, graph_prev, 0, ring_comm, &status);
    path_history.resize(path_size);
    MPI_Recv(path_history.data(), path_size, MPI_INT, graph_prev, 1, ring_comm, &status);

    int received_data = 0;
    MPI_Recv(&received_data, 1, MPI_INT, graph_prev, 2, ring_comm, &status);

    path_history.push_back(graph_rank);

    if (graph_rank == target) {
      // ========== Я ЦЕЛЬ ==========
      GetOutput() = path_history;
    } else {
      // ========== ПЕРЕДАЮ ДАЛЬШЕ ПО КОЛЬЦУ ==========
      path_size = path_history.size();
      MPI_Send(&path_size, 1, MPI_INT, graph_next, 0, ring_comm);
      MPI_Send(path_history.data(), path_size, MPI_INT, graph_next, 1, ring_comm);
      MPI_Send(&received_data, 1, MPI_INT, graph_next, 2, ring_comm);
    }
  }

  // Синхронизация всех процессов
  MPI_Barrier(ring_comm);

  // Освобождаем созданные коммуникаторы
  if (graph_result == MPI_SUCCESS && graph_comm != MPI_COMM_WORLD) {
    MPI_Comm_free(&graph_comm);
  }
  if (cart_result == MPI_SUCCESS && cart_comm != MPI_COMM_WORLD) {
    MPI_Comm_free(&cart_comm);
  }

  return true;
}

bool BorunovVRingSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace borunov_v_ring
