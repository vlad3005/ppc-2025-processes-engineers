#include "borunov_v_ring/mpi/include/ops_mpi.hpp"

#include <mpi.h>

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
  // As above, avoid calling MPI functions before MPI_Init. If MPI is not
  // initialized yet, only perform simple non-negativity checks so that
  // constructing task parameters or inspecting tasks doesn't require MPI.
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    return (GetInput().source_rank >= 0 && GetInput().target_rank >= 0);
  }

  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Проверяем, что указанные ранги существуют в текущем коммуникаторе
  if (GetInput().source_rank < 0 || GetInput().source_rank >= size) {
    return false;
  }
  if (GetInput().target_rank < 0 || GetInput().target_rank >= size) {
    return false;
  }

  return true;
}

bool BorunovVRingMPI::PreProcessingImpl() {
  return true;
}

bool BorunovVRingMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &input = GetInput();
  int source = input.source_rank;
  int target = input.target_rank;

  // ============================================================
  // СОЗДАНИЕ ВИРТУАЛЬНОЙ ТОПОЛОГИИ КОЛЬЦА С ИСПОЛЬЗОВАНИЕМ КОММУНИКАТОРОВ
  // ============================================================

  // Создаем группу всех процессов из MPI_COMM_WORLD
  MPI_Group world_group = MPI_GROUP_NULL;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  // Создаем новый коммуникатор для кольцевой топологии
  // Все процессы остаются в группе, но мы создаем отдельный коммуникатор
  // для явного представления топологии кольца
  MPI_Comm ring_comm = MPI_COMM_WORLD;
  MPI_Comm_dup(MPI_COMM_WORLD, &ring_comm);

  // Получаем ранг и размер в новом коммуникаторе кольца
  int ring_rank = 0;
  int ring_size = 0;
  MPI_Comm_rank(ring_comm, &ring_rank);
  MPI_Comm_size(ring_comm, &ring_size);

  // Определяем соседей в кольцевой топологии
  // next_rank - следующий процесс в кольце (по часовой стрелке)
  // prev_rank - предыдущий процесс в кольце (против часовой стрелки)
  int next_rank = (ring_rank + 1) % ring_size;
  int prev_rank = (ring_rank - 1 + ring_size) % ring_size;

  // Освобождаем группу (коммуникатор ring_comm будет освобожден позже)
  MPI_Group_free(&world_group);

  // ============================================================
  // ПЕРЕДАЧА ДАННЫХ ЧЕРЕЗ КОЛЬЦЕВУЮ ТОПОЛОГИЮ
  // ============================================================

  // Вектор, который будет хранить путь прохождения данных через кольцо
  std::vector<int> path_history;

  // Определяем, участвует ли текущий процесс в передаче данных
  // Данные передаются от source к target по кольцу в направлении по часовой стрелке
  bool is_participant = false;

  if (source == target) {
    // Если источник и цель совпадают, участвует только один процесс
    is_participant = (ring_rank == source);
  } else {
    // Определяем путь передачи по кольцу в направлении по часовой стрелке
    // Процесс участвует, если он находится на пути от source к target (включительно)

    if (source < target) {
      // Прямой путь без перехода через границу (например, 1 -> 2 -> 3)
      is_participant = (ring_rank >= source && ring_rank <= target);
    } else {
      // Путь с переходом через границу (например, 3 -> 0 -> 1)
      // Участвуют процессы: source, source+1, ..., size-1, 0, 1, ..., target
      is_participant = (ring_rank >= source || ring_rank <= target);
    }
  }

  if (ring_rank == source) {
    // ========== Я ИСТОЧНИК ==========
    path_history.push_back(ring_rank);

    if (ring_rank == target) {
      // Если источник и есть цель, задача выполнена сразу
      GetOutput() = path_history;
    } else {
      // Отправляем данные следующему процессу в кольце
      // Используем коммуникатор ring_comm для передачи
      int path_size = static_cast<int>(path_history.size());

      // Отправляем размер пути
      MPI_Send(&path_size, 1, MPI_INT, next_rank, 0, ring_comm);

      // Отправляем сам путь
      MPI_Send(path_history.data(), path_size, MPI_INT, next_rank, 1, ring_comm);

      // Отправляем данные (если нужно)
      MPI_Send(&input.data, 1, MPI_INT, next_rank, 2, ring_comm);
    }
  } else if (is_participant) {
    // ========== Я ПРОМЕЖУТОЧНЫЙ УЗЕЛ ИЛИ ПОЛУЧАТЕЛЬ ==========

    // Принимаем данные от предыдущего узла в кольце
    int path_size = 0;
    MPI_Status status;

    // Принимаем размер пути
    MPI_Recv(&path_size, 1, MPI_INT, prev_rank, 0, ring_comm, &status);

    // Принимаем путь
    path_history.resize(path_size);
    MPI_Recv(path_history.data(), path_size, MPI_INT, prev_rank, 1, ring_comm, &status);

    // Принимаем данные
    int received_data = 0;
    MPI_Recv(&received_data, 1, MPI_INT, prev_rank, 2, ring_comm, &status);

    // Добавляем себя в историю пути
    path_history.push_back(ring_rank);

    if (ring_rank == target) {
      // ========== Я ЦЕЛЬ ==========
      // Сохраняем итоговый результат с полным путем передачи
      GetOutput() = path_history;
    } else {
      // ========== ПЕРЕДАЮ ДАЛЬШЕ ПО КОЛЬЦУ ==========
      path_size = static_cast<int>(path_history.size());

      // Отправляем размер пути следующему процессу
      MPI_Send(&path_size, 1, MPI_INT, next_rank, 0, ring_comm);

      // Отправляем путь следующему процессу
      MPI_Send(path_history.data(), path_size, MPI_INT, next_rank, 1, ring_comm);

      // Отправляем данные следующему процессу
      MPI_Send(&received_data, 1, MPI_INT, next_rank, 2, ring_comm);
    }
  }

  // Синхронизация всех процессов перед завершением
  MPI_Barrier(ring_comm);

  // Освобождаем созданный коммуникатор
  MPI_Comm_free(&ring_comm);

  return true;
}

bool BorunovVRingMPI::PostProcessingImpl() {
  return true;
}

}  // namespace borunov_v_ring
