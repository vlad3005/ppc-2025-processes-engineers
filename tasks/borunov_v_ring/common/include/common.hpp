#pragma once

#include <vector>

#include "task/include/task.hpp"

namespace borunov_v_ring {

struct RingTaskData {
  int data;         // Данные для передачи
  int source_rank;  // Кто начинает передачу
  int target_rank;  // Кто должен получить данные

  // Добавляем конструктор, чтобы избежать подсказок о designated-initializers
  RingTaskData() = default;
  RingTaskData(int d, int s, int t) : data(d), source_rank(s), target_rank(t) {}
};

using InType = RingTaskData;
using OutType = std::vector<int>;  // Результат: список ранков (путь), через которые прошли данные
using TestType = int;              // Используется для тестов (можно игнорировать или адаптировать)
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace borunov_v_ring
