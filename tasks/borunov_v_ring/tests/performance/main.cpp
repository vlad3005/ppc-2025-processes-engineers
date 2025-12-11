#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <vector>

#include "borunov_v_ring/common/include/common.hpp"
#include "borunov_v_ring/mpi/include/ops_mpi.hpp"
#include "borunov_v_ring/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace borunov_v_ring {

// Эталонная функция расчета пути (взята из функциональных тестов)
OutType CalculateExpectedPath(const InType &input, int size) {
  std::vector<int> path_history;
  int current_rank = input.source_rank;
  int target_rank = input.target_rank;

  if (size <= 0) {
    return path_history;
  }

  int max_steps = size;
  int steps = 0;

  while (current_rank != target_rank && steps < max_steps) {
    path_history.push_back(current_rank);
    current_rank = (current_rank + 1) % size;
    steps++;
  }

  if (steps < max_steps || current_rank == target_rank) {
    path_history.push_back(current_rank);
  }

  return path_history;
}

class BorunovVRingPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  InType input_data_{};

 protected:
  void SetUp() override {
    // Устанавливаем параметры в точности как в тесте ZeroToLast_0_to_3
    // Значение: 50, Источник: 0, Цель: 3
    input_data_ = RingTaskData{50, 0, 3};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Нормализация цели (как в функциональных тестах)
    int normalized_target = input_data_.target_rank;
    if (size > 0) {
      normalized_target = normalized_target % size;
    }

    // Если вектор пустой, этот ранг НЕ должен быть целевым
    if (output_data.empty()) {
      return rank != normalized_target;
    }

    // Нормализация источника
    int normalized_source = input_data_.source_rank;
    if (size > 0) {
      normalized_source = normalized_source % size;
    }

    RingTaskData normalized_input = input_data_;
    normalized_input.source_rank = normalized_source;
    normalized_input.target_rank = normalized_target;

    // Расчет эталонного пути
    OutType expected_path = CalculateExpectedPath(normalized_input, size);

    // Сверка размера пути
    if (output_data.size() != expected_path.size()) {
      return false;
    }

    // Поэлементная сверка пути
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected_path[i]) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(BorunovVRingPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BorunovVRingMPI, BorunovVRingSEQ>(PPC_SETTINGS_borunov_v_ring);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BorunovVRingPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BorunovVRingPerfTest, kGtestValues, kPerfTestName);

}  // namespace borunov_v_ring
