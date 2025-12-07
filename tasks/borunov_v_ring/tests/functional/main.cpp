#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "borunov_v_ring/common/include/common.hpp"
#include "borunov_v_ring/mpi/include/ops_mpi.hpp"
#include "borunov_v_ring/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/perf_test_util.hpp"  // Для TupleToGTestValues

namespace borunov_v_ring {

// TestType для функциональных тестов: входные данные + имя
using FuncTestType = std::tuple<InType, std::string>;

namespace {
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
}  // namespace

class BorunovVRingFuncTestes : public ppc::util::BaseRunFuncTests<InType, OutType, FuncTestType> {
 public:
  static std::string PrintTestParam(const FuncTestType &test_param) {
    const auto &input_data = std::get<0>(test_param);
    const auto &name = std::get<1>(test_param);
    return "Source_" + std::to_string(input_data.source_rank) + "_Target_" + std::to_string(input_data.target_rank) +
           "_" + name;
  }

  bool CheckTestOutputData(OutType &output_data) override {
    // В MPI реализации только процесс-получатель (target) имеет результат
    // Если output_data пустой, значит этот процесс не является получателем
    // В этом случае проверка должна пройти (процесс не участвовал в получении результата)
    if (output_data.empty()) {
      // Проверяем, должен ли этот процесс иметь результат
      int rank = 0;
      int size = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);

      // Если этот процесс является целевым, он должен иметь результат
      return rank != input_data_.target_rank;
    }

    // Вычисляем ожидаемый путь
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    OutType expected_path = CalculateExpectedPath(input_data_, size);

    // Сравниваем полученный путь с ожидаемым
    if (output_data.size() != expected_path.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected_path[i]) {
        return false;
      }
    }

    return true;
  }

  // InType (RingTaskData) - маленькая структура, обычно возвращается по значению
  InType GetTestInputData() override {
    return input_data_;
  }

 protected:
  void SetUp() override {
    // Используем auto, чтобы не писать сложный тип TaskParamType вручную
    const auto &full_params = GetParam();
    // Пользовательские данные (FuncTestType) находятся третьим элементом
    const auto &user_test_data = std::get<2>(full_params);

    // Извлекаем InType из FuncTestType
    input_data_ = std::get<0>(user_test_data);
  }

 private:
  InType input_data_ = {0, 0, 0};
};

namespace {

TEST_P(BorunovVRingFuncTestes, RingPathTest) {
  ExecuteTest(GetParam());
}

// Тестовые наборы входных данных: {data, source_rank, target_rank}
const std::array<FuncTestType, 6> kRingTestParam = {
    FuncTestType({10, 0, 2}, "ShortPath_0_to_2"),  FuncTestType({20, 1, 1}, "FullCycle_1_to_1"),
    FuncTestType({30, 3, 1}, "WrapAround_3_to_1"), FuncTestType({40, 2, 3}, "Adjacent_2_to_3"),
    FuncTestType({50, 0, 3}, "ZeroToLast_0_to_3"), FuncTestType({60, 3, 0}, "LastToZero_3_to_0")};

const auto kFuncTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<BorunovVRingMPI, InType>(kRingTestParam, PPC_SETTINGS_borunov_v_ring),
                   ppc::util::AddFuncTask<BorunovVRingSEQ, InType>(kRingTestParam, PPC_SETTINGS_borunov_v_ring));

const auto kFuncGtestValues = ppc::util::TupleToGTestValues(kFuncTestTasksList);

// Используем PrintFuncTestName из базового класса, который включает имя задачи (MPI/SEQ)
INSTANTIATE_TEST_SUITE_P(BorunovVRingFuncTestInstantiation, BorunovVRingFuncTestes, kFuncGtestValues,
                         BorunovVRingFuncTestes::PrintFuncTestName<BorunovVRingFuncTestes>);

}  // namespace
}  // namespace borunov_v_ring
