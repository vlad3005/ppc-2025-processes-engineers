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
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace borunov_v_ring {

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
    int rank = 0;
    int size = 0;
    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);
    } else {
      rank = 0;
      size = ppc::util::GetNumProc();
    }

    int normalized_target = input_data_.target_rank;
    if (size > 0) {
      normalized_target = normalized_target % size;
    }

    if (output_data.empty()) {
      return rank != normalized_target;
    }

    int normalized_source = input_data_.source_rank;
    if (size > 0) {
      normalized_source = normalized_source % size;
    }

    RingTaskData normalized_input = input_data_;
    normalized_input.source_rank = normalized_source;
    normalized_input.target_rank = normalized_target;

    OutType expected_path = CalculateExpectedPath(normalized_input, size);

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

  InType GetTestInputData() override {
    return input_data_;
  }

 protected:
  void SetUp() override {
    const auto &full_params = GetParam();

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
