#include <gtest/gtest.h>

#include <numeric>
#include <random>
#include <vector>

#include "borunov_v_block_partitioning/common/include/common.hpp"
#include "borunov_v_block_partitioning/mpi/include/ops_mpi.hpp"
#include "borunov_v_block_partitioning/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace borunov_v_block_partitioning {

class BorunovVBlockPartitioningPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    int width = 4000;
    int height = 4000;

    input_data_.resize(2 + width * height);
    input_data_[0] = width;
    input_data_[1] = height;

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 255);
    for (int i = 0; i < width * height; ++i) {
      input_data_[2 + i] = dist(gen);
    }

    output_data_.resize(width * height);
  }

  void SetPerfAttributes(ppc::performance::PerfAttr &perf_attrs) override {
    ppc::util::BaseRunPerfTests<InType, OutType>::SetPerfAttributes(perf_attrs);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.size() == static_cast<size_t>(input_data_[0] * input_data_[1]);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType output_data_;
};

TEST_P(BorunovVBlockPartitioningPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BorunovVBlockPartitioningMPI, BorunovVBlockPartitioningSEQ>(
        PPC_SETTINGS_borunov_v_block_partitioning);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BorunovVBlockPartitioningPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BorunovVBlockPartitioningPerfTest, kGtestValues, kPerfTestName);

}  // namespace borunov_v_block_partitioning
