#include <gtest/gtest.h>

#include "borunov_v_ring/common/include/common.hpp"
#include "borunov_v_ring/mpi/include/ops_mpi.hpp"
#include "borunov_v_ring/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace borunov_v_ring {

class BorunovVRingPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
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
