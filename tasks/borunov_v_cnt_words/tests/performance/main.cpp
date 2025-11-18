#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "borunov_v_cnt_words/common/include/common.hpp"
#include "borunov_v_cnt_words/mpi/include/ops_mpi.hpp"
#include "borunov_v_cnt_words/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace borunov_v_cnt_words {

class BorunovVCntWordsPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    const size_t kNumWords = 10000000;
    const std::string kWord = "word ";

    input_data_.reserve(kNumWords * kWord.length());

    for (size_t i = 0; i < kNumWords; ++i) {
      input_data_ += kWord;
    }

    expected_count_ = kNumWords;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_count_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_count_ = 0;
};

TEST_P(BorunovVCntWordsPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BorunovVCntWordsMPI, BorunovVCntWordsSEQ>(PPC_SETTINGS_borunov_v_cnt_words);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BorunovVCntWordsPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BorunovVCntWordsPerfTests, kGtestValues, kPerfTestName);

}  // namespace borunov_v_cnt_words
