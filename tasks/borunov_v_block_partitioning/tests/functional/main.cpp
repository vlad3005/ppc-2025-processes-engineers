#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <ranges>
#include <string>
#include <tuple>

#include "borunov_v_block_partitioning/common/include/common.hpp"
#include "borunov_v_block_partitioning/mpi/include/ops_mpi.hpp"
#include "borunov_v_block_partitioning/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace borunov_v_block_partitioning {

class BorunovLinearFilterTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const testing::TestParamInfo<typename BorunovLinearFilterTest::ParamType> &info) {
    std::string task_name = std::get<1>(info.param);
    auto test_params = std::get<2>(info.param);

    int width = std::get<0>(test_params);
    int height = std::get<1>(test_params);

    std::string name = task_name + "_" + std::to_string(width) + "x" + std::to_string(height);

    std::ranges::replace(name, ':', '_');
    std::ranges::replace(name, '.', '_');
    std::ranges::replace(name, '/', '_');

    return name;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<2>(GetParam());
    int width = std::get<0>(params);
    int height = std::get<1>(params);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);

    input_data_.resize(2 + width * height);
    input_data_[0] = width;
    input_data_[1] = height;

    for (int i = 0; i < width * height; ++i) {
      input_data_[2 + i] = dist(gen);
    }

    CalculateReferenceOutput(width, height);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == reference_output_;
  }

 private:
  InType input_data_;
  OutType reference_output_;

  void CalculateReferenceOutput(int width, int height) {
    reference_output_.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));

    const std::array<std::array<float, 3>, 3> kernel = {{
        {1.0F / 16.0F, 2.0F / 16.0F, 1.0F / 16.0F},
        {2.0F / 16.0F, 4.0F / 16.0F, 2.0F / 16.0F},
        {1.0F / 16.0F, 2.0F / 16.0F, 1.0F / 16.0F},
    }};

    const int *pixels = input_data_.data() + 2;

    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        float sum = 0.0F;
        for (int ky = -1; ky <= 1; ++ky) {
          for (int kx = -1; kx <= 1; ++kx) {
            int nx = std::clamp(j + kx, 0, width - 1);
            int ny = std::clamp(i + ky, 0, height - 1);
            sum += static_cast<float>(pixels[(ny * width) + nx]) *
                   kernel[static_cast<std::size_t>(ky + 1)][static_cast<std::size_t>(kx + 1)];
          }
        }
        reference_output_[(i * width) + j] = static_cast<int>(std::round(sum));
      }
    }
  }
};

TEST_P(BorunovLinearFilterTest, RunFilter) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParams = {std::make_tuple(10, 10), std::make_tuple(20, 15), std::make_tuple(15, 20),
                                             std::make_tuple(32, 32)};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<BorunovVBlockPartitioningMPI, InType>(
                                               kTestParams, PPC_SETTINGS_borunov_v_block_partitioning),
                                           ppc::util::AddFuncTask<BorunovVBlockPartitioningSEQ, InType>(
                                               kTestParams, PPC_SETTINGS_borunov_v_block_partitioning));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

INSTANTIATE_TEST_SUITE_P(BorunovFilterTests, BorunovLinearFilterTest, kGtestValues,
                         BorunovLinearFilterTest::PrintTestParam);

}  // namespace borunov_v_block_partitioning
