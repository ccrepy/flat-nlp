#include "third_party/flat_nlp/lib/fftw_distance.h"

#include <algorithm>
#include <array>

#include "testing/base/public/benchmark.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/types/span.h"

namespace flat {

// Utility function to get a random initialized strided NdSignal (x in [-1;1[ )
std::vector<double> GenerateRandomNdSignal(int signal_dim, int signal_len) {
  std::vector<double> NdSignal(signal_dim * signal_len);
  std::generate_n(NdSignal.begin(), NdSignal.size(),
                  []() { return ((double)rand() / RAND_MAX - 0.5) * 2; });
  return NdSignal;
}

class FftwNlspDistanceTest : public ::testing::Test {
 protected:
  FftwNlspDistanceTest() {}
};

TEST_F(FftwNlspDistanceTest, NlspDistanceSingleStride) {
  StridedFlatDistanceFn distance = StridedFlatDistanceFn(1, 8);

  std::array<double, 8> s1{1, 2, 0, 0, 0, 0, 0, 0};
  std::array<double, 8> s2{0, 0, 1, 0, 0, 0, 0, 0};

  EXPECT_THAT(distance.Call(&s1, &s1), 0.);

  // We verify the value is equal to np_distance_test.py implementation.
  EXPECT_THAT(distance.Call(&s1, &s2), testing::DoubleNear(0.447213, 1e-5));
}

TEST_F(FftwNlspDistanceTest, NlspDistance) {
  StridedFlatDistanceFn distance = StridedFlatDistanceFn(2, 4);

  std::array<double, 2 * 4> s1{1, 0, 0, 0, 0, 1, 0, 0};
  std::array<double, 2 * 4> s2{1, 2, 3, 0, 2, 3, 4, 0};

  EXPECT_THAT(distance.Call(&s1, &s1), 0.);

  // We verify the value is equal to np_distance_test.py implementation.
  EXPECT_THAT(distance.Call(&s1, &s2), testing::DoubleNear(0.769496, 1e-5));
}

void BM_NlspDistance(benchmark::State& state) {
  const int signal_dim = state.range(0);
  const int signal_len = state.range(1);

  StridedFlatDistanceFn distance =
      StridedFlatDistanceFn(signal_dim, signal_len);

  std::vector<double> s1(GenerateRandomNdSignal(signal_dim, signal_len));
  std::vector<double> s2(GenerateRandomNdSignal(signal_dim, signal_len));

  for (auto s : state) {
    distance.Call(&s1, &s2);
  }
}
BENCHMARK(BM_NlspDistance)
    ->ArgPair(2, 4)
    ->ArgPair(16, 8)
    ->ArgPair(32, 32)
    ->ArgPair(32, 64);

TEST_F(FftwNlspDistanceTest, HalfComplexNlspDistanceSingleStride) {
  StridedHalfComplexFlatDistanceFn distance =
      StridedHalfComplexFlatDistanceFn(1, 8);

  std::array<double, 8> hc_s1{3, 2.4142, 1, -0.4142, -1, -1.4142, -2., -1.4142};
  std::array<double, 8> hc_s2{1., 0., -1., 0., 1, 1., -0., -1.};

  EXPECT_THAT(distance.Call(&hc_s1, &hc_s1), 0.);

  // We verify the value is equal to np_distance_test.py implementation.
  EXPECT_THAT(distance.Call(&hc_s1, &hc_s2),
              testing::DoubleNear(0.447213, 1e-5));
}

TEST_F(FftwNlspDistanceTest, HalfComplexNlspDistance) {
  StridedHalfComplexFlatDistanceFn distance =
      StridedHalfComplexFlatDistanceFn(2, 4);

  std::array<double, 8> hc_s1{1., 1., 1., 0., 1., 0., -1., -1.};
  std::array<double, 8> hc_s2{6., -2., 2., -2., 9., -2., 3., -3.};

  EXPECT_THAT(distance.Call(&hc_s1, &hc_s1), 0.);

  // We verify the value is equal to np_distance_test.py implementation.
  EXPECT_THAT(distance.Call(&hc_s1, &hc_s2),
              testing::DoubleNear(0.769496, 1e-5));
}

void BM_HalfComplexNlspDistance(benchmark::State& state) {
  const int signal_dim = state.range(0);
  const int signal_len = state.range(1);

  StridedHalfComplexFlatDistanceFn distance =
      StridedHalfComplexFlatDistanceFn(signal_dim, signal_len);

  std::vector<double> hc_s1(GenerateRandomNdSignal(signal_dim, signal_len));
  std::vector<double> hc_s2(GenerateRandomNdSignal(signal_dim, signal_len));

  for (auto s : state) {
    distance.Call(&hc_s1, &hc_s2);
  }
}
BENCHMARK(BM_HalfComplexNlspDistance)
    ->ArgPair(2, 4)
    ->ArgPair(16, 8)
    ->ArgPair(32, 32)
    ->ArgPair(32, 64);

}  // namespace flat
