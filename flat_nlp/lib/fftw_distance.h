#ifndef THIRD_PARTY_FLAT_NLP_LIB_FFTW_DISTANCE_H_
#define THIRD_PARTY_FLAT_NLP_LIB_FFTW_DISTANCE_H_

#include <vector>

#include "third_party/absl/types/span.h"
#include "third_party/fftw/api/fftw3.h"

// This library implement distances used by Flat NLP and is intended for fast
// execution, typically as as clif interface for python.
//
// It defines similar distances as np_distance.py but using FFTW implementation
// for fast convolution instead of numpy.
//
// The classes defined below are suffixed with 'Fn' as they are effectively
// parameterized functions.
namespace flat {

// Function object to make 'fftw_free' compatible with smartpointers delete.
struct FftwDeleter {
  inline void operator()(void* ptr) const { fftw_free(ptr); }
};

// Wrapper class holding the FFTW buffers and the flat distance implementation
// for inputs signals defined in the time domain.
// This class is thread-unsafe. Typically each thread or fiber should have a
// separate instance.
class StridedFlatDistanceFn {
 public:
  StridedFlatDistanceFn(int signal_dim, int signal_len);
  ~StridedFlatDistanceFn();

  // This function is equivalent to np_distance.strided_flat_distance(...).
  // s1 and s2 are expected to be defined in time domain and are strided
  // representations of 2 NdSignals (row major order).
  // Their lengths must be equal to signal_dim * signal_len.
  double Call(absl::Span<double> s1, absl::Span<double> s2);

 private:
  std::vector<double> ConvolveAndReduceNdSignal(absl::Span<double> s1,
                                                absl::Span<double> s2);

  const int signal_dim_;
  const int signal_len_;

  // Members below are used internally by fftw and are not thread safe.
  std::unique_ptr<fftw_complex[], FftwDeleter> fft_s1_;
  std::unique_ptr<fftw_complex[], FftwDeleter> fft_s2_;

  std::unique_ptr<fftw_complex[], FftwDeleter> fft_s_out_;
  std::unique_ptr<double[], FftwDeleter> s_out_;

  fftw_plan fft_plan_;
  fftw_plan ifft_plan_;
};

// Wrapper class holding the FFTW buffers and the flat distance implementation
// for inputs signals defined in the frequency domain in fftw halfcomplex
// format.
// This class is thread-unsafe. Typically each thread or fiber should have a
// separate instance.
class StridedHalfComplexFlatDistanceFn {
 public:
  StridedHalfComplexFlatDistanceFn(int signal_dim, int signal_len);
  ~StridedHalfComplexFlatDistanceFn();

  // This function is equivalent to np_distance.strided_hc_flat_distance(...).
  // hc_s1 and hc_s2 are expected to be defined in frequency domain as fftw
  // halfcomplex format and are strided representations of 2 NdSignals (row
  // major order).
  // Their lengths must be equal to signal_dim * signal_len.
  double Call(absl::Span<double> hc_s1, absl::Span<double> hc_s2);

 private:
  std::vector<double> ConvolveAndReduceNdSignal(absl::Span<double> hc_s1,
                                                absl::Span<double> hc_s2);

  const int signal_dim_;
  const int signal_len_;

  // Members below are used internally by fftw and are not thread safe.
  std::unique_ptr<double[], FftwDeleter> hc_s_out_;
  std::unique_ptr<double[], FftwDeleter> s_out_;

  fftw_plan ihc_plan_;
};

}  // namespace flat

#endif  // THIRD_PARTY_FLAT_NLP_LIB_FFTW_DISTANCE_H_
