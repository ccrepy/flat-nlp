#include "third_party/flat_nlp/lib/fftw_distance.h"

#include <algorithm>
#include <vector>

#include "third_party/fftw/api/fftw3.h"
#include "util/math/mathutil.h"

namespace flat {

double ComputeDistanceFromSummedConvolution(
    absl::Span<const double> summed_signal, int signal_dim) {
  double peak_convolution =
      *std::max_element(summed_signal.begin(), summed_signal.end()) /
      signal_dim;
  return sqrt(1. - MathUtil::IPow(std::fmin(1, peak_convolution), 2));
}

StridedFlatDistanceFn::StridedFlatDistanceFn(int signal_dim, int signal_len)
    : signal_dim_(signal_dim), signal_len_(signal_len) {
  this->fft_s1_.reset(fftw_alloc_complex(signal_len));
  this->fft_s2_.reset(fftw_alloc_complex(signal_len));

  this->fft_s_out_.reset(fftw_alloc_complex(signal_len));
  this->s_out_.reset(fftw_alloc_real(signal_len));

  this->fft_plan_ =
      fftw_plan_dft_r2c_1d(signal_len, nullptr, nullptr, FFTW_ESTIMATE);

  this->ifft_plan_ =
      fftw_plan_dft_c2r_1d(signal_len, nullptr, nullptr, FFTW_ESTIMATE);
}

StridedFlatDistanceFn::~StridedFlatDistanceFn() {
  fftw_destroy_plan(this->fft_plan_);
  fftw_destroy_plan(this->ifft_plan_);
}

std::vector<double> StridedFlatDistanceFn::ConvolveAndReduceNdSignal(
    absl::Span<double> s1, absl::Span<double> s2) {
  std::vector<double> reduced_signal(this->signal_len_);

  absl::Span<double> s1_stride(s1.data(), this->signal_len_);
  absl::Span<double> s2_stride(s2.data(), this->signal_len_);

  for (int n = 0; n < this->signal_dim_; n++) {
    double norm_s1 = 0.0, norm_s2 = 0.0;

    // forward transform
    fftw_execute_dft_r2c(this->fft_plan_, s1_stride.data(),
                         this->fft_s1_.get());
    fftw_execute_dft_r2c(this->fft_plan_, s2_stride.data(),
                         this->fft_s2_.get());

    // itemwise complex product in frequency domain, s2 is reversed by conjugate
    for (int ix = 0; ix < this->signal_len_; ix++) {
      // real part: re((a1 + ib1) * conj(a2 + ib2)) = a1 * a2 + b1 * b2
      this->fft_s_out_[ix][0] = this->fft_s1_[ix][0] * this->fft_s2_[ix][0] +
                                this->fft_s1_[ix][1] * this->fft_s2_[ix][1];

      // imag part: imag((a1 + ib1) * conj(a2 + ib2)) = b1 * a2 - a1 * b2
      this->fft_s_out_[ix][1] = this->fft_s1_[ix][1] * this->fft_s2_[ix][0] -
                                this->fft_s1_[ix][0] * this->fft_s2_[ix][1];

      // compute the energy of each signal
      norm_s1 += MathUtil::IPow(s1_stride[ix], 2);
      norm_s2 += MathUtil::IPow(s2_stride[ix], 2);
    }

    // backward transform
    fftw_execute_dft_c2r(this->ifft_plan_, this->fft_s_out_.get(),
                         this->s_out_.get());

    // accumulate into the reduced signal
    const double s_out_norm = sqrt(norm_s1 * norm_s2) * this->signal_len_;
    for (int ix = 0; ix < this->signal_len_; ix++) {
      reduced_signal[ix] += this->s_out_[ix] / s_out_norm;
    }

    // point to next stride
    s1_stride = absl::Span<double>(s1_stride.data() + s1_stride.size(),
                                   this->signal_len_);
    s2_stride = absl::Span<double>(s2_stride.data() + s2_stride.size(),
                                   this->signal_len_);
  }

  return reduced_signal;
}

double StridedFlatDistanceFn::Call(absl::Span<double> s1,
                                   absl::Span<double> s2) {
  std::vector<double> summed_signal(ConvolveAndReduceNdSignal(s1, s2));
  return ComputeDistanceFromSummedConvolution(summed_signal, this->signal_dim_);
}

StridedHalfComplexFlatDistanceFn::StridedHalfComplexFlatDistanceFn(
    int signal_dim, int signal_len)
    : signal_dim_(signal_dim), signal_len_(signal_len) {
  this->hc_s_out_.reset(fftw_alloc_real(signal_len));
  this->s_out_.reset(fftw_alloc_real(signal_len));

  this->ihc_plan_ =
      fftw_plan_r2r_1d(signal_len, nullptr, nullptr, FFTW_HC2R, FFTW_ESTIMATE);
}

StridedHalfComplexFlatDistanceFn::~StridedHalfComplexFlatDistanceFn() {
  fftw_destroy_plan(this->ihc_plan_);
}

std::vector<double> StridedHalfComplexFlatDistanceFn::ConvolveAndReduceNdSignal(
    absl::Span<double> hc_s1, absl::Span<double> hc_s2) {
  int half_len(this->signal_len_ / 2);
  std::vector<double> reduced_signal(this->signal_len_);

  absl::Span<double> hc_s1_stride(hc_s1.data(), this->signal_len_);
  absl::Span<double> hc_s2_stride(hc_s2.data(), this->signal_len_);

  for (int n = 0; n < this->signal_dim_; n++) {
    double norm_s1 = MathUtil::IPow(hc_s1_stride[0], 2) +
                     MathUtil::IPow(hc_s1_stride[half_len], 2);
    double norm_s2 = MathUtil::IPow(hc_s2_stride[0], 2) +
                     MathUtil::IPow(hc_s2_stride[half_len], 2);

    // itemwise complex product in frequency domain, s2 is reversed by conjugate
    this->hc_s_out_[0] = hc_s1_stride[0] * hc_s2_stride[0];
    this->hc_s_out_[half_len] = hc_s1_stride[half_len] * hc_s2_stride[half_len];
    for (int ix_real = 1, ix_imag = this->signal_len_ - 1; ix_real < ix_imag;
         ix_real++, ix_imag--) {
      // real part: re((a1 + ib1) * conj(a2 + ib2)) = a1 * a2 + b1 * b2
      this->hc_s_out_[ix_real] = hc_s1_stride[ix_real] * hc_s2_stride[ix_real] +
                                 hc_s1_stride[ix_imag] * hc_s2_stride[ix_imag];

      // imag part: imag((a1 + ib1) * conj(a2 + ib2)) = b1 * a2 - a1 * b2
      this->hc_s_out_[ix_imag] = hc_s1_stride[ix_imag] * hc_s2_stride[ix_real] -
                                 hc_s1_stride[ix_real] * hc_s2_stride[ix_imag];

      // compute the energy of each signal
      norm_s1 += 2 * (MathUtil::IPow(hc_s1_stride[ix_real], 2) +
                      MathUtil::IPow(hc_s1_stride[ix_imag], 2));
      norm_s2 += 2 * (MathUtil::IPow(hc_s2_stride[ix_real], 2) +
                      MathUtil::IPow(hc_s2_stride[ix_imag], 2));
    }
    norm_s1 /= this->signal_len_;
    norm_s2 /= this->signal_len_;

    // backward transform
    fftw_execute_r2r(this->ihc_plan_, this->hc_s_out_.get(),
                     this->s_out_.get());

    // accumulate into the reduced signal
    const double s_out_norm = sqrt(norm_s1 * norm_s2) * this->signal_len_;
    for (int ix = 0; ix < this->signal_len_; ix++) {
      reduced_signal[ix] += this->s_out_[ix] / s_out_norm;
    }

    // point to next stride
    hc_s1_stride = absl::Span<double>(hc_s1_stride.data() + hc_s1_stride.size(),
                                      this->signal_len_);
    hc_s2_stride = absl::Span<double>(hc_s2_stride.data() + hc_s2_stride.size(),
                                      this->signal_len_);
  }

  return reduced_signal;
}

double StridedHalfComplexFlatDistanceFn::Call(absl::Span<double> hc_s1,
                                              absl::Span<double> hc_s2) {
  std::vector<double> summed_signal(ConvolveAndReduceNdSignal(hc_s1, hc_s2));
  return ComputeDistanceFromSummedConvolution(summed_signal, this->signal_dim_);
}

}  // namespace flat
