#include <cuda/__stream/stream_ref.h> 
#include <cuda/__stream/invalid_stream.h>
#include <cuda/__launch/host_launch.h>
#include <iostream>
#include "cuda/__stream/stream.h"
#include <cuda/__driver/driver_api.h> // for __streamCreateWithPriority
#include <cuda/__device/device_ref.h>  // for device_ref
#include <cuda/__device/all_devices.h>
#include <iostream>
#include <optional>

static cuda::stream* create_stream_helper(cudaStream_t handle) {
    return new cuda::stream(cuda::stream::from_native_handle(handle));
}

inline cuda::stream make_test_stream()
{
    cuda::stream result = cuda::stream{cuda::devices[0]};
    return result;
}

inline bool is_valid_stream(const ::cuda::stream_ref& s) noexcept
{
    return s != ::cuda::invalid_stream_t{};
}

class StreamTester {
public:
  StreamTester() = default;  

  cuda::stream_ref get_stream_ref() noexcept {
    if (!stream_.has_value()) {
      throw std::runtime_error("Stream not initialized");
    }
    return *stream_;
  }

  void set_stream(cuda::stream s) {
    stream_ = std::move(s);
  }

  cuda::stream get_stream() {
    if (!stream_.has_value()) {
      throw std::runtime_error("Stream not initialized");
    }
    return std::move(*stream_);
  }

  StreamTester(const StreamTester&) = delete;
  StreamTester& operator=(const StreamTester&) = delete;

private:
  std::optional<cuda::stream> stream_;
};

static StreamTester* g_tester = nullptr;

extern "C" {

// callable from python - persist a StreamTester instance
// in heap memory.
void construct_global_stream_tester()
{
  if (!g_tester) g_tester = new StreamTester();
}

void destroy_global_stream_tester()
{
  delete g_tester;
  g_tester = nullptr;
}

cuda::stream_ref get_global_stream_ref()
{
  return g_tester->get_stream_ref();
}

void set_global_stream(cuda::stream s) {
  s.print_handle_as_int();
  g_tester->set_stream(std::move(s));
}

cuda::stream get_global_stream() {
  return std::move(g_tester->get_stream());
}

} // extern "C"
