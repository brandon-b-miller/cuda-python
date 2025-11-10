from cuda.core.experimental import Device
from cuda.core.experimental._stream import test_sink_python_stream, set_cuda_stream_from_core_stream, py_construct_global_stream_tester, test_get_stream_from_cpp, make_core_stream_from_cuda_stream


# 1. Create a python stream and test if a c++ API can use a stream::ref made from it
dev = Device()
dev.set_current()
py_construct_global_stream_tester()

# Test creating a python owned cuda.core.Stream, passing it to
# c++ as a reference
st = dev.create_stream()
orig_handle = int(st.handle)

valid = test_sink_python_stream(st)
assert valid # valled is_valid_stream 

# 2. Release ownership of the stream into a persisten c++ entity
set_cuda_stream_from_core_stream(st)
assert int(st.handle) == 0

# The c++ entity now owns the stream. Users of the original python
# stream object going forward will encounter an invalid stream error 
# from the runtime if used.

# 3. Wrap a reference to the c++ owned stream in a nonowning python stream
st2 = test_get_stream_from_cpp()


# 4. obtain ownership of a stream fully back from c++ into python
st3 = make_core_stream_from_cuda_stream()
breakpoint()
