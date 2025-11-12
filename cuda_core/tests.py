from cuda.core.experimental import Device
import numpy as np
from cuda.core.experimental._stream import (
    py_construct_global_stream_tester, 
    set_cpp_stream_from_core_stream,
    test_python_stream_to_cuda_stream_ref,
    get_core_stream_from_cpp_streamref,
    get_core_stream_from_cpp_stream,
)

# instantiate a StreamTester, a c++ entity that stores a cuda::stream internally
# instance lives on the heap and persists until containing cython extension unloads 
py_construct_global_stream_tester()

# 1. Create a python stream and test if a c++ API can use a stream::ref made from it
dev = Device()
dev.set_current()
st = dev.create_stream()
orig_handle = np.uint64(st.handle)

# Test creating a python owned cuda.core.Stream, passing it to
# c++ as a reference

valid = test_python_stream_to_cuda_stream_ref(st)
assert valid # valled is_valid_stream 

# 2. Release ownership of the stream into a persisten c++ entity
set_cpp_stream_from_core_stream(st)
#assert int(st.handle) == 0
del st

# The c++ entity now owns the stream. Users of the original python
# stream object going forward will encounter an invalid stream error 
# from the runtime if used.

# 3. Wrap a reference to the c++ owned stream in a nonowning python stream
nonowning_stream = get_core_stream_from_cpp_streamref()
del nonowning_stream

# 4. obtain ownership of a stream fully back from c++ into python
final_stream = get_core_stream_from_cpp_stream()

assert int(final_stream.handle) == orig_handle
print("passed!")
