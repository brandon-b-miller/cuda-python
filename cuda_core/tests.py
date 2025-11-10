from cuda.core.experimental import Device
from cuda.core.experimental._stream import test_sink_python_stream

dev = Device()
dev.set_current()

# Test creating a python owned cuda.core.Stream, passing it to
# c++ as a reference
st = dev.create_stream()

test_sink_python_stream(st)



