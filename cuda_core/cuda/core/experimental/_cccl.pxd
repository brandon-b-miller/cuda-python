cimport cuda.bindings.cyruntime as ccudart
cdef extern from "cuda/stream_ref" namespace "cuda" nogil:
    # stream_ref and the constructor for cudaStream_t
    cdef cppclass stream_ref:
        stream_ref() nogil
        stream_ref(ccudart.cudaStream_t stream_) nogil 
        ccudart.cudaStream_t get() nogil except + 

cdef extern from "cuda/__stream/stream.h" namespace "cuda" nogil:
    cdef cppclass stream:
        stream() nogil except +
        
        @staticmethod
        stream from_native_handle(ccudart.cudaStream_t handle) nogil except +

        ccudart.cudaStream_t release() nogil except +
