import os

# Suppress TensorFlow and C++ logging noise before any TF-related imports.
# These environment variables are read when TensorFlow's native libraries load,
# so they must be set at package import time.
defaults = {
    "TF_CPP_MIN_LOG_LEVEL": "3",  # 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR
    "ABSL_MIN_LOG_LEVEL": "2",  # Suppress absl INFO/WARNING before init
    "TF_ENABLE_ONEDNN_OPTS": "0",  # Disable oneDNN custom ops warning
    "GRPC_VERBOSITY": "ERROR",  # Suppress gRPC logs
    "GLOG_minloglevel": "2",  # Suppress glog INFO messages
}
for key, value in defaults.items():
    if not os.environ.get(key):
        os.environ[key] = value
