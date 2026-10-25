from .amplitude import AmplitudeDetector
from .cnn import PRESETS as CNN_PRESETS
from .cnn import build_cnn_detector, cnn_macs, cnn_param_count, make_cnn
from .dnn import PRESETS as DNN_PRESETS
from .dnn import build_dnn_detector, make_mlp, mlp_param_count
from .rfc import PRESETS as RFC_PRESETS
from .rfc import RandomForestDetector, build_rfc_detector