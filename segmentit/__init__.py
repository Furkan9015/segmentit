"""SegmentIt: Nanopore signal segmentation package."""

__version__ = "0.1.0"

from segmentit.data.fast5_reader import Fast5Reader
from segmentit.data.label_reader import LabelReader
from segmentit.data.dataset import NanoporeSegmentationDataset
from segmentit.models.segmentation_model import SegmentationModel, EfficientUNet, ResidualBlock
from segmentit.models.inference import SegmentationInference
from segmentit.models.trainer import Trainer
from segmentit.utils.metrics import compute_metrics, compute_tolerance_metrics
from segmentit.utils.signal_processing import normalize_signal, median_filter, detect_peaks
from segmentit.utils.losses import FocalLoss, DiceLoss, WeightedBCELoss, CombinedSegmentationLoss