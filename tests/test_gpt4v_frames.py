"""Offline unit tests for GPT-4V frame extraction.

Run with ``python -m unittest discover -s tests -p 'test_gpt4v_frames.py'``.
The full source module is loaded with its model/API dependencies stubbed, so
these tests need only NumPy and do not load models or contact an API.
"""

import base64
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "t2v_metrics/models/vqascore_models/gpt4v_model.py"
)


def load_frame_module():
    package_name = "_gpt4v_frame_test_models"
    package = ModuleType(package_name)
    package.__path__ = []
    vqa_model = ModuleType(f"{package_name}.vqa_model")
    vqa_model.VQAScoreModel = object
    cv2 = ModuleType("cv2")
    cv2.CAP_PROP_FRAME_COUNT = object()
    cv2.CAP_PROP_POS_FRAMES = object()
    cv2.VideoCapture = Mock()
    cv2.imencode = Mock(side_effect=lambda extension, frame: (True, frame))
    torch = ModuleType("torch")
    torch.Tensor = object
    openai = ModuleType("openai")
    openai.OpenAI = object
    stubs = {
        package_name: package,
        f"{package_name}.vqa_model": vqa_model,
        "cv2": cv2,
        "torch": torch,
        "openai": openai,
        "tiktoken": ModuleType("tiktoken"),
    }
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.gpt4v_model", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class Capture:
    def __init__(self, reported_count, frame_count):
        self.reported_count = reported_count
        self.frames = [f"frame-{index}".encode() for index in range(frame_count)]
        self.position = 0
        self.seeks = []
        self.reads = 0
        self.released = False

    def get(self, prop):
        return self.reported_count

    def set(self, prop, position):
        self.position = position
        self.seeks.append(position)
        return True

    def read(self):
        self.reads += 1
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def release(self):
        self.released = True


class ExtractFramesTests(unittest.TestCase):
    def setUp(self):
        self.module = load_frame_module()

    def extract(self, capture, count):
        self.module.cv2.VideoCapture.return_value = capture
        frames = self.module.extract_frames("video.mp4", count)
        self.assertTrue(capture.released)
        return [base64.b64decode(frame) for frame in frames]

    def test_unknown_counts_decode_distinct_frames_until_eof(self):
        for reported_count in (0, -1, np.nan, np.inf, -np.inf):
            with self.subTest(reported_count=reported_count):
                capture = Capture(reported_count, 3)
                self.assertEqual(self.extract(capture, 8), capture.frames)
                self.assertEqual(capture.seeks, [])
                self.assertEqual(capture.reads, 4)

    def test_unknown_count_limits_sequential_decoding(self):
        capture = Capture(0, 11)
        self.assertEqual(self.extract(capture, 4), capture.frames[:4])
        self.assertEqual(capture.seeks, [])
        self.assertEqual(capture.reads, 4)

    def test_unknown_count_empty_video_stops_after_first_read(self):
        capture = Capture(0, 0)
        self.assertEqual(self.extract(capture, 8), [])
        self.assertEqual(capture.reads, 1)

    def test_known_short_video_does_not_repeat_frames(self):
        capture = Capture(3, 3)
        self.assertEqual(self.extract(capture, 8), capture.frames)
        self.assertEqual(capture.seeks, [0, 1, 2])

    def test_known_video_samples_uniformly(self):
        capture = Capture(11, 11)
        self.assertEqual(
            self.extract(capture, 4),
            [capture.frames[index] for index in (0, 3, 6, 10)],
        )
        self.assertEqual(capture.seeks, [0, 3, 6, 10])

    def test_known_count_keeps_trying_after_failed_sample(self):
        capture = Capture(11, 11)
        original_read = capture.read

        def read():
            if capture.position == 3:
                return False, None
            return original_read()

        capture.read = read
        self.assertEqual(
            self.extract(capture, 4),
            [capture.frames[index] for index in (0, 6, 10)],
        )

    def test_zero_requested_frames_does_not_decode(self):
        for reported_count in (0, 3):
            with self.subTest(reported_count=reported_count):
                capture = Capture(reported_count, 3)
                self.assertEqual(self.extract(capture, 0), [])
                self.assertEqual(capture.reads, 0)

    def test_capture_released_after_decode_error(self):
        for reported_count in (0, 3):
            with self.subTest(reported_count=reported_count):
                capture = Capture(reported_count, 3)
                capture.read = Mock(side_effect=RuntimeError("decode failed"))
                self.module.cv2.VideoCapture.return_value = capture
                with self.assertRaisesRegex(RuntimeError, "decode failed"):
                    self.module.extract_frames("video.mp4", 3)
                self.assertTrue(capture.released)

    def test_capture_released_after_encode_error(self):
        for reported_count in (0, 3):
            with self.subTest(reported_count=reported_count):
                capture = Capture(reported_count, 3)
                self.module.cv2.VideoCapture.return_value = capture
                self.module.cv2.imencode.side_effect = RuntimeError("encode failed")
                with self.assertRaisesRegex(RuntimeError, "encode failed"):
                    self.module.extract_frames("video.mp4", 3)
                self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
