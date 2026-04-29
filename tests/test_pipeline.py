import sys
import unittest
from pathlib import Path
from unittest import mock

from transcribe.formats import Segment
from transcribe.pipeline import TranscriptionConfig, transcribe_audio


def _fake_whisperx(final_segments):
    fake = mock.MagicMock()
    fake.load_audio.return_value = "AUDIO"
    model = mock.MagicMock()
    model.transcribe.return_value = {"segments": [{"start": 0, "end": 1, "text": "x"}], "language": "en"}
    fake.load_model.return_value = model
    fake.load_align_model.return_value = (mock.MagicMock(name="align_model"), {"meta": True})
    fake.align.return_value = {"segments": [{"start": 0, "end": 1, "text": "x"}]}
    diar = mock.MagicMock()
    diar.return_value = "DIAR_DF"
    fake.DiarizationPipeline.return_value = diar
    fake.assign_word_speakers.return_value = {"segments": final_segments}
    return fake


class TranscribeAudioTests(unittest.TestCase):
    def setUp(self):
        self.final_segments = [
            {"start": 0.0, "end": 2.0, "text": "hi", "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "text": "hello", "speaker": "SPEAKER_01"},
        ]
        self.fake = _fake_whisperx(self.final_segments)
        self.modules_patch = mock.patch.dict(sys.modules, {"whisperx": self.fake})
        self.modules_patch.start()
        self.extract_patch = mock.patch(
            "transcribe.pipeline.extract_wav",
            return_value=Path("/tmp/x.wav"),
        )
        self.extract = self.extract_patch.start()

    def tearDown(self):
        self.modules_patch.stop()
        self.extract_patch.stop()

    def test_returns_segments_in_order(self):
        cfg = TranscriptionConfig(hf_token="tok")
        segs = transcribe_audio(Path("/tmp/x.mp3"), cfg)
        self.assertEqual(
            segs,
            [
                Segment(0.0, 2.0, "hi", "SPEAKER_00"),
                Segment(2.0, 4.0, "hello", "SPEAKER_01"),
            ],
        )

    def test_extracts_wav_for_non_wav_inputs(self):
        cfg = TranscriptionConfig(hf_token="tok")
        transcribe_audio(Path("/tmp/x.mp3"), cfg)
        self.extract.assert_called_once()
        called_with = self.extract.call_args.args[0]
        self.assertEqual(called_with, Path("/tmp/x.mp3"))

    def test_skips_extraction_for_wav_inputs(self):
        cfg = TranscriptionConfig(hf_token="tok")
        transcribe_audio(Path("/tmp/already.wav"), cfg)
        self.extract.assert_not_called()
        self.fake.load_audio.assert_called_once_with("/tmp/already.wav")

    def test_passes_model_device_and_compute_type(self):
        cfg = TranscriptionConfig(
            hf_token="tok", model="small", device="cuda", compute_type="float16"
        )
        transcribe_audio(Path("/tmp/x.mp3"), cfg)
        self.fake.load_model.assert_called_once()
        args, kwargs = self.fake.load_model.call_args
        self.assertEqual(args[0], "small")
        self.assertEqual(kwargs.get("device") or args[1], "cuda")
        self.assertEqual(kwargs.get("compute_type"), "float16")

    def test_passes_hf_token_to_diarization_pipeline(self):
        cfg = TranscriptionConfig(hf_token="my-token", device="cpu")
        transcribe_audio(Path("/tmp/x.mp3"), cfg)
        self.fake.DiarizationPipeline.assert_called_once()
        kwargs = self.fake.DiarizationPipeline.call_args.kwargs
        self.assertEqual(kwargs.get("use_auth_token"), "my-token")
        self.assertEqual(kwargs.get("device"), "cpu")

    def test_missing_hf_token_raises(self):
        with self.assertRaises(ValueError):
            transcribe_audio(Path("/tmp/x.mp3"), TranscriptionConfig(hf_token=None))

    def test_segments_missing_speaker_get_unknown_label(self):
        self.fake.assign_word_speakers.return_value = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hi", "speaker": "SPEAKER_00"},
                {"start": 1.0, "end": 2.0, "text": "?"},
            ]
        }
        segs = transcribe_audio(Path("/tmp/x.mp3"), TranscriptionConfig(hf_token="tok"))
        self.assertEqual(segs[1].speaker, "UNKNOWN")

    def test_passes_min_max_speakers_when_set(self):
        cfg = TranscriptionConfig(hf_token="tok", min_speakers=2, max_speakers=4)
        transcribe_audio(Path("/tmp/x.mp3"), cfg)
        diar = self.fake.DiarizationPipeline.return_value
        kwargs = diar.call_args.kwargs
        self.assertEqual(kwargs.get("min_speakers"), 2)
        self.assertEqual(kwargs.get("max_speakers"), 4)
