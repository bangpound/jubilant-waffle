import subprocess
import unittest
from pathlib import Path
from unittest import mock

from transcribe.audio import FFmpegError, extract_wav


class ExtractWavTests(unittest.TestCase):
    def _run_ok(self, *args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=b"", stderr=b"")

    def test_invokes_ffmpeg_with_16k_mono_wav_args(self):
        with mock.patch("transcribe.audio.subprocess.run", side_effect=self._run_ok) as run:
            out = extract_wav(Path("input.mp3"), Path("out.wav"))
        self.assertEqual(out, Path("out.wav"))
        run.assert_called_once()
        argv = run.call_args.args[0]
        self.assertEqual(argv[0], "ffmpeg")
        self.assertIn("-i", argv)
        self.assertEqual(argv[argv.index("-i") + 1], "input.mp3")
        self.assertIn("-ar", argv)
        self.assertEqual(argv[argv.index("-ar") + 1], "16000")
        self.assertIn("-ac", argv)
        self.assertEqual(argv[argv.index("-ac") + 1], "1")
        self.assertIn("-y", argv)
        self.assertEqual(argv[-1], "out.wav")

    def test_works_for_mp4(self):
        with mock.patch("transcribe.audio.subprocess.run", side_effect=self._run_ok) as run:
            extract_wav(Path("clip.mp4"), Path("clip.wav"))
        argv = run.call_args.args[0]
        self.assertEqual(argv[argv.index("-i") + 1], "clip.mp4")

    def test_raises_ffmpeg_error_on_nonzero_exit(self):
        failing = subprocess.CompletedProcess(
            args=["ffmpeg"], returncode=1, stdout=b"", stderr=b"boom"
        )
        with mock.patch("transcribe.audio.subprocess.run", return_value=failing):
            with self.assertRaises(FFmpegError) as ctx:
                extract_wav(Path("x.mp3"), Path("x.wav"))
        self.assertIn("boom", str(ctx.exception))

    def test_raises_ffmpeg_error_when_binary_missing(self):
        with mock.patch(
            "transcribe.audio.subprocess.run",
            side_effect=FileNotFoundError("no ffmpeg"),
        ):
            with self.assertRaises(FFmpegError):
                extract_wav(Path("x.mp3"), Path("x.wav"))
