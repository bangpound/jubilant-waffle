import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from transcribe.cli import main
from transcribe.formats import Segment


SEGS = [
    Segment(0.0, 2.0, "hi", "SPEAKER_00"),
    Segment(2.0, 4.0, "hello", "SPEAKER_01"),
]


class CLITests(unittest.TestCase):
    def setUp(self):
        self.transcribe_patch = mock.patch(
            "transcribe.cli.transcribe_audio", return_value=SEGS
        )
        self.transcribe = self.transcribe_patch.start()

    def tearDown(self):
        self.transcribe_patch.stop()

    def test_writes_text_to_stdout_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "x.mp3"
            inp.touch()
            stdout = io.StringIO()
            rc = main([str(inp), "--hf-token", "tok"], stdout=stdout)
        self.assertEqual(rc, 0)
        self.assertIn("[SPEAKER_00] hi", stdout.getvalue())

    def test_format_vtt_writes_webvtt_header(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "x.mp3"
            inp.touch()
            stdout = io.StringIO()
            rc = main([str(inp), "-f", "vtt", "--hf-token", "tok"], stdout=stdout)
        self.assertEqual(rc, 0)
        self.assertTrue(stdout.getvalue().startswith("WEBVTT"))

    def test_output_path_writes_file_and_infers_format(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "x.mp3"
            inp.touch()
            out = Path(td) / "out.vtt"
            rc = main([str(inp), "-o", str(out), "--hf-token", "tok"])
            self.assertEqual(rc, 0)
            body = out.read_text()
            self.assertTrue(body.startswith("WEBVTT"))

    def test_md_extension_yields_markdown(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "x.mp3"
            inp.touch()
            out = Path(td) / "out.md"
            rc = main([str(inp), "-o", str(out), "--hf-token", "tok"])
            self.assertEqual(rc, 0)
            self.assertIn("**SPEAKER_00**", out.read_text())

    def test_explicit_format_overrides_extension(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "x.mp3"
            inp.touch()
            out = Path(td) / "out.vtt"
            rc = main([str(inp), "-o", str(out), "-f", "text", "--hf-token", "tok"])
            self.assertEqual(rc, 0)
            body = out.read_text()
            self.assertFalse(body.startswith("WEBVTT"))
            self.assertIn("[SPEAKER_00] hi", body)

    def test_passes_config_to_transcribe_audio(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "x.mp3"
            inp.touch()
            main([
                str(inp),
                "--hf-token", "tok",
                "--model", "small",
                "--device", "cuda",
                "--compute-type", "float16",
                "--language", "en",
                "--min-speakers", "2",
                "--max-speakers", "3",
            ], stdout=io.StringIO())
        cfg = self.transcribe.call_args.args[1]
        self.assertEqual(cfg.hf_token, "tok")
        self.assertEqual(cfg.model, "small")
        self.assertEqual(cfg.device, "cuda")
        self.assertEqual(cfg.compute_type, "float16")
        self.assertEqual(cfg.language, "en")
        self.assertEqual(cfg.min_speakers, 2)
        self.assertEqual(cfg.max_speakers, 3)

    def test_hf_token_falls_back_to_env(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "x.mp3"
            inp.touch()
            with mock.patch.dict("os.environ", {"HF_TOKEN": "env-tok"}, clear=False):
                main([str(inp)], stdout=io.StringIO())
        cfg = self.transcribe.call_args.args[1]
        self.assertEqual(cfg.hf_token, "env-tok")

    def test_names_mapping_renames_speakers_in_output(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "x.mp3"
            inp.touch()
            stdout = io.StringIO()
            rc = main([
                str(inp),
                "--hf-token", "tok",
                "--names", "SPEAKER_00=Dave,SPEAKER_01=Sarah",
            ], stdout=stdout)
        self.assertEqual(rc, 0)
        out = stdout.getvalue()
        self.assertIn("[Dave] hi", out)
        self.assertIn("[Sarah] hello", out)

    def test_missing_input_returns_nonzero(self):
        stderr = io.StringIO()
        rc = main(["/nonexistent/file.mp3", "--hf-token", "tok"], stderr=stderr)
        self.assertNotEqual(rc, 0)

    def test_unknown_extension_with_no_format_errors(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "x.mp3"
            inp.touch()
            out = Path(td) / "out.xyz"
            stderr = io.StringIO()
            rc = main([str(inp), "-o", str(out), "--hf-token", "tok"], stderr=stderr)
        self.assertNotEqual(rc, 0)
