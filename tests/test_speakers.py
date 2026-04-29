import json
import tempfile
import unittest
from pathlib import Path

from transcribe.formats import Segment
from transcribe.speakers import apply_names, parse_names


class ApplyNamesTests(unittest.TestCase):
    def test_returns_input_unchanged_when_mapping_empty(self):
        segs = [Segment(0.0, 1.0, "hi", "SPEAKER_00")]
        self.assertEqual(apply_names(segs, {}), segs)

    def test_remaps_known_speakers(self):
        segs = [
            Segment(0.0, 1.0, "hi", "SPEAKER_00"),
            Segment(1.0, 2.0, "hello", "SPEAKER_01"),
        ]
        out = apply_names(segs, {"SPEAKER_00": "Dave", "SPEAKER_01": "Sarah"})
        self.assertEqual([s.speaker for s in out], ["Dave", "Sarah"])
        self.assertEqual([s.text for s in out], ["hi", "hello"])

    def test_leaves_unmapped_speakers_unchanged(self):
        segs = [Segment(0.0, 1.0, "hi", "SPEAKER_02")]
        out = apply_names(segs, {"SPEAKER_00": "Dave"})
        self.assertEqual(out[0].speaker, "SPEAKER_02")


class ParseNamesTests(unittest.TestCase):
    def test_none_returns_empty_dict(self):
        self.assertEqual(parse_names(None), {})

    def test_inline_kv_pairs(self):
        self.assertEqual(
            parse_names("SPEAKER_00=Dave,SPEAKER_01=Sarah"),
            {"SPEAKER_00": "Dave", "SPEAKER_01": "Sarah"},
        )

    def test_inline_strips_whitespace(self):
        self.assertEqual(
            parse_names(" SPEAKER_00 = Dave , SPEAKER_01 = Sarah "),
            {"SPEAKER_00": "Dave", "SPEAKER_01": "Sarah"},
        )

    def test_invalid_inline_raises(self):
        with self.assertRaises(ValueError):
            parse_names("not-a-pair")

    def test_json_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "names.json"
            p.write_text(json.dumps({"SPEAKER_00": "Dave"}))
            self.assertEqual(parse_names(str(p)), {"SPEAKER_00": "Dave"})

    def test_json_file_must_be_object(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "names.json"
            p.write_text("[1,2,3]")
            with self.assertRaises(ValueError):
                parse_names(str(p))
