import unittest

from transcribe.formats import Segment, to_markdown, to_text, to_vtt


class VTTTests(unittest.TestCase):
    def test_empty_segments_produces_only_header(self):
        self.assertEqual(to_vtt([]), "WEBVTT\n")

    def test_single_segment_uses_voice_tag_and_hms_timestamps(self):
        segments = [Segment(start=0.0, end=5.25, text="Hello world", speaker="SPEAKER_00")]
        expected = (
            "WEBVTT\n"
            "\n"
            "00:00:00.000 --> 00:00:05.250\n"
            "<v SPEAKER_00>Hello world\n"
        )
        self.assertEqual(to_vtt(segments), expected)

    def test_multiple_segments_separated_by_blank_line(self):
        segments = [
            Segment(start=0.0, end=2.0, text="Hi.", speaker="SPEAKER_00"),
            Segment(start=2.0, end=4.5, text="Hello.", speaker="SPEAKER_01"),
        ]
        out = to_vtt(segments)
        self.assertEqual(
            out,
            "WEBVTT\n"
            "\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "<v SPEAKER_00>Hi.\n"
            "\n"
            "00:00:02.000 --> 00:00:04.500\n"
            "<v SPEAKER_01>Hello.\n",
        )

    def test_timestamp_handles_hours_and_subsecond_precision(self):
        seg = Segment(start=3661.001, end=3725.999, text="x", speaker="A")
        out = to_vtt([seg])
        self.assertIn("01:01:01.001 --> 01:02:05.999", out)

    def test_text_is_stripped_of_surrounding_whitespace(self):
        seg = Segment(start=0.0, end=1.0, text="  hello  ", speaker="A")
        self.assertIn("<v A>hello\n", to_vtt([seg]))


class TextTests(unittest.TestCase):
    def test_empty_segments_produces_empty_string(self):
        self.assertEqual(to_text([]), "")

    def test_single_segment_renders_speaker_prefix(self):
        seg = Segment(start=0.0, end=1.0, text="Hello world", speaker="SPEAKER_00")
        self.assertEqual(to_text([seg]), "[SPEAKER_00] Hello world\n")

    def test_consecutive_lines_from_same_speaker_merge(self):
        segments = [
            Segment(start=0.0, end=1.0, text="Hi.", speaker="A"),
            Segment(start=1.0, end=2.0, text="How are you?", speaker="A"),
            Segment(start=2.0, end=3.0, text="Good.", speaker="B"),
        ]
        self.assertEqual(
            to_text(segments),
            "[A] Hi. How are you?\n[B] Good.\n",
        )

    def test_text_is_stripped(self):
        seg = Segment(start=0.0, end=1.0, text="  hello  ", speaker="A")
        self.assertEqual(to_text([seg]), "[A] hello\n")


class MarkdownTests(unittest.TestCase):
    def test_empty_segments_produces_empty_string(self):
        self.assertEqual(to_markdown([]), "")

    def test_single_segment_uses_bold_speaker_and_hms_range(self):
        seg = Segment(start=0.0, end=5.5, text="Hello world", speaker="SPEAKER_00")
        self.assertEqual(
            to_markdown([seg]),
            "**SPEAKER_00** _(00:00:00 – 00:00:05)_\n\nHello world\n",
        )

    def test_multiple_segments_separated_by_blank_lines(self):
        segments = [
            Segment(start=0.0, end=2.0, text="Hi.", speaker="A"),
            Segment(start=2.0, end=4.0, text="Hello.", speaker="B"),
        ]
        self.assertEqual(
            to_markdown(segments),
            "**A** _(00:00:00 – 00:00:02)_\n\nHi.\n"
            "\n"
            "**B** _(00:00:02 – 00:00:04)_\n\nHello.\n",
        )

    def test_consecutive_same_speaker_segments_merge(self):
        segments = [
            Segment(start=0.0, end=2.0, text="Hi.", speaker="A"),
            Segment(start=2.0, end=4.0, text="How are you?", speaker="A"),
        ]
        self.assertEqual(
            to_markdown(segments),
            "**A** _(00:00:00 – 00:00:04)_\n\nHi. How are you?\n",
        )
