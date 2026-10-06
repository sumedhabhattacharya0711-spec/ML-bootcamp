from types import SimpleNamespace

from meeting_assistant.hallucination import doubtful_indices, score_segments


def seg(text, start, no_speech_prob=0.05, avg_logprob=-0.3, compression_ratio=1.5):
    """A fake Whisper segment with good confidence unless told otherwise."""
    return SimpleNamespace(text=text, start=start, no_speech_prob=no_speech_prob,
                           avg_logprob=avg_logprob, compression_ratio=compression_ratio)


DURATION = 600.0  # a 10-minute meeting; the last 10% starts at 540 s


def test_youtube_phrase_is_flagged():
    flags = score_segments([seg("Thanks for watching!", 100)], DURATION)
    assert flags[0].flagged
    assert flags[0].score == 3  # contains subtitle phrase (+2) and the whole line is a BoH phrase (+1)
    assert "thanks for watching" in flags[0].reason_text


def test_subtitle_credit_is_flagged():
    flags = score_segments([seg("Subtitles by the Amara.org community", 100)], DURATION)
    assert flags[0].flagged


def test_repeated_thank_you_mid_meeting_is_flagged():
    segments = [seg("We should ship it.", 90)] + [seg("Thank you.", 100 + i) for i in range(3)]
    flags = score_segments(segments, DURATION)
    assert not flags[0].flagged
    for f in flags[1:]:
        assert f.flagged
        assert f.score == 3  # phantom + repeated + mid-meeting
        assert "repeated 3x mid-meeting" in f.reason_text


def test_single_thank_you_at_end_is_not_flagged():
    flags = score_segments([seg("Thank you.", 590)], DURATION)
    assert flags[0].score == 1
    assert not flags[0].flagged


def test_normal_line_is_not_flagged():
    flags = score_segments([seg("We moved the training jobs to Kubeflow.", 100)], DURATION)
    assert flags[0].score == 0
    assert flags[0].reasons == []


def test_two_plain_repeats_mid_meeting_are_not_flagged():
    # "Yeah." is not a phantom phrase and only repeats twice: repeat (+1) only.
    flags = score_segments([seg("Yeah.", 100), seg("Yeah.", 101)], DURATION)
    assert [f.score for f in flags] == [1, 1]
    assert not any(f.flagged for f in flags)


def test_two_thank_yous_mid_meeting_are_flagged():
    flags = score_segments([seg("Thank you.", 100), seg("Thank you.", 101)], DURATION)
    assert [f.score for f in flags] == [3, 3]  # phantom + repeated + mid-meeting


def test_plain_line_repeated_three_times_mid_meeting_is_flagged():
    flags = score_segments([seg("With menus.", 100 + i) for i in range(3)], DURATION)
    assert all(f.flagged and f.score == 2 for f in flags)
    assert "repeated 3x mid-meeting" in flags[0].reason_text


def test_looping_low_confidence_line_is_flagged():
    flags = score_segments(
        [seg("the the the the the the the the", 100, avg_logprob=-1.4, compression_ratio=3.1)],
        DURATION,
    )
    assert flags[0].flagged
    assert "low confidence" in flags[0].reason_text
    assert "looping text" in flags[0].reason_text


def test_provided_by_in_normal_speech_is_not_flagged():
    flags = score_segments([seg("The figures were provided by finance.", 100)], DURATION)
    assert not flags[0].flagged


def test_doubtful_indices():
    segments = [seg("Hello everyone.", 10), seg("Thanks for watching!", 20), seg("Let's start.", 30)]
    assert doubtful_indices(score_segments(segments, DURATION)) == {1}


def test_boh_list_is_loaded():
    from meeting_assistant.hallucination import PHANTOM_PHRASES, load_boh
    boh = load_boh()
    assert len(boh) == 294
    assert "the train is now moving towards the central station" in PHANTOM_PHRASES
    assert "i m sorry" in PHANTOM_PHRASES  # "I'm sorry" in BoH's format


def test_boh_phrase_with_low_confidence_is_flagged():
    flags = score_segments(
        [seg("The train is now moving towards the central station.", 100, no_speech_prob=0.8)],
        DURATION,
    )
    assert flags[0].score == 2  # phantom + low confidence
    assert flags[0].flagged
    assert "known Whisper phantom phrase" in flags[0].reason_text


def test_boh_phrase_only_counts_as_whole_line():
    flags = score_segments([seg("Woof, the dog in the demo video was loud.", 100)], DURATION)
    assert flags[0].score == 0


def test_meeting_backchannels_from_boh_are_not_flagged():
    flags = score_segments([seg("Mm-hmm.", 100), seg("Mm-hmm.", 101)], DURATION)
    assert not any(f.flagged for f in flags)
