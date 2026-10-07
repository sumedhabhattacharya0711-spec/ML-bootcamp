from meeting_assistant.segment import estimate_tokens, gap_similarities, split_segments

REMOTE = ["the remote control buttons need a bigger power button and a volume wheel",
          "the remote should have rubber buttons and a scroll wheel for channels",
          "buttons on the remote control must be easy for older users to press",
          "maybe the remote gets an lcd screen above the buttons and scroll wheel",
          "the power button on the remote control should glow in the dark",
          "fewer buttons on the remote make the scroll wheel easier to find"]
BUDGET = ["the budget is twelve euros fifty per unit for production costs",
          "selling price twenty five euros gives the profit margin finance wants",
          "production costs must stay under budget to hit the profit target",
          "finance says the selling price and margin depend on production volume",
          "the budget leaves little money for an expensive lcd on each unit",
          "profit target is fifteen million euros so production costs matter"]


def test_short_transcript_is_one_part():
    assert split_segments(["hello there", "short meeting"], max_tokens=4000) == [(0, 2)]


def test_parts_cover_every_line_in_order_and_fit_the_budget():
    lines = REMOTE + BUDGET + REMOTE
    budget = 60
    parts = split_segments(lines, budget)
    assert parts[0][0] == 0 and parts[-1][1] == len(lines)
    assert all(a < b for a, b in parts)
    assert all(parts[i][1] == parts[i + 1][0] for i in range(len(parts) - 1))
    assert all(sum(estimate_tokens(t) for t in lines[a:b]) <= budget for a, b in parts)


def test_cut_falls_on_the_topic_change():
    lines = REMOTE + BUDGET
    sims = gap_similarities(lines)
    assert min(range(1, len(lines)), key=lambda g: sims[g]) == len(REMOTE)
    budget = sum(estimate_tokens(t) for t in lines) * 2 // 3   # forces exactly two parts
    assert split_segments(lines, budget) == [(0, len(REMOTE)), (len(REMOTE), len(lines))]


def test_over_long_single_line_becomes_its_own_part():
    lines = ["short line", "word " * 500, "another short line"]
    parts = split_segments(lines, 50)
    assert (1, 2) in parts and parts[-1][1] == 3
