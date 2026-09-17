import polars as pl

from candgen.core.submission import answer_frame, read_answer, validate_answer

ITEMS = [f"{i:016x}" for i in range(60)]
QUERIES = ["70DfDUpwjxB4lzFd", "0000000000000001"]


def test_valid_answer_round_trips_through_csv(tmp_path):
    answer = answer_frame(QUERIES, [ITEMS[:50], [ITEMS[55]]])
    assert validate_answer(answer, QUERIES, ITEMS) == []
    path = tmp_path / "answer.csv"
    answer.write_csv(path)
    assert path.read_text().splitlines()[0] == "query_id,answer"
    written = read_answer(path)
    assert written.equals(answer)
    assert written["answer"][1] == "0000000000000037"


def test_answer_frame_keeps_first_fifty():
    answer = answer_frame(QUERIES[:1], [ITEMS])
    assert answer["answer"][0].split(" ") == ITEMS[:50]


def test_validator_reports_each_violation():
    answer = pl.DataFrame(
        {
            "query_id": [QUERIES[0], QUERIES[0], "short"],
            "answer": [ITEMS[0] + " " + ITEMS[0], "ffffffffffffffff", ""],
        }
    )
    errors = validate_answer(answer, QUERIES, ITEMS)
    joined = "\n".join(errors)
    assert "duplicate query_id" in joined
    assert "1 missing, 1 extra" in joined
    assert "length != 16" in joined
    assert "duplicate items" in joined
    assert "outside corpus" in joined
    assert "empty answer" in joined


def test_validator_rejects_extra_columns():
    answer = answer_frame(QUERIES, [ITEMS[:1], ITEMS[1:2]]).with_row_index()
    assert validate_answer(answer, QUERIES, ITEMS)
