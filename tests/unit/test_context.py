"""Context budgeting: trim_to_budget keeps recent turns, prunes the rest."""

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from app.agent.context import trim_to_budget


def _history() -> list:
    return [
        HumanMessage("a " * 100, id="1"),
        AIMessage("b " * 100, id="2"),
        HumanMessage("c " * 100, id="3"),
        AIMessage("d " * 100, id="4"),
        HumanMessage("recent question", id="5"),
    ]


def test_under_budget_keeps_everything():
    msgs = _history()
    kept, to_remove = trim_to_budget(msgs, max_tokens=100_000)
    assert [m.id for m in kept] == ["1", "2", "3", "4", "5"]
    assert to_remove == []


def test_over_budget_drops_oldest():
    msgs = _history()
    kept, to_remove = trim_to_budget(msgs, max_tokens=20)
    # the most recent message survives; older ones are pruned
    assert kept[-1].id == "5"
    assert len(kept) < len(msgs)
    removed_ids = {r.id for r in to_remove}
    assert all(isinstance(r, RemoveMessage) for r in to_remove)
    assert removed_ids == {m.id for m in msgs} - {m.id for m in kept}


def test_kept_window_starts_on_human():
    msgs = _history()
    kept, _ = trim_to_budget(msgs, max_tokens=60)
    assert kept[0].type == "human"  # never start mid-turn (e.g. on a tool/ai reply)


def test_empty_history():
    kept, to_remove = trim_to_budget([], max_tokens=100)
    assert kept == []
    assert to_remove == []


def test_removed_plus_kept_covers_all():
    msgs = _history()
    kept, to_remove = trim_to_budget(msgs, max_tokens=30)
    assert len(kept) + len(to_remove) == len(msgs)
