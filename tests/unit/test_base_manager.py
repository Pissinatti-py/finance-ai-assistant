"""BaseManager CRUD coverage via SemanticCacheManager (concrete subclass)."""

import pytest

from app.config import settings
from app.db.managers.cache_manager import semantic_cache_manager as mgr
from app.schemas.cache import SemanticCacheCreate

FAKE_VEC = [0.1] * settings.embedding_dim


def _obj(text: str = "q", vec: list[float] | None = None) -> SemanticCacheCreate:
    return SemanticCacheCreate(
        question_text=text,
        question_embedding=vec or FAKE_VEC,
        response_text=f"answer:{text}",
        sources=[],
    )


async def test_create_returns_persisted_instance(db):
    obj = await mgr.create(db, _obj("create_test"))
    assert obj.id is not None
    assert obj.question_text == "create_test"
    assert obj.response_text == "answer:create_test"


async def test_get_by_id(db):
    obj = await mgr.create(db, _obj("get_test"))
    fetched = await mgr.get(db, obj.id)
    assert fetched is not None
    assert fetched.id == obj.id


async def test_get_missing_returns_none(db):
    assert await mgr.get(db, 99999) is None


async def test_get_by_field(db):
    await mgr.create(db, _obj("field_target"))
    found = await mgr.get_by_field(db, "question_text", "field_target")
    assert found is not None
    assert found.question_text == "field_target"


async def test_get_by_field_invalid_raises(db):
    with pytest.raises(AttributeError):
        await mgr.get_by_field(db, "nonexistent", "x")


async def test_get_multi_returns_all(db):
    for i in range(3):
        await mgr.create(db, _obj(f"multi_{i}"))
    items = await mgr.get_multi(db, limit=100)
    assert len(items) == 3


async def test_get_multi_filter(db):
    await mgr.create(db, _obj("target"))
    await mgr.create(db, _obj("other"))
    items = await mgr.get_multi(db, filters={"question_text": "target"})
    assert len(items) == 1
    assert items[0].question_text == "target"


async def test_get_multi_order_asc(db):
    await mgr.create(db, _obj("zzz"))
    await mgr.create(db, _obj("aaa"))
    items = await mgr.get_multi(db, order_by="question_text")
    assert items[0].question_text == "aaa"
    assert items[-1].question_text == "zzz"


async def test_get_multi_order_desc(db):
    await mgr.create(db, _obj("zzz"))
    await mgr.create(db, _obj("aaa"))
    items = await mgr.get_multi(db, order_by="-question_text")
    assert items[0].question_text == "zzz"


async def test_get_multi_pagination(db):
    for i in range(5):
        await mgr.create(db, _obj(f"page_{i}"))
    page1 = await mgr.get_multi(db, skip=0, limit=2)
    page2 = await mgr.get_multi(db, skip=2, limit=2)
    assert len(page1) == 2
    assert len(page2) == 2
    assert {o.id for o in page1}.isdisjoint({o.id for o in page2})


async def test_count(db):
    for i in range(4):
        await mgr.create(db, _obj(f"count_{i}"))
    assert await mgr.count(db) == 4


async def test_count_with_filter(db):
    await mgr.create(db, _obj("counted"))
    await mgr.create(db, _obj("not_counted"))
    total = await mgr.count(db, filters={"question_text": "counted"})
    assert total == 1


async def test_paginate(db):
    for i in range(7):
        await mgr.create(db, _obj(f"pg_{i}"))
    result = await mgr.paginate(db, page=2, per_page=3)
    assert result.total == 7
    assert result.page == 2
    assert result.per_page == 3
    assert result.num_pages == 3
    assert len(result.items) == 3


async def test_paginate_last_page(db):
    for i in range(5):
        await mgr.create(db, _obj(f"lp_{i}"))
    result = await mgr.paginate(db, page=3, per_page=2)
    assert len(result.items) == 1


async def test_update_by_id(db):
    obj = await mgr.create(db, _obj("before"))
    updated = await mgr.update(db, obj.id, {"response_text": "after"})
    assert updated is not None
    assert updated.response_text == "after"


async def test_update_missing_returns_none(db):
    result = await mgr.update(db, 99999, {"response_text": "x"})
    assert result is None


async def test_update_instance(db):
    obj = await mgr.create(db, _obj("inst"))
    result = await mgr.update_instance(db, obj, {"response_text": "changed"})
    assert result.response_text == "changed"


async def test_delete_by_id(db):
    obj = await mgr.create(db, _obj("to_delete"))
    assert await mgr.delete(db, obj.id) is True
    assert await mgr.get(db, obj.id) is None


async def test_delete_missing_returns_false(db):
    assert await mgr.delete(db, 99999) is False


async def test_delete_by_field(db):
    await mgr.create(db, _obj("del_field"))
    n = await mgr.delete_by_field(db, "question_text", "del_field")
    assert n == 1
    assert await mgr.get_by_field(db, "question_text", "del_field") is None


async def test_delete_bulk(db):
    for i in range(3):
        await mgr.create(db, _obj(f"bulk_del_{i}"))
    # delete two of the three
    await mgr.create(db, _obj("keeper"))
    n = await mgr.delete_bulk(db, filters={"response_text": "answer:bulk_del_0"})
    assert n == 1
    assert await mgr.count(db) == 3  # 2 bulk_del + keeper


async def test_exists(db):
    obj = await mgr.create(db, _obj("exists"))
    assert await mgr.exists(db, obj.id) is True
    assert await mgr.exists(db, 99999) is False


async def test_exists_by_field(db):
    await mgr.create(db, _obj("exists_field"))
    assert await mgr.exists_by_field(db, "question_text", "exists_field") is True
    assert await mgr.exists_by_field(db, "question_text", "ghost") is False
