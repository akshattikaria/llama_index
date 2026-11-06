from typing import Any

import pytest
from sqlalchemy.orm import declarative_base

from llama_index.core.llms import ChatMessage
from llama_index.storage.chat_store.postgres.base import PostgresChatStore, get_data_model


class _Result:
    def scalar_one_or_none(self) -> list[dict[str, Any]]:
        return [ChatMessage(content="remove me", role="user").model_dump()]


class _Session:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def __enter__(self) -> "_Session":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def execute(self, statement: Any, params: Any = None) -> _Result:
        self.statements.append(str(statement))
        return _Result()

    def commit(self) -> None:
        pass


class _Store:
    schema_name = "configured_schema"
    _table_class = get_data_model(
        declarative_base(), "schema_test_table", schema_name
    )

    def __init__(self) -> None:
        self.session = _Session()

    def _session(self) -> _Session:
        return self.session


@pytest.mark.parametrize(
    ("method", "args"),
    [
        (PostgresChatStore.delete_message, ("session", 0)),
        (PostgresChatStore.delete_last_message, ("session",)),
    ],
)
def test_delete_updates_use_configured_schema(method: Any, args: tuple[Any, ...]):
    store = _Store()

    deleted = method(store, *args)

    assert deleted is not None
    update_statement = store.session.statements[-1]
    assert "UPDATE configured_schema.schema_test_table" in update_statement
    assert "UPDATE schema_test_table" not in update_statement
