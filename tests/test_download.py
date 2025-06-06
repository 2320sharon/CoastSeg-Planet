import pytest
import coastseg_planet.download
from coastseg_planet.download import get_ids
from coastseg_planet.download import get_order_ids_by_name
from unittest.mock import AsyncMock


def test_get_ids():
    items = [
        {"id": 5, "properties": {"acquired": "2023-06-27"}},
        {"id": 1, "properties": {"acquired": "2023-06-28"}},
        {"id": 3, "properties": {"acquired": "2023-06-29"}},
        {"id": 2, "properties": {"acquired": "2023-06-30"}},
        {"id": 4, "properties": {"acquired": "2023-07-01"}},
    ]
    expected_ids = [5, 1, 3, 2, 4]

    assert set(get_ids(items)) == set(expected_ids)
    assert get_ids(items) == expected_ids


def test_empty_list():
    assert get_ids([]) == []


def test_none():
    assert get_ids(None) == []


def test_valid_items(monkeypatch):
    items = [
        {"id": "1", "properties": {"acquired": "2023-01-01"}},
        {"id": "2", "properties": {"acquired": "2023-01-01"}},
        {"id": "3", "properties": {"acquired": "2023-01-02"}},
    ]

    def mock_get_acquired_date(item):
        return item["properties"]["acquired"]

    def mock_get_ids_by_date(items):
        return {"2023-01-01": ["1", "2"], "2023-01-02": ["3"]}

    monkeypatch.setattr(
        "coastseg_planet.download.get_acquired_date", mock_get_acquired_date
    )
    monkeypatch.setattr(
        "coastseg_planet.download.get_ids_by_date", mock_get_ids_by_date
    )

    expected_ids = ["1", "2", "3"]
    assert set(get_ids(items)) == set(expected_ids)
    assert get_ids(items) == expected_ids


def test_get_ids_by_date():
    items = [
        {"id": 1, "properties": {"acquired": "2023-06-27"}},
        {"id": 2, "properties": {"acquired": "2023-06-27"}},
        {"id": 3, "properties": {"acquired": "2023-06-28"}},
        {"id": 4, "properties": {"acquired": "2023-06-28"}},
        {"id": 5, "properties": {"acquired": "2023-06-29"}},
    ]

    expected_ids_by_date = {
        "2023-06-27": [1, 2],
        "2023-06-28": [3, 4],
        "2023-06-29": [5],
    }

    assert coastseg_planet.download.get_ids_by_date(items) == expected_ids_by_date


@pytest.mark.asyncio
async def test_get_order_ids_by_name():
    # Create a mock client
    client = AsyncMock()

    # Mock the list_orders method to return an async iterator
    async def mock_list_orders():
        orders = [
            {"name": "order1", "state": "success", "id": "order1_id"},
            {"name": "order2", "state": "success", "id": "order2_id"},
        ]
        for order in orders:
            yield order

    client.list_orders = mock_list_orders

    order_name = "order1"
    states = ["success"]

    result = await get_order_ids_by_name(client, order_name, states)

    assert isinstance(result, list)
    assert result == ["order1_id"]


@pytest.mark.asyncio
async def test_get_order_ids_by_name_state_not_matching():
    # Create a mock client
    client = AsyncMock()

    # Mock the list_orders method to return an async iterator
    async def mock_list_orders():
        orders = [
            {"name": "order1", "state": "success", "id": "order1_id"},
            {"name": "order2", "state": "success", "id": "order2_id"},
        ]
        for order in orders:
            yield order

    client.list_orders = mock_list_orders

    order_name = "order1"
    states = ["failed"]

    result = await get_order_ids_by_name(client, order_name, states)

    assert isinstance(result, list)
    assert result == []


@pytest.mark.asyncio
async def test_get_order_ids_by_name_not_existing():
    # Create a mock client
    client = AsyncMock()

    # Mock the list_orders method to return an async iterator
    async def mock_list_orders():
        orders = [
            {"name": "order1", "state": "success", "id": "order1_id"},
            {"name": "order2", "state": "success", "id": "order2_id"},
        ]
        for order in orders:
            yield order

    client.list_orders = mock_list_orders

    order_name = "non_existing_order"
    states = ["success"]

    result = await get_order_ids_by_name(client, order_name, states)

    assert isinstance(result, list)
    assert result == []
