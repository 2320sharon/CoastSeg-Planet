import pytest
import coastseg_planet.download
from coastseg_planet.download import get_ids
from coastseg_planet.download import get_order_ids_by_name
from unittest.mock import AsyncMock
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


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


def test_get_id_to_coregister_returns_empty_when_coregister_false():
    ids = ["1", "2", "3"]
    items = [
        {"id": "1", "properties": {"cloud_cover": 0.2}},
        {"id": "2", "properties": {"cloud_cover": 0.1}},
        {"id": "3", "properties": {"cloud_cover": 0.3}},
    ]
    result = coastseg_planet.download.get_id_to_coregister(ids, items, coregister=False)
    assert result == ""


def test_get_id_to_coregister_with_user_coregister_id_in_ids():
    """
    Test the case where the user provides a coregister ID that is in the list of IDs to download but has a higher cloud cover.
    The function should return the user-provided coregister ID.
    """
    ids = ["1", "2", "3"]
    items = [
        {"id": "1", "properties": {"cloud_cover": 0.2}},
        {"id": "2", "properties": {"cloud_cover": 0.8}},
        {"id": "3", "properties": {"cloud_cover": 0.3}},
    ]
    result = coastseg_planet.download.get_id_to_coregister(
        ids, items, coregister=True, user_coregister_id="2"
    )
    assert result == "2"


def test_get_id_to_coregister_with_user_coregister_id_not_in_ids():
    """
    Test the case where the user provides a coregister ID that is not in the list of IDs to download.
    The function should raise a ValueError.
    """
    ids = ["1", "2", "3"]
    items = [
        {"id": "1", "properties": {"cloud_cover": 0.2}},
        {"id": "2", "properties": {"cloud_cover": 0.1}},
        {"id": "3", "properties": {"cloud_cover": 0.3}},
    ]
    with pytest.raises(ValueError) as excinfo:
        coastseg_planet.download.get_id_to_coregister(
            ids, items, coregister=True, user_coregister_id="4"
        )
    assert "Coregister ID 4 not found in the list of IDs to download" in str(
        excinfo.value
    )


def test_get_id_to_coregister_selects_lowest_cloud_cover(monkeypatch):
    ids = ["1", "2", "3"]
    items = [
        {"id": "1", "properties": {"cloud_cover": 0.2}},
        {"id": "2", "properties": {"cloud_cover": 0.1}},
        {"id": "3", "properties": {"cloud_cover": 0.3}},
    ]
    # Patch get_image_id_with_lowest_cloud_cover to return "2"
    monkeypatch.setattr(
        coastseg_planet.download,
        "get_image_id_with_lowest_cloud_cover",
        lambda items: "2",
    )
    result = coastseg_planet.download.get_id_to_coregister(ids, items, coregister=True)
    assert result == "2"


def test_get_id_to_coregister_lowest_cloud_cover_not_in_ids(monkeypatch):
    ids = ["1", "2", "3"]
    items = [
        {"id": "2", "properties": {"cloud_cover": 0.1}},
        {"id": "3", "properties": {"cloud_cover": 0.3}},
        {"id": "4", "properties": {"cloud_cover": 0.0}},
    ]
    # Patch get_image_id_with_lowest_cloud_cover to return "4"
    monkeypatch.setattr(
        coastseg_planet.download,
        "get_image_id_with_lowest_cloud_cover",
        lambda items: "4",
    )
    with pytest.raises(ValueError) as excinfo:
        coastseg_planet.download.get_id_to_coregister(ids, items, coregister=True)
    assert "Coregister ID 4 not found in the list of IDs to download" in str(
        excinfo.value
    )
    items = [
        {"id": "1", "properties": {"cloud_cover": 0.2}},
        {"id": "2", "properties": {"cloud_cover": 0.1}},
        {"id": "3", "properties": {"cloud_cover": 0.3}},
        {"id": "4", "properties": {"cloud_cover": 0.0}},
    ]
    # Patch get_image_id_with_lowest_cloud_cover to return "4"
    monkeypatch.setattr(
        coastseg_planet.download,
        "get_image_id_with_lowest_cloud_cover",
        lambda items: "4",
    )
    with pytest.raises(ValueError) as excinfo:
        coastseg_planet.download.get_id_to_coregister(ids, items, coregister=True)
    assert "Coregister ID 4 not found in the list of IDs to download" in str(
        excinfo.value
    )


def test_build_tools_list_clip_and_toar(monkeypatch):
    """
    Test that the build_tools_list function returns both the clip and toar tools
    when provided with tools containing "clip" and "toar" and a valid roi_dict.
    """
    # Mock planet.order_request.clip_tool and toar_tool
    mock_clip_tool = object()
    mock_toar_tool = object()
    monkeypatch.setattr(
        "coastseg_planet.download.planet.order_request.clip_tool",
        lambda aoi: mock_clip_tool,
    )
    monkeypatch.setattr(
        "coastseg_planet.download.planet.order_request.toar_tool",
        lambda scale_factor: mock_toar_tool,
    )

    tools = ["clip", "toar"]
    roi_dict = {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 0]]]}
    result = coastseg_planet.download.build_tools_list(tools, roi_dict=roi_dict)
    assert mock_clip_tool in result
    assert mock_toar_tool in result
    assert len(result) == 2


def test_build_tools_list_coregister(monkeypatch):
    mock_coregister_tool = object()
    monkeypatch.setattr(
        "coastseg_planet.download.planet.order_request.coregister_tool",
        lambda id_to_coregister: mock_coregister_tool,
    )
    tools = ["coregister"]
    result = coastseg_planet.download.build_tools_list(tools, id_to_coregister="abc123")
    assert result == [mock_coregister_tool]


def test_build_tools_list_unknown_tool(monkeypatch, capsys):
    """
    Test that unknown tool names are skipped and a warning is printed,
    while valid tools are still included in the result.
    """
    # Only "clip" is valid, "foo" is not
    mock_clip_tool = object()
    monkeypatch.setattr(
        "coastseg_planet.download.planet.order_request.clip_tool",
        lambda aoi: mock_clip_tool,
    )
    tools = ["clip", "foo"]
    roi_dict = {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 0]]]}
    result = coastseg_planet.download.build_tools_list(tools, roi_dict=roi_dict)
    assert result == [mock_clip_tool]
    captured = capsys.readouterr()
    assert "Warning: Unknown tool 'foo' skipped." in captured.out


def test_build_tools_list_empty():
    """
    Test that providing an empty tool list returns an empty result list.
    """
    result = coastseg_planet.download.build_tools_list([])
    assert result == []


def test_build_tools_list_strip_and_case(monkeypatch):
    """
    Test that tool names are processed case-insensitively and with whitespace stripped.
    """
    mock_clip_tool = object()
    mock_toar_tool = object()
    monkeypatch.setattr(
        "coastseg_planet.download.planet.order_request.clip_tool",
        lambda aoi: mock_clip_tool,
    )
    monkeypatch.setattr(
        "coastseg_planet.download.planet.order_request.toar_tool",
        lambda scale_factor: mock_toar_tool,
    )
    tools = ["  CLIP  ", "ToAr"]
    roi_dict = {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 0]]]}
    result = coastseg_planet.download.build_tools_list(tools, roi_dict=roi_dict)
    assert mock_clip_tool in result
    assert mock_toar_tool in result
    assert len(result) == 2


@pytest.mark.asyncio
async def test_make_order_and_download_basic(monkeypatch):
    # Mocks
    roi = MagicMock()
    roi.to_json.return_value = (
        '{"type": "Polygon", "coordinates": [[[0,0],[1,1],[1,0],[0,0]]]}'
    )
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    order_name = "test_order"
    download_path = "/tmp"
    coregister_id = ""
    product_bundle = "ortho_analytic_4b"
    min_area_percentage = 0.5
    tools = {"clip", "toar"}

    # Patch planet.Session
    mock_session = MagicMock()
    mock_sess_instance = MagicMock()
    mock_session.__aenter__.return_value = mock_sess_instance
    monkeypatch.setattr("coastseg_planet.download.planet.Session", lambda: mock_session)

    # Patch query_planet_items
    items = [
        {"id": "1", "properties": {"cloud_cover": 0.2}},
        {"id": "2", "properties": {"cloud_cover": 0.1}},
    ]
    monkeypatch.setattr(
        "coastseg_planet.download.query_planet_items", AsyncMock(return_value=items)
    )

    # Patch filter_items_by_area
    monkeypatch.setattr(
        "coastseg_planet.download.filter_items_by_area",
        lambda roi, items, min_area: items,
    )

    # Patch get_ids
    monkeypatch.setattr(
        "coastseg_planet.download.get_ids",
        lambda items, month_filter=None: [item["id"] for item in items],
    )

    # Patch get_id_to_coregister
    monkeypatch.setattr(
        "coastseg_planet.download.get_id_to_coregister",
        lambda ids, items, coregister, user_coregister_id="": "",
    )

    # Patch build_tools_list
    monkeypatch.setattr(
        "coastseg_planet.download.build_tools_list",
        lambda tools, roi_dict=None, id_to_coregister="": ["tool1", "tool2"],
    )

    # Patch sess.client
    mock_order_client = MagicMock()
    mock_sess_instance.client.return_value = mock_order_client

    # Patch process_orders_in_batches
    monkeypatch.setattr(
        "coastseg_planet.download.process_orders_in_batches", AsyncMock()
    )

    # Run
    await coastseg_planet.download.make_order_and_download(
        roi,
        start_date,
        end_date,
        order_name,
        download_path,
        coregister_id=coregister_id,
        product_bundle=product_bundle,
        min_area_percentage=min_area_percentage,
        tools=tools,
    )

    coastseg_planet.download.query_planet_items.assert_awaited_once()
    coastseg_planet.download.process_orders_in_batches.assert_awaited_once()
    assert mock_sess_instance.client.called


@pytest.mark.asyncio
async def test_make_order_and_download_with_month_filter(monkeypatch):
    roi = MagicMock()
    roi.to_json.return_value = (
        '{"type": "Polygon", "coordinates": [[[0,0],[1,1],[1,0],[0,0]]]}'
    )
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    order_name = "test_order"
    download_path = "/tmp"
    tools = {"clip"}

    mock_session = MagicMock()
    mock_sess_instance = MagicMock()
    mock_session.__aenter__.return_value = mock_sess_instance
    monkeypatch.setattr("coastseg_planet.download.planet.Session", lambda: mock_session)

    items = [{"id": "1", "properties": {"cloud_cover": 0.2}}]
    monkeypatch.setattr(
        "coastseg_planet.download.query_planet_items", AsyncMock(return_value=items)
    )
    monkeypatch.setattr(
        "coastseg_planet.download.filter_items_by_area",
        lambda roi, items, min_area: items,
    )
    get_ids_mock = MagicMock(return_value=["1"])
    monkeypatch.setattr("coastseg_planet.download.get_ids", get_ids_mock)
    monkeypatch.setattr(
        "coastseg_planet.download.get_id_to_coregister",
        lambda ids, items, coregister, user_coregister_id="": "coregister-id",
    )
    monkeypatch.setattr(
        "coastseg_planet.download.build_tools_list",
        lambda tools, roi_dict=None, id_to_coregister="": ["clip_tool"],
    )
    mock_order_client = MagicMock()
    mock_sess_instance.client.return_value = mock_order_client
    process_orders_mock = AsyncMock()
    monkeypatch.setattr(
        "coastseg_planet.download.process_orders_in_batches", process_orders_mock
    )

    await coastseg_planet.download.make_order_and_download(
        roi,
        start_date,
        end_date,
        order_name,
        download_path,
        tools=tools,
        month_filter=["01"],
    )

    args, kwargs = get_ids_mock.call_args
    assert args[1] == ["01"]  # month_filter was passed correctly as positional arg

    process_args, process_kwargs = process_orders_mock.call_args
    assert process_args[0] == mock_order_client
    assert process_args[1] == ["1"]  # IDs to download
    assert process_args[2] == ["clip_tool"]  # tools_list
    assert process_args[3] == download_path
    assert process_args[4] == order_name
    assert process_kwargs["product_bundle"] == "ortho_analytic_4b"
    assert process_kwargs["id_to_coregister"] == "coregister-id"


@pytest.mark.asyncio
async def test_make_order_and_download_tools_none(monkeypatch):
    roi = MagicMock()
    roi.to_json.return_value = (
        '{"type": "Polygon", "coordinates": [[[0,0],[1,1],[1,0],[0,0]]]}'
    )
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    order_name = "test_order"
    download_path = "/tmp"

    mock_session = MagicMock()
    mock_sess_instance = MagicMock()
    mock_session.__aenter__.return_value = mock_sess_instance
    monkeypatch.setattr("coastseg_planet.download.planet.Session", lambda: mock_session)

    items = [{"id": "1", "properties": {"cloud_cover": 0.2}}]
    monkeypatch.setattr(
        "coastseg_planet.download.query_planet_items", AsyncMock(return_value=items)
    )
    monkeypatch.setattr(
        "coastseg_planet.download.filter_items_by_area",
        lambda roi, items, min_area: items,
    )
    monkeypatch.setattr(
        "coastseg_planet.download.get_ids", lambda items, month_filter=None: ["1"]
    )
    monkeypatch.setattr(
        "coastseg_planet.download.get_id_to_coregister",
        lambda ids, items, coregister, user_coregister_id="": "",
    )
    monkeypatch.setattr(
        "coastseg_planet.download.build_tools_list",
        lambda tools, roi_dict=None, id_to_coregister="": [],
    )
    mock_order_client = MagicMock()
    mock_sess_instance.client.return_value = mock_order_client
    process_orders_mock = AsyncMock()
    monkeypatch.setattr(
        "coastseg_planet.download.process_orders_in_batches", process_orders_mock
    )

    await coastseg_planet.download.make_order_and_download(
        roi,
        start_date,
        end_date,
        order_name,
        download_path,
        tools=None,
    )

    process_orders_mock.assert_awaited_once()
