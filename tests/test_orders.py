import os
import json
import tempfile
from coastseg_planet.orders import Order  # Replace with actual module if different

# Sample GeoJSON with a valid polygon
SAMPLE_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
                ],
            },
            "properties": {},
        }
    ],
}


def create_temp_geojson():
    """
    Create a temporary GeoJSON file with a valid Polygon geometry.

    Returns:
        str: Path to the created temporary GeoJSON file.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".geojson", mode="w")
    json.dump(SAMPLE_GEOJSON, tmp)
    tmp.close()
    return tmp.name


def test_order_creation_and_validation_with_tools():
    """
    Test that an Order object is created correctly with all expected attributes,
    including the optional 'tools' parameter, and that validations pass.
    """
    roi_path = create_temp_geojson()
    with tempfile.TemporaryDirectory() as destination:
        tools_set = {"clip", "toar"}

        # Create order with tools
        order = Order(
            order_name="test_order",
            roi_path=roi_path,
            start_date="2024-01-01",
            end_date="2024-12-31",
            destination=destination,
            cloud_cover=0.5,
            min_area_percentage=0.8,
            coregister_id="reference123",
            continue_existing=False,
            month_filter=["01", "02"],
            tools=tools_set,
        )

        order_dict = order.get_order()

        # Check basic fields
        assert order_dict["order_name"] == "test_order"
        assert order_dict["cloud_cover"] == 0.5
        assert order_dict["start_date"] == "2024-01-01"
        assert order_dict["end_date"] == "2024-12-31"
        assert order_dict["roi_path"] == roi_path
        assert order_dict["destination"] == destination
        assert order_dict["min_area_percentage"] == 0.8
        assert order_dict["coregister_id"] == "reference123"
        assert order_dict["continue_existing"] is False
        assert order_dict["month_filter"] == ["01", "02"]
        assert (
            set(order_dict["tools"]) == tools_set
        )  # tools returned as list, compare as set
        assert os.path.isdir(destination)
        assert os.path.isfile(roi_path)


def test_order_to_json_roundtrip_with_tools():
    """
    Test that the Order dictionary with tools can be serialized to JSON and
    loaded back without losing any information or structure.
    """
    roi_path = create_temp_geojson()
    with tempfile.TemporaryDirectory() as destination:
        tools_set = {"clip", "toar"}
        order = Order(
            order_name="json_test",
            roi_path=roi_path,
            start_date="2023-01-01",
            end_date="2023-12-31",
            destination=destination,
            tools=tools_set,
        )

        # Save to JSON file
        order_dict = order.get_order()
        json_path = os.path.join(destination, "order.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(order_dict, f)

        # Load back and compare
        with open(json_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        print(f"Loaded order: {loaded}")

        # Compare tools as sets to avoid order sensitivity
        assert set(loaded["tools"]) == tools_set
        # All other fields should match directly
        for key in order_dict:
            if key != "tools":
                assert loaded[key] == order_dict[key]
