from autocross.web import catalog, course_document


def test_catalog_lists_controllers_and_public_maps():
    data = catalog()
    ids = {item["id"] for item in data["controllers"]}
    assert {"baseline", "rl_longitudinal", "rl_drift"} <= ids
    tracks = {item["id"] for item in data["tracks"]}
    assert "hackathon_test" in tracks
    assert "practice_loop" in tracks
    assert all("/" not in item["id"] or item["group"] == "Generated" for item in data["tracks"])


def test_course_document_has_drawable_geometry():
    course = course_document("hackathon_test")
    assert course["name"] == "Hackathon test"
    assert len(course["center"]) > 20
    assert len(course["left"]) == len(course["center"]) or len(course["left"]) > 20
    assert course["cones"]
    assert len(course["bounds"]) == 4
    assert "x" in course["spawn"] and "yaw" in course["spawn"]
    assert course["vehicle"]["wheelbase"] > 0
    assert course["finish"] > course["start"]
