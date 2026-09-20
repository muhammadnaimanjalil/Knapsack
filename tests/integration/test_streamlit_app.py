"""Smoke-test the local Streamlit composition without a browser or deployment."""

from streamlit.testing.v1 import AppTest


def test_example_data_and_all_three_tabs_render_without_exceptions():
    app = AppTest.from_file("streamlit_app.py").run(timeout=20)
    assert [tab.label for tab in app.tabs] == [
        "Offline portfolio", "Online decisions", "Compare policies"
    ]
    app.button[0].click().run(timeout=20)
    assert not app.exception
    assert "60 items and 1,000 package observations" in app.success[0].value

    app.button[1].click().run(timeout=30)
    assert not app.exception
    assert any(metric.label == "Portfolio value" for metric in app.metric)

    app.button[2].click().run(timeout=30)
    assert not app.exception
    assert any(metric.label == "Accepted value" for metric in app.metric)

    next(field for field in app.number_input if field.label == "Simulation runs").set_value(10)
    app.button[3].click().run(timeout=45)
    assert not app.exception
    assert any(heading.value == "Policy comparison" for heading in app.subheader)
