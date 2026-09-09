from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_without_exception() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=20).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Esperimento",
        "Selezione",
        "QAOA",
        "Risultati",
        "Esportazioni",
    ]
    assert app.title[0].value == "Quantum Risk Lab"
    assert any("LASSO: L1-penalized logistic regression" in item.value for item in app.caption)
