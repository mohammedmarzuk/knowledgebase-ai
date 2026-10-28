from config import _bool_env, _int_env


def test_int_env_uses_valid_value(monkeypatch):
    monkeypatch.setenv("TEST_INT_SETTING", "7")
    assert _int_env("TEST_INT_SETTING", 3, minimum=1, maximum=10) == 7


def test_int_env_falls_back_for_invalid_or_out_of_range(monkeypatch):
    monkeypatch.setenv("TEST_INT_SETTING", "abc")
    assert _int_env("TEST_INT_SETTING", 3, minimum=1, maximum=10) == 3
    monkeypatch.setenv("TEST_INT_SETTING", "100")
    assert _int_env("TEST_INT_SETTING", 3, minimum=1, maximum=10) == 3


def test_bool_env(monkeypatch):
    monkeypatch.setenv("TEST_BOOL_SETTING", "yes")
    assert _bool_env("TEST_BOOL_SETTING") is True
    monkeypatch.setenv("TEST_BOOL_SETTING", "off")
    assert _bool_env("TEST_BOOL_SETTING") is False
