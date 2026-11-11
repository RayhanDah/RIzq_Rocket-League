import io
import sys

from rlagent.train import utf8_console


def test_stage_message_survives_cp1252_redirect(monkeypatch):
    buf = io.BytesIO()
    out = io.TextIOWrapper(buf, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", out)
    utf8_console()
    print("palier « maitrise » maîtrisé → palier 7/7")
    out.flush()
    assert "→".encode("utf-8") in buf.getvalue()
