"""Nightly schedule: the launchd job description and the .env check (installing needs macOS)."""
import plistlib

from matchup import schedule


def test_plist(tmp_path):
    p = plistlib.loads(schedule.plist(str(tmp_path), "ARK", 3, 15, ["AUB"], python="/venv/bin/python"))
    assert p["Label"] == schedule.LABEL
    assert p["ProgramArguments"] == ["/venv/bin/python", "-m", "matchup", "nightly", "--folder", str(tmp_path),
                                     "--home", "ARK", "--opponent", "AUB"]
    assert p["StartCalendarInterval"] == {"Hour": 3, "Minute": 15}
    assert p["WorkingDirectory"].endswith("pipeline")


def test_env_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(schedule, "PIPELINE", tmp_path)
    assert "doesn't exist" in schedule.env_ready()
    (tmp_path / ".env").write_text("# DATABASE_URL=commented\nREGISTRY_DATABASE_URL=x\n")
    assert "isn't in" in schedule.env_ready()
    (tmp_path / ".env").write_text("DATABASE_URL=postgresql://x\n")
    assert schedule.env_ready() is None
