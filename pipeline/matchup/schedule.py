"""Nightly refresh on the Mac: sync the Trackman folder, then publish if anything new arrived.

    python -m matchup schedule --folder ~/Trackman --home ARK            # install (3:15 AM daily)
    python -m matchup schedule --remove                                  # uninstall
    python -m matchup nightly --folder ~/Trackman --home ARK             # what the job runs (try it by hand)

Uses launchd (macOS's scheduler). If the Mac is asleep at the scheduled time, launchd runs the job when it
wakes. The job reads pipeline/.env, so DATABASE_URL (and REGISTRY_DATABASE_URL for ID recovery) must be
saved there. Output goes to ~/Library/Logs/softball-matchup-nightly.log.
"""
from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path

LABEL = "com.softball-matchup.nightly"
PIPELINE = Path(__file__).resolve().parents[1]
LOG = Path.home() / "Library" / "Logs" / "softball-matchup-nightly.log"
AGENT = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def plist(folder: str, home: str, hour: int, minute: int, opponents: list[str] = (), python: str = sys.executable) -> bytes:
    args = [python, "-m", "matchup", "nightly", "--folder", str(Path(folder).expanduser()), "--home", home]
    for t in opponents:
        args += ["--opponent", t]
    return plistlib.dumps({
        "Label": LABEL,
        "ProgramArguments": args,
        "WorkingDirectory": str(PIPELINE),
        "StartCalendarInterval": {"Hour": hour, "Minute": minute},
        "StandardOutPath": str(LOG),
        "StandardErrorPath": str(LOG),
        "EnvironmentVariables": {"PYTHONUNBUFFERED": "1", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin"},
        "ProcessType": "Background",
    })


def env_ready() -> str | None:
    """Problem with pipeline/.env for an unattended run, or None."""
    env = PIPELINE / ".env"
    if not env.exists():
        return f"{env} doesn't exist: the nightly job can't ask for the database password. Save DATABASE_URL there first."
    keys = {line.split("=", 1)[0].strip() for line in env.read_text().splitlines() if "=" in line and not line.lstrip().startswith("#")}
    if "DATABASE_URL" not in keys:
        return f"DATABASE_URL isn't in {env}."
    return None


def install(folder: str, home: str, at: str = "03:15", opponents: list[str] = ()) -> int:
    if sys.platform != "darwin":
        print("schedule uses macOS launchd; run it on your Mac.")
        return 1
    problem = env_ready()
    if problem:
        print(problem)
        return 1
    if not Path(folder).expanduser().is_dir():
        print(f"No folder at {folder}")
        return 1
    protected = [d for d in ("Desktop", "Documents", "Downloads", "Library/Mobile Documents")
                 if str(Path(folder).expanduser().resolve()).startswith(str(Path.home() / d))]
    if protected:
        print(f"Note: {folder} is inside ~/{protected[0]}, which macOS protects from background jobs. If the log shows\n"
              f"'Operation not permitted', add {sys.executable} under System Settings > Privacy & Security >\n"
              "Full Disk Access, or move the Trackman folder somewhere like ~/Trackman.")
    hour, minute = (int(x) for x in at.split(":"))
    AGENT.parent.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    uid = os.getuid()
    subprocess.run(["launchctl", "bootout", f"gui/{uid}", str(AGENT)], capture_output=True)   # replace an old one
    AGENT.write_bytes(plist(folder, home, hour, minute, list(opponents)))
    r = subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(AGENT)], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"launchctl couldn't load the job: {r.stderr.strip()}")
        return 1
    print(f"Nightly refresh installed: every day at {hour:02d}:{minute:02d} (or when the Mac wakes, if asleep).")
    print(f"Log: {LOG}")
    print(f"Run it now to test: launchctl kickstart gui/{uid}/{LABEL}")
    return 0


def remove() -> int:
    if AGENT.exists():
        subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}", str(AGENT)], capture_output=True)
        AGENT.unlink()
        print("Nightly refresh removed.")
    else:
        print("No nightly refresh installed.")
    return 0
