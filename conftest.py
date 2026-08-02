"""Project-wide pytest configuration.

The ``--durations=10`` flag (set as a default in ``pyproject.toml``) makes pytest
print the slowest tests to the terminal. When the suite runs on GitHub Actions we
also mirror that list into the job's step summary, so the timings show up on the
run's summary page instead of being buried in the raw logs.
"""

import os


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Append the slowest-test table to the GitHub Actions step summary.

    Does nothing outside CI (when ``GITHUB_STEP_SUMMARY`` is unset) or when
    ``--durations`` is disabled, so local runs are unaffected.
    """
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    # Under pytest-xdist this hook also fires on each worker, which only sees its
    # own shard of the tests. Write once, from the controller, using the fully
    # aggregated stats (workers set ``config.workerinput``; the controller never
    # does).
    if hasattr(config, "workerinput"):
        return

    durations = config.getoption("durations", default=None)
    if not durations:
        return

    durations_min = config.getoption("durations_min", default=None)
    if durations_min is None:
        # Matches pytest's own default threshold for the terminal report.
        durations_min = 0.005

    # Gather every phase report (setup/call/teardown) that carries a duration,
    # mirroring how pytest itself builds the "slowest durations" list.
    reports = [
        rep
        for replist in terminalreporter.stats.values()
        for rep in replist
        if hasattr(rep, "duration") and getattr(rep, "when", None) is not None
    ]
    if not reports:
        return

    reports.sort(key=lambda rep: rep.duration, reverse=True)
    if durations > 0:
        reports = reports[:durations]
    reports = [rep for rep in reports if rep.duration >= durations_min]
    if not reports:
        return

    lines = [
        f"### ⏱️ Slowest {len(reports)} test durations",
        "",
        "| Duration (s) | Phase | Test |",
        "| ---: | :--- | :--- |",
    ]
    for rep in reports:
        lines.append(f"| {rep.duration:.2f} | {rep.when} | `{rep.nodeid}` |")
    lines.append("")

    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
