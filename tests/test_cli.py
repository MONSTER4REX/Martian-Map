"""Offline tests for the `marswalk` CLI end-to-end report.

These assert the pipeline wires together and the report is well-formed; they do
not assert specific physics values (all placeholders).
"""

import json

from marswalk.cli import build_inputs, build_parser, main, run_mission


def _args(argv):
    parser = build_parser()
    return parser.parse_args(["mission", *argv])


def test_mission_report_structure():
    rep = run_mission(build_inputs(_args(["--stability-n", "4"])))
    assert set(rep) >= {
        "grid", "start", "goal", "routes", "routes_summary",
        "mission_scenarios", "stability", "caveat",
    }
    # three named routes, in the comparison order
    names = [r["name"] for r in rep["routes"]]
    assert names == [
        "Shortest (3-D distance)", "Slope-aware", "Full-constraint (mission)",
    ]
    # the three standard contingencies are present
    scen = [m["scenario"] for m in rep["mission_scenarios"]]
    assert scen == ["nominal", "rover_loss", "suit_degraded"]
    # stability ran the requested number of realizations
    assert rep["stability"]["n"] == 4
    assert rep["stability"]["n_feasible"] + rep["stability"]["n_infeasible"] == 4
    assert "PLACEHOLDER" in rep["caveat"].upper()


def test_ridge_mission_route_feasible_and_flips():
    # default ridge: shortest crosses the ridge (hard exclusion) -> mission detours
    rep = run_mission(build_inputs(_args(["--stability-n", "2"])))
    shortest = next(r for r in rep["routes"] if r["name"].startswith("Shortest"))
    mission = next(r for r in rep["routes"] if r["name"].startswith("Full-constraint"))
    assert shortest["mission_feasible"] is False
    assert mission["mission_feasible"] is True


def test_conservative_shrinks_budget():
    base = run_mission(build_inputs(_args(["--stability-n", "2"])))
    cons = run_mission(build_inputs(_args(["--stability-n", "2", "--conservative"])))
    b = base["mission_scenarios"][0]["walkback_budget_s"]
    c = cons["mission_scenarios"][0]["walkback_budget_s"]
    assert c < b  # conservative consumables bind sooner


def test_main_text_and_json(capsys):
    assert main(["mission", "--stability-n", "2"]) == 0
    text = capsys.readouterr().out
    assert "THREE-ROUTE COMPARISON" in text
    assert "ROUTE STABILITY" in text

    assert main(["mission", "--json", "--stability-n", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stability"]["n"] == 2


def test_main_defaults_to_mission(capsys):
    # no subcommand -> mission
    assert main(["--stability-n", "2"]) == 0
    assert "MARSWALK PLANNER" in capsys.readouterr().out
