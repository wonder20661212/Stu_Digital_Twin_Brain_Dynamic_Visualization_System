"""Command-line demos for the NPI Brain SDK."""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from . import __version__
from .client import NPIClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="npi-demo",
        description="Run quick demos for the NPI Brain SDK.",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Optional external project/data root. Defaults to bundled wheel resources.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON for check/info.")

    subparsers = parser.add_subparsers(dest="command")

    check = subparsers.add_parser("check", help="Run the SDK quick-check demo.")
    check.add_argument("--strength", type=float, default=0.2, help="Perturbation strength to inspect.")
    check.add_argument("--algorithm", default="pcmci", help="Causal mask algorithm to summarize.")
    check.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    web = subparsers.add_parser("web", help="Start the web visualizer demo.")
    web.add_argument("--host", default="127.0.0.1", help="Backend host.")
    web.add_argument("--port", type=int, default=5002, help="Backend port.")

    info = subparsers.add_parser("info", help="Print bundled SDK resource information.")
    info.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    return parser


def run_check(args: argparse.Namespace) -> int:
    client = NPIClient(project_root=args.project_root)
    strengths = client.available_strengths()
    ec = client.get_ec_matrix(args.strength)
    mask = client.compute_mask(args.algorithm)
    summary = client.mask_summary(mask)
    payload = {
        "sdk_version": __version__,
        "project_root": str(client.project_root),
        "available_strengths": strengths,
        "selected_strength": args.strength,
        "ec_shape": list(ec.shape),
        "algorithm": args.algorithm,
        "mask_summary": summary,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("NPI Brain SDK demo check")
        print(f"SDK version: {payload['sdk_version']}")
        print(f"Project root: {payload['project_root']}")
        print(f"Available strengths: {payload['available_strengths']}")
        print(f"EC shape at {args.strength}: {tuple(payload['ec_shape'])}")
        print(f"{args.algorithm} mask: {payload['mask_summary']}")
        print("Result: OK")
    return 0


def run_info(args: argparse.Namespace) -> int:
    client = NPIClient(project_root=args.project_root)
    payload = {
        "sdk_version": __version__,
        "project_root": str(client.project_root),
        "available_strengths": client.available_strengths(),
        "available_ec_results": client.available_ec_results(),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("NPI Brain SDK information")
        print(f"SDK version: {payload['sdk_version']}")
        print(f"Project root: {payload['project_root']}")
        print(f"Available strengths: {payload['available_strengths']}")
        print("Available EC result modes:")
        for item in payload["available_ec_results"]:
            print(f"- {item.get('mode')}: {item.get('label')}")
    return 0


def run_web(args: argparse.Namespace) -> int:
    print(f"Starting NPI web demo at http://{args.host}:{args.port}/")
    print("Press Ctrl+C to stop the backend service.")
    client = NPIClient(project_root=args.project_root)
    client.run_web_app(host=args.host, port=args.port)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "check"
    if command == "check":
        if not hasattr(args, "strength"):
            args.strength = 0.2
        if not hasattr(args, "algorithm"):
            args.algorithm = "pcmci"
        return run_check(args)
    if command == "info":
        return run_info(args)
    if command == "web":
        return run_web(args)
    parser.error(f"Unknown command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
