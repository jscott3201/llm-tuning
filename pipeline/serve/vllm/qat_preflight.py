"""Check an explicitly prebuilt image on CPU, or recover its saved resources."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import signal
import sys

PIPELINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE))


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    subcommands = command.add_subparsers(dest="action", required=True)
    run = subcommands.add_parser("run", help="Allocate one bounded CPU preflight for a prebuilt image")
    run.add_argument("--image-id", required=True)
    for child in (run, subcommands.add_parser("recover", help="Reconcile saved resources without allocation")):
        child.add_argument("--environment", required=True)
        child.add_argument("--receipt", type=Path, required=True)
    return command


async def execute(args):
    from _common.qat_preflight import Backend, preflight, recover
    from _common.qat_preflight_receipt import LIMITS, Receipt, validator_source

    final_deadline = asyncio.get_running_loop().time() + LIMITS["total_seconds"]
    receipt = Receipt(args.receipt)
    handlers = []
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, task.cancel)
            handlers.append(sig)
        if args.action == "run":
            source = validator_source()
            receipt.create(args.environment, args.image_id, source)
            code = await preflight(receipt, Backend(args.environment), source, final_deadline=final_deadline)
        else:
            receipt.load(args.environment)
            code = await recover(receipt, Backend(args.environment))
        print(json.dumps({key: receipt.data[key] for key in ("status", "cleanup", "app_id", "sandbox_id")},
                         separators=(",", ":")))
        return code
    finally:
        for sig in handlers:
            loop.remove_signal_handler(sig)
        receipt.close()


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return asyncio.run(execute(args))
    except Exception:
        # Exception strings can contain private config, inventory or paths.
        print('{"status":"failed","reason":"preflight_command_failed"}', file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
