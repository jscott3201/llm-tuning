"""Private atomic receipts and fixed resource policy for CPU preflight."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

SOURCE_LIMIT = 65536
LIMITS = {"cpu": [1, 1], "memory_mib": [4096, 4096], "sandbox_seconds": 90,
          "total_seconds": 180, "cleanup_seconds": 45, "stdout_bytes": 32768, "stderr_bytes": 8192}
IDENTITIES = {"app": r"ap-[A-Za-z0-9]{1,64}", "sandbox": r"sb-[A-Za-z0-9]{1,64}",
              "image": r"im-[A-Za-z0-9]{1,64}", "environment": r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}"}


def identity(kind, value):
    """Reject malformed targets before any resource operation."""
    if not isinstance(value, str) or re.fullmatch(IDENTITIES[kind], value) is None:
        raise ValueError("invalid_" + kind)
    return value


def validator_source():
    """Read only the fixed tracked validator, with a strict source-size cap."""
    with Path(__file__).with_name("qat_stack.py").open("rb") as source:
        value = source.read(SOURCE_LIMIT + 1)
    if not value or len(value) > SOURCE_LIMIT:
        raise ValueError("invalid_validator_source_size")
    return value


def validate(data, environment):
    """Validate recovery authority, including unchanged bounds and environment."""
    if data.get("schema") != "qat-cpu-preflight-v1" or data.get("environment") != identity("environment", environment):
        raise ValueError("receipt_environment_or_schema_mismatch")
    identity("image", data.get("image_id"))
    run_id = data.get("run_id", "")
    if re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise ValueError("invalid_receipt_run_id")
    if data.get("description") != "qat-cpu-preflight-" + run_id or data.get("tag") != {"qat-preflight": run_id}:
        raise ValueError("invalid_receipt_ownership")
    if data.get("limits") != LIMITS or re.fullmatch(r"[0-9a-f]{64}", data.get("validator_sha256", "")) is None:
        raise ValueError("invalid_receipt_policy")
    if type(data.get("validator_bytes")) is not int or not 0 < data["validator_bytes"] <= SOURCE_LIMIT:
        raise ValueError("invalid_receipt_source_size")
    if type(data.get("baseline_absent")) is not bool:
        raise ValueError("invalid_receipt_baseline")
    for kind in ("app", "sandbox"):
        state = data.get("allocation", {}).get(kind)
        if state not in {"not_requested", "unresolved", "acknowledged"}:
            raise ValueError("invalid_receipt_allocation")
        value = data.get(kind + "_id")
        if value is not None:
            identity(kind, value)
        if state == "acknowledged" and value is None:
            raise ValueError("missing_acknowledged_identity")
        if state != "not_requested" and not data["baseline_absent"]:
            raise ValueError("allocation_without_baseline")
    if data["allocation"]["sandbox"] != "not_requested" and data["allocation"]["app"] != "acknowledged":
        raise ValueError("sandbox_without_acknowledged_app")


class Receipt:
    """Hold one local owner lock while atomically saving selected private state."""
    def __init__(self, path):
        self.path = Path(path).absolute()
        self.lock_fd = os.open(str(self.path) + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(self.lock_fd)
            raise ValueError("receipt_in_use") from None
        self.data = None

    def create(self, environment, image_id, source):
        """Reserve the receipt exclusively before listing or allocating remotely."""
        identity("environment", environment)
        identity("image", image_id)
        run_id = uuid.uuid4().hex
        self.data = {"schema": "qat-cpu-preflight-v1", "run_id": run_id,
                     "created_at": datetime.now(timezone.utc).isoformat(),
                     "environment": environment, "image_id": image_id,
                     "description": "qat-cpu-preflight-" + run_id, "tag": {"qat-preflight": run_id},
                     "validator_sha256": hashlib.sha256(source).hexdigest(), "validator_bytes": len(source),
                     "limits": dict(LIMITS), "baseline_absent": False,
                     "allocation": {"app": "not_requested", "sandbox": "not_requested"},
                     "app_id": None, "sandbox_id": None, "status": "prepared", "cleanup": "pending"}
        validate(self.data, environment)
        descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(self._bytes())
            output.flush()
            os.fsync(output.fileno())
        self._sync_directory()

    def load(self, environment):
        """Read a bounded existing receipt, refusing symlinks and stale authority."""
        descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as source:
            content = source.read(65537)
        if len(content) > 65536:
            raise ValueError("receipt_size_limit")
        self.data = json.loads(content)
        if not isinstance(self.data, dict):
            raise ValueError("invalid_receipt")
        validate(self.data, environment)

    def _bytes(self):
        content = (json.dumps(self.data, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if len(content) > 65536:
            raise ValueError("receipt_size_limit")
        return content

    def _sync_directory(self):
        descriptor = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def save(self):
        """Publish a complete update atomically; keep the stable sidecar lock."""
        content = self._bytes()
        descriptor, temporary = tempfile.mkstemp(prefix=".qat-receipt-", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
            self._sync_directory()
        finally:
            Path(temporary).unlink(missing_ok=True)

    def close(self):
        """Release the local lock; retain the receipt and stable lock file."""
        os.close(self.lock_fd)
