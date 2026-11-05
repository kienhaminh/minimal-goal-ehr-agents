"""Run-owned disposable MedAgentBench FHIR servers for parallel workers."""

import hashlib
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path



def atomic_json(path, data, indent=2):
    """`indent=None` writes compact JSON, several times faster for a large file (results.json)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A short temporary name: a long file name plus a suffix exceeded the Windows 260-character path limit
    # (2026-10-15, a journal archive of a run folder with a long name).
    temporary = path.with_name("." + uuid.uuid4().hex[:12] + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=indent, allow_nan=False, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        # On Windows, a reader without delete sharing (antivirus, a monitor) makes replace fail briefly.
        for wait in (0.1, 0.2, 0.5, 1, 2, 0):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if not wait:
                    raise
                time.sleep(wait)
    finally:
        temporary.unlink(missing_ok=True)


IMAGE = "jyxsu6/medagentbench@sha256:3fb83d7ed71c5476f9eb6212bd440a909ef7505922bbc757dc488a8fc0701966"
FHIR_PORT = 8080
OWNER_LABEL = "dev.clinical-agent.owner"
# Without a limit each JVM sizes its heap from the whole Docker VM, so four
# workers exhausted the VM (run100 attempts 4 and 5).
CONTAINER_MEMORY = "3g"
JAVA_HEAP_OPTIONS = "-Xmx2g"
DOCKER_TIMEOUT_SECONDS = 30
READINESS_REQUEST_TIMEOUT_SECONDS = 2


class FhirWorkersError(RuntimeError):
    """A worker failed to start or an owned container could not be removed."""


class FhirWorkers:
    def __init__(self, image, workers, run_id, out, *, timeout_seconds=900):
        if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
            raise ValueError("workers must be a positive integer")
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not image or not run_id:
            raise ValueError("image and run_id are required")

        self.image = str(image)
        self.workers = workers
        self.run_id = str(run_id)
        self.timeout_seconds = float(timeout_seconds)
        self.out = Path(out)
        self.receipt_path = self.out / "fhir-resources.json"
        self.owner = uuid.uuid4().hex
        self.owner_label = f"{OWNER_LABEL}={self.owner}"
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._creation_attempted = False
        self._closing = False
        self._cleanup_persistence_errors = []
        self._removed_endpoints = set()
        self._state = "new"
        safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "-", self.run_id).strip("-.") or "run"
        safe_run = safe_run[:48]
        name_suffix = uuid.uuid4().hex[:10]
        self._resources = [
            {
                "worker": worker,
                "container_id": None,
                "name": f"autonomy-fhir-{safe_run}-{worker}-{name_suffix}",
                "endpoint": None,
                "status": "pending",
            }
            for worker in range(workers)
        ]
        self._receipt = {
            "schema_version": 1,
            "run_id": self.run_id,
            "image": self.image,
            "workers": workers,
            "owner_label": self.owner_label,
            "timeout_seconds": self.timeout_seconds,
            "status": self._state,
            "resources": self._resources,
            "commands": [],
            "cleanup": {"attempts": [], "removed_endpoints": [], "errors": []},
            "created_at": _now(),
            "updated_at": _now(),
        }

    @property
    def removed_endpoints(self):
        with self._lock:
            return frozenset(self._removed_endpoints)

    def start(self):
        """Start one independent server per worker and return their FHIR endpoints."""
        with self._lifecycle_lock:
            if self._state == "ready":
                return [resource["endpoint"] for resource in self._resources]
            if self._state != "new":
                raise FhirWorkersError(f"Cannot start workers in state {self._state}")
            self._set_state("starting")
            self._creation_attempted = True
            try:
                self._parallel(self._create, range(self.workers), "create")
                self._parallel(self._start, range(self.workers), "start")
                self._wait_ready()
                endpoints = [resource["endpoint"] for resource in self._resources]
                if len(set(endpoints)) != self.workers:
                    raise FhirWorkersError("Docker assigned duplicate FHIR endpoints")
                self._set_state("ready")
                return endpoints
            except BaseException as error:
                self._set_state("startup_failed", error=str(error))
                raise

    def relocate(self, out):
        """Move the receipt into `out`, a run folder created after the workers started; logs go there at close."""
        with self._lock:
            old_dir, old_path = self.out, self.receipt_path
            self.out = Path(out)
            self.receipt_path = self.out / "fhir-resources.json"
            self._save()
            old_path.unlink(missing_ok=True)
            try:
                old_dir.rmdir()  # only if empty
            except OSError:
                pass

    def close(self):
        """Save logs and remove every container verified with this owner's unique label."""
        with self._lifecycle_lock:
            if self._state == "closed":
                return
            self._closing = True
            attempt = {"started_at": _now(), "errors": []}
            self._receipt["cleanup"]["attempts"].append(attempt)
            self._save()
            if not self._creation_attempted:
                self._set_state("closed")
                attempt["finished_at"] = _now()
                self._save()
                return

            ids = set()
            discovery_succeeded = False
            try:
                listed = self._docker(
                    [
                        "docker", "ps", "--all", "--quiet", "--no-trunc",
                        "--filter", f"label={self.owner_label}",
                    ],
                    "discover owned containers",
                )
                if listed["return_code"] == 0:
                    discovery_succeeded = True
                    ids.update(line.strip() for line in listed["stdout"].splitlines() if line.strip())
                else:
                    self._cleanup_error(attempt, f"Container discovery failed: {listed['stderr'].strip()}")
            except Exception as error:
                self._cleanup_error(attempt, f"Container discovery failed: {error}")

            ids.update(r["container_id"] for r in self._resources if r["container_id"])
            for container_id in sorted(ids):
                try:
                    self._cleanup_container(container_id, attempt)
                except Exception as error:
                    self._cleanup_error(attempt, f"Cleanup failed for {container_id}: {error}")

            if not discovery_succeeded:
                self._cleanup_error(
                    attempt,
                    "Could not verify the full set of containers after a possibly ambiguous create",
                )
            attempt["finished_at"] = _now()
            self._receipt["cleanup"]["removed_endpoints"] = sorted(self._removed_endpoints)
            self._save()
            errors = attempt["errors"] + self._cleanup_persistence_errors
            if errors:
                self._set_state("cleanup_failed")
                raise FhirWorkersError("FHIR cleanup failed: " + "; ".join(errors))
            self._set_state("closed")

    def _create(self, worker):
        resource = self._resources[worker]
        command = [
            "docker", "create", "--pull", "never", "--platform", "linux/amd64",
            "--publish", f"127.0.0.1::{FHIR_PORT}",
            "--memory", CONTAINER_MEMORY, "--memory-swap", CONTAINER_MEMORY,
            "--env", f"JAVA_TOOL_OPTIONS={JAVA_HEAP_OPTIONS}",
            "--label", self.owner_label,
            "--label", f"dev.clinical-agent.run={self.run_id}",
            "--label", f"dev.clinical-agent.worker={worker}",
            "--name", resource["name"], self.image,
        ]
        result = self._docker(command, f"create worker {worker}")
        self._expect(result, f"create worker {worker}")
        ids = result["stdout"].strip().splitlines()
        container_id = ids[-1] if ids else ""
        if not re.fullmatch(r"[0-9a-f]{12,64}", container_id):
            raise FhirWorkersError(f"Docker returned an invalid ID for worker {worker}")
        with self._lock:
            resource.update(container_id=container_id, status="created")
            self._save()

    def _start(self, worker):
        resource = self._resources[worker]
        container_id = resource["container_id"]
        result = self._docker(["docker", "start", container_id], f"start worker {worker}")
        self._expect(result, f"start worker {worker}")
        with self._lock:
            resource["status"] = "started"
            self._save()

        result = self._docker(
            ["docker", "port", container_id, f"{FHIR_PORT}/tcp"],
            f"resolve worker {worker} endpoint",
        )
        self._expect(result, f"resolve worker {worker} endpoint")
        port = _loopback_port(result["stdout"])
        if port is None:
            raise FhirWorkersError(f"Docker did not bind worker {worker} to loopback")
        with self._lock:
            resource.update(endpoint=f"http://127.0.0.1:{port}/fhir", status="waiting_for_fhir")
            self._save()

    def _wait_ready(self):
        deadline = time.monotonic() + self.timeout_seconds
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            while True:
                pending = [r for r in self._resources if r["status"] != "ready"]
                if not pending:
                    return
                futures = {
                    executor.submit(_fhir_4_0_1, r["endpoint"]): r for r in pending
                }
                for future in as_completed(futures):
                    try:
                        ready = future.result()
                    except Exception:
                        ready = False
                    if ready:
                        with self._lock:
                            futures[future]["status"] = "ready"
                            self._save()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    workers = [str(r["worker"]) for r in pending if r["status"] != "ready"]
                    with self._lock:
                        for resource in pending:
                            if resource["status"] != "ready":
                                resource["status"] = "readiness_timeout"
                        self._save()
                    raise FhirWorkersError(
                        "FHIR metadata version 4.0.1 was not ready for worker(s): "
                        + ", ".join(workers)
                    )
                time.sleep(min(1, remaining))

    def _cleanup_container(self, container_id, attempt):
        verified = self._docker(
            [
                "docker", "inspect", "--format",
                f"{{{{.Id}}}}\t{{{{.Name}}}}\t{{{{index .Config.Labels \"{OWNER_LABEL}\"}}}}",
                container_id,
            ],
            f"verify owner for {container_id}",
        )
        if verified["return_code"] != 0:
            if _missing(verified["stderr"]):
                return
            self._cleanup_error(attempt, f"Owner check failed for {container_id}: {verified['stderr'].strip()}")
            return
        fields = verified["stdout"].rstrip("\r\n").split("\t")
        if len(fields) != 3 or fields[0] != container_id or fields[2] != self.owner:
            self._cleanup_error(attempt, f"Owner check failed for {container_id}; left untouched")
            return

        name = fields[1].lstrip("/")
        with self._lock:
            resource = next((r for r in self._resources if r["container_id"] == container_id), None)
            if resource is None:
                resource = next((r for r in self._resources if r["name"] == name), None)
                if resource is None:
                    resource = {"worker": None, "container_id": container_id, "name": name,
                                "endpoint": None, "status": "discovered_owned"}
                    self._resources.append(resource)
                else:
                    resource.update(container_id=container_id, status="discovered_owned")
            resource["name"] = name or resource["name"]
            self._save()

        log_path = self.out / "fhir-logs" / (_safe_filename(resource["name"]) + ".log")
        try:
            logs = self._docker(
                ["docker", "logs", "--timestamps", "--tail", "1000", container_id],
                f"collect logs for {container_id}",
                stdout_path=log_path,
            )
            if logs["return_code"] != 0:
                self._cleanup_error(attempt, f"Log collection failed for {container_id}: {logs['stderr'].strip()}")
            else:
                with self._lock:
                    resource["log_file"] = str(log_path)
                    self._save()
        except Exception as error:
            self._cleanup_error(attempt, f"Log collection failed for {container_id}: {error}")

        removed = self._docker(
            ["docker", "rm", "--force", "--volumes", container_id],
            f"remove owned container {container_id}",
        )
        if removed["return_code"] != 0:
            if not _missing(removed["stderr"]):
                self._cleanup_error(attempt, f"Removal failed for {container_id}: {removed['stderr'].strip()}")
            return
        with self._lock:
            resource["status"] = "removed"
            if resource.get("endpoint"):
                self._removed_endpoints.add(resource["endpoint"])
                self._receipt["cleanup"]["removed_endpoints"] = sorted(self._removed_endpoints)
            self._save()

    def _parallel(self, function, values, operation):
        errors = []
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(function, value): value for value in values}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    errors.append(f"worker {futures[future]} {operation}: {error}")
        if errors:
            raise FhirWorkersError("; ".join(errors))

    def _docker(self, argv, purpose, *, stdout_path=None):
        record = {"purpose": purpose, "argv": argv, "started_at": _now(), "status": "running"}
        with self._lock:
            self._receipt["commands"].append(record)
            self._save()
        output = b""
        error_output = b""
        try:
            result = subprocess.run(
                argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=DOCKER_TIMEOUT_SECONDS, check=False,
            )
            output, error_output = result.stdout or b"", result.stderr or b""
            status, return_code = ("succeeded" if result.returncode == 0 else "failed"), result.returncode
        except subprocess.TimeoutExpired as error:
            output, error_output = _bytes(error.output), _bytes(error.stderr)
            status, return_code = "timed_out", None
        except OSError as error:
            error_output = str(error).encode("utf-8", errors="replace")
            status, return_code = "error", None

        stdout = output.decode("utf-8", errors="replace")
        stderr = error_output.decode("utf-8", errors="replace")
        fields = {"stderr": stderr}
        if stdout_path is None:
            fields["stdout"] = stdout
        else:
            try:
                stdout_path.parent.mkdir(parents=True, exist_ok=True)
                stdout_path.write_bytes(output)
                fields.update(
                    stdout_file=str(stdout_path),
                    stdout_bytes=len(output),
                    stdout_sha256=hashlib.sha256(output).hexdigest(),
                )
            except OSError as error:
                fields["stdout_file_error"] = str(error)
        with self._lock:
            record.update(fields, status=status, return_code=return_code, finished_at=_now())
            self._save()
        if status == "timed_out":
            raise FhirWorkersError(f"Docker command timed out ({purpose})")
        if status == "error":
            raise FhirWorkersError(f"Could not run Docker command ({purpose}): {stderr}")
        if "stdout_file_error" in fields:
            raise FhirWorkersError(f"Could not save Docker logs ({purpose}): {fields['stdout_file_error']}")
        return {"return_code": return_code, "stdout": stdout, "stderr": stderr}

    def _expect(self, result, purpose):
        if result["return_code"] != 0:
            detail = result["stderr"].strip() or result["stdout"].strip()
            raise FhirWorkersError(f"Docker could not {purpose}: {detail or result['return_code']}")

    def _set_state(self, state, **fields):
        with self._lock:
            self._state = state
            self._receipt.update(status=state, **fields)
            self._save()

    def _cleanup_error(self, attempt, message):
        with self._lock:
            attempt["errors"].append(message)
            self._receipt["cleanup"]["errors"].append(message)
            self._save()

    def _save(self):
        with self._lock:
            self._receipt["updated_at"] = _now()
            try:
                atomic_json(self.receipt_path, self._receipt)
            except OSError as error:
                if not self._closing:
                    raise
                # An unavailable artifact directory must not prevent owned resource removal.
                message = "Could not persist cleanup receipt: " + str(error)
                if message not in self._cleanup_persistence_errors:
                    self._cleanup_persistence_errors.append(message)


def _loopback_port(output):
    for line in output.splitlines():
        host, separator, text = line.strip().rpartition(":")
        if separator and host == "127.0.0.1" and text.isdigit() and 0 < int(text) < 65536:
            return int(text)
    return None


def _fhir_4_0_1(endpoint):
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/metadata",
        headers={"Accept": "application/fhir+json, application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=READINESS_REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read()) if response.status == 200 else {}
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("resourceType") == "CapabilityStatement"
        and payload.get("fhirVersion") == "4.0.1"
    )


def _missing(message):
    message = message.lower()
    return "no such object" in message or "no such container" in message


def _safe_filename(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.") or "container"


def _bytes(value):
    if value is None:
        return b""
    return value if isinstance(value, bytes) else str(value).encode("utf-8", errors="replace")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
