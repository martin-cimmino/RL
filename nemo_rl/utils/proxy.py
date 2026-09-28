"""Optional Squid forward-proxy keep-alive for air-gapped bare-metal training.

boost_usr_prod compute nodes have no outbound internet access, which breaks anything
that needs it (web-search tool servers, HF Hub downloads, ...). squidward
(https://github.com/igeniusai/squidward) submits a Squid proxy as a SLURM job on a
network-enabled partition (e.g. lrd_all_serial) and exposes standard HTTP_PROXY/
HTTPS_PROXY env vars for it.

Sometimes different partitions cap walltime under a training run's own slurm_time (e.g.
lrd_all_serial has a 4h walltime), so a single squidward job can't cover a whole run.
squidward's own `ForwardProxy` renews the underlying job before it expires, but only
when something actively calls `ensure_ready()` — it has no background timer of its own,
and its `run()`/`arun()` wrappers are meant for wrapping individual short calls, not one
call that blocks for the training run's whole duration.

This module instead starts a lightweight background thread, once, that calls
`ensure_ready()` on a fixed interval for as long as this process (the main training
script, which already runs for the whole job) is alive — proactive renewal with zero
per-request overhead and no call sites to wrap. `node`/`port` stay fixed across
renewals, so the HTTP_PROXY value exported once at startup remains valid throughout.
"""

import atexit
import logging
import threading
from typing import Any, NotRequired, Optional, TypedDict

logger = logging.getLogger(__name__)


class ProxyConfig(TypedDict):
    enabled: bool
    account: str
    node: str
    partition: str
    qos: str
    port: NotRequired[Optional[int]]
    time_limit: str
    renew_before: int
    ping_interval_s: int
    username: NotRequired[Optional[str]]
    password: NotRequired[Optional[str]]
    job_name: str


DEFAULT_PROXY_CONFIG: ProxyConfig = {
    "enabled": False,
    "account": "",
    "node": "login13",
    "partition": "lrd_all_serial",
    "qos": "normal",
    "port": None,
    "time_limit": "04:00:00",
    "renew_before": 900,  # 15 minutes
    "ping_interval_s": 60,
    "username": None,
    "password": None,
    "job_name": "squidward-proxy",
}


class ProxyKeepAlive:
    """Owns a `squidward.ForwardProxy` plus the background renewal thread.

    Constructing this enters the proxy context (submits the job if not already
    reachable, sets HTTP_PROXY/HTTPS_PROXY/NO_PROXY in `os.environ`) and starts the
    keep-alive thread. Call `stop()` to cancel the job and stop the thread; this is
    also registered via `atexit` so a normal process exit cleans up without an
    explicit call.
    """

    def __init__(self, proxy: Any, ping_interval_s: int):
        self._proxy = proxy
        self._ping_interval_s = ping_interval_s
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="squidward-keepalive", daemon=True
        )
        self._thread.start()
        atexit.register(self.stop)

    def _run(self) -> None:
        while not self._stop_event.wait(self._ping_interval_s):
            try:
                self._proxy.ensure_ready()
            except Exception:
                logger.exception(
                    "squidward keep-alive: ensure_ready() failed, will retry"
                )

    def stop(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        self._proxy.__exit__(None, None, None)


def maybe_start_proxy(config: dict) -> Optional[ProxyKeepAlive]:
    """Start the squidward keep-alive if `config["proxy"]["enabled"]` is set.

    Returns None (no-op) if the `proxy` block is absent or `enabled` is false, so
    existing configs with no `proxy` key are unaffected. Call this once, early in
    the main training script, before anything that needs outbound internet access
    (web-search resource servers, HF Hub downloads, ...).
    """
    proxy_config: ProxyConfig = {**DEFAULT_PROXY_CONFIG, **config.get("proxy", {})}
    if not proxy_config["enabled"]:
        return None

    # Deferred import: squidward is only installed under the `proxy` extra (see
    # pyproject.toml) — importing it unconditionally would break every venv that
    # doesn't opt into this extra.
    import squidward

    if not proxy_config["account"]:
        raise ValueError("proxy.enabled=true requires proxy.account to be set.")
    if not proxy_config["password"]:
        raise ValueError(
            "proxy.enabled=true requires proxy.password to be set (e.g. SQUIDWARD_PASSWORD "
            "in .env — see .env.example) — this runs non-interactively inside a SLURM batch "
            "job, so squidward's interactive password prompt is not usable here."
        )

    logger.info(
        "Starting squidward forward-proxy keep-alive (node=%s partition=%s time_limit=%s renew_before=%ds "
        "ping_interval=%ds)",
        proxy_config["node"],
        proxy_config["partition"],
        proxy_config["time_limit"],
        proxy_config["renew_before"],
        proxy_config["ping_interval_s"],
    )

    proxy = squidward.ForwardProxy(
        account=proxy_config["account"],
        node=proxy_config["node"],
        partition=proxy_config["partition"],
        qos=proxy_config["qos"],
        port=proxy_config["port"],
        time_limit=proxy_config["time_limit"],
        renew_before=proxy_config["renew_before"],
        username=proxy_config["username"],
        password=proxy_config["password"],
        job_name=proxy_config["job_name"],
    )
    proxy.__enter__()
    logger.info(
        "squidward forward-proxy up — HTTP_PROXY/HTTPS_PROXY exported for this process."
    )

    return ProxyKeepAlive(proxy, ping_interval_s=proxy_config["ping_interval_s"])
