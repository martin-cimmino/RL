"""Opt-in forward-proxy env vars for jobs that need outbound internet access.

The proxy itself is NOT managed from here: it's a standing outside service. This 
module's only job is the per-experiment opt-in: copy the PROXY_URL already present in
this process's environment into HTTP_PROXY/HTTPS_PROXY/NO_PROXY (+lowercase) ONLY when
`config["proxy"]["enabled"]` is true for this experiment.
"""

import logging
import os
from typing import NotRequired, TypedDict

logger = logging.getLogger(__name__)

# Covers every cluster-internal address (all node IPs here are 10.x.x.x), so
# internal traffic (vLLM policy, Ray, NemoGym) is never accidentally routed
# through the proxy -- only genuine internet-bound requests are affected.
_NO_PROXY = "localhost,127.0.0.1,10.0.0.0/8"


class ProxyConfig(TypedDict):
    enabled: bool
    url: NotRequired[str | None]


DEFAULT_PROXY_CONFIG: ProxyConfig = {
    "enabled": False,
    "url": None,
}


def maybe_enable_proxy(config: dict) -> None:
    """Set HTTP_PROXY/HTTPS_PROXY/NO_PROXY (+lowercase) if `config["proxy"]["enabled"]`.

    No-op if the `proxy` block is absent or `enabled` is false, so existing configs
    with no `proxy` key are unaffected. Call this once, early in the main training
    script, before anything that needs outbound internet access (web-search
    resource servers, HF Hub downloads, ...).
    """
    proxy_config: ProxyConfig = {**DEFAULT_PROXY_CONFIG, **config.get("proxy", {})}
    if not proxy_config["enabled"]:
        return

    proxy_url = proxy_config["url"] or os.environ.get("PROXY_URL")
    if not proxy_url:
        raise ValueError(
            "proxy.enabled=true requires a proxy URL -- set PROXY_URL in .env "
            "(see squid-proxy-kit/print_proxy_env.sh) or proxy.url in this config."
        )

    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ[var] = proxy_url
    os.environ["NO_PROXY"] = _NO_PROXY
    os.environ["no_proxy"] = _NO_PROXY

    logger.info("Forward-proxy enabled for this run (NO_PROXY=%s).", _NO_PROXY)
