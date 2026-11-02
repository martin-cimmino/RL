"""[CUSTOM] Enable proxy connection via env variable passthrough.

This module's only job is, when explicitly enabled in config, to read the proxy's URL
from the environment (PROXY_URL env var) and export it as HTTP_PROXY/HTTPS_PROXY for
this process -- nemo_rl's own actor creation propagates it to every Ray actor's
runtime_env from there.
"""

import logging
import os
from typing import NotRequired, TypedDict

logger = logging.getLogger(__name__)


class ProxyConfig(TypedDict):
    enabled: bool
    url_env_var: NotRequired[str]


DEFAULT_PROXY_CONFIG: ProxyConfig = {
    "enabled": False,
    "url_env_var": "PROXY_URL",
}


def maybe_enable_proxy(config: dict) -> None:
    """Export HTTP_PROXY/HTTPS_PROXY from the configured env var, if `config["proxy"]["enabled"]`.

    No-op if the `proxy` block is absent or `enabled` is false. Call this once, early
    in the main training script, before anything that needs outbound internet access.
    """
    proxy_config: ProxyConfig = {**DEFAULT_PROXY_CONFIG, **config.get("proxy", {})}
    if not proxy_config["enabled"]:
        return

    url_env_var = proxy_config["url_env_var"]
    proxy_url = os.environ.get(url_env_var)
    if not proxy_url:
        raise ValueError(
            f"proxy.enabled=true requires {url_env_var} to be set (e.g. in .env, see "
            ".env.example) -- the forward proxy itself is a standing job managed outside this "
            "repo; this only needs its address."
        )

    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    logger.info("Forward-proxy enabled: HTTP_PROXY/HTTPS_PROXY exported from %s.", url_env_var)
