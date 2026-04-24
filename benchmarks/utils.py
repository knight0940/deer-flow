"""
Shared utilities for all benchmarks.

Provides:
- Network connectivity detection and auto-recovery
- API quota/rate-limit detection and pause logic
- Unified error classification: network vs quota vs other

All benchmarks should use these instead of local implementations.
"""

import socket
import time


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


def is_quota_error(error: str | Exception) -> bool:
    """Check if error is quota/rate-limit/billing related."""
    msg = str(error).lower()
    patterns = [
        "429", "rate limit", "rate_limit", "quota", "insufficient",
        "billing", "capacity", "overloaded", "too many requests",
        "resource_exhausted", "tokens exhausted", "account limit",
        "spending limit",
    ]
    return any(p in msg for p in patterns)


def is_network_error(error: str | Exception) -> bool:
    """Check if error is caused by network connectivity issues."""
    msg = str(error).lower()
    error_type = type(error).__name__.lower()

    # Check exception type name
    network_types = [
        "connectionerror", "timeouterror", "sslerror",
        "connecttimeout", "readtimeout", "connectionreseterror",
        "gaierror",
    ]
    if any(t in error_type for t in network_types):
        return True

    # Check error message
    patterns = [
        "connection error", "connection refused", "connection reset",
        "timed out", "timeout", "name resolution", "name or service",
        "network unreachable", "no route to host", "broken pipe",
        "ssl error", "certificate verify",
        "could not resolve", "temporary failure",
        "nodename nor servname", "getaddrinfo failed",
        "couldn't resolve host",
    ]
    if any(p in msg for p in patterns):
        return True

    return False


# ---------------------------------------------------------------------------
# Network connectivity check
# ---------------------------------------------------------------------------


def check_network(host: str = "8.8.8.8", port: int = 53, timeout: int = 3) -> bool:
    """Check network connectivity by connecting to a DNS server."""
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except (socket.timeout, OSError):
        return False


def wait_for_network(poll_interval: int = 15):
    """Block until network is restored. Auto-retries with polling."""
    print("  Waiting for network recovery...", flush=True)
    while not check_network():
        print(f"  No network. Retrying in {poll_interval}s...", flush=True)
        time.sleep(poll_interval)
    print("  Network restored!", flush=True)


# ---------------------------------------------------------------------------
# Quota recovery
# ---------------------------------------------------------------------------


def wait_for_quota_recovery(poll_interval: int = 60):
    """Block until quota/rate-limit recovers.

    For rate limits (429), these usually recover in seconds to minutes.
    For quota exhaustion, the user needs to top up manually.
    We just keep retrying periodically.
    """
    print(f"  Will retry in {poll_interval}s...", flush=True)
    time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Unified pause handler
# ---------------------------------------------------------------------------


def handle_pauseable_error(
    error: str | Exception,
    context: str = "",
) -> str:
    """Handle network/quota errors with pause-and-retry logic.

    Returns:
        "network" — network error, waited for recovery, caller should retry
        "quota"   — quota/rate-limit error, waited, caller should retry
        "other"   — non-retryable error, caller should record and move on

    Usage in benchmark main loop:
        while True:
            result = run_one_task(...)
            if result.error:
                action = handle_pauseable_error(result.error, context=task_id)
                if action != "other":
                    continue  # retry same task
            break  # success or non-retryable error
    """
    if is_network_error(error):
        msg = str(error)[:100]
        print(f"\n  NETWORK ERROR ({context}): {msg}", flush=True)
        save_hint = "Checkpoint saved. "
        print(f"  {save_hint}Pausing until network recovers...", flush=True)
        wait_for_network()
        return "network"

    if is_quota_error(error):
        msg = str(error)[:100]
        print(f"\n  QUOTA/RATE LIMIT ({context}): {msg}", flush=True)
        print("  Pausing. Please resolve the quota issue.", flush=True)
        wait_for_quota_recovery()
        return "quota"

    return "other"
