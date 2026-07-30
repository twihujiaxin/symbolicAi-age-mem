# -*- coding: utf-8 -*-
"""For distributed training with multiple process groups."""
import ipaddress
import socket
import subprocess
from datetime import timedelta
from typing import Any, Optional, Union

import torch
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)


def _detect_default_network_interface() -> str:
    """Return the network interface used for the default route.

    NCCL falls back to socket communication when IB is disabled, but its
    auto-detection may pick a non-existent interface (e.g. eth0) on machines
    that use names like ens7f0.  This helper reads the actual default route
    so the caller can set NCCL_SOCKET_IFNAME before init_process_group.
    """
    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.split()
            if "dev" in parts:
                return parts[parts.index("dev") + 1]
    except Exception:
        pass

    # Fallback: first non-loopback interface with an IPv4 address
    try:
        import psutil

        for iface, addrs in psutil.net_if_addrs().items():
            if iface == "lo":
                continue
            for addr in addrs:
                if addr.family == socket.AF_INET and not addr.address.startswith("127."):
                    return iface
    except Exception:
        pass

    return ""


def is_ipv6_address(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
        return isinstance(ip, ipaddress.IPv6Address)
    except ValueError:
        return False


def get_available_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def is_port_available(port: int, host="127.0.0.1") -> bool:
    with socket.socket() as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def init_process_group(
    host: str,
    port: int,
    group_name: str,
    backend: Union[str, Backend] = "nccl",
    timeout: Optional[float] = None,
    world_size: int = -1,
    rank: int = -1,
    pg_options: Optional[Any] = None,
    device_id: Optional[torch.device] = None,
):
    import os
    
    assert backend == "nccl", "Only nccl backend is supported for now."

    # Set NCCL env vars to avoid P2P access issues.
    # Must be set before creating the process group.
    if "NCCL_P2P_DISABLE" not in os.environ:
        os.environ["NCCL_P2P_DISABLE"] = "1"
    if "NCCL_IB_DISABLE" not in os.environ:
        os.environ["NCCL_IB_DISABLE"] = "1"
    # Disable shared memory transport; fall back to socket communication.
    if "NCCL_SHM_DISABLE" not in os.environ:
        os.environ["NCCL_SHM_DISABLE"] = "1"
    # After disabling IB, NCCL falls back to sockets but may auto-detect a
    # non-existent interface (e.g. eth0).  Pin it to the actual default route
    # interface to prevent "bind: No such device" errors.
    if "NCCL_SOCKET_IFNAME" not in os.environ:
        iface = _detect_default_network_interface()
        if iface:
            os.environ["NCCL_SOCKET_IFNAME"] = iface
    if "GLOO_SOCKET_IFNAME" not in os.environ and "NCCL_SOCKET_IFNAME" in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"]

    from torch.distributed.distributed_c10d import is_nccl_available

    assert is_nccl_available()

    init_method = (
        f"tcp://[{host}]:{port}" if is_ipv6_address(ip_str=host) else f"tcp://{host}:{port}"
    )

    backend = Backend(backend)

    if timeout is None:
        timeout = default_pg_timeout
    else:
        timeout = timedelta(seconds=timeout)

    # backward compatible API
    store, rank, world_size = next(rendezvous(init_method, rank, world_size, timeout=timeout))
    store.set_timeout(timeout)

    # Use a PrefixStore to avoid accidental overrides of keys used by
    # different systems (e.g. RPC) in case the store is multi-tenant.
    prefix_store = PrefixStore(group_name, store)

    pg_options_param_name = "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    pg, _ = _new_process_group_helper(
        group_size=world_size,
        group_rank=rank,
        global_ranks_in_group=[],
        backend=backend,
        store=prefix_store,
        group_name=group_name,
        timeout=timeout,
        device_id=device_id,
        **{pg_options_param_name: pg_options},
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg
