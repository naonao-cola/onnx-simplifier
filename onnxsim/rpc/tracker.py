"""A small tracker: servers register under a key, clients ask for a server by key.

Like TVM's tracker in spirit (device keys instead of addresses), but deliberately simple: no
priorities and no exclusive locking. ``request`` hands out matching servers round-robin; a server
disappears from the registry as soon as its registration connection closes.
"""

from __future__ import annotations

import socketserver
import threading
from typing import Any, Dict, List, Optional, Tuple

from . import _protocol as proto


class _TrackerHandler(socketserver.BaseRequestHandler):
    server: "Tracker"

    def handle(self) -> None:
        sock = self.request
        registered: Optional[Tuple[str, Tuple[str, int]]] = None
        try:
            header, _ = proto.recv_message(sock)
            op = header.get("op")
            if op == "register":
                entry = (
                    str(header["key"]),
                    (str(header["addr"][0]), int(header["addr"][1])),
                )
                registered = entry
                self.server.add(*entry)
                proto.send_message(sock, {"ok": True})
                try:  # hold the registration open; liveness = this connection staying up
                    while sock.recv(1):
                        pass
                except OSError:
                    pass
            elif op == "request":
                addr = self.server.pick(str(header["key"]))
                if addr is None:
                    proto.send_message(
                        sock,
                        {"ok": False, "error": f"no server with key {header['key']!r}"},
                    )
                else:
                    proto.send_message(sock, {"ok": True, "addr": list(addr)})
            elif op == "summary":
                proto.send_message(sock, {"ok": True, "servers": self.server.summary()})
            else:
                proto.send_message(
                    sock, {"ok": False, "error": f"unknown operation {op!r}"}
                )
        except (ConnectionError, proto.RPCError, OSError):
            pass
        finally:
            if registered is not None:
                self.server.remove(*registered)


class Tracker(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        super().__init__((host, port), _TrackerHandler)
        self._lock = threading.Lock()
        self._servers: Dict[str, List[Tuple[str, int]]] = {}
        self._cursor: Dict[str, int] = {}

    @property
    def address(self) -> Tuple[str, int]:
        return self.server_address[0], self.server_address[1]

    def add(self, key: str, addr: Tuple[str, int]) -> None:
        with self._lock:
            servers = self._servers.setdefault(key, [])
            if addr not in servers:
                servers.append(addr)

    def remove(self, key: str, addr: Tuple[str, int]) -> None:
        with self._lock:
            servers = self._servers.get(key, [])
            if addr in servers:
                servers.remove(addr)
            if not servers:
                self._servers.pop(key, None)

    def pick(self, key: str) -> Optional[Tuple[str, int]]:
        with self._lock:
            servers = self._servers.get(key)
            if not servers:
                return None
            index = self._cursor.get(key, 0) % len(servers)
            self._cursor[key] = index + 1
            return servers[index]

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                key: [list(a) for a in addrs] for key, addrs in self._servers.items()
            }

    def start(self) -> "Tracker":
        threading.Thread(target=self.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self.shutdown()
        self.server_close()
