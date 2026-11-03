"""テスト中のネットワーク接続を禁止する番兵。"""

import socket

import pytest


def _forbidden(*args, **kwargs):
    raise RuntimeError("テスト中のネットワーク接続は禁止")


@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch):
    """integration マーク付きのテストでは差し替えない。"""
    if request.node.get_closest_marker("integration") is not None:
        return
    monkeypatch.setattr(socket.socket, "connect", _forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)
