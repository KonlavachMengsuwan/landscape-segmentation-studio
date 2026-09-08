"""Process identity and loopback restart checks; never signals a real process."""
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import launch


def record():
    return dict(pid=12345, instance="a" * 32, root=str(launch.ROOT), workspace_id=launch.WORKSPACE_ID)


def command():
    return f"python {launch.SCRIPT} serve --studio-instance={'a' * 32} --workspace-id={launch.WORKSPACE_ID}"


def test_accepts_only_complete_instance_and_workspace_markers(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, command(), ""))
    assert launch.owned_process(record()) is True


@pytest.mark.parametrize("changed", [lambda s: s.replace("a" * 32, "b" * 32),
                                      lambda s: s.replace("a" * 32, "a" * 33),
                                      lambda s: s.replace(launch.WORKSPACE_ID, "0" * 64),
                                      lambda s: s.replace(str(launch.SCRIPT), "/another/app.py"),
                                      lambda s: "python unrelated.py"])
def test_rejects_other_processes(monkeypatch, changed):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, changed(command()), ""))
    assert launch.owned_process(record()) is False


def test_rejects_other_workspace_before_process_lookup(monkeypatch):
    def should_not_run(*args, **kwargs):
        pytest.fail("A mismatched workspace must not be inspected or signaled")
    monkeypatch.setattr(subprocess, "run", should_not_run)
    item = record()
    item["root"] = "/somewhere/else"
    assert launch.owned_process(item) is False


def test_live_listener_is_rejected_and_left_running(monkeypatch):
    import socket
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', 0)); listener.listen(1)
        monkeypatch.setattr(launch, 'PORT', listener.getsockname()[1])
        with pytest.raises(RuntimeError, match='already in use'):
            launch.check_port_available()
        with socket.create_connection(listener.getsockname(),timeout=2) as client:
            accepted,_=listener.accept()
            with accepted:
                accepted.sendall(b'alive')
                assert client.recv(5)==b'alive'


def test_immediate_restart_accepts_closed_connection_time_wait(monkeypatch):
    import socket
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', 0)); listener.listen(1)
        address=listener.getsockname()
        monkeypatch.setattr(launch, 'PORT', address[1])
        with socket.create_connection(address, timeout=2) as client:
            accepted,_=listener.accept()
            with accepted:
                accepted.shutdown(socket.SHUT_WR)
                assert client.recv(1)==b''
    launch.check_port_available()
