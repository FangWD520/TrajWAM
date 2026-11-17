"""Start and clean up only the CARLA process owned by this worker."""
import atexit
import os
from pathlib import Path
import shlex
import signal
import subprocess


def stop_carla(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    except ProcessLookupError:
        pass


def start_carla(root, port, gpu):
    script = Path(root).resolve() / "CarlaUE4.sh"
    if not script.is_file():
        raise FileNotFoundError(script)
    cmd = [str(script), "-RenderOffScreen", "-nosound",
           f"-carla-rpc-port={port}", f"-graphicsadapter={gpu}"]
    if os.geteuid() == 0:
        import pwd
        user = os.environ.get("CARLA_USER")
        if not user or pwd.getpwnam(user).pw_uid == 0:
            raise RuntimeError("Run as a non-root user, or set CARLA_USER to a non-root account")
        cmd = ["runuser", "-u", user, "--", *cmd]
    print("Starting:", shlex.join(cmd), flush=True)
    process = subprocess.Popen(cmd, start_new_session=True)
    atexit.register(stop_carla, process)
    return process
