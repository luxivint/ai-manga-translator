import importlib.util
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/container_entrypoint.py"
spec = importlib.util.spec_from_file_location("container_entrypoint", SCRIPT)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class ContainerTests(unittest.TestCase):
    def test_restart_clears_old_drain_and_generates_unique_server_identity(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"WORKER_INSTANCE_ID": "", "WORKER_DRAIN_FILE": ""}), patch.object(entry, "ROOT", Path(tmp)), patch.object(entry.socket, "gethostname", return_value="container-abc"):
            path = entry.configure_runtime()
            self.assertEqual(os.environ["WORKER_INSTANCE_ID"], "translator-server-container-abc")
            entry.request_drain(path)
            self.assertTrue(path.exists())
            entry.configure_runtime()
            self.assertFalse(path.exists())

    def test_stop_signal_only_requests_drain_without_killing_active_job(self):
        worker = Mock()
        worker.main.return_value = 0
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"WORKER_INSTANCE_ID": "translator-server-test", "WORKER_DRAIN_FILE": str(Path(tmp) / "stop")}), patch.dict(sys.modules, {"worker": worker}), patch.object(entry.signal, "signal") as handlers:
            self.assertEqual(entry.main(), 0)
            callback = next(c.args[1] for c in handlers.call_args_list if c.args[0] == signal.SIGTERM)
            callback(signal.SIGTERM, None)
            self.assertTrue((Path(tmp) / "stop").exists())
            worker.main.assert_called_once()


if __name__ == "__main__":
    unittest.main()
