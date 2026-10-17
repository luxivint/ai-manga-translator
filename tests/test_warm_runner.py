"""Warm in-process worker mode (WORKER_INPROCESS) with fake jobs only: real child process, fake job script.
No database, no R2, no gateway, no models."""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from batch_process import WarmRunner, env_fingerprint  # noqa: E402

FAKE_JOB = textwrap.dedent('''
    import os, sys, time
    STATE = {"calls": 0, "loaded": False, "resets": 0, "unloads": 0}

    def reset_job_state():
        STATE["resets"] += 1

    def loaded_models():
        return ["fake-model"] if STATE["loaded"] else []

    def unload_models():
        STATE["unloads"] += 1
        released, STATE["loaded"] = loaded_models(), False
        open(os.environ["FAKE_MARK"], "a").write("unloaded\\n")
        return released

    def main():
        STATE["calls"] += 1
        first_load = not STATE["loaded"]
        STATE["loaded"] = True
        mode = sys.argv[1] if len(sys.argv) > 1 else "ok"
        print("job=%s pid=%s calls=%s resets=%s first_load=%s knob=%s cwd=%s" % (
            os.environ.get("PIPELINE_JOB_ID"), os.getpid(), STATE["calls"], STATE["resets"], first_load,
            os.environ.get("FAKE_KNOB"), os.getcwd()), flush=True)
        os.write(2, b"native-stderr-line\\n")
        if mode == "fail":
            return 3
        if mode == "raise":
            raise RuntimeError("boom")
        if mode == "exit":
            sys.exit(4)
        if mode == "crash":
            os._exit(9)
        if mode == "hang":
            time.sleep(60)
        return 0
''')


class WarmBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.script = self.dir / "fake_job.py"
        self.script.write_text(FAKE_JOB, encoding="utf-8")
        self.mark = self.dir / "mark.txt"
        self.runner = WarmRunner(ROOT, idle_seconds=0, recycle_jobs=0, poll=0.3, child_log=self.dir / "child.log")
        self.addCleanup(self.runner.close)

    def env(self, job="j1", **extra):
        env = os.environ.copy()
        env.update({"PIPELINE_JOB_ID": job, "FAKE_MARK": str(self.mark), "FAKE_KNOB": "a"})
        env.update(extra)
        return env

    def run_job(self, mode="ok", job="j1", **extra):
        log = self.dir / f"{job}.log"
        self.runner.run([sys.executable, str(self.script), mode], self.dir, self.env(job, **extra), log, stall_seconds=30)
        return log.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def field(text, name):
        return next(p.split("=", 1)[1] for p in text.split() if p.startswith(name + "="))


class WarmRunnerTests(WarmBase):
    def test_second_job_reuses_the_process_and_module_state(self):
        first = self.run_job(job="j1")
        second = self.run_job(job="j2")
        self.assertEqual(self.field(first, "pid"), self.field(second, "pid"))
        self.assertEqual(self.field(first, "calls"), "1")
        self.assertEqual(self.field(second, "calls"), "2")
        self.assertEqual(self.field(first, "first_load"), "True")
        self.assertEqual(self.field(second, "first_load"), "False")  # models were still loaded
        self.assertEqual(self.field(second, "resets"), "2")  # per-job state reset hook ran each time
        self.assertEqual(self.runner.spawned, 1)

    def test_each_job_gets_its_own_env_cwd_and_log(self):
        a = self.run_job(job="j1")
        sub = self.dir / "elsewhere"
        sub.mkdir()
        log = self.dir / "j2.log"
        self.runner.run([sys.executable, str(self.script)], sub, self.env("j2"), log, stall_seconds=30)
        b = log.read_text(encoding="utf-8")
        self.assertIn("job=j1", a)
        self.assertIn("job=j2", b)
        self.assertNotIn("job=j1", b)
        self.assertEqual(Path(self.field(b, "cwd")).resolve(), sub.resolve())
        self.assertIn("native-stderr-line", a)  # fd-level stderr lands in the job log, like a subprocess

    def test_changed_knob_starts_a_fresh_child(self):
        a = self.run_job(job="j1", FAKE_KNOB="a")
        b = self.run_job(job="j2", FAKE_KNOB="b")
        self.assertNotEqual(self.field(a, "pid"), self.field(b, "pid"))
        self.assertEqual(self.field(b, "knob"), "b")
        self.assertEqual(self.runner.spawned, 2)

    def test_nonzero_exit_raises_with_code_and_recycles(self):
        a = self.run_job(job="j1")
        with self.assertRaises(subprocess.CalledProcessError) as ctx:
            self.run_job("fail", job="j2")
        self.assertEqual(ctx.exception.returncode, 3)
        self.assertIn("job=j2", ctx.exception.output)
        b = self.run_job(job="j3")
        self.assertNotEqual(self.field(a, "pid"), self.field(b, "pid"))  # failed job => clean child next time

    def test_exception_in_job_is_a_failed_job_not_a_dead_worker(self):
        with self.assertRaises(subprocess.CalledProcessError) as ctx:
            self.run_job("raise", job="j1")
        self.assertEqual(ctx.exception.returncode, 1)
        self.assertIn("RuntimeError: boom", ctx.exception.output)
        self.assertIn("job=j2", self.run_job(job="j2"))

    def test_sys_exit_code_is_preserved(self):
        with self.assertRaises(subprocess.CalledProcessError) as ctx:
            self.run_job("exit", job="j1")
        self.assertEqual(ctx.exception.returncode, 4)

    def test_native_crash_surfaces_exit_code_and_next_job_respawns(self):
        with self.assertRaises(subprocess.CalledProcessError) as ctx:
            self.run_job("crash", job="j1")
        self.assertEqual(ctx.exception.returncode, 9)
        self.assertIn("job=j2", self.run_job(job="j2"))
        self.assertEqual(self.runner.spawned, 2)

    def test_stall_kills_child_and_next_job_recovers(self):
        log = self.dir / "j1.log"
        started = time.monotonic()
        with self.assertRaises(RuntimeError) as ctx:
            self.runner.run([sys.executable, str(self.script), "hang"], self.dir, self.env("j1"), log, stall_seconds=1)
        self.assertIn("no output", str(ctx.exception))
        self.assertLess(time.monotonic() - started, 20)
        self.assertFalse(self.runner.alive())
        self.assertIn("job=j2", self.run_job(job="j2"))

    def test_cancel_from_tick_kills_child(self):
        class Cancelled(RuntimeError):
            pass

        def tick():
            raise Cancelled("Job cancelled by admin")

        with self.assertRaises(Cancelled):
            self.runner.run([sys.executable, str(self.script), "hang"], self.dir, self.env("j1"),
                            self.dir / "j1.log", tick, stall_seconds=30)
        self.assertFalse(self.runner.alive())
        self.assertIn("job=j2", self.run_job(job="j2"))

    def test_recycle_after_n_jobs(self):
        self.runner.recycle_jobs = 2
        pids = [self.field(self.run_job(job=f"j{i}"), "pid") for i in range(1, 4)]
        self.assertEqual(pids[0], pids[1])
        self.assertNotEqual(pids[1], pids[2])

    def test_recycle_on_rss_threshold(self):
        self.runner.recycle_rss_mb = 1  # any real process is above 1 MB
        a = self.field(self.run_job(job="j1"), "pid")
        b = self.field(self.run_job(job="j2"), "pid")
        self.assertNotEqual(a, b)

    def test_idle_unload_releases_models_but_keeps_child_for_next_job(self):
        self.runner.idle_seconds = 1
        first = self.run_job(job="j1")
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not self.mark.exists():
            time.sleep(0.2)
        self.assertTrue(self.mark.exists(), "idle watchdog never unloaded the models")
        second = self.run_job(job="j2")
        self.assertEqual(self.field(first, "pid"), self.field(second, "pid"))
        self.assertEqual(self.field(second, "first_load"), "True")  # reloaded after the unload

    def test_idle_child_exits_after_second_idle_window_and_next_job_respawns(self):
        self.runner.idle_seconds = 1
        first = self.run_job(job="j1")
        proc = self.runner.process
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.2)
        self.assertIsNotNone(proc.poll(), "idle child never exited (GPU contexts + RAM would stay held)")
        second = self.run_job(job="j2")
        self.assertNotEqual(self.field(first, "pid"), self.field(second, "pid"))
        self.assertEqual(self.runner.spawned, 2)

    def test_exiting_announcement_before_next_job_respawns_cleanly(self):
        self.run_job(job="j1")
        self.runner.events.put({"event": "exiting"})  # idle exit announced while we were not looking
        self.assertIn("job=j2", self.run_job(job="j2"))
        self.assertEqual(self.runner.spawned, 2)

    def test_explicit_unload_request(self):
        self.run_job(job="j1")
        self.runner.unload()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not self.mark.exists():
            time.sleep(0.2)
        self.assertTrue(self.mark.exists())

    def test_close_stops_the_child(self):
        self.run_job(job="j1")
        proc = self.runner.process
        self.runner.close()
        self.assertIsNotNone(proc.poll())
        self.runner.close()  # idempotent


FAKE_RETIRING_CHILD = textwrap.dedent('''
    import json, os, sys
    marker = os.environ["FAKE_MARKER"]
    out = os.fdopen(os.dup(1), "w")
    def send(ev):
        out.write(json.dumps(ev) + chr(10)); out.flush()
    send({"event": "ready"})
    for line in sys.stdin:
        req = json.loads(line)
        if req["op"] == "quit":
            break
        if not os.path.exists(marker):  # first child: the idle exit wins the race against the request
            open(marker, "w").close()
            send({"event": "exiting"})
            sys.exit(0)
        send({"event": "done", "code": 0})
''')


class IdleExitRaceTests(unittest.TestCase):
    def test_request_lost_to_idle_exit_is_resent_to_a_fresh_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = Path(tmp) / "child.py"
            child.write_text(FAKE_RETIRING_CHILD, encoding="utf-8")
            marker = Path(tmp) / "marker"
            runner = WarmRunner(ROOT, idle_seconds=0, recycle_jobs=0, poll=0.3, runner=child)
            self.addCleanup(runner.close)
            with mock.patch.dict(os.environ, {"FAKE_MARKER": str(marker)}):
                runner.run([sys.executable, "x.py"], tmp, {"A": "1"}, Path(tmp) / "log.txt", stall_seconds=30)
            self.assertEqual(runner.spawned, 2)
            self.assertEqual(runner.last.get("code"), 0)


class RealLocalBatchSmokeTests(unittest.TestCase):
    """The real scripts/local_batch.py inside the warm child, on an EMPTY input folder: no image, so no model and no
    network call. Proves the import, the argv/env/cwd hand-over, the exit code and the state hooks."""

    def test_local_batch_runs_twice_in_one_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "in").mkdir()
            runner = WarmRunner(ROOT, idle_seconds=0, recycle_jobs=0, poll=0.5, child_log=tmp / "child.log")
            self.addCleanup(runner.close)
            codes = []
            for n in (1, 2):
                log = tmp / f"j{n}.log"
                cmd = [sys.executable, str(ROOT / "scripts" / "local_batch.py"), "--input", str(tmp / "in"),
                       "--output", str(tmp / "out"), "--no-safe-recut", "--no-gpu"]
                env = os.environ.copy()
                env.update({"PIPELINE_JOB_ID": f"j{n}", "GATEWAY_QUOTA_FILE": str(tmp / "no-quota.json"),
                            "OPENAI_API_KEY": "unused-test-key"})
                try:
                    runner.run(cmd, ROOT, env, log, stall_seconds=120)
                    codes.append(0)
                except subprocess.CalledProcessError as exc:
                    codes.append(exc.returncode)
                    self.assertIn("No input images found", exc.output)
        self.assertEqual(codes, [3, 3])  # EXIT_VALIDATION_FAILED, exactly what the subprocess would exit with
        self.assertEqual(runner.spawned, 2)  # a failed job retires the child; the exit code survived the trip


class FingerprintTests(unittest.TestCase):
    def test_ignores_per_chapter_keys_only(self):
        base = {"A": "1", "PIPELINE_JOB_ID": "x", "TRANSLATION_RAW_PREFIX": "p1"}
        other_job = {"A": "1", "PIPELINE_JOB_ID": "y", "TRANSLATION_RAW_PREFIX": "p2"}
        other_knob = {"A": "2", "PIPELINE_JOB_ID": "x", "TRANSLATION_RAW_PREFIX": "p1"}
        self.assertEqual(env_fingerprint(base), env_fingerprint(other_job))
        self.assertNotEqual(env_fingerprint(base), env_fingerprint(other_knob))

    def test_dotenv_change_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            dotenv = Path(tmp) / ".env"
            dotenv.write_text("X=1\n")
            one = env_fingerprint({"A": "1"}, dotenv)
            dotenv.write_text("X=2\n")
            self.assertNotEqual(one, env_fingerprint({"A": "1"}, dotenv))
            self.assertEqual(env_fingerprint({"A": "1"}, Path(tmp) / "missing"), env_fingerprint({"A": "1"}))


class WorkerWiringTests(unittest.TestCase):
    def setUp(self):
        import worker
        self.worker = worker
        self.addCleanup(setattr, worker, "_WARM", worker._WARM)
        worker._WARM = None

    def _tmp(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        return holder.name

    def test_default_is_classic_subprocess(self):
        with mock.patch.dict(os.environ, {"WORKER_INPROCESS": "false"}), \
                mock.patch.object(self.worker, "run_batch_process") as classic:
            self.worker.run_translation_command(["python", "x.py"], {}, "job1")
        classic.assert_called_once()
        self.assertIsNone(self.worker._WARM)

    def test_knob_on_routes_to_warm_runner(self):
        env = {"WORKER_INPROCESS": "true", "WORKER_RECYCLE_JOBS": "7", "WORKER_MODEL_IDLE_SECONDS": "42"}
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(self.worker, "WarmRunner") as cls, \
                mock.patch.object(self.worker, "run_batch_process") as classic:
            self.worker.run_translation_command(["python", "x.py"], {"A": "1"}, "job1")
        classic.assert_not_called()
        kwargs = cls.call_args.kwargs
        self.assertEqual((kwargs["recycle_jobs"], kwargs["idle_seconds"]), (7, 42))
        cls.return_value.run.assert_called_once()

    def test_dry_run_processes_fake_queue_without_database(self):
        script = Path(self._tmp()) / "fake.py"
        script.write_text("import sys\ndef main():\n    print('dry ok')\n    return 0 if sys.argv[1:] != ['bad'] else 5\n"
                          "if __name__ == '__main__':\n    raise SystemExit(main())\n")
        queue = Path(self._tmp())
        (queue / "a.json").write_text(json.dumps({"script": str(script), "args": []}))
        (queue / "b.json").write_text(json.dumps({"script": str(script), "args": ["bad"]}))
        drain = queue / "stop.drain"
        env = os.environ.copy()
        env.update({"WORKER_DRY_RUN": str(queue), "WORKER_IDLE_SECONDS": "1", "WORKER_DRAIN_FILE": str(drain),
                    "DATABASE_URL": "postgres://never-used/none", "WORKER_INPROCESS": "false"})
        proc = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "worker.py")], cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and (queue / "a.json").exists() or (queue / "b.json").exists():
                time.sleep(0.5)
            drain.write_text("x")
            out, _ = proc.communicate(timeout=30)
        finally:
            if proc.poll() is None:
                proc.kill()
        self.assertTrue((queue / "a.done").exists(), out.decode(errors="replace"))
        self.assertTrue((queue / "b.failed").exists(), out.decode(errors="replace"))


if __name__ == "__main__":
    unittest.main()
