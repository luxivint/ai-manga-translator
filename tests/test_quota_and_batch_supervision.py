import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from modules.utils import gateway
from batch_process import run_batch_process

class QuotaTests(unittest.TestCase):
    def test_nested_usage_limit_stops_retry_and_blocks_following_calls(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'GATEWAY_QUOTA_FILE':tmp+'/quota.json'}):
            response=Mock(status_code=503,text='[429]: The usage limit has been reached (reset after 29m 30s)')
            post=Mock(return_value=response)
            for _ in range(2):
                with self.assertRaises(gateway.GatewayQuotaError):
                    gateway.resilient_post('http://test',headers={},data='',timeout=1,post=post)
            self.assertEqual(post.call_count,1)
            self.assertGreater(gateway.quota_remaining(),1700)

    def test_transient_429_without_account_quota_uses_existing_retries(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'GATEWAY_QUOTA_FILE':tmp+'/quota.json'}):
            bad=Mock(status_code=429,text='too many requests')
            good=Mock(status_code=200);good.json.return_value={}
            post=Mock(side_effect=[bad,good])
            gateway.resilient_post('http://test',headers={},data='',timeout=1,post=post,backoffs=(0,),sleep=lambda _:None,breaker=gateway.CircuitBreaker())
            self.assertEqual(post.call_count,2)

class WorkerQuotaTests(unittest.TestCase):
    def test_waiting_for_quota_does_not_claim_or_consume_retry(self):
        import worker
        with patch.object(worker,'quota_remaining',return_value=100), patch.object(worker,'quota_path') as path:
            path.return_value.exists.return_value=False
            conn=Mock()
            self.assertIsNone(worker.claim_job(conn));conn.cursor.assert_not_called()

    def test_successful_probe_releases_deferred_jobs_before_old_reset_time(self):
        import worker,json,time
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'GATEWAY_QUOTA_FILE':tmp+'/quota.json'}):
            path=Path(tmp)/'quota.json';path.write_text(json.dumps({'until':time.time()+1800}))
            conn=MagicMock();conn.cursor.return_value.__enter__.return_value.fetchone.return_value={'owned':True}
            response=Mock(status_code=200);response.json.return_value={'choices':[{'message':{'content':'OK'}}]}
            with patch('requests.post',return_value=response) as post:
                worker.probe_quota(conn)
            self.assertFalse(path.exists());self.assertEqual(post.call_count,1)
            sql=' '.join(str(c) for c in conn.cursor.return_value.__enter__.return_value.execute.call_args_list)
            self.assertIn('next_retry_at=NULL',sql)

    def test_probe_failure_waits_and_does_not_repeat_for_every_worker(self):
        import worker,json,time
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'GATEWAY_QUOTA_FILE':tmp+'/quota.json'}):
            path=Path(tmp)/'quota.json';path.write_text(json.dumps({'until':time.time()-1}))
            conn=MagicMock();conn.cursor.return_value.__enter__.return_value.fetchone.return_value={'owned':True}
            with patch('requests.post',return_value=Mock(status_code=429)) as post:
                worker.probe_quota(conn);worker.probe_quota(conn)
            self.assertEqual(post.call_count,1);self.assertGreater(worker.quota_remaining(),50)

    def test_other_worker_owns_job_so_cleanup_cannot_requeue_it(self):
        import worker
        from unittest.mock import MagicMock
        conn=MagicMock();conn.cursor.return_value.__enter__.return_value.fetchone.return_value={'owned':False}
        with patch.object(worker,'mark_failed_or_retry') as fail:
            self.assertFalse(worker.recover_unowned_job(conn,'test-owner','restart'))
            fail.assert_not_called()

class SupervisionTests(unittest.TestCase):
    def test_progress_follows_batch_stages(self):
        from batch_process import batch_progress
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'log'
            path.write_text('R2 download: raw/test\n[RECUT] protect 2/10')
            self.assertEqual(batch_progress(path),15)
            path.write_text('R2 upload: output\nR2 uploaded: 12 images')
            self.assertEqual(batch_progress(path),95)

    def test_success_and_nonzero_exit_preserve_diagnostic_log(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'batch.log'
            run_batch_process([sys.executable,'-c','print("done")'],ROOT,os.environ.copy(),path)
            self.assertIn('done',path.read_text())
            with self.assertRaises(subprocess.CalledProcessError) as ctx:
                run_batch_process([sys.executable,'-c','print("FAILED: safe cut");exit(3)'],ROOT,os.environ.copy(),path)
            self.assertIn('FAILED: safe cut',ctx.exception.output)

    def test_idle_process_is_stopped_and_heartbeat_called(self):
        import subprocess
        process=Mock();process.wait.side_effect=[subprocess.TimeoutExpired('x',5),subprocess.TimeoutExpired('x',5)]
        with tempfile.TemporaryDirectory() as tmp, patch('batch_process.subprocess.Popen',return_value=process),patch('batch_process.time.monotonic',side_effect=[0,0,5,601]),patch('batch_process.stop_tree') as stop:
            tick=Mock()
            with self.assertRaisesRegex(RuntimeError,'no output'):
                run_batch_process(['x'],ROOT,{},Path(tmp)/'log',tick)
            self.assertEqual(tick.call_count,2);stop.assert_called_once_with(process)
