import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from modules.utils import stage_trace as trace
from batch_process import LogActivity, WarmRunner, env_fingerprint
from scripts.job_timing_report import collect

@contextlib.contextmanager
def capture():
    output=io.StringIO()
    with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):yield output

class TraceTests(unittest.TestCase):
    def test_union_clips_overlapping_intervals(self):
        self.assertEqual(trace.union_seconds([(0,4),(2,6),(8,10)],1,9),6)

    def test_parallel_hierarchy_and_exclusive_time(self):
        with tempfile.TemporaryDirectory() as directory,capture():
            with trace.job(directory=directory,job_id='parallel',force=True,sample_seconds=.02):
                with trace.span('phase'):
                    barrier=threading.Barrier(2)
                    def work(page):
                        with trace.span('page',page=page):barrier.wait(timeout=3);time.sleep(.06)
                    with ThreadPoolExecutor(2) as pool:list(trace.map_in_context(pool,work,['1.png','2.png']))
            summary=json.loads(next(Path(directory).glob('*.summary.json')).read_text())
            events=[json.loads(x) for x in next(Path(directory).glob('*.jsonl')).read_text().splitlines()]
            ends=[x for x in events if x['event']=='end']
            phase=next(x for x in ends if x['stage']=='phase');pages=[x for x in ends if x['stage']=='page']
            self.assertTrue(all(p['parent_span_id']==phase['span_id'] for p in pages))
            self.assertLess(phase['own_wall_s'],.05)
            self.assertGreater(sum(p['wall_s'] for p in pages),phase['wall_s']*1.4)
            self.assertEqual(summary['unfinished_spans'],0)
            self.assertGreater(summary['samples'],0)
            self.assertIsNone(trace.current())

    def test_exception_and_next_job_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory,capture():
            with self.assertRaisesRegex(RuntimeError,'original'):
                with trace.job(directory=directory,job_id='first',force=True):
                    with trace.span('failed'):raise RuntimeError('original')
            with trace.job(directory=directory,job_id='second',force=True):pass
            summaries=[json.loads(p.read_text()) for p in Path(directory).glob('*.summary.json')]
            self.assertEqual({s['job_id']:s['status'] for s in summaries},{'first':'error','second':'ok'})
            self.assertIsNone(trace.current())

    def test_disabled_has_no_files_or_console(self):
        with tempfile.TemporaryDirectory() as directory,mock.patch.dict(os.environ,{'PIPELINE_TRACE':'false'}),capture() as output:
            with trace.job(directory=directory):
                with trace.span('unused'):pass
            self.assertFalse(list(Path(directory).iterdir()));self.assertEqual(output.getvalue(),'')

    def test_metadata_allowlist_and_exception_message_not_recorded(self):
        with tempfile.TemporaryDirectory() as directory,capture():
            with self.assertRaises(ValueError):
                with trace.job(directory=directory,force=True):
                    with trace.span('test',page='x.png',api_key='secret-A',url='secret-B',body='secret-C'):
                        raise ValueError('secret-D')
            text=next(Path(directory).glob('*.jsonl')).read_text()
            for secret in ('secret-A','secret-B','secret-C','secret-D'):self.assertNotIn(secret,text)
            self.assertIn('ValueError',text)

    def test_telemetry_io_failure_does_not_fail_processing(self):
        with tempfile.TemporaryDirectory() as directory,capture():
            blocker=Path(directory)/'file';blocker.write_text('x')
            with trace.job(directory=blocker/'child',force=True):value=42
            self.assertEqual(value,42)
            with trace.job(directory=directory,force=True):
                trace.current().file.close()
                with trace.span('after_disk_failure'):value=43
            self.assertEqual(value,43);self.assertIsNone(trace.current())

    def test_nonzero_return_is_error_but_positive_worker_count_is_success(self):
        with tempfile.TemporaryDirectory() as directory,mock.patch.dict(os.environ,{'PIPELINE_TRACE':'true','PIPELINE_TRACE_DIR':directory}),capture():
            @trace.traced_job('worker',job_arg='job')
            def worker(job):return 1
            @trace.traced_job('cli',exit_code=True)
            def cli():return 3
            self.assertEqual(worker({'id':'worker'}),1);self.assertEqual(cli(),3)
            summaries=[json.loads(p.read_text()) for p in Path(directory).glob('*.summary.json')]
            self.assertEqual({x['job_id']:x['status'] for x in summaries},{'worker':'ok','manual':'error'})

    def test_heartbeat_is_not_progress_including_split_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'job.log';path.write_bytes(b'page done\n')
            activity=LogActivity(path);before=activity.size()
            with path.open('ab') as f:f.write(b'[TIME LI')
            self.assertEqual(activity.size(),before)
            with path.open('ab') as f:f.write(b'VE] waiting\n[TIME LIVE] still waiting\n')
            self.assertEqual(activity.size(),before)
            with path.open('ab') as f:f.write(b'[TIME] END api.http\n')
            self.assertGreater(activity.size(),before)

    def test_gateway_trace_no_extra_request_and_retry_is_visible(self):
        from modules.utils import gateway
        response=mock.Mock(status_code=200);response.json.return_value={'ok':True}
        failed=mock.Mock(status_code=503,text='unavailable')
        post=mock.Mock(side_effect=[failed,response])
        with tempfile.TemporaryDirectory() as directory,capture():
            with trace.job(directory=directory,job_id='api',force=True):
                result=gateway.resilient_post('https://invalid.test',headers={},data='private',timeout=1,
                    kind='detection',post=post,sleep=lambda _:None,backoffs=(0,),rand=lambda:0,
                    breaker=gateway.CircuitBreaker(threshold=99),limiter=gateway.AdaptiveLimiter(1))
            self.assertIs(result,response);self.assertEqual(post.call_count,2)
            events=[json.loads(line) for line in next(Path(directory).glob('*.jsonl')).read_text().splitlines()]
            ends=[x for x in events if x['event']=='end']
            self.assertEqual([x['http_status'] for x in ends if x['stage']=='api.http'],[503,200])
            self.assertEqual([x['status'] for x in ends if x['stage']=='api.http'],['error','ok'])
            self.assertIn('api.retry_wait',{x['stage'] for x in ends})

    def test_warm_real_subprocess_two_jobs_with_trace_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);script=folder/'fake.py'
            script.write_text('def main():\n    print("page processed",flush=True)\n    return 0\n')
            env={**os.environ,'PIPELINE_TRACE':'true','PIPELINE_TRACE_DIR':str(folder/'timings')}
            runner=WarmRunner(ROOT,idle_seconds=60,poll=.05)
            try:
                for job_id in ('fake_a','fake_b'):
                    job_env={**env,'PIPELINE_JOB_ID':job_id,'PIPELINE_TRACE_PARENT_RUN':job_id+'-parent'}
                    runner.run([sys.executable,str(script)],ROOT,job_env,folder/(job_id+'.log'),stall_seconds=10,
                               fingerprint=env_fingerprint(job_env))
                    result=collect(folder/'timings',job_id)
                    self.assertEqual(len(result),1);self.assertEqual(result[0][1]['status'],'ok')
                    self.assertEqual(result[0][1]['parent_run_id'],job_id+'-parent')
                self.assertEqual(runner.spawned,1)
            finally:runner.close()

    def test_forward_only_timing_and_preserve_stall_detection(self):
        with tempfile.TemporaryDirectory() as directory,capture() as output:
            path=Path(directory)/'job.log'
            path.write_bytes(b'private application body\n[TIME] START ocr\r\n[TIME LIVE] ocr 10s\n')
            activity=LogActivity(path,forward_timing=True)
            before=activity.size()
            self.assertNotIn('private application body',output.getvalue())
            self.assertIn('[TIME] START ocr\n',output.getvalue())
            with path.open('ab') as stream:stream.write(b'[TIME LIVE] ocr 20s\n')
            self.assertEqual(activity.size(),before)

    def test_live_heartbeat_identifies_page_and_stage(self):
        with tempfile.TemporaryDirectory() as directory,capture() as output:
            with trace.job(directory=directory,force=True,heartbeat=.05,sample_seconds=.02):
                with trace.span('wait.stage',page='013.png'):time.sleep(.09)
            self.assertIn('[TIME LIVE]',output.getvalue())
            self.assertIn('wait.stage(013.png)',output.getvalue())

    def test_live_report_reads_incomplete_last_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'job_x-1.jsonl'
            path.write_text(json.dumps({'event':'start','job_id':'job_x','run_id':'r','span_id':'s','stage':'wait'})+'\n{"part')
            rows=collect(directory,'job_x');self.assertEqual(rows[0][1]['status'],'running_or_interrupted')
            self.assertEqual(len(rows[0][1]['open_spans']),1)
            with self.assertRaises(ValueError):collect(directory,'../secret')

if __name__=='__main__':unittest.main()
