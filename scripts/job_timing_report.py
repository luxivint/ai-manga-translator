"""Read-only timing report and diagnostic bundle. Never reads environment files."""
from __future__ import annotations
import argparse
import json
import re
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def collect(directory, job_id):
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', job_id):
        raise ValueError('Invalid job ID')
    files = sorted(Path(directory).glob(job_id + '-*.jsonl'))
    # Prefix matches must not include another job (e.g. job_x-extra).
    runs=[]
    for path in files:
        summary=path.with_suffix('.summary.json')
        if summary.exists():
            row=json.loads(summary.read_text(encoding='utf8'))
            if row.get('job_id')==job_id:runs.append((path,row,summary))
            continue
        # A running/crashed job still has its completed spans and latest active snapshot.
        completed=[];active={};sample=None;identity=None
        with path.open(encoding='utf8') as stream:
            for line in stream:
                try:event=json.loads(line)
                except ValueError:continue  # last line may still be writing
                if event.get('job_id')!=job_id:continue
                identity=event
                if event.get('event')=='start':active[event['span_id']]=event
                elif event.get('event')=='end':
                    active.pop(event['span_id'],None);completed.append(event)
                elif event.get('event')=='sample':sample=event
        if identity:
            runs.append((path,{'job_id':job_id,'run_id':identity['run_id'],'status':'running_or_interrupted',
                'completed_spans':len(completed),'open_spans':list(active.values()),'latest_sample':sample},None))
    return runs

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job_id')
    parser.add_argument('--directory',type=Path,default=ROOT/'work/job-timings')
    parser.add_argument('--bundle',action='store_true',help='Archive job log, timing events and summaries only')
    args=parser.parse_args(argv)
    try:runs=collect(args.directory,args.job_id)
    except ValueError as exc:parser.error(str(exc))
    if not runs:
        print('No stage telemetry found for this job. Old jobs cannot be broken down retrospectively.')
        return 1
    for path,row,summary in runs:
        print(f"\nRUN {row['run_id']} status={row['status']}")
        if summary:
            print(f"Elapsed: {row['wall_s']:.3f}s; process CPU: {row['process_cpu_s']:.3f}s; RSS peak sampled: {row['rss_peak_mib']} MiB")
            print(f"Parent run: {row.get('parent_run_id') or '-'}; root time outside child spans: {row['unattributed_root_wall_s']:.3f}s")
            print(f"{'Stage':38} {'Calls':>6} {'Active(s)':>12} {'Own work(s)':>12} {'Errors':>7}")
            for item in sorted(row['stages'],key=lambda s:s['own_work_wall_s'],reverse=True):
                print(f"{item['stage']:38} {item['count']:6} {item['active_wall_s']:12.3f} {item['own_work_wall_s']:12.3f} {item['errors']:7}")
        else:
            print(f"Completed spans: {row['completed_spans']}; open spans: {len(row['open_spans'])}")
            sample=row.get('latest_sample') or {}
            print(f"Last sample: {sample.get('utc','-')}; process CPU cores: {sample.get('cpu_cores','-')}; RSS MiB: {sample.get('rss_mib','-')}")
            for item in sample.get('active',[]):print(f"  {item['stage']} page={item.get('page')} elapsed={item['elapsed_s']:.1f}s")
    print('\nDo not add nested/parallel stage times or parent/child process durations together. CPU cores is not machine CPU percent.')
    if args.bundle:
        output=ROOT/'work/diagnostics'/f'{args.job_id}-timing-{time.time_ns()}.tar.gz'
        output.parent.mkdir(parents=True,exist_ok=True)
        paths=[p for path,_,summary in runs for p in (path,summary) if p is not None]
        log=ROOT/'work/job-logs'/f'{args.job_id}.log'
        if log.is_file():paths.append(log)
        with tarfile.open(output,'w:gz') as archive:
            for path in paths:archive.add(path,arcname=path.name,recursive=False)
        print(f'Bundle: {output}')
    return 0

if __name__=='__main__':raise SystemExit(main())
