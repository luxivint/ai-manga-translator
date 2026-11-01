"""Lightweight job/stage telemetry. No payloads, URLs, credentials or model calls.

Wall intervals are unioned, never blindly added across concurrent/nested spans.
Per-span process CPU is observed across ALL threads; it is not attributable CPU.
"""
from __future__ import annotations
import builtins
import contextlib,contextvars,functools,inspect,json,os,re,sys,threading,time,uuid
from datetime import datetime,timezone
from pathlib import Path

def print(*args, **kwargs):
    # Broken output pipes must never turn a translated page into a failed job.
    kwargs.setdefault('file', sys.stderr)
    try: builtins.print(*args, **kwargs)
    except (OSError, ValueError): pass

_session=contextvars.ContextVar('trace_session',default=None)
_parent=contextvars.ContextVar('trace_parent',default=None)
_global=None
_guard=threading.RLock()
_SAFE={'page','kind','attempt','count','bytes','http_status','exit_code','cache_hit','decision','reason_code'}

def _meta(values):
    return {k:(v[:160] if isinstance(v,str) else v) for k,v in values.items()
            if k in _SAFE and (v is None or type(v) in (str,int,float,bool))}

def _utc():return datetime.now(timezone.utc).isoformat(timespec='milliseconds')
def enabled():return os.environ.get('PIPELINE_TRACE','true').lower() not in ('0','false','off','no')
def current():return _session.get() or _global

def union_seconds(intervals,start=None,end=None):
    intervals=[(max(a,start) if start is not None else a,min(b,end) if end is not None else b) for a,b in intervals]
    intervals=sorted((a,b) for a,b in intervals if b>a)
    total=0.;last_a=last_b=None
    for a,b in intervals:
        if last_b is None:last_a,last_b=a,b
        elif a<=last_b:last_b=max(last_b,b)
        else:total+=last_b-last_a;last_a,last_b=a,b
    return total+(last_b-last_a if last_b is not None else 0.)

class NullSpan:
    def __init__(self):self.meta={}
    def __enter__(self):return self
    def __exit__(self,*args):return False

class Span:
    def __init__(self,session,name,meta,console):
        self.session=session;self.name=name;self.meta=_meta(meta);self.console=console;self.id=uuid.uuid4().hex[:12]
        self.children=[];self.child_thread_cpu=0.;self.rss_peak=None;self.done=False
    def __enter__(self):
        self.start=time.perf_counter();self.proc_start=time.process_time();self.thread_start=time.thread_time();self.thread=threading.get_ident()
        with self.session.lock:
            self.parent=_parent.get() or self.session.root_id
            self.session.active[self.id]=self
            if self.session.root_id is None:self.session.root_id=self.id;self.parent=None
            parent=self.session.active.get(self.parent)
            if parent:
                self.meta={**{k:v for k,v in parent.meta.items() if k in ('page','kind')},**self.meta}
        self.token=_parent.set(self.id)
        self.session.emit('start',self,console=self.console)
        return self
    def __exit__(self,typ,value,tb):
        if self.done:return False
        self.done=True;self.end=time.perf_counter();self.wall=max(0.,self.end-self.start)
        self.thread_cpu=max(0.,time.thread_time()-self.thread_start);self.proc_cpu=max(0.,time.process_time()-self.proc_start)
        self.status='error' if typ is not None or self.meta.get('exit_code',0) not in (0,None) or (self.meta.get('http_status') or 0)>=400 else 'ok'
        with self.session.lock:
            self.own=max(0.,self.wall-union_seconds(self.children,self.start,self.end))
            self.own_thread_cpu=max(0.,self.thread_cpu-self.child_thread_cpu)
            parent=self.session.active.get(self.parent)
            if parent:
                parent.children.append((self.start,self.end))
                if parent.thread==self.thread:parent.child_thread_cpu+=self.thread_cpu
            self.session.active.pop(self.id,None)
            self.session.finished(self)
        self.session.emit('end',self,error_type=typ.__name__ if typ else None,console=self.console)
        _parent.reset(self.token)
        return False

class Session:
    def __init__(self,job_id,directory,heartbeat=10,sample_seconds=1):
        self.job_id=re.sub(r'[^A-Za-z0-9_.-]','_',str(job_id or 'manual'))[:100]
        self.run_id=f'{self.job_id}-{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex[:6]}'
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/(self.run_id+'.jsonl');self.summary_path=self.directory/(self.run_id+'.summary.json')
        self.file=self.path.open('a',encoding='utf8');self.lock=threading.RLock();self.active={};self.stats={};self.root_id=None
        self.started=time.perf_counter();self.process_cpu_started=time.process_time();self.stop=threading.Event()
        self.heartbeat=max(.05,float(heartbeat));self.sample_seconds=max(.02,float(sample_seconds));self.samples=0;self.rss_peak=None;self.cpu_cores_peak=0.;self.warning=False
        self.parent_run_id=os.environ.get('PIPELINE_TRACE_PARENT_RUN') or None
        self.parent_span_id=os.environ.get('PIPELINE_TRACE_PARENT_SPAN') or None
        self.root=None;self.summary=None
        try:
            import psutil
            self.process=psutil.Process()
        except Exception:self.process=None
    def write(self,data):
        try:
            with self.lock:self.file.write(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n');self.file.flush()
        except (OSError,ValueError,TypeError):
            if not self.warning:
                self.warning=True;print('[TRACE WARNING] telemetry_write_failed; processing continues',flush=True)
    def emit(self,event,span,*,error_type=None,console=False):
        data={'version':1,'event':event,'utc':_utc(),'job_id':self.job_id,'run_id':self.run_id,'pid':os.getpid(),
              'parent_run_id':self.parent_run_id,'span_id':span.id,'parent_span_id':span.parent,'stage':span.name,'thread_id':span.thread,**_meta(span.meta)}
        if event=='end':data.update(wall_s=span.wall,own_wall_s=span.own,thread_cpu_s=span.thread_cpu,
                                   own_thread_cpu_s=span.own_thread_cpu,process_cpu_s_observed=span.proc_cpu,
                                   rss_peak_mib_observed=span.rss_peak,status=span.status,error_type=error_type)
        self.write(data)
        if console:
            page=f" page={data['page']}" if data.get('page') else ''
            if event=='start':print(f"[TIME] START {span.name}{page}",flush=True)
            else:print(f"[TIME] END {span.name}{page} wall={span.wall:.3f}s own={span.own:.3f}s "
                       f"cpu_process_observed={span.proc_cpu:.3f}s status={span.status}",flush=True)
    def finished(self,span):
        row=self.stats.setdefault(span.name,{'count':0,'errors':0,'work_wall_s':0.,'own_work_wall_s':0.,'thread_cpu_s':0.,'intervals':[]})
        row['count']+=1;row['errors']+=span.status=='error';row['work_wall_s']+=span.wall;row['own_work_wall_s']+=span.own;row['thread_cpu_s']+=span.thread_cpu;row['intervals'].append((span.start,span.end))
    def monitor(self):
        last=time.perf_counter();last_cpu=time.process_time();last_console=last
        while not self.stop.wait(self.sample_seconds):
            now=time.perf_counter();cpu=time.process_time();cores=max(0.,cpu-last_cpu)/max(.000001,now-last)
            rss=None
            try:
                if self.process:rss=self.process.memory_info().rss/1048576
            except Exception:pass
            with self.lock:
                active=[{'stage':s.name,'page':s.meta.get('page'),'elapsed_s':max(0,now-s.start)} for s in self.active.values()]
                self.samples+=1;self.cpu_cores_peak=max(self.cpu_cores_peak,cores)
                if rss is not None:
                    self.rss_peak=max(self.rss_peak or 0,rss)
                    for s in self.active.values():s.rss_peak=max(s.rss_peak or 0,rss)
            self.write({'version':1,'event':'sample','utc':_utc(),'job_id':self.job_id,'run_id':self.run_id,'pid':os.getpid(),
                        'cpu_cores':cores,'rss_mib':rss,'active':active})
            if now-last_console>=self.heartbeat:
                stages=', '.join(f"{a['stage']}"+(f"({a['page']})" if a['page'] else '')+f" {a['elapsed_s']:.0f}s" for a in active[-6:])
                print(f'[TIME LIVE] job={self.job_id} cpu_cores={cores:.2f} rss_mib={rss if rss is not None else "unknown"} active={stages}',flush=True)
                last_console=now
            last,last_cpu=now,cpu
    def close(self):
        self.stop.set()
        if hasattr(self,'thread') and self.thread.ident is not None:self.thread.join(timeout=2)
        end=time.perf_counter();rows=[]
        with self.lock:
            for stage,row in self.stats.items():rows.append({'stage':stage,**{k:v for k,v in row.items() if k!='intervals'},'active_wall_s':union_seconds(row['intervals'])})
        root=self.root
        self.summary={'version':1,'job_id':self.job_id,'run_id':self.run_id,'pid':os.getpid(),'parent_run_id':self.parent_run_id,'parent_span_id':self.parent_span_id,
                      'wall_s':root.wall if root else end-self.started,'process_cpu_s':time.process_time()-self.process_cpu_started,
                      'unattributed_root_wall_s':root.own if root else None,'status':root.status if root else 'incomplete',
                      'samples':self.samples,'rss_peak_mib':self.rss_peak,'cpu_cores_peak_sampled':self.cpu_cores_peak,'unfinished_spans':len(self.active),
                      'stages':rows,'notes':['work_wall_s sums work and can exceed wall time under concurrency',
                      'active_wall_s is interval union within ONE stage; do not sum nested/different stage coverage',
                      'own_work_wall_s excludes overlapping direct children by interval union',
                      'process_cpu_s_observed per span includes ALL process threads; never sum it across spans',
                      'thread_cpu_s omits native worker thread CPU; job process_cpu_s includes it',
                      'cpu_cores is process CPU seconds per elapsed second, not a machine CPU percentage',
                      'RSS is observed process memory, not allocated-by-stage memory']}
        try:self.summary_path.write_text(json.dumps(self.summary,ensure_ascii=False,indent=2),encoding='utf8')
        except OSError:print('[TRACE WARNING] summary_write_failed',flush=True)
        self.write({'version':1,'event':'summary','utc':_utc(),**self.summary})
        print(f"[TIME SUMMARY] job={self.job_id} wall={self.summary['wall_s']:.3f}s cpu_process={self.summary['process_cpu_s']:.3f}s "
              f"unattributed={self.summary['unattributed_root_wall_s']}s status={self.summary['status']} summary={self.summary_path}",flush=True)
        for row in sorted(rows,key=lambda r:r['own_work_wall_s'],reverse=True)[:15]:
            print(f"[TIME STAGE] {row['stage']} calls={row['count']} active={row['active_wall_s']:.3f}s work={row['work_wall_s']:.3f}s own_work={row['own_work_wall_s']:.3f}s errors={row['errors']}",flush=True)
        self.file.close()

def span(name,*,console=True,**meta):
    session=current()
    return Span(session,name,meta,console) if session is not None else NullSpan()

@contextlib.contextmanager
def job(name='pipeline.job',*,job_id=None,directory=None,force=False,heartbeat=None,sample_seconds=1):
    global _global
    if current() is not None:
        with span(name) as nested:yield nested
        return
    if not force and not enabled():yield NullSpan();return
    directory=directory or os.environ.get('PIPELINE_TRACE_DIR') or Path(__file__).resolve().parents[2]/'work/job-timings'
    try:session=Session(job_id or os.environ.get('PIPELINE_JOB_ID'),directory,heartbeat or os.environ.get('PIPELINE_TRACE_HEARTBEAT_SECONDS','10'),sample_seconds)
    except (OSError,ValueError):
        print('[TRACE WARNING] telemetry_initialization_failed; processing continues',flush=True);yield NullSpan();return
    token=_session.set(session)
    with _guard:_global=session
    root=Span(session,name,{},True);session.root=root
    print(f'[TRACE] job={session.job_id} run={session.run_id} events={session.path}',flush=True)
    try:
        root.__enter__()
        session.thread=threading.Thread(target=session.monitor,daemon=True,name='job-telemetry')
        try:session.thread.start()
        except RuntimeError:print('[TRACE WARNING] sampler_start_failed; stage timing continues',flush=True)
        try:yield root
        except BaseException:
            root.__exit__(*sys.exc_info());raise
        else:root.__exit__(None,None,None)
    finally:
        try:
            session.close()
        except Exception:
            print('[TRACE WARNING] telemetry_close_failed; processing continues',flush=True)
        finally:
            with _guard:_global=None
            _session.reset(token)

def set_exit_code(code):
    session=current()
    if session is not None and session.root is not None:session.root.meta['exit_code']=int(code or 0)

def parent_env(env):
    session=current()
    if session:
        env['PIPELINE_TRACE_PARENT_RUN']=session.run_id
        env['PIPELINE_TRACE_PARENT_SPAN']=_parent.get() or session.root_id or ''
    return env

def traced(name,*,page_arg=None,task_page=False,fields=(),console=True):
    def decorate(fn):
        signature=inspect.signature(fn)
        @functools.wraps(fn)
        def wrapped(*args,**kwargs):
            if current() is None:return fn(*args,**kwargs)
            meta={}
            try:
                bound=signature.bind_partial(*args,**kwargs).arguments
                if page_arg and page_arg in bound:meta['page']=Path(str(bound[page_arg])).name
                if task_page:meta['page']=Path(str(next(iter(bound.values()))[1])).name
                meta.update({f:bound[f] for f in fields if f in bound})
            except (TypeError,ValueError,IndexError):pass
            with span(name,console=console,**meta):return fn(*args,**kwargs)
        return wrapped
    return decorate

def traced_job(name,*,job_arg=None,exit_code=False):
    def decorate(fn):
        signature=inspect.signature(fn)
        @functools.wraps(fn)
        def wrapped(*args,**kwargs):
            identity=None
            if job_arg:
                value=signature.bind_partial(*args,**kwargs).arguments.get(job_arg)
                if isinstance(value,dict):identity=value.get('id')
            with job(name,job_id=identity):
                result=fn(*args,**kwargs)
                if exit_code:set_exit_code(result)
                return result
        return wrapped
    return decorate

def map_in_context(executor,fn,*iterables):
    context=contextvars.copy_context()
    def call(*args):return context.copy().run(fn,*args)
    return executor.map(call,*iterables)

def submit_in_context(executor,fn,*args,**kwargs):
    context=contextvars.copy_context()
    return executor.submit(context.run,functools.partial(fn,*args,**kwargs))


def call(name, fn, *args, **kwargs):
    with span(name):
        return fn(*args, **kwargs)
