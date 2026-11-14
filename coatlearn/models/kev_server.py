"""Optional loopback Python process owning a single pinned Kev GPU model.

Importing this file is lightweight. Use serve(checkpoint=..., base_model=...,
source_dir=..., model=...) in a dedicated torch environment. There is no download,
credential lookup, truncation, CPU fallback or automatic checkpoint selection.
"""
import json
import math
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def full_precision_answers(probs, metadata):
    """Native typed answers, retaining probabilities before wire rounding."""
    from math import fsum
    if len(probs) != len(metadata):
        raise ValueError("Answer count mismatch")
    answers = {}
    for values, item in zip(probs, metadata):
        p = [float(x) for x in values]
        if (len(p) != len(item['keys']) or not p or
                any(not math.isfinite(x) or not 0 <= x <= 1 for x in p)
                or abs(fsum(p)-1) > 1e-5):
            raise ValueError("Invalid model probabilities")
        # Remove only floating point summation error, retaining full precision.
        p = [x/fsum(p) for x in p]
        kind=item['type'];best=max(range(len(p)),key=p.__getitem__)
        if kind == 'noul':
            answer={'type':kind,'noul':p[1]}
        else:
            answer={'type':kind,'probabilities':dict(zip(item['keys'],p))}
            if kind == 'choice':
                answer.update(choice=item['keys'][best],confidence=(max(p)-1/len(p))/(1-1/len(p)) if len(p)>1 else 1.0)
            else:
                mean=sum(i*x for i,x in enumerate(p))
                spread=sum(x*abs(i-best) for i,x in enumerate(p))
                uniform=sum(abs(i-(len(p)-1)/2) for i in range(len(p)))/len(p)
                answer.update(score=mean,confidence=max(0.,1-spread/uniform) if uniform else 1.,legend=item['legend'])
        answers[item['id']]=answer
    return answers


def serve(*, checkpoint, base_model, source_dir, model='kev-9b-v1.0', port=8019,
          max_state_tokens=8192, row_token_budget=2048, identity=None):
    """Serve one pinned local checkpoint; serialize GPU work across all roles."""
    import os
    import sys
    import time
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                      HF_HUB_DISABLE_TELEMETRY='1',USE_TF='0',USE_FLAX='0')
    sys.path.insert(0,str(Path(source_dir).resolve()))
    import torch
    from kev.api import SystemOneRequest,to_record
    from kev.checkpoint import Checkpoint,LoadOptions
    from kev.model import encode,rows_of,ContextOverflow
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    ck=Checkpoint(str(Path(checkpoint).resolve()))
    cache=Path(os.environ['HF_HOME'])/'hub'/('models--'+ck.meta.base.replace('/','--'))/'snapshots'/ck.meta.base_revision
    cache.parent.mkdir(parents=True,exist_ok=True)
    if not cache.exists():cache.symlink_to(Path(base_model).resolve(),target_is_directory=True)
    if cache.resolve()!=Path(base_model).resolve():raise ValueError('Unexpected model cache')
    torch.cuda.reset_peak_memory_stats()
    tok,network=ck.load('cuda',LoadOptions(dtype=torch.bfloat16,merge=True,fused=False,cuda_graphs=False,backend='torch'))
    assert network.hybrid and all(p.device.type=='cuda' for p in network.parameters())
    # Bound eager question batches for long judge states and up to 16 questions.
    # Row splitting is native Kev execution, not a rubric/method change.
    import kev.model as implementation
    original_rows_per_pass=implementation.rows_per_pass
    def bounded_rows(rows, prefix_len=0, budget=row_token_budget):
        return original_rows_per_pass(rows, prefix_len, budget)
    implementation.rows_per_pass=bounded_rows
    lock=threading.Lock();counts={'requests':0,'failures':0,'input_tokens':0,'max_state_tokens':0,'max_row_tokens':0}
    details={'model':model,'checkpoint':str(Path(checkpoint).resolve()),'base_revision':ck.meta.base_revision,
             'temperature':network.head.temperature,'dtype':network.dtype,'device':torch.cuda.get_device_name(0),
             'max_state_tokens':max_state_tokens,'row_token_budget':row_token_budget,'identity':identity,
             'fused':False,'cuda_graphs':False,'quantization':None,'cpu_offload':False}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,status,data):
            body=json.dumps(data,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if self.path!='/health':return self.send(404,{'error':'not_found'})
            with lock:
                self.send(200,{**details,'counts':dict(counts),'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                               'gpu_reserved_bytes':torch.cuda.memory_reserved(),'gpu_free_bytes':torch.cuda.mem_get_info()[0]})
        def do_POST(self):
            if self.path!='/v1/systemone':return self.send(404,{'error':'not_found'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=8*1024*1024:raise ValueError('Request size')
                req=SystemOneRequest.model_validate(json.loads(self.rfile.read(length)))
                if req.model!=model:raise ValueError('Checkpoint identity mismatch')
                if len(req.questions)>16:raise ValueError('At most 16 questions per request')
                rec,meta=to_record(req)
                enc=encode(tok,rec,max_state=max_state_tokens,max_branch=max_state_tokens+8192,strict=True)
                state,_,rows=rows_of(enc)
            except Exception as exc:
                return self.send(422,{'error_type':type(exc).__name__})
            with lock:
                counts['requests']+=1
                counts['max_state_tokens']=max(counts['max_state_tokens'],len(state))
                counts['max_row_tokens']=max(counts['max_row_tokens'],max(len(state)+len(r['ids']) for r in rows))
                try:
                    start=time.perf_counter()
                    probs=[p.tolist() for p in network.probs(enc)]
                    answers=full_precision_answers(probs,meta)
                    used=len(state)+sum(len(r['ids']) for r in rows);counts['input_tokens']+=used
                    response={'model':model,'answers':answers,'usage':{'input_tokens':used,'output_tokens':0},
                              'latency_ms':1000*(time.perf_counter()-start)}
                except Exception as exc:
                    counts['failures']+=1
                    return self.send(500,{'error_type':type(exc).__name__})
            self.send(200,response)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(json.dumps({'ready':True,**details}),flush=True)
    try:server.serve_forever()
    finally:server.server_close()
