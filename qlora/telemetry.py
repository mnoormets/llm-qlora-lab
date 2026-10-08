"""Training callbacks record CUDA allocator bytes, not total device VRAM."""
import json
from pathlib import Path
from transformers import TrainerCallback

class MemoryTrace(TrainerCallback):
    def __init__(self,path):
        import torch
        self.torch=torch;self.path=Path(path);self.profiler=None;self.global_peak_allocated=0;self.global_peak_reserved=0
    def sample(self,phase,step):
        c=self.torch.cuda;row={'phase':phase,'step':step,'cuda_memory_available':c.is_available()}
        if c.is_available():
            row.update(allocated_bytes=c.memory_allocated(),reserved_bytes=c.memory_reserved(),interval_peak_allocated_bytes=c.max_memory_allocated(),interval_peak_reserved_bytes=c.max_memory_reserved())
            self.global_peak_allocated=max(self.global_peak_allocated,row['interval_peak_allocated_bytes']);self.global_peak_reserved=max(self.global_peak_reserved,row['interval_peak_reserved_bytes'])
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
    def on_step_begin(self,args,state,control,**kwargs):
        if self.torch.cuda.is_available():self.torch.cuda.reset_peak_memory_stats()
        self.sample('optimizer_interval_begin',state.global_step)
    def on_substep_end(self,args,state,control,**kwargs):self.sample('accumulated_microbatch_end',state.global_step)
    def on_pre_optimizer_step(self,args,state,control,**kwargs):self.sample('before_optimizer',state.global_step)
    def on_optimizer_step(self,args,state,control,**kwargs):self.sample('after_optimizer',state.global_step)
    def on_step_end(self,args,state,control,**kwargs):
        self.sample('optimizer_interval_end',state.global_step)
        if self.profiler is not None:self.profiler.step()
    def on_evaluate(self,args,state,control,**kwargs):self.sample('evaluation_end',state.global_step)

def training_profiler(torch,directory,steps):
    activities=[torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():activities.append(torch.profiler.ProfilerActivity.CUDA)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    def export(prof):
        prof.export_chrome_trace(str(directory/'training-trace.json'))
        (directory/'operators.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=40),encoding='utf-8')
    return torch.profiler.profile(activities=activities,schedule=torch.profiler.schedule(wait=0,warmup=1,active=steps,repeat=1),record_shapes=True,profile_memory=True,on_trace_ready=export)
