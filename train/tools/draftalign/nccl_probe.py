import os, time, torch, torch.distributed as dist
dist.init_process_group("nccl", timeout=__import__("datetime").timedelta(seconds=120))
r=dist.get_rank(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
x=torch.ones(64*1024*1024, device="cuda")  # 256 MB
dist.all_reduce(x); torch.cuda.synchronize()
t=time.time()
for _ in range(5): dist.all_reduce(x)
torch.cuda.synchronize(); dt=(time.time()-t)/5
if r==0: print(f"PROBE_OK allreduce 256MB x16 ranks: {dt*1000:.0f} ms/iter  (~{2*0.256*15/16/dt:.1f} GB/s bus)", flush=True)
dist.destroy_process_group()
