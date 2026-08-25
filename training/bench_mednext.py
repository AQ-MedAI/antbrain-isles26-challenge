import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'mednext_v2', 'trainers'))
import torch
from nnUNetTrainerMedNeXt import nnUNetTrainerMedNeXt_B_kernel3, nnUNetTrainerMedNeXt_B_kernel5

torch.backends.cudnn.benchmark = True

def bench(cls, use_compile, iters=20, warmup=5):
    net = cls.build_network_architecture(None, None, 1, 2, enable_deep_supervision=True).cuda()
    net.train()
    if use_compile:
        net = torch.compile(net)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, eps=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    x = torch.randn(2, 1, 128, 128, 128, device='cuda')
    y = [torch.randint(0, 2, (2, 2, s, s, s), device='cuda').float() for s in (128, 64, 32, 16, 8)]
    loss_fn = torch.nn.L1Loss()
    for i in range(warmup + iters):
        if i == warmup:
            torch.cuda.synchronize(); t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', enabled=True):
            out = net(x)
            l = sum(loss_fn(o, t) for o, t in zip(out, y))
        scaler.scale(l).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    print(f'{cls.__name__} compile={use_compile}: {dt:.3f} s/iter  (epoch~250 it: {dt*250:.0f}s)')
    del net, opt
    torch.cuda.empty_cache()
    torch._dynamo.reset()

bench(nnUNetTrainerMedNeXt_B_kernel3, False)
bench(nnUNetTrainerMedNeXt_B_kernel5, False)
bench(nnUNetTrainerMedNeXt_B_kernel5, True)
