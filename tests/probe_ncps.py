import inspect, sys
import torch
import ncps
from ncps.torch import CfC
from ncps.torch.cfc_cell import CfCCell

print("torch", torch.__version__, "| ncps", getattr(ncps, "__version__", "?"))
print("CfC.__init__", inspect.signature(CfC.__init__))
print("CfC.forward ", inspect.signature(CfC.forward))
print("CfCCell.__init__", inspect.signature(CfCCell.__init__))
print(inspect.getsource(CfCCell))          # cell layers + forward, needed for the MAC formula

def npar(m): return sum(p.numel() for p in m.parameters())

# parameter count of CfC + Linear(h,1) head for variant W (F=1) and E (F=4)
for F in (1, 4):
    for h in (8, 12, 15, 16, 24, 40, 49, 50, 64):
        m = CfC(F, h, batch_first=True, return_sequences=False)  # adjust to printed signature
        print(f"F={F} h={h:3d} cfc={npar(m):6d} +head={npar(m)+h+1:6d}")

m = CfC(1, 16, batch_first=True, return_sequences=False)
for k, v in m.state_dict().items(): print(k, tuple(v.shape))
x = torch.randn(2, 398, 1)
out = m(x); print(type(out), [o.shape for o in out] if isinstance(out, tuple) else out.shape)