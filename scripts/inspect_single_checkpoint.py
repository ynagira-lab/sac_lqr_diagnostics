from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

# NO RETRAINING: Q1, Q2, min(Q1,Q2) at exact x=[1,0]
MODEL_PATH=str(Path(__file__).resolve().parent.parent / "models" / "sac_lqr_trained_300k.zip")
x=np.array([1.,0.],dtype=np.float32)
ugrid=np.linspace(-10.,10.,401)
U_LQR=-2.5853072593
U_TRUE=-2.700

model=SAC.load(MODEL_PATH)
device=model.device

obs=np.repeat(x[None,:],len(ugrid),axis=0).astype(np.float32)
uphys=ugrid[:,None].astype(np.float32)
asc=model.policy.scale_action(uphys).astype(np.float32)

with torch.no_grad():
    q1t,q2t=model.critic(
        torch.tensor(obs,dtype=torch.float32,device=device),
        torch.tensor(asc,dtype=torch.float32,device=device))
q1=q1t.cpu().numpy().ravel()
q2=q2t.cpu().numpy().ravel()
qmin=np.minimum(q1,q2)

u1=ugrid[np.argmax(q1)]
u2=ugrid[np.argmax(q2)]
umin=ugrid[np.argmax(qmin)]
actor,_=model.predict(x,deterministic=True)
uactor=float(np.asarray(actor).ravel()[0])

def val(a,u): return float(np.interp(u,ugrid,a))

print("="*72)
print("TWIN CRITIC LANDSCAPE AT EXACT x=[1,0] -- NO RETRAINING")
print("="*72)
print(f"Q1 argmax             = {u1:.3f}")
print(f"Q2 argmax             = {u2:.3f}")
print(f"min(Q1,Q2) argmax     = {umin:.3f}")
print(f"Actor                 = {uactor:.3f}")
print(f"LQR                   = {U_LQR:.3f}")
print("\nreward-Q values (larger is better)")
print(f"{'u':>10}{'Q1':>14}{'Q2':>14}{'min':>14}{'Q1-Q2':>14}")
for u in [U_LQR,U_TRUE,uactor,umin]:
    a,b=val(q1,u),val(q2,u)
    print(f"{u:10.3f}{a:14.4f}{b:14.4f}{min(a,b):14.4f}{a-b:14.4f}")

plt.figure(figsize=(11,6))
plt.plot(ugrid,q1,label="Q1")
plt.plot(ugrid,q2,label="Q2")
plt.plot(ugrid,qmin,linewidth=2.2,label="min(Q1,Q2)")
plt.axvline(U_LQR,linestyle="--",label=f"LQR={U_LQR:.3f}")
plt.axvline(U_TRUE,linestyle=":",label=f"True Qpi={U_TRUE:.3f}")
plt.axvline(uactor,linestyle="-.",label=f"Actor={uactor:.3f}")
plt.axvline(umin,linestyle="--",label=f"min-Q argmax={umin:.3f}")
plt.xlabel("Physical action u"); plt.ylabel("Reward-Q")
plt.title("Twin Critic landscape at exact x=[1,0]")
plt.grid(True); plt.legend(); plt.tight_layout()

plt.figure(figsize=(11,5))
plt.plot(ugrid,q1-q2)
plt.axhline(0,linestyle="--")
plt.axvline(U_LQR,linestyle="--",label=f"LQR={U_LQR:.3f}")
plt.axvline(uactor,linestyle="-.",label=f"Actor={uactor:.3f}")
plt.axvline(umin,linestyle="--",label=f"min-Q argmax={umin:.3f}")
plt.xlabel("Physical action u"); plt.ylabel("Q1 - Q2")
plt.title("Twin Critic disagreement at exact x=[1,0]")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.show()
