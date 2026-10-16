"""Compare Timestral inputs on three 30-row holdouts.

Run from the project root: uv run python scripts/evaluate_covariates.py
Uses cached model weights. Future covariates are actual held-out inputs and
represent an idealized experiment, not operational forecast performance.
Historical inputs are assumed available at the cutoff; the CSV has no issue times.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from timestral import TimestralForecaster
torch.set_num_threads(4)
d=pd.read_csv('data/spg.csv'); y=d.generated_power_kw.to_numpy(dtype=np.float32)
cols=['shortwave_radiation_backwards_sfc','total_cloud_cover_sfc','temperature_2_m_above_gnd','relative_humidity_2_m_above_gnd','zenith','azimuth']
x=d[cols].to_numpy(dtype=np.float32).T
m=TimestralForecaster.from_pretrained('mistralai/Timestral-209M',device='cpu',local_files_only=True)
rows=[]
for end in [len(y)-90,len(y)-60,len(y)-30]:
 for mode in ['target only','past covariates','idealized future covariates']:
  kwargs={}
  if mode=='past covariates': kwargs['past_covariates']=x[None,:,:end]
  if mode=='idealized future covariates': kwargs['known_covariates']=x[None,:,:end+30]
  start=time.time()
  f=m.forecast(y[None,None,:end],prediction_length=30,**kwargs)
  p=f.median[0,0].numpy(); err=y[end:end+30]-p
  r=dict(origin=end,mode=mode,mae_kw=float(np.abs(err).mean()),rmse_kw=float(np.sqrt((err**2).mean())))
  rows.append(r); print(r,'seconds',round(time.time()-start,2),flush=True)
Path('reports').mkdir(exist_ok=True)
r=pd.DataFrame(rows);r.to_csv('reports/covariate_comparison.csv',index=False)
print(r.groupby('mode')[['mae_kw','rmse_kw']].mean().round(2).to_string(),flush=True)
