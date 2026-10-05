"""CPU checks of the opt-in correction, without loading production models."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

source=ast.parse(Path(__file__).with_name('corrected_probe.py').read_text())
functions=ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef)],type_ignores=[])

def original_values(trainer,args,dataloader,batch):
    updates=max(len(dataloader)//args.gradient_accumulation_steps,1)
    maximum=math.ceil(args.num_train_epochs*updates) if args.max_steps<0 else args.max_steps
    return (10,updates,len(dataloader)*4,400,args.max_steps<0,len(dataloader),maximum)

events=[]
def original_step(trainer,model,inputs,*args,**kwargs):
    events.append(('backward',model.boundary));return 123

scope={'math':math,'original_values':original_values,'original_step':original_step}
exec(compile(functions,'corrected_probe.py','exec'),scope)
for batches,updates in [(10,20),(50,70),(100,130)]:
    trainer=SimpleNamespace()
    args=SimpleNamespace(gradient_accumulation_steps=8,num_train_epochs=10,max_steps=-1)
    plan=scope['corrected_values'](trainer,args,list(range(batches)),32)
    assert plan[-1]==updates
    assert plan[1]==math.ceil(batches/8)
    args.max_steps=4
    smoke=scope['corrected_values'](trainer,args,list(range(batches)),32)
    assert smoke[-1]==4 and smoke[0]==math.ceil(4/smoke[1])
trainer=SimpleNamespace(is_deepspeed_enabled=True,accelerator=SimpleNamespace(sync_gradients=False),_npo_corrected_boundaries=0)
class Engine:
    boundary=None
    def set_gradient_accumulation_boundary(self,value):self.boundary=value;events.append(('boundary',value))
engine=Engine()
for value in [False,True]:
    trainer.accelerator.sync_gradients=value
    assert scope['corrected_step'](trainer,engine,{})==123
assert events==[('boundary',False),('backward',False),('boundary',True),('backward',True)]
assert trainer._npo_corrected_boundaries==1
print('PASS: ceil planning, explicit smoke limit, boundary set before backward')
