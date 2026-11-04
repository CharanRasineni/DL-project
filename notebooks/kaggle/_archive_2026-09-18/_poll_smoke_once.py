import json, os, time
from pathlib import Path
b=Path(r'c:\\Users\\rshit\\OneDrive\\Documents\\Charan DL project\\notebooks\\kaggle')
os.environ['KAGGLE_API_TOKEN']=json.loads((Path.home()/'.kaggle/kaggle.json').read_text())['key']
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
api=KaggleApi(); api.authenticate()
ref='rscharan/fathomnet-research-smoke-minimal'
dest=b/'research_results'/'smoke_minimal'; dest.mkdir(parents=True,exist_ok=True)
deadline=time.time()+40*60
while time.time()<deadline:
    try:
        s=api.kernels_status(ref)
        st=str(getattr(s.status,'value',s.status)).upper().split('.')[-1]
        print(time.strftime('%H:%M:%S'),st,flush=True)
        (dest/'status.txt').write_text(st)
        if st in ('COMPLETE','ERROR','CANCEL_ACKNOWLEDGED','CANCELED'):
            try: api.kernels_output(ref,str(dest))
            except Exception as e: print('download error',repr(e),flush=True)
            try:
                req=ApiListKernelSessionOutputRequest(); req.user_name='rscharan'; req.kernel_slug='fathomnet-research-smoke-minimal'
                with api.build_kaggle_client() as c: r=c.kernels.kernels_api_client.list_kernel_session_output(req)
                (dest/'inline.log').write_text(r.log or '',encoding='utf-8',errors='replace')
                print('inline log chars',len(r.log or ''),flush=True)
            except Exception as e: print('log fetch error',repr(e),flush=True)
            print('FINAL',st,flush=True); break
    except Exception as e:
        print('status error',type(e).__name__,str(e),flush=True)
    time.sleep(45)
