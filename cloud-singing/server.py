"""Authenticated, single-job, synchronous GPU cover endpoint. No idle jobs."""
import hashlib,hmac,json,os,pathlib,subprocess,sys,tempfile,threading,time
from fastapi import FastAPI,File,Form,Header,HTTPException,UploadFile
from fastapi.responses import Response
app=FastAPI(title='QQ Cloud Singing')
lock=threading.Lock()
deadline=0.0
LIMIT=40*1024*1024
def authorize(value):
    key=os.environ.get('SINGING_API_KEY','')
    if not key or not hmac.compare_digest(key,value or ''):raise HTTPException(401,'Unauthorized')
def run(args):
    remaining=deadline-time.monotonic() if deadline else 450
    if remaining<=0:raise subprocess.TimeoutExpired(args,0)
    result=subprocess.run([str(x) for x in args],capture_output=True,timeout=min(450,remaining))
    if result.returncode:raise RuntimeError('Audio processing failed')
async def upload(file,dest):
    n=0
    with dest.open('xb') as out:
        while data:=await file.read(1024*1024):
            n+=len(data)
            if n>LIMIT:raise HTTPException(413,'Audio upload too large')
            out.write(data)
    if n<100:raise HTTPException(400,'Audio upload is empty')
@app.get('/health')
def health():return {'ok':True,'service':'qq-cloud-singing','version':1,'mode':'on_demand','modelLoaded':False}
@app.post('/v1/audio/cover')
async def cover(source:UploadFile=File(...),reference:UploadFile=File(...),seconds:int=Form(30),x_singing_key:str|None=Header(None)):
    global deadline
    authorize(x_singing_key)
    if not 5<=seconds<=30:raise HTTPException(400,'Expected 5..30 seconds')
    if not lock.acquire(False):raise HTTPException(409,'Another cover is running')
    started=time.monotonic()
    deadline=started+540
    try:
        with tempfile.TemporaryDirectory(prefix='qq-cover-') as temp:
            d=pathlib.Path(temp)
            await upload(source,d/'source.bin');await upload(reference,d/'reference.bin')
            run(['ffmpeg','-v','error','-n','-i',d/'source.bin','-t',seconds,'-ar','44100','-ac','2',d/'source.wav'])
            run(['ffmpeg','-v','error','-n','-i',d/'reference.bin','-t','20','-ar','44100','-ac','1',d/'reference.wav'])
            run([sys.executable,'/opt/singing/server.py','separate',d/'source.wav',d])
            from huggingface_hub import hf_hub_download
            ckpt=hf_hub_download('Plachta/Seed-VC','DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth',revision='257283f9f41585055e8f858fba4fd044e5caed6e',cache_dir='./checkpoints',local_files_only=True)
            run([sys.executable,'inference.py','--source',d/'vocals.wav','--target',d/'reference.wav','--output',d/'converted','--checkpoint',ckpt,'--config','configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml','--diffusion-steps','50','--f0-condition','True','--auto-f0-adjust','False','--fp16','True'])
            converted=list((d/'converted').glob('*.wav'))
            if len(converted)!=1:raise RuntimeError('Missing converted audio')
            run(['ffmpeg','-v','error','-n','-i',converted[0],'-i',d/'no_vocals.wav','-filter_complex','[0:a][1:a]amix=inputs=2:duration=shortest:normalize=0,alimiter=limit=0.95','-c:a','libmp3lame','-b:a','256k',d/'cover.mp3'])
            data=(d/'cover.mp3').read_bytes()
            return Response(data,media_type='audio/mpeg',headers={'X-Generation-Seconds':str(round(time.monotonic()-started,2)),'X-Content-SHA256':hashlib.sha256(data).hexdigest()})
    except HTTPException:raise
    except subprocess.TimeoutExpired:raise HTTPException(504,'Generation timeout; do not retry automatically')
    except Exception:raise HTTPException(502,'Generation failed; do not retry automatically')
    finally:lock.release()
def separate(source,directory):
    import torch,torchaudio,soundfile as sf
    torch.set_num_threads(4)
    x,sr=sf.read(source,dtype='float32',always_2d=True);wave=torch.from_numpy(x.T.copy())
    assert sr==44100 and wave.shape[0]==2
    model=torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS.get_model().eval().cuda()
    mean=wave.mean();std=wave.std().clamp_min(1e-6);norm=(wave-mean)/std
    length=wave.shape[-1];window=10*sr;stride=8*sr
    result=torch.zeros(4,2,length);weights=torch.zeros(length)
    with torch.inference_mode():
        for start in range(0,length,stride):
            end=min(start+window,length);n=end-start
            chunk=torch.nn.functional.pad(norm[:,start:end],(0,window-n))
            out=model(chunk[None].cuda())[0,:,:,:n].cpu()
            weight=torch.ones(n);fade=min(2*sr,n)
            if start:weight[:fade]=torch.linspace(1e-4,1,fade)
            if end<length:weight[-fade:]=torch.linspace(1,1e-4,fade)
            result[:,:,start:end]+=out*weight;weights[start:end]+=weight
            if end==length:break
    vocals=(result/weights.clamp_min(1e-6)*std+mean/4)[model.sources.index('vocals')]
    d=pathlib.Path(directory)
    sf.write(d/'vocals.wav',vocals.T.numpy(),sr,subtype='FLOAT')
    sf.write(d/'no_vocals.wav',(wave-vocals).T.numpy(),sr,subtype='FLOAT')
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='separate':separate(sys.argv[2],sys.argv[3])
