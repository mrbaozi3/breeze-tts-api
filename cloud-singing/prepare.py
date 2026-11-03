import os,pathlib
from huggingface_hub import hf_hub_download,snapshot_download
root=pathlib.Path('/opt/seed-vc');os.chdir(root)
model='DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth'
hf_hub_download('Plachta/Seed-VC',model,revision='257283f9f41585055e8f858fba4fd044e5caed6e',cache_dir='./checkpoints')
for repo,name in [('lj1995/VoiceConversionWebUI','rmvpe.pt'),('funasr/campplus','campplus_cn_common.bin')]:
    hf_hub_download(repo,name,cache_dir='./checkpoints')
for repo,rev,files in [
 ('nvidia/bigvgan_v2_44khz_128band_512x','95a9d1dcb12906c03edd938d77b9333d6ded7dfb',['config.json','bigvgan_generator.pt']),
 ('openai/whisper-small','973afd24965f72e36ca33b3055d56a652f456b4d',['config.json','preprocessor_config.json','model.safetensors'])]:
    snapshot_download(repo,revision=rev,allow_patterns=files,cache_dir='./checkpoints/hf_cache')
file=root/'modules/length_regulator.py'
text=file.read_text().replace('from dac.nn.quantize import VectorQuantize','').replace('            if vector_quantize:','            if vector_quantize:\n                from dac.nn.quantize import VectorQuantize')
file.write_text(text)
file=root/'inference.py';text=file.read_text()
text=text.replace('bigvgan_name, use_cuda_kernel=False','bigvgan_name, revision="95a9d1dcb12906c03edd938d77b9333d6ded7dfb", use_cuda_kernel=False')
text=text.replace('WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)','WhisperModel.from_pretrained(whisper_name, revision="973afd24965f72e36ca33b3055d56a652f456b4d", torch_dtype=torch.float16)')
text=text.replace('del whisper_model.decoder','del whisper_model.decoder\n        whisper_model = whisper_model.to(device)')
text=text.replace('AutoFeatureExtractor.from_pretrained(whisper_name)','AutoFeatureExtractor.from_pretrained(whisper_name, revision="973afd24965f72e36ca33b3055d56a652f456b4d")')
text=text.replace('max_seq_length=8192','max_seq_length=4096')
file.write_text(text)
import torch,torchaudio
torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS.get_model()
print('All inference models cached; no runtime model downloads required')
