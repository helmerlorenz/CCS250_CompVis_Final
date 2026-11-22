"""Retrieve trusted upstream architecture/checkpoint files and resolve active weights."""
from pathlib import Path
import hashlib
import os
import shutil
import tempfile
from urllib.request import Request, urlopen
from .models import ROOT, checkpoint_path

CACHE=ROOT/'cache'
UPSTREAM=CACHE/'upstream'
SOURCE_URLS={
 'restormer':'https://raw.githubusercontent.com/swz30/Restormer/main/basicsr/models/archs/restormer_arch.py',
 'nafnet':'https://raw.githubusercontent.com/megvii-research/NAFNet/main/basicsr/models/archs/NAFNet_arch.py',
}
WEIGHT_URLS = {
    'restormer': 'https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth',
    'dncnn': 'https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth',
WEIGHT_URLS={
 'restormer':'https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_color_denoising_blind.pth',
 'dncnn':'https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth',
}
NAF_REPO='mikestealth/nafnet-models'
NAF_FILE='NAFNet-SIDD-width32.pth'
NAF_SHA256='89c70e808d1783b6c07911306e106aaf0d4f7f3da8c61078b99ff7f8929a26f4'


def download(url: str, destination: Path, min_bytes: int=1024, sha256: str|None=None):
 destination.parent.mkdir(parents=True,exist_ok=True)
 if destination.exists() and destination.stat().st_size>=min_bytes:
  if sha256 is None or hashlib.sha256(destination.read_bytes()).hexdigest()==sha256:return destination
 with tempfile.NamedTemporaryFile(dir=destination.parent,prefix='partial_',delete=False) as temp: part=Path(temp.name)
 try:
  request=Request(url,headers={'User-Agent':'iMAGE-research/1.0'})
  with urlopen(request,timeout=120) as response,part.open('wb') as output:shutil.copyfileobj(response,output)
  if part.stat().st_size<min_bytes:raise IOError(f'Incomplete download ({part.stat().st_size} bytes): {url}')
  if sha256 and hashlib.sha256(part.read_bytes()).hexdigest()!=sha256:raise IOError('Downloaded file failed SHA-256 verification.')
  part.replace(destination);return destination
 finally:part.unlink(missing_ok=True)


def architecture_path(name: str)->Path:
 if name=='dncnn':raise ValueError('DnCNN is defined locally.')
 if name not in SOURCE_URLS:raise ValueError(name)
 return download(SOURCE_URLS[name],UPSTREAM/(name+'_arch.py'),min_bytes=3000)


def pretrained_path(name: str) -> Path:
    if name == 'nafnet':
        destination = CACHE / 'pretrained' / NAF_FILE
        # HF Hub caches the downloaded model; this app keeps a stable path too.
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == NAF_SHA256:
            return destination
        from huggingface_hub import hf_hub_download
        location = hf_hub_download(repo_id=NAF_REPO, filename=NAF_FILE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(location, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != NAF_SHA256:
            destination.unlink(missing_ok=True)
            raise IOError('NAFNet checkpoint mirror has an unexpected SHA-256 digest.')
        return destination
    if name not in WEIGHT_URLS:
        raise ValueError(name)
    filename = 'restormer_real.pth' if name == 'restormer' else name + '.pth'
    return download(WEIGHT_URLS[name], CACHE / 'pretrained' / filename, min_bytes=1_000_000)
def pretrained_path(name: str)->Path:
 if name=='nafnet':
  destination=CACHE/'pretrained'/NAF_FILE
  if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest()==NAF_SHA256:return destination
  from huggingface_hub import hf_hub_download
  location=hf_hub_download(repo_id=NAF_REPO,filename=NAF_FILE)
  destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(location,destination)
  if hashlib.sha256(destination.read_bytes()).hexdigest()!=NAF_SHA256:
   destination.unlink(missing_ok=True);raise IOError('NAFNet checkpoint mirror has unexpected SHA-256.')
  return destination
 if name not in WEIGHT_URLS:raise ValueError(name)
 return download(WEIGHT_URLS[name],CACHE/'pretrained'/(name+'.pth'),min_bytes=1_000_000)


def get_active_checkpoint(name: str)->Path:
 """Use explicitly selected folder strictly; otherwise deployed weights then baseline."""
 selected=os.environ.get('IMAGE_CHECKPOINT_DIR','').strip()
 if selected:
  path=checkpoint_path(name)
  if not path.is_file():
   raise FileNotFoundError(f'IMAGE_CHECKPOINT_DIR is set to {path.parent}, but {path.name} is missing. Refusing to silently load another model version.')
  return path
 local=ROOT/'weights'/(name+'.pth')
 return local if local.is_file() else pretrained_path(name)
