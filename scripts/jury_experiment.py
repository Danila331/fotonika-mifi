"""Reproducible validation selection, frozen held-out comparison and PNG gallery.

Run from repository root: python scripts/jury_experiment.py validate|test|gallery
Test refuses to run until validation has frozen the configuration. Proxy weights
are not an official jury score. All numerical metrics use quantized phase PNGs.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
import evaluate as independent
from doe.infer import NeuralDOE
from doe.io import phase_to_u8
from doe.metrics import evaluate, crosstalk
from doe.quality import QualityConfig, refine_quality, exported_metrics, proxy_score
from doe.visualize import method_grid

OUT = ROOT / 'out/jury'
KEYS = ('pcc','ssim_global','ssim_skimage','ssim_gauss','pcc_worst','selectivity','proxy')
CONFIGS = {
    'long_balanced': QualityConfig(alpha=1.0,beta=.5),
    'global_emphasis': QualityConfig(alpha=.5,beta=1.0),
    'peak_finish': QualityConfig(alpha=.5,beta=1.0,peak_final=True),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync(device):
    if device=='mps': torch.mps.synchronize()


def metrics(phases, targets):
    rows=[]
    for p,t in zip(phases,targets):
        m=exported_metrics(p,t); m['proxy']=proxy_score(m); rows.append(m)
    return rows


def means(rows):
    return {k:float(np.nanmean([np.nan if r[k] is None else r[k] for r in rows])) for k in KEYS}


def run(net, targets, mode, config=None, bases=None, bs=4):
    phases=[]; choices=[]; sync(str(net.device)); start=time.perf_counter()
    for i in range(0,len(targets),bs):
        t=targets[i:i+bs]
        if mode=='fast': p=net.predict_batch(t)
        elif mode=='baseline': p=net.predict_refined_batch(t,100,xtalk=.25,alpha=1,beta=.5)
        else:
            p,c=refine_quality(net,t,config,baseline=bases[i:i+bs] if bases is not None else None)
            choices.extend(c)
        phases.extend(p)
        print(f'{mode}: {min(i+bs,len(targets))}/{len(targets)}',flush=True)
    sync(str(net.device))
    return np.array(phases), choices, time.perf_counter()-start


def save_run(folder,name,phases,targets,seconds,choices=None):
    folder.mkdir(parents=True,exist_ok=True)
    np.save(folder/f'{name}.npy',phases)
    rows=metrics(phases,targets)
    # Verify EVERY sample and mode from on-disk phase and reconstruction PNGs.
    for i,(phase,target) in enumerate(zip(phases,targets)):
        case=folder/name/f'{i:03d}';case.mkdir(parents=True,exist_ok=True)
        Image.fromarray(phase_to_u8(phase)).save(case/'doe.png')
        q=independent.read_png(case/'doe.png',True)
        ref=evaluate(q,target);check,recon=independent.evaluate(q,target)
        assert check['planes']==ref['planes'] and check['summary']==ref['summary']
        for j,img in enumerate(ref['recon'],1):
            Image.fromarray(img).save(case/f'recon_{j}.png')
            assert np.array_equal(independent.read_png(case/f'recon_{j}.png',True),recon[j-1])
            Image.fromarray(target[j-1]).save(case/f'target_{j}.png')
        (case/'metrics.json').write_text(json.dumps(check,indent=2))
    rec={'rows':rows,'mean':means(rows),'seconds':seconds,'seconds_per_quad':seconds/len(targets),'choices':choices or [],'independent_png_checks':len(targets)}
    (folder/f'{name}.json').write_text(json.dumps(rec,indent=2))
    print(name,rec['mean'],f'{seconds:.2f}s',flush=True)
    return rec


def prepare(device=None):
    from doe.data import QuadGenerator, ExternalBank
    if (OUT/'dataset-manifest.json').exists() or (OUT/'frozen.json').exists():
        raise SystemExit('Dataset already prepared; preserve the frozen experiment.')
    sources=ExternalBank.get().counts()
    if not all(k in sources for k in ('c101_photo','c101_obj','bsds_photo')):
        raise SystemExit('First run scripts/prepare_external.py --only c101 bsds')
    manifest={'sources':['Caltech-101','BSDS500'],
              'missing_sources':['Caltech-256','Flowers-102','DTD','Fashion-MNIST'],
              'note':'New quads, existing source split rules; not a pristine new-source benchmark.'}
    for split,seed in [('val',791901),('test',791902)]:
        arr,families=QuadGenerator(split,seed=seed).sample(256,return_families=True)
        np.save(ROOT/'data'/f'{split}.npy',arr)
        manifest[split]={'seed':seed,'shape':list(arr.shape),
                         'sha256':hashlib.sha256(arr.tobytes()).hexdigest(),
                         'families':families,'used_n':32 if split=='val' else 64}
    (OUT/'dataset-manifest.json').write_text(json.dumps(manifest,indent=2))


def validate(device):
    T=np.load(ROOT/'data/val.npy')[:32]; net=NeuralDOE.load(device=device)
    d=OUT/'validation'
    all_results={}
    for mode in ('fast','baseline'):
        p,c,s=run(net,T,mode)
        all_results[mode]=save_run(d,mode,p,T,s,c)
    base=np.load(d/'baseline.npy')
    for name,config in CONFIGS.items():
        p,c,s=run(net,T,name,config,base)
        all_results[name]=save_run(d,name,p,T,s,c)
    chosen=max(CONFIGS,key=lambda k:all_results[k]['mean']['proxy'])
    frozen={'quality_name':chosen,'config':asdict(CONFIGS[chosen]),'checkpoint':'checkpoints/best.pt',
            'baseline_sha256':digest(ROOT/'checkpoints/best.pt'),'val_sha256':digest(ROOT/'data/val.npy'),
            'proxy':'0.4 Pearson + 0.3 global SSIM + 0.3 Gaussian SSIM (renormalize undefined Pearson)',
            'validation':{k:v['mean'] for k,v in all_results.items()},'device':device}
    (OUT/'frozen.json').write_text(json.dumps(frozen,indent=2))
    print('FROZEN',chosen,flush=True)


def compare_finetune(device,checkpoint):
    T=np.load(ROOT/'data/val.npy')[:32]; net=NeuralDOE.load(checkpoint,device=device)
    d=OUT/'validation'; report={}
    for mode in ('fast','baseline'):
        p,c,s=run(net,T,mode)
        report[mode]=save_run(d,'finetuned_'+mode,p,T,s,c)
    frozen=json.loads((OUT/'frozen.json').read_text())
    frozen['finetuned_checkpoint']=str(Path(checkpoint).resolve().relative_to(ROOT))
    frozen['finetuned_sha256']=digest(checkpoint)
    frozen['finetuned_validation']={k:v['mean'] for k,v in report.items()}
    frozen['finetuned_fast_improves']=report['fast']['mean']['proxy']>frozen['validation']['fast']['proxy']
    # Quality continues from the original checkpoint to isolate the optimizer gain.
    (OUT/'frozen.json').write_text(json.dumps(frozen,indent=2))


def test(device):
    f=json.loads((OUT/'frozen.json').read_text())
    if (OUT/'test/summary.json').exists(): raise SystemExit('Held-out results already exist; do not retune/re-evaluate them.')
    assert digest(ROOT/f['checkpoint'])==f['baseline_sha256']
    T=np.load(ROOT/'data/test.npy')[:64];net=NeuralDOE.load(device=device);d=OUT/'test'; res={}
    for mode in ('fast','baseline'):
        p,c,s=run(net,T,mode);res[mode]=save_run(d,mode,p,T,s,c)
    p,c,s=run(net,T,'quality',QualityConfig(**f['config']),np.load(d/'baseline.npy'))
    res['quality']=save_run(d,'quality',p,T,s,c)
    if f.get('finetuned_checkpoint'):
        assert digest(ROOT/f['finetuned_checkpoint'])==f['finetuned_sha256']
        tuned=NeuralDOE.load(ROOT/f['finetuned_checkpoint'],device=device)
        for mode in ('fast','baseline'):
            p,c,s=run(tuned,T,mode);res['finetuned_'+mode]=save_run(d,'finetuned_'+mode,p,T,s,c)
    rng=np.random.default_rng(791903); idx=rng.integers(0,len(T),(4000,len(T)))
    comparisons={}
    for mode,reference in [('quality','baseline'),('finetuned_fast','fast'),('finetuned_baseline','baseline')]:
        if mode not in res: continue
        delta={}
        for k in KEYS:
            a=np.array([r[k] if r[k] is not None else np.nan for r in res[mode]['rows']])
            b=np.array([r[k] if r[k] is not None else np.nan for r in res[reference]['rows']])
            diff=a-b; ci=np.nanpercentile(np.nanmean(diff[idx],axis=1),[2.5,97.5])
            delta[k]={'mean':float(np.nanmean(diff)),'ci95':ci.tolist(),'wins':int((diff>1e-9).sum()),'losses':int((diff < -1e-9).sum())}
        comparisons[mode+'_vs_'+reference]=delta
    summary={'n':len(T),'device':device,'test_sha256':digest(ROOT/'data/test.npy'),'frozen':f,
             'means':{k:v['mean'] for k,v in res.items()},'paired_differences':comparisons,
             'runtime':{k:v['seconds_per_quad'] for k,v in res.items()},
             'quality_total_seconds_per_quad':res['baseline']['seconds_per_quad']+res['quality']['seconds_per_quad'],
             'quality_time_note':'quality runtime entry is additional; use quality_total_seconds_per_quad for total. Batched throughput, not single-image latency.'}
    manifest=json.loads((OUT/'dataset-manifest.json').read_text())
    from collections import Counter
    groups={}
    for i,families in enumerate(manifest['test']['families'][:len(T)]):
        family,count=Counter(families).most_common(1)[0]
        group=family if count>=3 else 'mixed'
        groups.setdefault(group,[]).append(i)
    summary['per_family']={g:{'n':len(ids),'means':{name:means([rec['rows'][i] for i in ids]) for name,rec in res.items()}} for g,ids in groups.items()}
    (d/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


def gallery(device):
    f=json.loads((OUT/'frozen.json').read_text()); config=QualityConfig(**f['config'])
    net=NeuralDOE.load(device=device); dest=OUT/'gallery';dest.mkdir(exist_ok=True)
    cases=[]
    tuned=NeuralDOE.load(ROOT/f['finetuned_checkpoint'],device=device) if f.get('finetuned_checkpoint') else None
    # Fixed first eight test examples, then ALL existing demo sets. No best-case selection.
    targets=np.load(ROOT/'data/test.npy')[:8]
    base=np.load(OUT/'test/baseline.npy')[:8];better=np.load(OUT/'test/quality.npy')[:8]
    for i,t in enumerate(targets): cases.append((f'test_{i:02}',t,base[i],better[i]))
    for demo in sorted((ROOT/'data/demo').iterdir()):
        if not demo.is_dir(): continue
        t=np.stack([np.asarray(Image.open(demo/f'target_{i}.png')) for i in range(1,5)])
        b=net.predict_refined_batch(t[None],100,xtalk=.25,alpha=1,beta=.5)
        p,_=refine_quality(net,t[None],config,b)
        cases.append((demo.name,t,b[0],p[0]))
    cards=[]
    for name,t,b,p in cases:
        rows=[];case_dir=dest/name;case_dir.mkdir(exist_ok=True)
        for i,img in enumerate(t,1): Image.fromarray(img).save(case_dir/f'target_{i}.png')
        modes=[('База +100','baseline',b),('Качество +300','quality',p)]
        if tuned is not None:
            modes=[('Исходная сеть','fast',net.predict(t)),('Дообученная сеть','finetuned_fast',tuned.predict(t))]+modes
        for label,tag,phase in modes:
            q=phase_to_u8(phase);Image.fromarray(q).save(case_dir/f'{tag}_doe.png')
            rep=evaluate(q,t);ev,rec=independent.evaluate(independent.read_png(case_dir/f'{tag}_doe.png',True),t)
            assert ev['summary']==rep['summary'] and np.array_equal(rec,rep['recon'])
            for i,img in enumerate(rec,1): Image.fromarray(img).save(case_dir/f'{tag}_recon_{i}.png')
            rows.append({'label':label,**rep,'crosstalk':crosstalk(t,rec)})
        method_grid(t,rows,case_dir/'comparison.png',title=name+' · изображения без постобработки')
        cards.append(f'<article><h2>{name}</h2><a href="{name}/comparison.png"><img loading="lazy" src="{name}/comparison.png" alt="Сравнение {name}"></a></article>')
    summary=json.loads((OUT/'test/summary.json').read_text())
    trs=''.join('<tr><td>'+k+'</td>'+''.join(f'<td>{v[m]:.4f}</td>' for m in ('pcc','ssim_global','ssim_gauss','selectivity'))+'</tr>' for k,v in summary['means'].items())
    (dest/'index.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Сравнение синтеза ДОЭ</title><style>body{font:16px system-ui;margin:32px auto;padding:0 20px;max-width:1100px;background:#fafaf8;color:#161616}img{width:100%;height:auto}article{margin:36px 0}td,th{padding:10px;border-bottom:1px solid #ccc;text-align:left}table{border-collapse:collapse}p{line-height:1.6}</style><h1>Сравнение синтеза ДОЭ</h1><p>64 отложенных комплекта. Выбор параметров: отдельные 32 валидационных комплекта. Ниже — первые 8 тестовых примеров и все 12 демо, без отбора по выигрышу. Восстановления непосредственно из сохранённого 8-битного ДОЭ; единая шкала 0–255, без ретуши. Дополнительное уточнение требует больше времени. Итоговая формула балла жюри неизвестна.</p><table><tr><th>Режим</th><th>Pearson</th><th>SSIM глоб.</th><th>SSIM окно</th><th>Селективность</th></tr>'+trs+'</table>'+''.join(cards)+'</html>')
    print(dest/'index.html',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','validate','finetune','test','gallery']);ap.add_argument('--device',default='mps');ap.add_argument('--checkpoint')
    a=ap.parse_args();torch.set_num_threads(4); OUT.mkdir(parents=True,exist_ok=True)
    if a.action=='finetune': compare_finetune(a.device,a.checkpoint)
    else: globals()[a.action](a.device)
