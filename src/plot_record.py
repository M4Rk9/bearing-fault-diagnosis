"""Reproduce a raw-record waveform and spectrum using a prepared M4 manifest."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.data_pipeline import digest, raw_path, read_signal, validate_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--raw-dir',default='data/raw')
    parser.add_argument('--record-id',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    rows=json.loads(Path(args.manifest).read_text())
    validate_manifest(rows,require_coverage=False)
    matches=[r for r in rows if r['record_id']==args.record_id]
    if len(matches)!=1:
        raise ValueError('Choose exactly one existing record_id')
    row=matches[0]
    path=raw_path(args.raw_dir,row)
    if digest(path)!=row['sha256']:
        raise ValueError('Hash mismatch')
    x=read_signal(path,row['channel_key'])
    fs=row['sampling_rate_hz']
    shown=min(len(x),round(.2*fs))
    n=min(len(x),fs)
    centered=x[:n]-x[:n].mean()
    taper=np.hanning(n)
    spectrum=2*np.abs(np.fft.rfft(centered*taper))/taper.sum()
    spectrum[0]/=2
    if n%2==0:
        spectrum[-1]/=2
    fig,axes=plt.subplots(2,1,figsize=(10,6),layout='constrained')
    axes[0].plot(np.arange(shown)/fs,x[:shown],lw=.75,color='#1b6c8e')
    axes[0].set(xlabel='Time (s)',ylabel=f"Amplitude ({row['units']})",title='Raw waveform — first 0.2 s')
    axes[1].plot(np.fft.rfftfreq(n,1/fs),spectrum,lw=.75,color='#a94c25')
    axes[1].set(xlabel='Frequency (Hz)',ylabel=f"Amplitude ({row['units']})",title='First 1 s — mean removed, Hann, one-sided amplitude spectrum')
    fig.suptitle(f"CWRU {row['record_id']} | {row['class_name']} | {row['load_hp']} HP | {fs} Hz")
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=150)
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps({'record':row,'waveform_samples':shown,'fft_samples':n,
        'fft_taper':'symmetric Hann','fft_scaling':'one-sided amplitude / taper sum; DC and Nyquist not doubled'},indent=2)+'\n')


if __name__=='__main__':
    main()
