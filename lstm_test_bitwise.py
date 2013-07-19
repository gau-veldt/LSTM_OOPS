import sys
import math
import subprocess
import pygame
import random
from pybrain.structure import LSTMLayer,LinearLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SequentialDataSet
from pybrain.supervised.trainers import BackpropTrainer,RPropMinusTrainer

pygame.init()
size=(800,400)
visual=pygame.display.set_mode(size,pygame.DOUBLEBUF)
entireWindow=pygame.Rect(0,0,800,400)
elapsed=0
since=pygame.time.get_ticks()
begin=since
total=0
framerate=1000.0/20.0
font=pygame.font.SysFont("courier",18)
_term=False
_render=False
noskip=True

Entropy=random.SystemRandom()

def quitWasRequested():
    global _term
    return _term

def wantRender():
    global _render
    return _render

def pumpGui():
    global _term,_render,begin,since,elapsed,total,framerate
    _render=False
    for evt in pygame.event.get():
        if evt.type==pygame.QUIT:
            _term=True
        if evt.type==pygame.KEYDOWN:
            global noskip
            noskip=False

    now=pygame.time.get_ticks()
    total=now-begin
    delta=now-since
    elapsed+=delta
    since=now
    if elapsed>framerate:
        _render=True
        while elapsed>framerate:
            elapsed-=framerate

epochCount=2500000
errGoal=0.001

try:
    n=buildNetwork(1,83,8,hiddenclass=LSTMLayer,outclass=LinearLayer,peepholes=True,recurrent=True)
    n.sortModules()

    ds=SequentialDataSet(1,8)
    stories=[
        "See Bill.\nSee Bill run.\nSee Bill run Windows.\nSee Windows crash.\nCrash Windows, Crash!\n",
    ]
    targets=[
        list(s) for s in stories
    ]
    
    expect=[0]*8
    for t in targets:
        ds.newSequence()
        target=t
        for ch in target:
            curCh=ord(ch)
            for b in range(8):
                expect[7-b]=((2**b) & curCh )/2**b
            ds.addSample([0.0],[]+expect)
            #print sentenceIndex,curCh

    print "Training..."
    #trainer=BackpropTrainer(n,dataset=ds,learningrate=.01,momentum=.2,verbose=False)
    trainer=RPropMinusTrainer(n,dataset=ds,learningrate=.01,momentum=.2,verbose=False)
    
    epochs=0
    emse=float("Inf")
    
    # train to learn the sequence in time. in theory it should not need input at all
    # but I give the last character output as input to sequence t+1
    # limiting epochs just to see if it is training at all within in a fixed runtime
    while emse>=errGoal and epochs<epochCount and noskip:
        epochs+=1
        smse=0
        target=targets[-1]
        tCount=len(target)

        pumpGui()
        rendering=wantRender()
        if quitWasRequested():
            raise KeyboardInterrupt

        if rendering:
            pygame.draw.rect(visual,(96,96,96),entireWindow)
            cap="Epochs: %s/%s" % (epochs,epochCount)
            text=font.render(cap,True,(255,255,255))
            visual.blit(text,(4,4))
            eps=epochs*60000/total
            tm=total
            tm=int(tm/1000)
            s=str(tm%60).rjust(2,'0')
            tm=int(tm/60)
            m=str(tm%60).rjust(2,'0')
            tm=int(tm/60)
            h=str(tm%24).rjust(2,'0')
            tm=int(tm/24)
            d=str(tm).rjust(3,'0')
            cap="  Time: %sd%sh%sm%ss  %s epochs/min" % (d,h,m,s,eps)
            text=font.render(cap,True,(255,255,255))
            visual.blit(text,(4,24))
            eta=(total*epochCount/epochs)-total
            tm=eta
            tm=int(tm/1000)
            s=str(tm%60).rjust(2,'0')
            tm=int(tm/60)
            m=str(tm%60).rjust(2,'0')
            tm=int(tm/60)
            h=str(tm%24).rjust(2,'0')
            tm=int(tm/24)
            d=str(tm).rjust(3,'0')
            cap="   ETA: %sd%sh%sm%ss" % (d,h,m,s)
            text=font.render(cap,True,(255,255,255))
            visual.blit(text,(4,44))
        
        n.reset()  # reset to t0 for training sequence
        emse=trainer.train()

        if rendering:
            xOff=4+11*23
            eRect=pygame.Rect(xOff,44,11*17,20)
            pygame.draw.rect(visual,(96,96,96),eRect)
            frac=str(emse).split('.')
            if len(frac)<2:
                frac.append('')
            (w,f)=frac
            w=w.rjust(3,'0')
            f=f.ljust(11,'0')
            cap="e=%s.%s" % (w,f)
            text=font.render(cap,True,(255,255,255))
            visual.blit(text,(xOff,44))
            pygame.display.flip()

    print emse,epochs,noskip
    
    n.reset()
    sentenceIndex=0
    iters=0
    print "Test run:"
    sys.stdout.write("%s: " % sentenceIndex)
    sys.stdout.flush()
    while sentenceIndex<5 and iters<100:
        iters+=1
        out=n.activate([0])
        #out=n.activate((inp/255.0,lastGuess))
        #lastGuess=out[0]
        rv=[out[b] for b in range(8)]
        rc=0
        for b in range(8):
            rc=rc+(2**b)*min(1,max(0,int(round(rv[7-b]))))
        rCh=chr(rc)
        prCh=rCh
        if rc>=32 and rc<128:
            prCh="'"+chr(rc)+"'"
            sys.stdout.write(rCh)
        else:
            prCh='non-printing'
            sys.stdout.write(chr(127))
        sys.stdout.flush()
        #ch=hex(rc)[2:].rjust(4,'0')
        #print "t%s: i=%s o=%s %s" % (str(iters).rjust(4,'0'),sentenceIndex,ch,prCh)
        if rCh=='\n':
            sentenceIndex+=1
            iters=0
            sys.stdout.write("\n%s: " % sentenceIndex)
            sys.stdout.flush()
    print "[EOF]"

except KeyboardInterrupt:
    print "Quitting."
finally:
    pygame.quit()
