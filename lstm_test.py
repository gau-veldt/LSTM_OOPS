import math
import subprocess
import pygame
import random
from pybrain.structure import LSTMLayer,LinearLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
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


reportInterval=1000
epochCount=2500000
EOL=13

try:
    n=buildNetwork(1,30,1,hiddenclass=LSTMLayer,outclass=LinearLayer,peepholes=True,recurrent=True)
    n.sortModules()
    
    ds=SupervisedDataSet(1,1)
    targets=[
        list('Hi!')
        #list('Hello, World!'),
        #list('Division by zero'),
        #list('Stack overflow'),
        #list('Null pointer'),
        #list('Segment violation'),
        #list('Not a number'),
        #list('File not found'),
        #list('Permission denied'),
        #list('Out of memory'),
        #list('Illegal instruction')"""
    ]
    exampleCount=len(targets)
    
    
    print "Training sets:"
    for target in targets:
        prevCh=EOL
        for t in range(len(target)):
            targetCh=target[t]
            targetOrd=ord(targetCh)
            print "at t%s: i=%s t=%s '%s'" % (str(t).rjust(4,'0'),hex(prevCh)[2:].rjust(4,'0'),hex(targetOrd)[2:].rjust(4,'0'),targetCh)
            prevCh=targetOrd
    
    print "Training..."
    #trainer=BackpropTrainer(n,dataset=ds,learningrate=.01,momentum=.2,verbose=False)
    trainer=RPropMinusTrainer(n,dataset=ds,learningrate=.01,momentum=.2,verbose=False)
    
    epochs=0
    emse=float("Inf")
    
    # train to learn the sequence in time. in theory it should not need input at all
    # but I give the last character output as input to sequence t+1
    # limiting epochs just to see if it is training at all within in a fixed runtime
    while emse>=0.5 and epochs<epochCount and noskip:
        epochs+=1
        sTargs=[]+targets
        random.shuffle(sTargs)
        smse=0
        for target in sTargs:
            tCount=len(target)

            pumpGui()
            rendering=wantRender()
            if quitWasRequested():
                raise KeyboardInterrupt
    
            inpTicker=[' ']*tCount
            inpTicker=[' ']*tCount
            tgtTicker=[' ']*tCount
            outTicker=[' ']*tCount
            outVTicker=[0]*tCount
            
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
            
            n.reset()
            prevCh=EOL
            prevGuess=0.0
            se=0
            n.reset()  # reset to t0 for training sequence
            for i in range(len(target)):
                curCh=ord(target[i])
                tgtTicker[i]=chr(curCh)
                inpTicker[i]=chr(max(1,prevCh))
                ds.clear()
                # scales bytes to range [0,1]
                ds.addSample((prevCh/255.0),(curCh/255.0,))
                #ds.addSample((prevCh/255.0,prevGuess),(curCh/255.0,))
                err=trainer.train()
                prevGuess=n['out'].outputbuffer[0][0]
                outVh=min(255.0,max(0.0,255.0*prevGuess))
                outCh=int(round(outVh))
                outVh=outVh-outCh
                if 32>outCh or 127<outCh:
                    outCh=1.0
                outTicker[i]=chr(int(outCh))
                outVTicker[i]=round(20.0*outVh)
                se+=(255.0*err)**2
                prevCh=curCh
            mse=math.sqrt(se)
            smse+=mse**2
            
            if rendering:
                frac=str(mse).split('.')
                if len(frac)<2:
                    frac.append('')
                (w,f)=frac
                w=w.rjust(3,'0')
                f=f.ljust(11,'0')
                cap="e=%s.%s" % (w,f)
                text=font.render(cap,True,(255,255,255))
                visual.blit(text,(4+11*23,44))
                for i in range(tCount):
                    x=4+11*i
                    v=outVTicker[i]
                    abv=pygame.Rect(4,124,11*tCount,30)
                    blw=pygame.Rect(4,174,11*tCount,30)
                    text=font.render(inpTicker[i],True,(255,255,255))
                    visual.blit(text,(x,84))
                    text=font.render(tgtTicker[i],True,(255,255,255))
                    visual.blit(text,(x,104))
                    co=ord(outTicker[i])
                    text=font.render(chr(min(255,1+co)),True,(255,255,255))
                    visual.blit(text,(x,134+v))
                    text=font.render(chr(co),True,(255,255,255))
                    visual.blit(text,(x,154+v))
                    text=font.render(chr(max(1,co-1)),True,(255,255,255))
                    visual.blit(text,(x,174+v))
                    pygame.draw.rect(visual,(96,96,96),abv)
                    pygame.draw.rect(visual,(96,96,96),blw)
                    pygame.display.flip()
                    
        emse=math.sqrt(smse)
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
    
    n.reset()
    print "Test run:"
    inp=EOL
    lastGuess=0.0
    target=targets[0]
    for k in range(len(target)):
        out=n.activate((inp/255.0))
        #out=n.activate((inp/255.0,lastGuess))
        #lastGuess=out[0]
        rc=int(round(out[0]*255.0))
        irc=inp
        rCh='>255'
        if rc>0 and rc<256:
            if rc>=32 and rc<128:
                rCh="'"+chr(rc)+"'"
            else:
                rCh='non-printing'
        else:
            if rc<0:
                rc=65536+rc
                rCh='<0'
        ich=hex(irc)[2:].rjust(4,'0')
        ch=hex(rc)[2:].rjust(4,'0')
        print "t%s: i=%s o=%s %s" % (str(k).rjust(4,'0'),ich,ch,rCh)
        inp=ord(target[k])

    z=raw_input("Press enter to continue.")

except KeyboardInterrupt:
    print "Quitting."
finally:
    pygame.quit()
