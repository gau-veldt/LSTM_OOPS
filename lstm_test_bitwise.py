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
    n=buildNetwork(8,20,8,hiddenclass=LSTMLayer,outclass=LinearLayer,peepholes=True,recurrent=True)
    n.sortModules()
    
    ds=SupervisedDataSet(8,8)
    targets=[
        list('Hello!'),
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
    while emse>=0.01 and epochs<epochCount and noskip:
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
            tgtTicker=[' ']*tCount
            outTicker=[' ']*tCount
            innies=[0]*8
            expect=[0]*8
            
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
            for i in range(tCount):
                curCh=ord(target[i])
                tgtTicker[i]=chr(curCh)
                inpTicker[i]=chr(max(1,prevCh))
                for b in range(8):
                    innies[7-b]=((2**b) & prevCh)/2**b
                    expect[7-b]=((2**b) & curCh )/2**b
                #print "I:",bin(prevCh),innies
                #print "T:",bin(curCh) ,expect
                ds.clear()
                # scales bytes to range [0,1]
                ds.addSample(innies,expect)
                #ds.addSample((prevCh/255.0,prevGuess),(curCh/255.0,))
                err=trainer.train()
                prevGuess=[n['out'].outputbuffer[0][b] for b in range(8)]
                outCh=0
                for b in range(8):
                    bv=min(1,max(0,int(round(prevGuess[7-b]))))
                    outCh=outCh+(2**b)*bv
                    if bv==1:
                        db=min(1.0,prevGuess[7-b])
                        fb=2.0*(db-0.5)
                        cv=108+int(round(147.0*fb))
                        cv=(cv,cv,cv)
                        cap="1"
                        x=4+11*i
                        y=290-20*b
                        sfc=font.render(cap,True,cv)
                        visual.blit(sfc,(x,y))
                    else:
                        db=max(0.0,prevGuess[7-b])
                        fb=2.0*(0.5-db)
                        cv=108+int(round(147.0*fb))
                        cv=(cv,cv,cv)
                        cap="0"
                        x=4+11*i
                        y=290-20*b
                        sfc=font.render(cap,True,cv)
                        visual.blit(sfc,(x,y))
                    
                if 32>outCh or 127<outCh:
                    outCh=1
                outTicker[i]=chr(outCh)
                se+=err**2
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
                    text=font.render(inpTicker[i],True,(255,255,255))
                    visual.blit(text,(x,84))
                    text=font.render(tgtTicker[i],True,(255,255,255))
                    visual.blit(text,(x,104))
                    text=font.render(outTicker[i],True,(255,255,255))
                    visual.blit(text,(x,124))
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
        for b in range(8):
            innies[7-b]=((2**b) & inp)/2**b
        out=n.activate(innies)
        #out=n.activate((inp/255.0,lastGuess))
        #lastGuess=out[0]
        rv=[out[b] for b in range(8)]
        rc=0
        for b in range(8):
            rc=rc+(2**b)*min(1,max(0,int(round(rv[7-b]))))
        rCh=chr(rc)
        irc=inp
        if rc>=32 and rc<128:
            rCh="'"+chr(rc)+"'"
        else:
            rCh='non-printing'
        ich=hex(irc)[2:].rjust(4,'0')
        ch=hex(rc)[2:].rjust(4,'0')
        print "t%s: i=%s o=%s %s" % (str(k).rjust(4,'0'),ich,ch,rCh)
        inp=ord(target[k])

    z=raw_input("Press enter to continue.")

except KeyboardInterrupt:
    print "Quitting."
finally:
    pygame.quit()
