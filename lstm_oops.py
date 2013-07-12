#! /usr/bin/python
"""
    Python LSTM (Long short term memory)
    with OOPS training (Optimal Ordering Problem Solver)
    implementation.

    OOPS is an alternative training method that avoids some
    of the drawbacks in back-propogation (ie: local minima)
    training.  An OOPS trainer also allows both supervised
    (training examples) and unsupervised (reinforcement learning)
    training.

    Copyright (C) 2013 Christopher BRIAN Jack (gau_veldt@hotmail.com)
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import pygame,os
pygame.init()
size=(640,320)
visual=pygame.display.set_mode(size,pygame.DOUBLEBUF)
elapsed=0
since=pygame.time.get_ticks()
framerate=1000.0/60.0
font=pygame.font.SysFont("courier",18)
textregion=pygame.Rect(0,230,640,30)

import math
import random
import sys
import pprint
from functools import partial

NDEBUG=False

EntropySource=random.SystemRandom()
Formatter=pprint.PrettyPrinter(indent=2)

def blackhole(*args,**kwargs):
    pass
if NDEBUG:
    debug=blackhole
else:
    debug=print

def log(self,msg,which='testLog'):
    target=self.logs[which]
    target.append(msg)
    target=target[-100:]

def last(self,which='testLog'):
    try:
        return self.logs[which][-1]
    except:
        pass

_serNo=0
def serNo():
    global _serNo
    _serNo+=1
    return _serNo

log.logs={}
log.logs['testLog']=[]
log.logs['solveLog']=[]
log.last=partial(last,log)
log.log=partial(log,log)

halfPi=math.pi/2.0
twoPi=math.pi*2.0

def sigmoid(x):
    """
    Sigmoid function
    """
    rc=1/(1+math.exp(-x))
    return rc

def searchCurve(a):
    if a<halfPi:
        return 1.0-math.sin(a)
    else:
        return -1.0-math.sin(a)

class TopologyError(Exception):
    """ When something goes wrong in topology """
    def __init__(self,val):
        self.value=val
    def __str__(self):
        return self.value


class Terminal:
    def __init__(self,*args,**kwargs):
        self.value=0.0
        self.serNo=serNo()
    def write(self,val,**kwargs):
        self.value=val
    def read(self,**kwargs):
        return self.value        
    def __eq__(self,other):
        return hash(self)==hash(other)
    def __hash__(self):
        return self.serNo
    def __lt__(self,other):
        return self.serNo<other.serNo

class Input(Terminal):
    def __init__(self,*args,**kwargs):
        super(Input,self).__init__(*args,**kwargs)
    def __str__(self):
        return "ITERM+%s=%s" % (self.serno,self.value)

class Output(Terminal):
    count=0
    def __init__(self,*args,**kwargs):
        super(Output,self).__init__(*args,**kwargs)
    def __str__(self):
        return "OTERM%s=%s" % (self.serno,self.value)


class Topology:
    """
    Maintains ANN/RNN network topology
    
    Manages node-to-node connections,
    connection weights and activation traversal.
    Connections may contain cycles (recurrent network).
    
    When an input changes network will reflect it
    immediately.  The nodes will be activated in
    hop count order recursively so long as it has
    not already activated on this input's time step.
    This allows local adaptation to any asymmetry of
    indvidual input's information flow and timing of
    individual input change.
        
    Nodes must provide methods AvailableConnectionPoints(),
    Activate() and read("connName").
    
    AvailableConnectionPoints():
        returns a dictionary of connection types:
        { "connName" : connType, ...}
        connType is one of:
        0: input  - connName is a sink
        1: output - connName is a source
    
    Activate() will be called to run node cycle
    
    read("connName")
        - reads value connName
    
    """
    def __init__(self,*args,**kwargs):
        self.connections={}
        self.nodeRefs={}
        self.outRefs={}
        self.SquishOutput=True

    def enableOutputLogistic(enable=True):
        """
        Enables or disables logistical
        squishing of output terminal values
        """
        self.SquishOutput=enable

    def Connect(self,source,sink):
        """
        Connects specified connection points
        
            source: 2-tuple or list of 2-tuples or None
                    signal source (from an output port)
                    When source is None creates an input terminal

            sink: 2-tuple or list of 2-tuples or None
                  signal destination (to an input port)
                  When sink is None creates an output terminal

        Each specified source will be connected all specified sinks

        Input or output terminal creation returns an appropriate object

        Duplicated connection edges are silently filtered
        """
        if source is None and sink is None:
            # I/O terminal pair with no intervening nodes is likely a user error
            raise TopologyError("Attempt to create I/O terminal pair that bypasses nodes.")

        # Create input or output terminal
        newbie=None
        if source is None:
            newbie=Input()
            source=(newbie,Input)
        if sink is None:
            newbie=Output()
            sink=(newbie,Output)

        # make sure Inputs and Outputs aren't incorrectly placed
        if type(source[0])==Output:
            raise TopologyError("Output terminals as sources not permitted.")
        if type(sink[0])==Input:
            raise TopologyError("Input terminals as sinks not permitted.")

        # normalize when terminals provided
        if type(sink[0])==Output:
            # the sneaky caller didn't use auto-create for
            # output terminals but we still need outRefs to them
            self.outRefs[sink[0]]=1
            # normalize output terminal's channel designation
            sink=(sink[0],Output)

        if type(source[0])==Input:
            # normalize input terminal's channel designation
            source=(source[0],Input)

        # make single tuples lists of tuples
        if not isinstance(source,list):
            origPoints=[source]
        else:
            origPoints=source
        if not isinstance(sink,list):
            destPoints=[sink]
        else:
            destPoints=sink

        # validate connection specs
        origCount=len(origPoints)
        idx=0
        for CP in origPoints+destPoints:
            # origPoints want an output channel
            isSource=1
            if idx>=origCount:
                # destPoints want an input channel
                isSource=0
            try:
                cpNode,cpChan=CP
            except TypeError:
                raise TopologyError("Connect: %s not of form (x,y) or [x,y]" % CP)
            if not cpChan in [Input,Output]:
                try:
                    aCPs=cpNode.availableConnectionPoints()
                except AttributeError:
                    raise TopologyError("Connect: %s: %s is not a valid node" % (CP,cpNode))
                if cpChan not in aCPs.keys():
                    raise TopologyError("Connect: node %s: no such port %s" % (cpNode,cpChan))
                if aCPs[cpChan]!=isSource:
                    raise TopologyError("Connect: connection %s is not an %s" % (\
                        CP, ['input','output'][isSource]))
            else:
                if cpChan==Input and not isSource==1:
                    raise TopologyError("Connect: illegal incoming connection to input terminal")
                if cpChan==Output and isSource==1:
                    raise TopologyError("Connect: illegal outgoing connection from output terminal")
            idx=idx+1

        # make connections
        for orig in origPoints:
            for dest in destPoints:
                # store connection
                self.connections[(orig,dest)]=1.0
                # memoize nodes involved with connection
                # to improve performance of Activate()
                if orig[1]!=Input:
                    self.nodeRefs[orig[0]]=True
                if dest[1]!=Output:
                    self.nodeRefs[dest[0]]=True
        # returns the input or output terminal if one was created
        # otherwise None
        return newbie

    def getTargets(self,source,channel):
        """
        Get list of connection destinations for the specified source
        """
        found=[]
        for (src,srcChan),(dest,destChan) in self.connections:
            if src==source and srcChan==channel:
                found.append((dest,destChan))
        return found

    def getInputs(self,sink,channel):
        """
        Gets list of all weighted values feeding to the specified sink
        """
        found=[]
        sep=''
        for (src,srcChan),(dest,destChan) in self.connections:
            if dest==sink and destChan==channel:
                w=self.connections[((src,srcChan),(dest,destChan))]
                value=w*src.read(channel=srcChan)
                found.append(value)
                sep=','
        return found

    def Activate(self):
        """
        Activates network nodes

        Input and Output terminals are not passes in themselves but
        nodes with sinks connected to Input terminals will receive
        the values of the corresponding terminals.  The output
        levels are sampled (ie: the Output nodes written with
        values) after all nodes activate.            
        """
        # activate each node
        for n in self.nodeRefs:
            n.Activate(self)
        # node activations are done now activate each output
        for o in self.outRefs:
            sigma=sum(self.getInputs(o,Output))
            if self.SquishOutput:
                o.write(sigmoid(sigma))
            else:
                o.write(sigma)


class NodeError(Exception):
    """ when something goes wrong in a node """
    def __init__(self,val):
        self.value=val
    def __str__(self):
        return self.value


class LSTM_Node:
    """
    Long Short Term Memory node

    The above extra gates could be useful in some topologies to enable
    localized problem space searching within the netowrk itself.

    has inputs:
        input       - input
        inputGate   - input attenuator
        forgetGate  - internal state attenuator
        outputGate  - output attenuator

    has outputs:
        output   - node's output
        peephole - node's internal state

    optional:
        output activation function
        
    """
    iConns=["input","inputGate","outputGate","forgetGate"]
    oConns=["output","peephole"]
    connMap={
            k:v for (k,v) in \
            [(x,0) for x in iConns] + \
            [(x,1) for x in oConns]
        }
    def __init__(self,*args,**kwargs):
        """ sets up node """
        self.CEC=0.0
        self.serNo=serNo()
        self.states={k:v for (k,v) in [(x,0.0) for x in LSTM_Node.oConns]}
        for chan in LSTM_Node.iConns:
            self.states[chan]={
                "value":None,
                }
    def __lt__(self,n):
        return self.serNo<n.serNo
    def __str__(self):
        info="{'CEC':%s,'peephole':%s,'output':%s,'inputs':%s}" % (\
        self.CEC,self.states['peephole'],\
        self.states['output'],\
        { k:self.states[k] for k in LSTM_Node.iConns})
        return Formatter.pformat(eval(info))
        
    def availableConnectionPoints(self,**kwargs):
        """ enumerate connection points """
        if "InputOnly" in kwargs:
            return LSTM_Node.iConns
        if "OutputOnly" in kwargs:
            return LSTM_Node.oConns
        return LSTM_Node.connMap
    
    def read(self,**kwargs):
        """
        Query a channel's value
        """
        if "channel" in kwargs:
            ch=kwargs["channel"]
            if ch in LSTM_Node.connMap.keys():
                if ch in LSTM_Node.oConns:
                    return self.states[ch]
                else:
                    return self.states[ch]['value']
            else:
                raise NodeError("LSTM_Node: no such channel '%s'" % ch)
        else:
            raise NodeError("LSTM_Node: read() must specify a channel")
    
    def Activate(self,net):
        """
        perform activation pass
        """
        # activate input and scale to [-2,2]
        self.states['input']['value']=4.0*sigmoid(sum(net.getInputs(self,'input')))-2.0
        # activate inputGate
        self.states['inputGate']['value']=sigmoid(sum(net.getInputs(self,'inputGate')))
        # compute gated input
        gatedInput=self.states['input']['value']*self.states['inputGate']['value']
        # apply input to internal state
        self.CEC=self.CEC+gatedInput
        # activate forget gate
        self.states['forgetGate']['value']=sigmoid(sum(net.getInputs(self,'forgetGate')))
        # gate internal state (applies forgetfulness)
        self.CEC=self.CEC*self.states['forgetGate']['value']
        # squish the ungated output (peephole)
        self.states['peephole']=sigmoid(self.CEC)
        # activate output gate
        # NB: This is done after squished peephole value is known 
        # so that the fresh CEC state is visible to the output gate
        self.states['outputGate']['value']=sigmoid(sum(net.getInputs(self,'outputGate')))
        # gate (already squished) output
        self.states['output']=self.states['peephole']*self.states['outputGate']['value']


class OOPS:
    """
    OOPS - Optimal Ordered Problem Solver
    A little about OOPS and how it is being implemented:

    A true OOPS is more complicated and I've simplified it
    to work in constant storage with fixed search cycles.
    Some of the theoretical aspects like changing the search
    algorithm are not imeplemented.  A general OOPS blows up
    in storage requirement due to the need to store *all*
    prior solutions.  I am using a fixed solution store
    ordered by increasing error (or decreasing fitness).

    OOPS basically solves a sequence problems and may use
    previous solutions as the basis to solve future problems.
    
    For this trainer case the goal (problem) is defined as:
        Yielding a weight vector that reduces training error
        (supervised) or increases fitness score (unsupervised).

    So any vector of weights that achieves the goal is a solution
    to the problem.  The problem is defined incrementally to make
    the implementation on-line friendly.

    Once a solution is found OOPS algorithm is to add the solution
    to a holding space for solution examples then present the solution.
    The implementation of presenting the solution is to apply the
    weights of the solution vector to the current Topology.
    
    OOPS would then solve the next problem.  Here we obtain a new
    problem from the incremental definition by rememebring the gain
    made by the solution (reduced error or gained reward).  This
    effectively yields the next problem with "the bar slightly raised"
    whose goal is a further reduction of error or gain of reward.  By
    repeating the cycle over many epochs it becomes possible to train
    the network to a desired behavior.

    When a new network is created it has no previous solutions on record.
    Furthermore OOPS specifies that timeslices should both look through
    solutions on record AND explore the problem space for novel solutions.

    I implement this search by copying a weight vector then applying
    a random number of randomly chosen evolution operators (radical,
    sign-invert, splice, swap and transpose):

        radical: some weight in the vector is replaced by a whole new
                 randomly chosen value.

        sign-invert: one weight in the vector's sign is inverted.

        splice: 50% of the vector is randomly overwritten with the
                contents of a weighted random selection of a previous
                solution's vector.  The weighting curve (1.0-cos(x),
                0.0<=x<=Pi/2.0) favors elements at the top of the list
                since they are the highest fitness.  Splice will not
                be chosen when there is only one solution in the solution
                store (first solution is initialized to the initial
                network when trainer is created).

        swap: swap some weight in the vector with some other weight.

        transpose: one weight in the vector is swapped with a neighbour.

    -- Backtracking --
    
    Since LSTM is recurrent and has states that change over time (the
    internal LSTM states, known as Constant Error Carousels or CECs for
    short) provides a unique source of possible solutions by introspecting
    the states of the Topology's CECs in solutions.

    I implement a very simple form of backtracking.  Whenever a solution
    is recorded, so is the current state of CECs in the Topology,
    essentially, the timestamp the solution was found.
    I will call it as such.

    NB: My search algorith is sterile right now it doesn't
        actually move in time between searches effectively.
        Might be faster to have less cross passes and rely more on the
        affector.

    I do four different search operations in an epoch:
    
        1. evaluate all previous soltuions at current timestamp
        2. generate a number of mutated weight vectors
        3. test the vectors in (2) at current timestamp
        4. test the vectors in (2) at timestamps of all previous solutions

    I can describe these steps in humorous layman:
        1. See what happens if ancient mutants were living now.
        2. "Honey, I want some mutants! Bring forth the plutonium
            and let's make some passionate glowing radioactive love."
        3. "Honey, how well did our mutant kids do in the educatatron
            today?"
        4. "Honey get the time machine!  I want to see how our precious
            mutant kids do in the past."

    So it can be gleaned the the search is going to check for new solutions
    that might work better if they were done in the previous timestamps.

        - First off the current timestamp is saved: TS_now
        - search result initialized to Topology's current weighting,
          timestamp (TS_now), and fitness level
        - Each previous solution is tried at TS_now
            - any better performing result replaces current search result
        - generate some number mutants
        - for each mutant
            - test mutant at TS_now
            - if better the current search result is replaced
            - test mutant at all past timestamps (of previous solutions)
                - any better-performing result replaces current search
                  result
                  
    After the above procedure the best search result is moved to the top
    of the solution store and when the store is already at capcity the
    worst (bottom) entry is first discarded to make room.  The result
    is also stored into the Topology's weights and CECs and the error
    or fitness of the result is remembered for the next epoch.  It is
    possible all terms fail to find a better alternative and the result
    will be that the Topology is unmodified (since the best found result
    is initialized with the current Topology's state and fitness first).

    Viola!
    We have an OOPS/EVOLINO hybrid.

    Some apporaches to evaluate each search term:
        1 evaluate a training set (invert sign of error so that negative
          inidicates higher error)
        2 run a simulation that can yield a performance factor
        3 if your net writes stories put em on a website and get
          crowdsourced ratings.  Speed things up a bit by using an
          SVR to approximate ratings using regression (human crowd
          sourcing greatly slows down the training process otherwise).
        4 similar process to 3 if your net draws art, makes music, etc.

    Note: It is possible to change fitness test regime if desired.

    If the net is forgetting too soon it needs more nodes, better
    connectivity, or an alternate topology.  Try arranging with nodes in
    hypercubes where nodes connect only to immediate neighbours in the
    hypercube, input terminals, or output terminals; however, be certain
    each node connects to its neighbours on all axes (in a cube each node
    has 3 neighbours, 4 in a tesseract, 5 in a pentaract, etc.)

    The hypercube formation forces the network to learn to cluster
    information and flow it towards the outputs.  BTW the number of CECs
    in a hypercube Topology will be 2^N where N is the dimension of the
    hypecube.  To keep connectivity reasonable try only connecting
    neighbouring LSTM gates, excepting the input terminals which should
    connect to all gates of gates of all nodes connected to input
    terminals.  A similar crossing should exist on the output.  Each
    output terminal should get as input the outputs of all nodes that feed
    an output terminal.
    """
    def __init__(self,*args,**kwargs):
        """
        Create the OOPS Trainer

            NB: !!! Do not add or modify connections on the
                    Topology once a trainer is created for it !!!

                The stored solutions list only weights and CEC states to
                minimize storage, and the length and ordering of these lists
                depends on the ordering found in net.connections and
                net.nodeRefs when the Trainer was created.
        
            Arguments:
                Topology     - The Topology for trainer to operate on
                maxSolutions - Solution store maximum size (default 1000)        
        """
        self.maxSolutions=1000
        if 'maxSolutions' in kwargs:
            self.maxSolutions=kwargs['maxSolutions']
        self.maxSolutions=max(1,self.maxSolutions)

        # make list of mutation operator references
        self.mutationOps=[getattr(self,'mutate%s'%i) for i in [
            'Splice','Radical','Sign','Swap','Transpose','Tumor']]
        
        self.net=None
        if 'Topology' in kwargs:
            self.net=kwargs['Topology']
        if self.net is None:
            raise TypeError("OOPS: No network specified.")

        # randomize initial weights
        for c in self.net.connections:
            self.net.connections[c]=EntropySource.uniform(-6,6)
        for n in self.net.nodeRefs:
            n.CEC=EntropySource.uniform(-6,6)
            n.output=EntropySource.uniform(0,1)

        """
        arguments to evaluator:
            topology: the network's Topology object
            
        returns:
            resulting rank of solution
            
        NB: network state will be configured for the
            candidate and timestamp (CEC state), and thus
            completely ready to test, when called
        """
        self.solutions=[]
        self.evalfunc=None
        if 'Evaluator' in kwargs:
            self.changeEvaluator(kwargs['Evaluator'])
        if self.evalfunc is None:
            raise TypeError("OOPS: No evaluator specified.")

        """
        when net improves we remember which weights were changed
        and the activity is weighted by the degree of change that
        resulted.  Whenever updated the vector is normalized
        to have maximum activity 1.0

        so genarlly whenever a weight participates in a better result
        its activity increases by 1.0 however we will scale the change
        by the net change in fitness from the previous best

        also 1.0 assumes weights are always modified by a fixed constant
        or not at all so I further scale by the distance of the change
        made to the weight.

        the noise inducers have a choice to either cause more noise
        to affective "good" weights that have raised fitness or cause noise
        on "bad" weights that have lowered fitness.

        The affect will basically allow filtering modifications to allow
        those that have the best history of improving fitness and attenuate
        the ones that negated the fitness.  It's a very poor man's analogue
        to a kalman filter and as such uses no matrix math nor derivatives.

        Again I want to avoid gradient based training since it has the
        issue of local minima.  I also want something adaptive (the fitness
        function is ALLOWED to change).  The ability of the fitness metric
        to change is exactly why 50% of the search should be in "bad"
        weightspace and 50% in known "good" weightspace.

        We will do 50% of each

        Ideally affect would be managed separate such that one affect vector
        is tracker for every internal RNN state reached.  This has intractable
        storage requirements.

        the affect is a global training aspect and should
        be updated by all tests (every mutant uddates this metric)

        we are trying to make the mutator smarter over time by coming up with
        some data about where in the weightspace to do mutations
        """
        self.resetAffect()
        
        self.currentSolves=0

    def evaluator(self,net,**kwargs):
        global visual,elapsed,since,framerate,font,textregion
        
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                raise KeyboardInterrupt
        now=pygame.time.get_ticks()
        delta=now-since
        elapsed+=delta
        since=now
            
        weightCount=len(self.net.connections)
        
        newRk=self.evalfunc(net)
        self.minFitness=min(self.minFitness,newRk)
        self.maxFitness=max(self.maxFitness,newRk)
        
        if 'original' in kwargs and 'current' in kwargs and 'originalFitness' in kwargs:
            org=kwargs['original']
            cur=kwargs['current']
            oldRk=kwargs['originalFitness']
            self.updateAffect(org,cur,newRk-oldRk)
            
            if elapsed>framerate:
                
                while elapsed>framerate:
                    elapsed-=framerate

                for x in range(weightCount):
                    bar=100*self.weightAffect[x]
                    xbar=100-bar
                    pygame.draw.line(visual,(0,0,0),(16+x*6,10),(16+x*6,10+xbar),5)
                    pygame.draw.line(visual,(0,255,255),(16+x*6,110),(16+x*6,110-bar),5)
                    prevBar=100*sigmoid(org[x])
                    xPrevBar=100-prevBar
                    curBar=100*sigmoid(cur[x])
                    xCurBar=100-curBar
                    pygame.draw.line(visual,(0,0,0),  (16+x*6,120),(16+x*6,120+xPrevBar),2)
                    pygame.draw.line(visual,(255,0,255),(16+x*6,220),(16+x*6,220-prevBar),2)
                    pygame.draw.line(visual,(0,0,0),  (19+x*6,120),(19+x*6,120+xCurBar),2)
                    pygame.draw.line(visual,(128,0,255),(19+x*6,220),(19+x*6,220-curBar),2)

                nameImg=font.render(self.testId,True,(160,160,224))
                pygame.draw.rect(visual,(0,0,0),textregion)
                
                visual.blit(nameImg,(16,234))

            pygame.display.flip()
        
        return newRk

    def resetAffect(self):
        self.weightAffect=[1.0]*len(self.net.connections)
        self.affectInit=True

    def updateAffect(self,priorWts,currentWts,netFitness):
        """
        Updates weight affects then renormalizes to [0,1]
        where 0 is worst affect, 1 is best
        """
        weightCount=len(priorWts)
        self.affectInit=False
        if len(priorWts)-len(currentWts)+len(self.weightAffect)-len(priorWts)!=0:
            raise TypeErorr("updateActivity: Incompatible weightspaces (sizes differ).")
        topChg=float("-Inf")
        btmChg=float("Inf")
        fScale=self.maxFitness-self.minFitness
        if fScale==0.0:
            fScale=1.0
        for idx in range(weightCount):
            magnitude=math.fabs(currentWts[idx]-priorWts[idx])
            change=magnitude*(netFitness/fScale)
            topChg=max(topChg,magnitude)
            btmChg=min(btmChg,magnitude)
        offset=btmChg
        scale=topChg-btmChg
        if scale==0.0:
            # prevent dividum byzeroum
            scale=1e-300
        minAff=0
        maxAff=0
        for idx in range(weightCount):
            magnitude=math.fabs(currentWts[idx]-priorWts[idx])
            change=(magnitude/scale)*(netFitness/fScale)
            affect=self.weightAffect[idx]+change
            self.weightAffect[idx]=affect
            minAff=min(minAff,affect)
            maxAff=max(maxAff,affect)
        offset=minAff
        scale=maxAff-minAff
        for idx in range(weightCount):
            nAffect=(self.weightAffect[idx]-offset)/scale
            self.weightAffect[idx]=nAffect

    def changeEvaluator(self,testFunc):
        self.evalfunc=testFunc
        self.minFitness=float("Inf")
        self.maxFitness=float("-Inf")
        if self.solutions==[]:
            save=self.saveSnapshot()
            rank=self.evalfunc(self.net)
            self.minFitness=min(self.minFitness,rank)
            self.maxFitness=max(self.minFitness,rank)
            log.log(log.last(),which='solveLog')
            self.solutions=[(save,rank)]
            self.rank=rank
            self.loadSnapshot(save)
        else:
            save=self.saveSnapshot()
            self.rank=self.evalfunc(self.net)
            # to change evaluator we need to reevaluate solutions
            # then resort them by descenidng fitness
            #print("*** Trainer changed - reevaluating solutions")
            for idx in range(len(self.solutions)):
                ((sW,sS),sR)=self.solutions[idx]
                self.loadSnapshot((sW,sS))
                sR=self.evalfunc(self.net)
                #print("  %s" % log.last())
                self.solutions[idx]=((sW,sS),sR)                
                self.minFitness=min(self.minFitness,sR)
                self.maxFitness=max(self.minFitness,sR)
            self.loadSnapshot(save)
            # re-sort solutions by descending fitness of new evaluation regime
            self.solutions=sorted(self.solutions,key=lambda s:s[1],reverse=True)

    def TrainingEpoch(self):
        sol=self.solutions[0]
        self.loadSnapshot(self.solutions[0][0])
        TS_now=self.saveState()
        curTerm={'w':self.saveWeights(),'s':TS_now,'r':self.rank}
        searchTerm={'w':self.saveWeights(),'s':TS_now,'r':self.rank}
        # create some mutations
        mutantCount=1000
        # maximum random mutation operators per gene
        #mCount=len(self.solutions)+len(self.net.connections)
        alternate=0
        # will cylce good/bad affects
        oscillateAlternate=1
        if self.affectInit:
            # if affect is in a reset state
            # we don't want to oscillate
            # since it means 50% of mutants
            # won't be modified at all on first
            # pass and thus wasted
            oscillateAlternate=0
        mCount=len(self.net.connections)+len(self.net.nodeRefs)
        for mutantId in range(mutantCount):
            self.testId="Mutant_%s" % (str(1000-mutantId).rjust(4,"0"))
            # pick a random first parent
            mutant=[]+self.solutions[
                round((len(self.solutions)-1)*(1.0-math.cos(EntropySource.uniform(0.0,halfPi))))
                ][0][0]
            #mutant=[]+self.solutions[0][0][0]
            mutationCount=round(EntropySource.uniform(1,mCount))
            # splice (mating to second random parent)
            self.mutationOps[0](mutant)
            egg=[]+mutant
            # mutate mutant
            for mutations in range(mutationCount):
                # apply randomly chosen mutation operator (other than splice)
                op=round(EntropySource.uniform(1,len(self.mutationOps)-1))
                self.mutationOps[op](mutant)
            for idx in range(len(mutant)):
                new=EntropySource.uniform(-6,6)
                org=mutant[idx]
                aff=self.weightAffect[idx]**2.0
                if (alternate==0):
                    # mutate "good" weights
                    # aff=0.0 is org, aff=1.0 is new
                    mutant[idx]=org*(1.0-aff)+new*aff
                else:
                    # mutate "bad" weights
                    # aff=1.0 is org, aff=0.0 is new
                    mutant[idx]=org*aff+new*(1.0-aff)
            alternate=oscillateAlternate-alternate
            # test at TS_now
            self.loadWeights(mutant)
            self.loadState(TS_now)
            rk=self.evaluator(self.net,
                              original=searchTerm['w'],
                              current=mutant,
                              originalFitness=searchTerm['r'])
            if rk>searchTerm['r']:
                searchTerm['w']=[]+mutant
                searchTerm['s']=TS_now
                searchTerm['r']=rk
                log.log(log.last(),which='solveLog')
                self.solutions=[((searchTerm['w'],searchTerm['s']),searchTerm['r'])]+\
                                self.solutions[0:self.maxSolutions-1]
                self.rank=rk
                self.currentSolves+=1
        # if we found anything better store the best solution
        if searchTerm['r']>curTerm['r']:
            self.loadWeights(searchTerm['w'])
            self.loadState(searchTerm['s'])
            self.rank=searchTerm['r']

    def mutateTumor(self,chrom):
        # similar to Radical but affects a
        # randomly chosen section of the victim
        p1=round(EntropySource.uniform(0,len(chrom)))
        p2=p1
        while p2==p1:
            p2=round(EntropySource.uniform(0,len(chrom)))
        lhs=min(p1,p2)
        rhs=max(p1,p2)
        ugly=[0.0]*(rhs-lhs)
        for c in range(rhs-lhs):
            radical=EntropySource.uniform(-6.0,6.0)
            ugly[c]=radical
        chrom[lhs:rhs]=ugly
    def mutateRadical(self,chrom):
        where=round(EntropySource.uniform(0,len(chrom)-1))
        radical=EntropySource.uniform(-6.0,6.0)
        chrom[where]=radical
        return chrom
    def mutateSign(self,chrom):
        where=round(EntropySource.uniform(0,len(chrom)-1))
        chrom[where]=-chrom[where]
        return chrom
    def mutateSplice(self,chrom):
        nSol=len(self.solutions)
        which=1.0-math.cos(EntropySource.uniform(0.0,halfPi))
        which=round(which*float(nSol-1))
        ((other,sS),sR)=self.solutions[which]
        picked=[False]*len(chrom)
        for transcribe in range(int(len(chrom)/2)):
            k=round(EntropySource.uniform(0,len(chrom)-1))
            while picked[k]:
                k=round(EntropySource.uniform(0,len(chrom)-1))
            picked[k]=True
            chrom[k]=other[k]
        return chrom
    def mutateSwap(self,chrom):
        a=round(EntropySource.uniform(0,len(chrom)-1))
        b=a
        while b==a:
            b=round(EntropySource.uniform(0,len(chrom)-1))
        temp=chrom[a]
        chrom[a]=chrom[b]
        chrom[b]=temp
        return chrom
    def mutateTranspose(self,chrom):
        a=round(EntropySource.uniform(0,len(chrom)-1))
        b=(a+1) % len(chrom)
        temp=chrom[a]
        chrom[a]=chrom[b]
        chrom[b]=temp
        return chrom
            
    def saveWeights(self):
        return []+[self.net.connections[edge] for edge in sorted(self.net.connections)]
    def saveState(self):
        nodes=self.net.nodeRefs
        return [n.CEC for n in nodes]+[n.states['output'] for n in nodes]
    def saveSnapshot(self):
        return (self.saveWeights(),self.saveState())
    
    def loadWeights(self,Wts):
        idx=0
        for edge in self.net.connections:
            self.net.connections[edge]=Wts[idx]
            idx=idx+1
    def loadState(self,innerState):
        skip=int(len(innerState)/2)
        idx=0
        for n in self.net.nodeRefs.keys():
            n.CEC=innerState[idx]
            n.states['peephole']=sigmoid(n.CEC)
            n.states['output']=innerState[idx+skip]
            idx=idx+1
    def loadSnapshot(self,snap):
        Wts,CECs=snap
        self.loadWeights(Wts)
        self.loadState(CECs)


if __name__ == "__main__":

    try:

        from pprint import PrettyPrinter    
        fmt=PrettyPrinter(indent=2,width=40)

        net=Topology()
        inputs={}
        outputs={}
        node_labels="A,B,C,D"

        connections=[
                "AA","AB","AC","AD",
                "BA","BB","BC","BD",
                "CA","CB","CC","CD",
                "DA","DB","DC","DD"
            ]

        input_connections=[
            "0A"
            ]

        output_connections=[
            "D0"
            ]
        
        nodes   = { idx      : LSTM_Node() for idx in node_labels}
        nodenames={ nodes[k] : k           for k   in nodes }

        for nm in nodes:
            n=nodes[nm]
            net.Connect((n,'peephole'),(n,'inputGate'))
            net.Connect((n,'peephole'),(n,'forgetGate'))
            net.Connect((n,'peephole'),(n,'outputGate'))

        for c in connections:
            net.Connect((nodes[c[0]],"output"),(nodes[c[1]],"input"))
            net.Connect((nodes[c[0]],"output"),(nodes[c[1]],"inputGate"))
            net.Connect((nodes[c[0]],"output"),(nodes[c[1]],"forgetGate"))
            net.Connect((nodes[c[0]],"output"),(nodes[c[1]],"outputGate"))
            
        for c in input_connections:
            idx=int(c[0])
            inputs[idx]=net.Connect(None,(nodes[c[1]],"input"))
            
        for c in output_connections:
            idx=int(c[1])
            outputs[idx]=net.Connect((nodes[c[0]],"output"),None)

        """
        print("Inputs:")
        fmt.pprint(inputs)
        print("Outputs:")
        fmt.pprint(outputs)
        print("Nodes:")
        fmt.pprint(nodes)
        fmt.pprint(nodenames)
        print("Connections:")
        for c in net.connections.keys():
            n1,c1=c[0]
            n2,c2=c[1]
            w=net.connections[c]
            if n1 in nodenames.keys():
                n1="Node "+nodenames[n1]
            if n2 in nodenames.keys():
                n2="Node "+nodenames[n2]
            if c1==Input:
                c1="*"
            if c2==Output:
                c2="*"
            lhs="%s:%s" % (n1,c1)
            rhs="%s:%s" % (n2,c2)
            print("%s --> %s W=%s" % (lhs.rjust(15),rhs.ljust(15),w))
        """

        def Tester(theNet,test=""):
            global testlog
            # learn test string
            # the outputs scaled to range 0..255 and rounded to get ASCII
            eTerms=0.0
            inputs[0].write(0) # no input used for this test
            result=""
            for c in test:
                theNet.Activate()
                # compare output to expect and calculate error squares
                o1=ord(c)
                o2=round(outputs[0].read()*255)
                # noramlize the ord to a fraction so that unrounded
                # output may be used for error calculation
                of1=float(o1)
                of2=outputs[0].read()*255.0
                eTerms+=(of2-of1)*(of2-of1)
                result=result+chr(o2)
            eDist=math.sqrt(float(eTerms))
            fitness=-eDist
            log.log("'%s':'%s', fitness=%s" % (test,result,fitness))
            # negate so higher error = lower fitness
            return fitness
        
        Trainer=OOPS(Topology=net,Evaluator=Tester)

        test="Hello, World!"

        #prefixes=[]
        epoch=1
        print("Goal sequence: %s" % test)
        for pfx in range(len(test)):
        #for pfx in [len(test)-1]:
            subTest=partial(Tester,test=test[0:pfx+1])
            Trainer.changeEvaluator(subTest)
            while round(Trainer.solutions[0][1])<0:
                solves=Trainer.currentSolves
                Trainer.TrainingEpoch()
                newSolves=Trainer.currentSolves-solves
                lastSolve=log.last('solveLog')
                if newSolves>0:
                    gotcha=["+","-"][round(Trainer.solutions[0][1])<0]
                    print("Epoch %s %s %s (%s solutions)" % (str(epoch).rjust(12,'0'),
                          gotcha,lastSolve,len(Trainer.solutions)))
                epoch+=1
            #prefixes.append(Trainer.solutions[0])
            #Trainer.solutions=[]+prefixes

    finally:
        pygame.quit()
