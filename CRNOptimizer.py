##### Chemical Reaction Network Optimizer #####
#                                             #
#               By: Colin Yancey              #
#              Created April 2025             #
#            Last Edited: 05/14/2025          #
#                  Schulman Lab               #
#            Johns Hopkins University         #
#                                             #
###############################################

import numpy as np
from scipy.optimize import minimize
import time

import GeneralUse as GU

def printProgress(progressNum, totalIter, barMaxSize = 20):
    barLen = int(np.floor(progressNum/totalIter*barMaxSize))
    print("["+("#"*barLen)+("-"*(barMaxSize-barLen))+"] "+str(progressNum)+" / "+str(totalIter))

class CRNOptimizer:
    def __init__(self, rNwks, modOpts, fittingData, simTimePts, fpFormula, timeDelay = 0):
        self.ReactionNetworks = rNwks
        self.ModOptions = modOpts
        self.FittingData = fittingData
        self.SimTimePts = simTimePts
        self.TimeDelay = timeDelay
        
        self.calculateTimeProfile = fpFormula

    def calcPredCost(self, y):
        totCost = 0
        for j, x in enumerate(self.ModOptions):
            x.changeValue(np.exp(y[j]))
        for i, rN in enumerate(self.ReactionNetworks):
            rN.simulateReactionFn(np.max(self.SimTimePts)+self.TimeDelay, evalPts = self.SimTimePts+self.TimeDelay, continueFromLast=False)
            totCost += np.sum(
                (
                    self.calculateTimeProfile(rN.SimResults[0]) - self.FittingData[i]
                )**2
            )
        return totCost
    
    def optimize(self, method = "Nelder-Mead", maxIter = 1000, boundSizes = 1e3, tickSize=5, options={}):
        start_time = time.time()
        
        lnModOpts = np.log(self.ModOptions)
        
        if isinstance(boundSizes, (int, float, complex)):
            boundSize = np.log(boundSizes)
            expBoundSizes = [boundSize for _ in np.arange(len(self.ModOptions))]
        else:
            if len(boundSizes) != len(self.ModOptions):
                raise ValueError("'boundSizes' provided as a list must be equal in length to the list of provided 'modOpts'")
            
            expBoundSizes = [np.log(x) for x in boundSizes]
        
        self.Iters = 0
        self.ProgressTicks = 0
        def callbackFn(_):
            self.Iters += 1
            curTickCount = np.floor(self.Iters/tickSize)
            if curTickCount > self.ProgressTicks:
                self.ProgressTicks = curTickCount
                printProgress(self.Iters, maxIter)

        res = minimize(
            self.calcPredCost,
            lnModOpts,
            method=method,
            options={'maxiter': maxIter} | options,
            bounds=[(x-expBoundSizes[i], x+expBoundSizes[i]) for i, x in enumerate(lnModOpts)],
            callback = callbackFn
        )
        
        if not res.success:
            print(res.message)
        
        self.OptimizerResults = res.x

        print("Time passed: " + str(GU.roundToSigFig((time.time() - start_time)/60, 4)) + " minutes")

    
    def checkIfOptimized(self):
        if not hasattr(self, "OptimizerResults"):
            raise ValueError("Optimization must be run once to obtain optimizer results.")
    
    def getOptimizedParameters(self):
        self.checkIfOptimized()

        return np.exp(self.OptimizerResults)

    def getOptimizedSimulation(self):
        self.checkIfOptimized()

        for j, x in enumerate(self.ModOptions):
            x.changeValue(np.exp(self.OptimizerResults[j]))
        
        results = []
        for rN in self.ReactionNetworks:
            rN.simulateReactionFn(np.max(self.SimTimePts), evalPts = self.SimTimePts, continueFromLast=False)
            results.append(self.calculateTimeProfile(rN.SimResults[0]))

        return results

    def printOptimizerResults(self, names = [], sigFigs = 4):
        self.checkIfOptimized()

        print("- Optimization Results -")
        for i in np.arange(len(self.OptimizerResults)):
            print(("" if len(names) <= i else names[i] + ": ") + str(GU.roundToSigFig(np.exp(self.OptimizerResults[i]), sigFigs)))