#### Chemical Reaction Network Constructor ####
#                                             #
#               By: Colin Yancey              #
#             Created February 2025           #
#            Last Edited: 06/02/2025          #
#                 Schulman Lab                #
#           Johns Hopkins University          #
#                                             #
###############################################

import re
import numpy as np
import collections
import scipy.integrate as spi

import GeneralUse as GU

removeChars = " ,<->+"


def getRxnStr(substrates, products, backwardRate=False):
    substrates.sort()
    products.sort()
    return " + ".join(substrates) + (" <-> " if backwardRate else " -> ") + " + ".join(products)


# Creates a Component object representing an item in a reaction with a Name and a Concentration.
class Component:
    def __init__(self, compName, compConc):
        self._Name = compName
        self.Concentration = GU.floatSanitize(compConc)
    
    @property
    def Name(self):
        return self._Name

    def __str__(self):
        return self.Name


# Creates a Reaction object representing chemical kinetics. It has a ForwardRate and a BackwardRate, as well as a list of substrates and products.
class Reaction:
    def __init__(self, forwardRate, rxnStr = None, backwardRate = 0.0, substrates = [], products = []):
        if not forwardRate:
            raise ValueError("Missing forward rate.")
        
        if rxnStr:
            if backwardRate:
                eqnParse = re.match(r"(.+)<->(.+)", rxnStr)
            else:
                eqnParse = re.match(r"(.+)->(.*)", rxnStr)
            
            if eqnParse:
                substrates = eqnParse.group(1).split("+")
                products = eqnParse.group(2).split("+")
            else:
                raise ValueError("The chemical equation is not in the correct format.")
        
        if len(substrates)==0:
            raise ValueError("Missing substrates; each reaction must have a substrate.")
        
        substrates = [x.strip(removeChars) for x in substrates]
        if len(products) > 0:
            products = [x.strip(removeChars) for x in products]
        
        if collections.Counter(substrates) == collections.Counter(products):
            raise ValueError("This reaction appears to do nothing. Check your substrates and products.")
        
        self.ForwardRate = GU.floatSanitize(forwardRate)
        self.BackwardRate = GU.floatSanitize(backwardRate)
        self._RxnStr = getRxnStr(substrates, products, backwardRate > 0.0)
        self._Substrates = substrates
        self._Products = products

    # Offer soft protection to the "immutable" reaction properties of an object, which include the substrate and product identities and their associated reaction string.
    @property
    def RxnStr(self):
        return self._RxnStr
    
    @property
    def Substrates(self):
        return self._Substrates
    
    @property
    def Products(self):
        return self._Products
    
    # Returns the reaction string
    def __str__(self):
        return self.RxnStr


# Creates a chemical reaction network capable of storing reactions and components. Can run simulations once the network is properly assembled.
# This class does not currently support coefficients in stoichiometry directly. To implement stoichiometry, add multiple of the same substrate or product together.
class ReactionNetwork:
    def __init__(self):
        self.Reactions = {}
        self.Components = {}
        self.SimResults = []
        self.SimLastPoint = None
    
    #  Adds a unique Reaction object to ReactionNetwork.
    def addRxn(self, rxnObj):
        for curRxn in self.Reactions.values():
            if collections.Counter(curRxn.Substrates) == collections.Counter(rxnObj.Substrates) and collections.Counter(curRxn.Products) == collections.Counter(rxnObj.Products):
                raise ValueError("This reaction already exists in the network.")
            elif collections.Counter(curRxn.Substrates) == collections.Counter(rxnObj.Products) and collections.Counter(curRxn.Products) == collections.Counter(rxnObj.Substrates):
                raise ValueError("This reaction's inverse already exists in the network. Add a backward rate instead.")
        self.Reactions[rxnObj.RxnStr] = rxnObj

    # Adds a list of Reaction objects to ReactionNetwork.
    def addRxns(self, rxnObjList):
        for rxn in rxnObjList:
            self.addRxn(rxn)
    
    # Adds a unique Component object to ReactionNetwork.
    def addComponent(self, componentObj):
        if componentObj.Name in self.Components:
            raise ValueError("Component '" + componentObj.Name + "' already exists in the reaction network.")
        elif componentObj.Name == "Time":
            raise ValueError("'Time' is a protected name used to reference time stamps for a solution; please use a different component name.")
        self.Components[componentObj.Name] = componentObj
    
    # Adds a list of Component objects to ReactionNetwork.
    def addComponents(self, compObjList):
        for comp in compObjList:
            self.addComponent(comp)

    # Returns a list of components based solely on what is used in the current list of Reaction objects.
    def getListOfReactionComponents(self):
        rxnComponentList = []
        for rxn in self.Reactions.values():
            for comp in rxn.Substrates+rxn.Products:
                if not comp in rxnComponentList:
                    rxnComponentList.append(comp)
        return rxnComponentList


    def getCompiledReactionODEFunction(self):
        if not self.SimLastPoint:
            rxnComps = self.getListOfReactionComponents()
            curComps = self.Components.keys()
            for compName in rxnComps:
                if not compName in curComps:
                    self.Components[compName] = Component(compName, 0.0)

        self.ComponentCount = len(self.Components)
        compList = list(self.Components.keys())

        compiledRateList = []
        compiledRxnRateMatrix = []
        compiledModifierMatrix = []

        def matrixAssembly(substrs, prods):
            subsList = [0]*(self.ComponentCount)
            negSubsPosProdsList = [0]*(self.ComponentCount)
            
            for substr in substrs:
                curInd = compList.index(substr)
                subsList[curInd] += 1
                negSubsPosProdsList[curInd] -= 1
            
            for prod in prods:
                curInd = compList.index(prod)
                negSubsPosProdsList[curInd] += 1
            
            compiledRxnRateMatrix.append(subsList)
            compiledModifierMatrix.append(negSubsPosProdsList)
        
        for rxn in self.Reactions.values():
            compiledRateList.append(rxn.ForwardRate)
            matrixAssembly(rxn.Substrates, rxn.Products)

            if rxn.BackwardRate > 0:
                compiledRateList.append(rxn.BackwardRate)
                matrixAssembly(rxn.Products, rxn.Substrates)

        compiledRateList = np.array(compiledRateList)
        compiledRxnRateMatrix = np.array(compiledRxnRateMatrix)
        compiledModifierMatrix = np.array(compiledModifierMatrix)

        if np.max(compiledRxnRateMatrix) > 1: # Stoichiometries present > 1, so exponentiation is required
            def calculateDerivative(tVal, dVals):
                return np.matmul( # Distribute and sum the rates, with signs according to their status as a substrate or product of each reaction
                    compiledModifierMatrix.T,
                    np.multiply( # Multiply the kinetic rates by the np.prod result to obtain total rates for each reaction
                        compiledRateList,
                        np.prod( # Multiply concentrations of substrates consumed
                            np.power( # Distribute the substrates' concentrations into the sparse matrix format and exponentiate based on stoichiometry
                                dVals,
                                compiledRxnRateMatrix
                            ),
                            axis=1
                        )
                    )
                )
        else:
            compiledRxnRateMatrix = np.array(compiledRxnRateMatrix, dtype=np.bool)
            compiledRxnRateMatrixToggled = ~compiledRxnRateMatrix # Apply NOT filter

            def calculateDerivative(tVal, dVals):
                return np.matmul( # Distribute and sum the rates, with signs according to their status as a substrate or product of each reaction
                    compiledModifierMatrix.T,
                    np.multiply( # Multiply the kinetic rates by the np.prod result to obtain total rates for each reaction
                        compiledRateList,
                        np.prod( # Multiply concentrations of substrates consumed
                            compiledRxnRateMatrix * dVals + compiledRxnRateMatrixToggled,
                            axis=1
                        )
                    )
                )
            
        return calculateDerivative
    

    # Simulates the reaction
    def simulateReactionFn(self, simTime, continueFromLast = True, method="LSODA", simDataResolution = 0, evalPts = []):
        if simDataResolution == 0 and len(evalPts) == 0:
                raise ValueError("'evalPts' must have at least one time entry or 'simDataresolution' must be > 0 to record simulation results.")
        elif len(evalPts) == 0:
            evalPts = np.linspace(0, simTime, simDataResolution)
        else:
            for x in evalPts:
                if x > simTime:
                    raise ValueError("All evaluated time points in 'evalPts' must be within the designated simulation time.")

        reactionODEFn = self.getCompiledReactionODEFunction()

        if not continueFromLast or not self.SimLastPoint:
            self.SimLastPoint = ComponentGroup({comp.Name: comp.Concentration for comp in self.Components.values()})
        
        compList = list(self.Components.keys()) # Maintain a pre-set order of components when transferring the data into a linear vector for the ODE calculation, as to not mis-map output concentrations.

        rawSolution = spi.solve_ivp(
            reactionODEFn,
            (0, simTime),
            [self.SimLastPoint.Components[compName] for compName in compList],
            method=method,
            t_eval=evalPts
        )

        self.SimResults.append({compList[i]: x for i,x in enumerate(rawSolution.y)})
        self.SimResults[-1]["Time"] = rawSolution.t
        self.SimLastPoint = ComponentGroup({compName: self.SimResults[-1][compName][-1] for compName in compList})


    # Allows a simulation result to have a modification made to the last simulation's concentration profile.
    # Useful if you want to simulate spiking a component in a solution (simulating a sudden manual/external adjustment in concentration)
    def adjustResultComponent(self, compName, compConc):
        if not self.SimLastPoint:
            raise ValueError("A prior simulation has not yet been recorded.")
        self.SimLastPoint.Components[compName] = GU.floatSanitize(compConc)

    def clearSimResults(self):
        self.SimResults = []


# Implicitly creates a string of contents of component concentrations in the correct formatting while also maintaining the data structure.
class ComponentGroup:
    def __init__(self, componentConcDict):
        self.Components = componentConcDict

    def __str__(self):
        itmsInDict = self.Components.keys()
        return "\n".join([itm + " = " + str(GU.roundToSigFig(self.Components[itm], sigFig=4)) for itm in itmsInDict])


if __name__=="__main__":
    import matplotlib.pyplot as plt

    simTime = 1000 # In units of sec
    simResolution = 201 # Number of points retrieved

    rxn1 = Reaction(5e-5, "A + B -> C") # In units of 1/sec/uM
    rxn2 = Reaction(1e-2, "C + D <-> CD", backwardRate = 1e-0) # In units of 1/sec/uM, and in units of 1/sec
    rxn3 = Reaction(3e-2, "CD -> D + E") # In units of 1/sec

    compA = Component("A", 100) # In units of uM
    compB = Component("B", 80) # In units of uM
    compD = Component("D", 5) # In units of uM

    newNetwork = ReactionNetwork()
    newNetwork.addRxns([rxn1, rxn2, rxn3])
    newNetwork.addComponents([compA, compB, compD])
    newNetwork.simulateReactionFn(simTime, simDataResolution=simResolution)

    comps = list(newNetwork.Components.keys())

    print("--- Final Concentrations ---\n" + str(newNetwork.SimLastPoint))

    # Plot each simulation result vs. the time points recorded
    for comp in comps:
        plt.plot(newNetwork.SimResults[0]["Time"], newNetwork.SimResults[0][comp])

    plt.xlabel("Time (seconds)")
    plt.ylabel("Concentration (uM)")
    plt.legend(comps)

    plt.show()