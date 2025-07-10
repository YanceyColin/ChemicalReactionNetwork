######### Genelet Network Constructor #########
#                                             #
#               By: Colin Yancey              #
#               Created July 2025             #
#            Last Edited: 07/09/2025          #
#                 Schulman Lab                #
#           Johns Hopkins University          #
#                                             #
###############################################

import numpy as np
import re

from CRNConstructor import Component, Reaction, ReactionNetwork
import GeneralUse as GU


componentDelimiter = ":"
defaultRNAPConc = 4 # U/uL, as determined by Thermofischer Scientific
defaultEnzymeMethod = "MM" # Michaelis-Menten Kinetics


class DefaultNucleicAcidRates:
    DNADNABind = 1e-5 # 1/nM/s
    DNARNABind = 1e-5 # 1/nM/s
    RNARNABind = 1e-5 # 1/nM/s

    DNADNADisplace = 5e-6 # 1/nM/s
    DNARNADisplace = 5e-6 # 1/nM/s

class DefaultLinearEnzymeRates:
    RNAP = GU.FloatWrapper(0.005) # 1/s * (uL/U)
    RNaseH = GU.FloatWrapper(0.03) # 1/s * (uL/U)
    RNaseR = GU.FloatWrapper(0) # 1/s * (uL/U)


class DefaultMMEnzymeRates:
    RNAPBind = GU.FloatWrapper(1e-1) # 1/nM/s
    RNAPUnbind = GU.FloatWrapper(1e-6) # 1/s
    RNAPProduce = GU.FloatWrapper(0.07) # 1/s

    RNaseHBind = GU.FloatWrapper(1e-4) # 1/nM/s
    RNaseHUnbind = GU.FloatWrapper(0.03) # 1/s
    RNaseHDeg = GU.FloatWrapper(0.006) # 1/s

    RNaseRBind = GU.FloatWrapper(1.2e-6) # 1/nM/s
    RNaseRUnbind = GU.FloatWrapper(0.05) # 1/s
    RNaseRDeg = GU.FloatWrapper(0.0101) # 1/s


class DefaultGNRates:
    ActGeneletBind = GU.FloatWrapper(DefaultNucleicAcidRates.DNADNABind)
    BlkGeneletBind = GU.FloatWrapper(DefaultNucleicAcidRates.DNADNABind)
    BlkDisplaceAct = GU.FloatWrapper(DefaultNucleicAcidRates.DNADNADisplace)
    RepDisplaceAct = GU.FloatWrapper(DefaultNucleicAcidRates.DNARNADisplace)
    CoactDisplaceBlk = GU.FloatWrapper(DefaultNucleicAcidRates.DNARNADisplace)

    ActRepBind = GU.FloatWrapper(DefaultNucleicAcidRates.DNARNABind)
    BlkCoactBind = GU.FloatWrapper(DefaultNucleicAcidRates.DNARNABind)
    RepRepSilBind = GU.FloatWrapper(DefaultNucleicAcidRates.RNARNABind)
    CoactCoactSilBind = GU.FloatWrapper(DefaultNucleicAcidRates.RNARNABind)

defaultGeneletReactionTemplates = [
    Reaction(DefaultGNRates.ActGeneletBind, "A + G -> A:G"),
    Reaction(DefaultGNRates.BlkGeneletBind, "B + G -> B:G"),
    Reaction(DefaultGNRates.BlkDisplaceAct, "A:G + B -> B:G + A"),
    Reaction(DefaultGNRates.RepDisplaceAct, "A:G + R -> A:R + G"),
    Reaction(DefaultGNRates.CoactDisplaceBlk, "B:G + C -> B:C + G")
]

defaultOCEReactionTemplates = [
    Reaction(DefaultGNRates.ActRepBind, "A + R -> A:R"),
    Reaction(DefaultGNRates.BlkCoactBind, "B + C -> B:C"),
    Reaction(DefaultGNRates.RepRepSilBind, "R + RS -> R:RS"),
    Reaction(DefaultGNRates.CoactCoactSilBind, "C + CS -> C:CS")
]



class EnzymeRxnTemplates:
    RNAP = {
        "Linear": lambda template, rnaProd, enzymeConc:
            [
                Reaction(DefaultLinearEnzymeRates.RNAP*enzymeConc, f"{template} -> {template} + {rnaProd}")
            ],

        "MM": lambda template, rnaProd, __:
            [
                Reaction(DefaultMMEnzymeRates.RNAPBind, f"RNAP + {template} <-> RNAP:{template}", DefaultMMEnzymeRates.RNAPUnbind),
                Reaction(DefaultMMEnzymeRates.RNAPProduce, f"RNAP:{template} -> RNAP + {template} + {rnaProd}")
            ]
    }

    RNaseH = {
        "Linear": lambda dna, rna, enzymeConc:
            [
                Reaction(DefaultLinearEnzymeRates.RNaseH*enzymeConc, f"{dna}:{rna} -> {dna}")
            ],

        "MM": lambda dna, rna, __:
            [
                Reaction(DefaultMMEnzymeRates.RNaseHBind, f"RNase H + {dna}:{rna} <-> RNase H:{dna}:{rna}", DefaultMMEnzymeRates.RNaseHUnbind),
                Reaction(DefaultMMEnzymeRates.RNaseHDeg, f"RNase H:{dna}:{rna} -> RNase H + {dna}")
            ]
    }

    RNaseR = {
        "Linear": lambda rna, enzymeConc:
            [
                Reaction(DefaultLinearEnzymeRates.RNaseDeg*enzymeConc, f"{rna} ->")
            ],
            
        "MM": lambda rna, __:
            [
                Reaction(DefaultMMEnzymeRates.RNaseRBind, f"RNase R + {rna} <-> RNase R:{rna}", DefaultMMEnzymeRates.RNaseRUnbind),
                Reaction(DefaultMMEnzymeRates.RNaseRDeg, f"RNase R:{rna} -> RNase R")
            ]
    }

class Genelet:
    def __init__(self, concentration, state):
        self.Concentration = concentration
        self.State = state

class OCE:
    def __init__(self):
        self.Products = {}


class GeneletNetwork(ReactionNetwork):
    def __init__(self, geneletReactionTemplates = defaultGeneletReactionTemplates, OCEReactionTemplates = defaultOCEReactionTemplates, indexingBlacklist = ["RNAP", "RNase H", "RNase R"]):
        super().__init__()
        
        self.GeneletReactionTemplates = {rxnObj.RxnStr: rxnObj for rxnObj in geneletReactionTemplates}
        self.OCEReactionTemplates = {rxnObj.RxnStr: rxnObj for rxnObj in OCEReactionTemplates}
        self.IndexingBlacklist = indexingBlacklist

        self.ActiveOCEs = {}
        self.ModifiedConcs = {}
        self.Reporters = []


    def proposeAddComponent(self, comp):
        if not isinstance(comp, Component):
            raise TypeError("All added components must be in the form of a Component object.")
        if comp.Name in self.Components:
            self.Components[comp.Name].Concentration = comp.Concentration
        else:
            self.addComponent(comp)
    
    def proposeAddRxnTemplate(self, rxnObj, rxnTemplateList):
        if not isinstance(rxnObj, Reaction):
            raise TypeError("All added reaction templates must be in the form of a Reaction object.")
        if not rxnObj.RxnStr in rxnTemplateList:
            rxnTemplateList[rxnObj.RxnStr] = rxnObj
        else:
            raise UserWarning(f"'{rxnObj.RxnStr}' is already present in the reaction template list. The duplicate was discarded.")
    
    def proposeAddRxnTemplates(self, rxnObjsList, rxnTemplateList):
        for rxnObj in rxnObjsList:
            self.proposeAddRxnTemplate(rxnObj, rxnTemplateList)
    

    def proposeAddRxn(self, rxnObj):
        if not isinstance(rxnObj, Reaction):
            raise TypeError("All added reactions must be in the form of a Reaction object.")
        if not rxnObj.RxnStr in self.Reactions:
            self.addRxn(rxnObj)
        else:
            raise UserWarning(f"'{rxnObj.RxnStr}' is already present in the reaction list. The duplicate was discarded.")
    
    def proposeAddRxns(self, rxnObjsList):
        for rxnObj in rxnObjsList:
            self.proposeAddRxn(rxnObj)
    
    def registerOCE(self, potentialNewOCE):
        if not potentialNewOCE in self.ActiveOCEs:
            self.ActiveOCEs[potentialNewOCE] = OCE()
    
    def registerReporter(self, potentialNewSignal):
        if not potentialNewSignal in self.Reporters:
            self.Reporters.append(potentialNewSignal)

    def addGenelet(self, geneletStr, conc, state = "OFF"):
        potentialGeneletInfo = re.match(r"G(\d+)([A-Z]+)(\d+)", geneletStr)
        
        if not potentialGeneletInfo:
            raise TypeError("Genelets must be given in the form 'G[OCE number][Symbol indicating the product output, i.e. RS for repressor silencer][OCE number]', i.e. 'G2C1'. Please check the input.")
        
        inputOCE = int(potentialGeneletInfo.group(1))
        outputType = potentialGeneletInfo.group(2)
        outputOCE = int(potentialGeneletInfo.group(3))

        self.registerOCE(inputOCE)
        if outputType == "S":
            self.registerReporter(outputOCE)
        else:
            self.registerOCE(outputOCE)

        self.ActiveOCEs[inputOCE].Products[outputType + str(outputOCE)] = Genelet(conc, state)

    def setInitialConc(self, rxnComponent, newConc):
        self.proposeAddComponent(Component(rxnComponent, newConc))
    
    def addRNaseHRxns(self, RNaseHConc, method = defaultEnzymeMethod):
        self.proposeAddComponent(Component("RNaseH", RNaseHConc))
        DNARNAElementGroups = [("A", "R"), ("B", "C")]
        for (dna, rna) in DNARNAElementGroups:
            self.proposeAddRxnTemplates(EnzymeRxnTemplates.RNaseH[method](dna, rna, RNaseHConc), self.OCEReactionTemplates)
    
    def addRNaseRRxns(self, RNaseRConc, method = defaultEnzymeMethod):
        self.proposeAddComponent(Component("RNaseR", RNaseRConc))
        RNAElements = ["R", "C", "RS", "CS"]
        for rna in RNAElements:
            self.proposeAddRxnTemplates(EnzymeRxnTemplates.RNaseR[method](rna, RNaseRConc), self.OCEReactionTemplates)


    # Split a string using the designated component delimiter.
    def parseComposites(self, inputStr):
        return inputStr.split(componentDelimiter)

    # Append a suffix to each element included in a string and give back that string using the component delimiter. Can be given a whitelist and will use the class's indexing blacklist.
    def appendSuffix(self, inputStr, suffix, whitelist = None):
        return componentDelimiter.join(
            [ comp + str(suffix) if not comp in self.IndexingBlacklist and (whitelist==None or comp in whitelist) else comp for comp in self.parseComposites(inputStr)]
        )

    def compileTotalReactionList(self, RNAPMethod = defaultEnzymeMethod, RNAPConc = defaultRNAPConc): # RNAPConc in U/uL, as determined by Thermofischer Scientific
        self.proposeAddComponent(Component("RNAP", RNAPConc))
        
        for rxn in self.OCEReactionTemplates.values():
            for oceNum in self.ActiveOCEs.keys():
                substrs = [self.appendSuffix(x, oceNum) for x in rxn.Substrates]
                prods = [self.appendSuffix(x, oceNum) for x in rxn.Products]
                
                self.proposeAddRxn(Reaction(rxn.ForwardRate, backwardRate = rxn.BackwardRate, substrates = substrs, products = prods))
        
        for rxn in self.GeneletReactionTemplates.values():
            for oceNum in self.ActiveOCEs.keys():
                substrs = [self.appendSuffix(x, oceNum) for x in rxn.Substrates]
                prods = [self.appendSuffix(x, oceNum) for x in rxn.Products]
                
                for rnaProd in self.ActiveOCEs[oceNum].Products.keys():
                    prodAmendSubstrs = [self.appendSuffix(x, rnaProd, whitelist = [f"G{oceNum}"]) for x in substrs]
                    prodAmendProds = [self.appendSuffix(x, rnaProd, whitelist = [f"G{oceNum}"]) for x in prods]

                    self.proposeAddRxn(Reaction(rxn.ForwardRate, backwardRate = rxn.BackwardRate, substrates = prodAmendSubstrs, products = prodAmendProds))
        for oceNum in self.ActiveOCEs.keys():
            for rnaProd in self.ActiveOCEs[oceNum].Products.keys():
                self.proposeAddRxns(EnzymeRxnTemplates.RNAP[RNAPMethod](f"A{oceNum}:G{oceNum}{rnaProd}", rnaProd, RNAPConc))

    def compileGeneletNetwork(self, RNAPMethod = defaultEnzymeMethod, RNAPConc = defaultRNAPConc):
        self.compileTotalReactionList(RNAPMethod, RNAPConc)

        for oceNum in self.ActiveOCEs.keys():
            oce = self.ActiveOCEs[oceNum]
            for rnaProd in oce.Products.keys():
                geneletName = f"G{oceNum}{rnaProd}"
                genelet = oce.Products[rnaProd]
                if genelet.State == "OFF":
                    self.proposeAddComponent(Component(geneletName, genelet.Concentration))
                elif genelet.State == "ON":
                    self.proposeAddComponent(Component(f"A{oceNum}:{geneletName}", genelet.Concentration))
                elif genelet.State == "BLK":
                    self.proposeAddComponent(Component(f"B{oceNum}:{geneletName}", genelet.Concentration))
                else:
                    raise TypeError(f"The state '{genelet.State}' given for '{geneletName}' is invalid. Only 'ON', 'OFF', or 'BLK' are valid initial states.")
        
        for comp in self.ModifiedConcs.values():
            self.proposeAddComponent(comp)





if __name__ == "__main__":
    import matplotlib.pyplot as plt

    GN = GeneletNetwork()
    GN.addGenelet("G2C1", 25, "OFF") # Concentrations in units of nM since default second-order rates are in units of 1/nM/sec.
    GN.addGenelet("G2C3", 5, "OFF")
    GN.addGenelet("G3R1", 25, "BLK")
    GN.addGenelet("G1S1", 25, "BLK")

    GN.setInitialConc("A1", 250)
    GN.setInitialConc("A2", 250)
    GN.setInitialConc("A3", 250)

    GN.setInitialConc("B1", 12.5)
    GN.setInitialConc("B3", 12.5)

    GN.compileGeneletNetwork(RNAPMethod = "Linear")

    GN.simulateReactionFn(3600*2, simDataResolution = 1000) # Time in seconds since default rates are in units of 1/sec.

    comps = ["A2:G2C1", "A2:G2C3", "A3:G3R1", "A1:G1S1"]

    print("--- Final Concentrations ---\n" + str(GN.SimLastPoint))

    # Plot each simulation result vs. the time points recorded
    for comp in comps:
        plt.plot(GN.SimResults[0]["Time"]/60, GN.SimResults[0][comp])

    plt.xlabel("Time (minutes)")
    plt.ylabel("Concentration (uM)")
    plt.legend(comps)

    plt.show()

