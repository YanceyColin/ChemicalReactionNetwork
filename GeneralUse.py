###### General Use Tools ######
#                             #
#      By: Colin Yancey       #
#      Created Feb. 2022      #
#    Last Edited: 2/25/2025   #
#        Schulman Lab         #
#  Johns Hopkins University   #
#                             #
###############################

import numpy as np

# Sets a floor and ceiling for a number
# num: Value to be buffered
# minVal: The returned value's lower bound
# maxVal: The returned value's upper bound
def bufferNumber(num, minVal=0, maxVal=1):
    return max(min(num, maxVal), minVal)

# Rounds to a specified count of significant figures.
# x: Value to be rounded
# sigFig: Number of significant figures to round to
def roundToSigFig(x, sigFig=2):
    return 0 if x==0 else round(x, sigFig-int(np.floor(np.log10(abs(x))))-1)

# Returns a list with specific values from the original, specified by the requested indexes.
# listObj: List of interest
# inds: Indexes of interest to pull from the list
def filterList(listObj, inds):
    return [x for i,x in enumerate(listObj) if i in inds]

# Sanitize an addendum to a dictionary's list entry that may or may not exist (checks for the existence of a list for the given key first).
# dictObj: The dictionary of interest
# key: The key for the given value
# addlList: The intended additional list to be appended
def safeAppend(dictObj, key, addlList, rmInBetween=0):
    if key in dictObj:
        dictObj[key] = np.append(dictObj[key][:(len(dictObj[key]) - rmInBetween)], addlList)
    else:
        dictObj[key] = addlList

# Sanitizes inputs that are default set as int's or complex's to be floats. Does not write to float directly in case of situations like the use of the FloatWrapper class.
# val: Number or number-presenting class to be sanitized for int or complex status
def floatSanitize(val):
    if type(val) == int or type(val) == complex:
        return float(val)
    else:
        return val

# (EXPERIMENTAL, MAY BREAK IN CERTAIN USE CASES)
# Creates a doppleganger float class that does almost everything a float does except the quantity can be modified directly.
# Current list of magic methods is not exhaustive. Exercise caution in use.
class FloatWrapper:
    def __init__(self, val):
        self.Value = floatSanitize(val)
    
    def __float__(self):
        return self.Value
    
    def __str__(self):
        return str(self.Value)
    
    def __round__(self, digitNum = None):
        if digitNum == None:
            round(self.Value)
        return round(self.Value, digitNum)
    
    def changeValue(self, val):
        self.Value = floatSanitize(val)
    
    def __int__(self):
        return int(self.Value)
    
    def __add__(self, other): return float(self) + other
    def __radd__(self, other): return other + float(self)
    def __sub__(self, other): return float(self) - other
    def __rsub__(self, other): return other - float(self)
    def __mul__(self, other): return float(self) * other
    def __rmul__(self, other): return other * float(self)
    def __truediv__(self, other): return float(self) / other
    def __rtruediv__(self, other): return other / float(self)
    def __floordiv__(self, other): return float(self) // other
    def __rfloordiv__(self, other): return other // float(self)
    def __mod__(self, other): return float(self) % other
    def __rmod__(self, other): return other % float(self)
    def __pow__(self, other): return float(self) ** other
    def __rpow__(self, other): return other ** float(self)
    def __lt__(self, other): return float(self) < other
    def __le__(self, other): return float(self) <= other
    def __eq__(self, other): return float(self) == other
    def __ne__(self, other): return float(self) != other
    def __gt__(self, other): return float(self) > other
    def __ge__(self, other): return float(self) >= other

    def log(self): return np.log(self.Value)
    def exp(self): return np.exp(self.Value)
    def __abs__(self): return abs(self.Value)



# Using the doppleganger float class, creates a wrapper that can have an adjustable multiplier alongside the modifiable float quantity.
class DynamicFloat(FloatWrapper):
    def __init__(self, val, mult=1.0):
        super().__init__(val)
        self.Mult = floatSanitize(mult)
    
    def __float__(self):
        return self.Value*self.Mult
    
    def __str__(self):
        return str(self.Value*self.Mult)
    
    def changeMult(self, mult):
        self.Mult = floatSanitize(mult)