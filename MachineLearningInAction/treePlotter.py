import matplotlib.pyplot as plt

decisionNode = dict(boxstyle = "sawtooth", fc = "0.8")
leafNode = dic(boxstyle = "round4", fc = "0.8")
arrow_args = dict(arrowstyle = "<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy = parentPt,\
        xycoords = 'axes fraction',\
        xytext = centerPt, textcoords = 'axes fraction',\
        va = "center", ha = "center", bbox = nodeType, arrowprops = arrow_args)