import numpy as np

def Topology(Agents, Top):

    Bias    = []
    Weights = []
    New_Bias = []
    New_Weights = []
    np.array(Bias)
    np.array(Weights)

    for i in range(len(Agents)):
        Bias.append([])
        Weights.append([])

    for i in range(len(Agents)):
        for L in range(len(Agents[i].net.layers)):
            Bias[i].append(np.array(Agents[i].net.layers[L].return_bias()))
            Weights[i].append(np.array(Agents[i].net.layers[L].return_weight()))

    if Top == "Ring":
        for i in range(len(Agents)):
            New_Bias.append([])
            New_Weights.append([])

        for i in range(len(Agents)):
            previous = i-1
            if previous<0:
                previous = len(Agents) - 1

            next = i+1
            if next>=len(Agents):
                next = 0

            for L in range(len(Agents[i].net.layers)):
                New_Bias[i]     = np.divide(Bias[next][L] + Bias[i][L] + Bias[previous][L], 3)
                New_Weights[i]  = np.divide(Weights[next][L] + Weights[i][L] + Weights[previous][L], 3)
                Agents[i].net.layers[L].weights = New_Weights[i]
                Agents[i].net.layers[L].bias = New_Bias[i]

    if Top == "Full":
        for L in range(len(Agents[0].net.layers)):
            np.array(New_Weights)
            np.array(New_Bias)

            for i in range(len(Bias[0][L])):
                New_Bias.append(0)
                New_Weights.append(0)

            for i in range(len(Agents)):
                New_Bias        += Bias[i][L]
                New_Weights     += Weights[i][L]         

            New_Bias = np.divide(New_Bias, len(Agents))
            New_Weights = np.divide(New_Weights, len(Agents))

            for i in range(len(Agents)):
                Agents[i].net.layers[L].weights = New_Weights
                Agents[i].net.layers[L].bias    = New_Bias

            New_Weights = []
            New_Bias = []

    if Top == "Ring_Laplacian":
        for i in range(len(Agents)):
            New_Bias.append([])
            New_Weights.append([])

        for i in range(len(Agents)):
            previous = i-1
            if previous<0:
                previous = len(Agents) - 1

            next = i+1
            if next>=len(Agents):
                next = 0

            for L in range(len(Agents[i].net.layers)):
                New_Bias[i]     = np.divide(3*Bias[next][L] + Bias[i][L] + Bias[previous][L], 5)
                New_Weights[i]  = np.divide(3*Weights[next][L] + Weights[i][L] + Weights[previous][L], 5)
                Agents[i].net.layers[L].weights = New_Weights[i]
                Agents[i].net.layers[L].bias = New_Bias[i]