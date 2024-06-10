import pyn

network = pyn.NN([2,2,1])

network.load("model.pyn")

print("0,0")
network.prediction([0,0])
network.print_output()

print("1,0")
network.prediction([1,0])
network.print_output()

print("0,1")
network.prediction([0,1])
network.print_output()

print("1,1")
network.prediction([1,1])
network.print_output()