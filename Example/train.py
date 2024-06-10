import  pyn

dataset = [
    [[0,0],[0]],
    [[1,0],[1]],
    [[0,1],[1]],
    [[1,1],[0]],
]


network = pyn.NN([2,2,1])

network.randomize_values()

network.load_dataset(dataset)

network.train(5000, 0.1, printdata=True)

network.save("model.pyn")
