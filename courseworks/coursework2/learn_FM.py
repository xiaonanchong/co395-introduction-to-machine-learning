import numpy as np
from simulator import RobotArm
from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)
from illustrate import illustrate_results_FM

## given the network and a instance of Preprocessor built from training data
## generate nb_pos test samples through robot simulation 
## according to these new test samples, 
## return the average distance between target [x,y,z] and prediction [x_, y_, z_]
def evaluate_architecture(network,prep, nb_pos=10):
    data = ((np.random.rand(nb_pos + 1, 6) * 2 - 1) * np.pi / 2)  
    data[0, :] = 0
    data = prep.apply(data)
    results = network(data[1:, 0:3])
    robot = RobotArm()
    data[1:, 3:6] = results
    data = prep.revert(data)
    prediction = data[1:, 3:6]
    angles = np.zeros((nb_pos + 1, 6))
    angles[:, 0:3] = data[:, 0:3]
    dist = 0
    for i in range(nb_pos):
        print(robot.forward_model(angles[i + 1, :]))
        print(prediction[i, :])
        print(np.sqrt(sum((robot.forward_model(angles[i + 1, :]) - prediction[i, :])**2)))
        dist += np.sqrt(sum((robot.forward_model(angles[i + 1, :]) - prediction[i, :])**2))
    distance = dist / nb_pos
    print('Average Distance Between Prediction and Target ---', distance)    
    return distance


def predict_hidden(dataset): 
    # preprocessing steps
    x = dataset[:, :3]
    x_train = np.loadtxt("FM_dataset.dat")[:, :3]
    y_train = np.loadtxt("FM_dataset.dat")[:, 3:]
    prep_data_x = Preprocessor(x_train)
    prep_data_y = Preprocessor(y_train)

    x_pre = prep_data_x.apply(x)
    # load the best model
    net = load_network("net")	
    preds = net(x_pre)
    preds_revert = prep_data_y.revert(preds)
    return preds_revert


def main():
    
    dataset = np.loadtxt("FM_dataset.dat") # [15625-6]
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    #'''
    # define the network
    input_dim = 3
    neurons = [80, 200, 3]
    activations = ["relu","relu","identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    # data preprocessing
    np.random.shuffle(dataset)

    x = dataset[:, :3]
    y = dataset[:, 3:]
    prep_input = Preprocessor(x)
    prep_output = Preprocessor(y)
    x_train_pre = prep_input.apply(x) 
    y_train_pre = prep_output.apply(y)
    trainer = Trainer(
        network=net,
        batch_size=100,
        nb_epoch=1000, 
        learning_rate=0.1,
        loss_fun="mse",
        shuffle_flag=True,
    )
    trainer.train(x_train_pre, y_train_pre)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train_pre))
    save_network(net, "net")
    print('best network saved')
    prep = Preprocessor(dataset)

    score = evaluate_architecture(net.forward,prep, 1000)
    print('EVALUATE ARCHITECTURE:', score)
    illustrate_results_FM(net, prep)

    #print(predict_hidden(dataset))
    #print(dataset[:, 3:])



if __name__ == "__main__":
    main()





















