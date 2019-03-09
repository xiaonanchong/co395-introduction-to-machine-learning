import numpy as np
from sklearn.metrics import confusion_matrix
from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
 
from illustrate import illustrate_results_ROI

area_map = {
    k: v
    for k, v in zip(range(4), ("Ground", "Zone 1", "Zone 2", "Unlabelled area"))
}

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def evaluate_architecture(network, prep_x, xtest, ytest):
    x_pre = prep_x.apply(xtest)
    results = network(x_pre)
    y_true = [area_map[x] for x in np.argmax(ytest, axis=1)]

    err = 0
    N = len(ytest)
    predictions = [area_map[x] for x in np.argmax(results, axis=1)]
   
    _, _, f1, _=precision_recall_fscore_support(y_true, predictions, average='weighted', labels=["Ground", "Zone 1", "Zone 2", "Unlabelled area"])

    cm = confusion_matrix(y_true, predictions)
    print(cm)
    for i in range(N):
        #print('pred: ',predictions[i],' true: ', y_true[i])
        if y_true[i] != predictions[i]:
            err = err+1
    err_rate = err / N
    print('The error rate of the predictions by the model ---', err_rate)
    return f1

def predict_hidden(dataset): 
    # preprocessing steps
    np.random.shuffle(dataset)
    x_test = dataset[:, :3]
    y_test = dataset[:, 3:]
   
    traindata = np.loadtxt("ROI_dataset.dat")
    np.random.shuffle(traindata)
    x = traindata[:, :3]
    y = traindata[:, 3:]
    X_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42)
    
    y_train_normal = np.array([x for x in np.argmax(y_train, axis=1)])
    i_class0 = np.where(y_train_normal==0)[0]
    i_class1 = np.where(y_train_normal==1)[0]
    i_class2 = np.where(y_train_normal==2)[0]
    i_class3 = np.where(y_train_normal==3)[0]
    
    
    n_need = 6000
    i_class0_upsampled = np.random.choice(i_class0, size=n_need, replace=True)
    i_class1_upsampled = np.random.choice(i_class1, size=n_need, replace=True)
    i_class2_upsampled = np.random.choice(i_class2, size=n_need, replace=True)
    i_class3_downsampled = np.random.choice(i_class3, size=n_need, replace=False)
    
    total = np.concatenate((i_class0_upsampled, i_class1_upsampled, i_class2_upsampled, i_class3_downsampled))
    
    X_train_balanced = X_train[total]
    
    y_train_balanced = y_train[total]

    prep_data = Preprocessor(X_train_balanced)
    train_x = prep_data.apply(X_train_balanced)
    x_pre = prep_data.apply(x_test)

    # load the best model
    net = load_network("net3")
    trainer = Trainer(
        network=net,
        batch_size=32,
        nb_epoch=800,
        learning_rate=0.1,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )
    trainer.train(train_x, y_train_balanced)
    preds = net(x_pre)
    pre = np.argmax(preds, axis=1)
    preds_onehot = indices_to_one_hot(pre, 4)

    return preds_onehot


def main():
    dataset = np.loadtxt("ROI_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    input_dim = 3
    neurons = [20, 30,20, 4]
    activations = ["relu","relu","relu","sigmoid"]
    network = MultiLayerNetwork(input_dim, neurons, activations)
    np.random.shuffle(dataset)
    x = dataset[:, :3]
    y = dataset[:, 3:]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    y_train_normal = np.array([x for x in np.argmax(y_train, axis=1)])
    i_class0 = np.where(y_train_normal==0)[0]
    i_class1 = np.where(y_train_normal==1)[0]
    i_class2 = np.where(y_train_normal==2)[0]
    i_class3 = np.where(y_train_normal==3)[0]
    
    n_class0 = len(i_class0)
    n_class1 = len(i_class1)
    
    n_need = 6000
    i_class0_upsampled = np.random.choice(i_class0, size=n_need, replace=True)
    i_class1_upsampled = np.random.choice(i_class1, size=n_need, replace=True)
    i_class2_upsampled = np.random.choice(i_class2, size=n_need, replace=True)
    i_class3_downsampled = np.random.choice(i_class3, size=n_need, replace=False)
    
    total = np.concatenate((i_class0_upsampled, i_class1_upsampled, i_class2_upsampled, i_class3_downsampled))
    
    X_train_balanced = X_train[total]
    
    y_train_balanced = y_train[total]
    y_t = [x for x in np.argmax(y, axis=1)]
    print(y_train_balanced.shape)
    c1 = y_t.count(0)
    c2 = y_t.count(1)
    c3 = y_t.count(2)
    c4 = y_t.count(3)

    print(c1)
    print(c2)
    print(c3)
    print(c4)
    
    prep_input = Preprocessor(X_train_balanced)
    x_train_pre = prep_input.apply(X_train_balanced)

    trainer = Trainer(
        network=network,
        batch_size=32,
        nb_epoch=800,
        learning_rate=0.1,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )
    trainer.train(x_train_pre, y_train_balanced)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train_balanced))
    
    prep_input = Preprocessor(X_test)
    x_test_pre = prep_input.apply(X_test)
    print("Test loss = ", trainer.eval_loss(x_test_pre, y_test))
    
    
    save_network(network, "net4")
    print('best network saved')
    
    prepx = Preprocessor(X_test)
    score = evaluate_architecture(network, prepx, X_test, y_test)
    print('weighted average f1_score is ', score)
    
#'''
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    illustrate_results_ROI(network, prepx)


if __name__ == "__main__":
    main()
