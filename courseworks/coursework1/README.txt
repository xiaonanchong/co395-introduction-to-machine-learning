put this python file and data folder: wifi in the this directory.
in the shell type python FINAL.py to run the file.

the default test codes print out the average classification rate before pruing and after pruning using ten-fold-cross-validation.

To use the codes on other dataset:
1. to build a decision tree given dataset: call function - decision_tree_learning(dataset, depth)
2. to prune a tree based on validation set: call function - pruning(tree, training_data, validation_data)
3. to visualize the tree: call function - visualize(tree)
4. to evaluate the tree model performance on test dataset: call function - evaluate(tree, test_data) returns the number of mis-classifies data samples
