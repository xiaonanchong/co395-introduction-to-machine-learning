########## Step 1: Load data #########

import numpy as np
#np.random.seed(2)
data = np.loadtxt('wifi_db/clean_dataset.txt')
# shuffle data
idx = np.random.permutation(data.shape[0])
data = data[idx]


########## Step 2: Create decision tree ##########

# create decision tree using training data set
# return the root node of the tree which is a dictionary object, and the total depth of the tree
def decision_tree_learning(dataset, depth):
    diction = {'atri':None,'value':None,'l': None,'r': None,'leaf':None}
    NumS = dataset.shape[0]
    Num = dataset.shape[1]
    label = dataset[:,Num-1]
    data = dataset[:,0:Num-1] 
    clas = np.unique(label)
    Nclas = clas.shape[0]
    H = infromation_entropy(label)
    record_right = []
    record_left = [] 
    record_atri = 0
    record_num = 0	
    record = H
    ReR = 0
    ReL	= 0	

    for i in range(0,Num-1): 
      data_use = data[:,i] 
      dataunique = np.unique(data_use)      
      for j2 in dataunique:
          left = []
          right = [] 
          left_data = []
          right_data = []
	  H_left = 0
          H_right = 0
          H_record = 0

          for k in range(0,NumS):
	    if data_use[k] <= j2:
	       left.append(label[k])
	       left_data.append(dataset[k,:])		
	    else: 	
	       right.append(label[k])
	       right_data.append(dataset[k,:])	
	  H_left, H_right, H_record = left_right_entropy(left, right)

          if H_record <= record:
            record = H_record
            ReR =  H_right
            ReL =  H_left
            record_atri = i
            record_num = j2
    	    record_right = np.array(right_data)
            record_left = np.array(left_data) 	
    
    information_gain = H - record
    diction['atri'] = record_atri
    diction['value'] = record_num
    if record > 0.:
      if ReL > 0:		
        diction['leaf'] = -1
        L_depth = depth+1
        dicleft, L_depth=decision_tree_learning(record_left, L_depth)
        diction['l'] = dicleft  
      else:
        if len(record_left) != 0:
           diction['l'] = {'atri':None,'value':None,'l': None,'r': None,'leaf':majority(record_left)}
           diction['leaf'] = -1
           L_depth = depth+1
      if ReR > 0:
        diction['leaf'] = -1	
        R_depth = depth+1	
        dicright, R_depth=decision_tree_learning(record_right, R_depth) 
        diction['r'] = dicright 	 
      else:	
        if len(record_right) != 0:
           diction['r'] = {'atri':None,'value':None,'l': None,'r': None,'leaf':majority(record_right)}
           diction['leaf'] = -1
           R_depth = depth+1
    else:
      diction['leaf'] = -1
      diction['l'] = {'atri':None,'value':None,'l': None,'r': None,'leaf':majority(record_left)}
      diction['r'] = {'atri':None,'value':None,'l': None,'r': None,'leaf':majority(record_right)}
      R_depth = depth+1
      L_depth = depth+1
    return diction, max(L_depth,R_depth)

# Output the entropy
def infromation_entropy(label):
    H = 0
    clas = np.unique(label)
    Num = label.shape[0]
    for j in clas:
      Numj = np.count_nonzero(label == j)
      pj = float(Numj)/float(Num)
      H -= pj* np.log2(pj)  
    return H

# Calculate the sum of entropy
def left_right_entropy(left, right):
    H_left = infromation_entropy(np.array(left))	
    H_right = infromation_entropy(np.array(right))
    Num_left = float(len(left))
    Num_right = float(len(right))
    Num = Num_left + Num_right
    H_record = Num_left/Num*H_left + Num_right/Num*H_right
    return H_left, H_right, H_record

# How to define the label of the leaf
def majority(dataset): 
    Num = dataset.shape[1]	
    label = np.unique(dataset[:,Num-1])
    record_label = 0
    record_num = 0
    for i in label:	
      num_label = np.count_nonzero(dataset[:,Num-1]==i)
      if num_label > record_num:
        record_label = i
	record_num = num_label
    return record_label 


########## Step 3: Evaluation ##########

# given test data and the root of the tree
# return the total number if mis-classification
def evaluate(node, split_data): 
  d = np.array(split_data)
  atri_index = node['atri']
  split_value = node['value']
  left_node = node['l']
  right_node = node['r']
  leaf = node['leaf']
  if leaf == -1:  
    left_data, right_data = extract_data(split_data, atri_index, split_value)
    e1 = evaluate(left_node, left_data)
    e2 = evaluate(right_node, right_data)
    error = e1 + e2
  else:
    count = 0
    for i in range(split_data.shape[0]):
      if int(split_data[i,split_data.shape[1]-1]) != int(leaf):
        count = count + 1
    error = count
  return error

# ten fold cross validation 
# return the classification rate
def evaluate_ten_fold_cross_validation(data):
  N = 10
  total_len = data.shape[0]
  one_fold = int(total_len/10)

  cr_1 = 0
  cr_2 = 0
  cr_3 = 0

  for i in range(N):        
    test_data = data[one_fold*i:one_fold*(i+1)]
    train_data = np.concatenate((data[0: one_fold*i], data[one_fold*(i+1):]), axis = 0)

    d = train_data
    l6 = int(8*(d.shape[0]/10.0))
    d0_6 = train_data[0:l6]
    d6_10 = train_data[l6:]
 
    # building a tree without pruning
    # use all the training data
    root_node1, depth1 = decision_tree_learning(d, 0) 
    cr1 = 1 - float(evaluate(root_node1, test_data)) / one_fold
    #visualize(root_node1)


    # building a tree and prune it
    # use 8/10 of the traning data to build a tree and use 2/10 the data to prune
    root_node2, depth2 = decision_tree_learning(d0_6, 0)
    cr2 = 1 - float(evaluate(root_node2, test_data)) / one_fold
    #visualize(root_node2)

    root_node3 = pruning(root_node2, d0_6, d6_10)
    depth3 = maxDepth(root_node3)
    cr3 = 1 - float(evaluate(root_node3, test_data)) / one_fold
    #visualize(root_node3)

    print(depth1, cr1, '|', depth2, cr2, '|', depth3, cr3)

    cr_1 = cr_1 + cr1
    cr_2 = cr_2 + cr2
    cr_3 = cr_3 + cr3

  print('average classification rate without pruning:', cr_1/N)
  print('average classification rate before pruning:', cr_2/N)
  print('average classification rate after pruning:', cr_3/N)



# given a tree, predict on a set of data
# return a list of results
def predict(node, data):
  predict = np.zeros((data.shape[0], 1))
  for i in range(data.shape[0]):
    predict[i] = p(node, data[i])
  return predict

# given a tree, predict on a single data
def p(node, d):
  atri = node['atri']
  value = node['value']
  l = node['l']
  r = node['r']
  leaf = node['leaf']
  if leaf == -1:
      if d[atri] <= value:
          return p(l,d)
      else:
          return p(r,d)
  else:
    return leaf

# given attribute and split value
# return two datasets which are split by the criteria
def extract_data(data, atri_index, split_value):
  data = np.array(data)
  right_data = data[np.where(data[:,atri_index]>split_value)]
  left_data = data[np.where(data[:,atri_index]<=split_value)]
  return left_data, right_data


########## Step 4: Pruning ##########
def pruning(diction, train_data, val_data): 
  accu = evaluate(diction, val_data)
  new_diction = dfs(diction, train_data, val_data)
  new_accu = evaluate(new_diction, val_data)
  while new_accu<accu:
        accu = new_accu
        diction = new_diction
	new_diction = dfs(diction, train_data, val_data)
        new_accu = evaluate(new_diction, val_data)
  return diction   

# use depth first search to find pruning candidates
# training data is used to determine the new label of new generated leaf
# validation data is used to decide whether each pruning will improve the model preformance 
def dfs(node, training_data, validation_data):
  l = node['l']
  r = node['r']
  atri_index = node['atri']
  split_value = node['value']
  leaf = node['leaf']

  #new_node = None
  
  if leaf != -1:
    #new_node = node
    return node
  else:
    training_data_l, training_data_r = extract_data(training_data, atri_index, split_value)
    validation_data_l, validation_data_r = extract_data(validation_data, atri_index, split_value)

    # if find the pattern to prune
    l_leaf = l['leaf']
    r_leaf = r['leaf']
    if l_leaf != -1 and r_leaf != -1:

      error1 = error(validation_data_l, l_leaf)
      error2 = error(validation_data_r, r_leaf)

      new_leaf = majority(training_data)
      error3 = error(validation_data, new_leaf)

      if True: #error3 < error1 + error2:
        # prune      
        new_node = {'atri':None,'value':None,'l': None,'r': None,'leaf':new_leaf}
        #print('----- pruing once', new_node == node)
        return new_node
        
      else:
        # do not prune
        return node
    else:
      # combine two branch and check again
      new_l = dfs(l, training_data_l, validation_data_l)
      new_r = dfs(r, training_data_r, validation_data_r)
      new_node = {'atri':atri_index, 'value':split_value, 'l': new_l, 'r': new_r, 'leaf': leaf}
      
      #------------------------------------------------------------------------------------
      l_leaf = new_l['leaf']
      r_leaf = new_r['leaf']
      if l_leaf != -1 and r_leaf != -1:        
    
        error1 = error(validation_data_l, l_leaf)
        error2 = error(validation_data_r, r_leaf)

        new_leaf = majority(training_data)
        error3 = error(validation_data, new_leaf)

        if True: #error3 < error1 + error2:
          # prune again    
          #print('----- pruing on pruned node')
          new_node = {'atri':None, 'value':None, 'l': None, 'r': None, 'leaf': new_leaf}
        
        else:
          pass
      #------------------------------------------------------------------------------------    
      return new_node
  
def error(data, leaf):
  e = np.count_nonzero(data[:,7]!=leaf)    
  return e

# calculate the depth of a tree after pruning
def maxDepth(root):
  if root['leaf'] != -1:
    return 0
  else:
    l = root['l']
    r = root['r']
    depth_l = maxDepth(l)
    depth_r = maxDepth(r)
    return max(depth_l, depth_r) + 1

########## Step 5: Visualization #########
from collections import deque
import matplotlib.pyplot as plt

def bfs(ax, root, xmin, xmax, ymin, ymax):
  depth = maxDepth(root)
  gap = 1.0/depth
  queue = deque([(root, xmin, xmax, ymin, ymax)])
  while len(queue) > 0:
    e = queue.popleft()
    node = e[0]
    xmin = e[1]
    xmax = e[2]
    ymin = e[3]
    ymax = e[4]
    atri = node['atri']
    val = node['value']
    text = '['+str(atri)+']:'+str(val)
    #---------------------
    center = xmin+(xmax-xmin)/2.0
    d = (center-xmin)/2.0        
    #---------------------
    if node['l'] != None:
      queue.append((node['l'], xmin, center, ymin, ymax-gap))
      ax.annotate(text, xy=(center-d, ymax-gap), xytext=(center, ymax),
            arrowprops=dict(arrowstyle="->"),
            )
    if node['r'] != None:
      queue.append((node['r'], center, xmax, ymin, ymax-gap))
      ax.annotate(text, xy=(center+d, ymax-gap), xytext=(center, ymax),
            arrowprops=dict(arrowstyle="->"),
            )
    #---------------------
    if node['leaf'] != -1:    
      an1 = ax.annotate(node['leaf'], xy=(center, ymax), xycoords="data",
                  va="bottom", ha="center",
                  bbox=dict(boxstyle="round", fc="w"))
    #---------------------

def visualize(root):
  fig, ax = plt.subplots(figsize=(18, 8))
  bfs(ax, root, 0.0, 1.0, 0.0, 1.0)
  fig.subplots_adjust(top=0.9)
  plt.show()

########## Step 4: testing ##########
evaluate_ten_fold_cross_validation(data)
'''
data = np.loadtxt('wifi_db/noisy_dataset.txt')
#np.random.seed(2)
idx = np.random.permutation(data.shape[0])
data = data[idx]


l8 = int(8*(data.shape[0]/10.0))
l9 = int(9*(data.shape[0]/10.0))
training_data = data[0:l9]
training_data2 = data[0:l8]
validation_data = data[l8:l9]
test_data = data[l9:]

d1 = 0
root1, d = decision_tree_learning(training_data, d1)
#cr1 = evaluate(root1, test_data)/float(len(test_data))
#print('----before pruing', 1-cr1, 'depth: ', d)

d2 = 0
root2= pruning(root1, training_data2, validation_data)
print(root2)
#cr2 = evaluate(root2, test_data)/float(len(test_data))
#print('----after pruing:', 1-cr2, 'depth: ', d2)
'''

