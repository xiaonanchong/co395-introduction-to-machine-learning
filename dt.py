import numpy as np

########## step 1 Loading data ##########

data = np.loadtxt('WIFI_db/clean_dataset.txt') ## data-[2000, 8]

########## step 2 Creating Decision Tree ##########

smallest_num = -1000000

## return dictionary object ({'atri', 'value', 'l', 'r', leaf}, depth)
## value >, <=
def decision_tree_learning(training_data, depth):
  if same_label(training_data):

    return ({'atri': None, 'value':None, 'l':None, 'r': None, 'leaf': data[0,7]}, depth)

  else: 
    split_attri, split_value = find_split(training_data)
    if split_attri == -100:
      node = {'atri': None, 'value': None, 'l': None, 'r': right_None, 'leaf': majority(training_data)}
    else:
	    left, right = extract_data(training_data, split_attri, split_value)
	    print(len(left), len(right))
	    (left_branch, l_depth) = decision_tree_learning(left, depth +1)
	    (right_branch, r_depth) = decision_tree_learning(right, depth +1)
	    node = {'atri': split_attri, 'value': split_value, 'l': left_branch, 'r': right_branch, 'leaf': -1}

    return (node, max(l_depth, r_depth))


def find_split(training_data):
  flag = False
  (attri, value, gain) = (-1, smallest_num, smallest_num)

  for attribute in range(training_data.shape[1]-1): # for each attribute 0-6
    sorted_data = sort(training_data, attribute)

    last_split_value = smallest_num
    for i in range(training_data.shape[0]-1): # for each possible split
      if training_data[i, 7] != training_data[i+1, 7]:
        split_value = (training_data[i, attribute] + training_data[i+1, attribute])/float(2)

        if split_value > last_split_value:

          left, right = extract_data(training_data, attribute, split_value)
          if len(right) != 0 and len(left) != 0:
                  flag = True
		  g = cal_gain(training_data, left, right)
		  if g>gain and g > 0: #########
		    attri = attribute
		    value = split_value
		    gain = g 
                    flag = True
  if flag==False:
    print('no proper split')
    print(sorted_data)
    print(split_value)
    print(same_label(training_data))
    return -100, 0
  return attri, value


def majority(label):
    clas = np.unique(label)
    Num = label.shape[0]
    leaf = -1
    for j in clas:
      Numj = np.count_nonzero(label == j)
      pj = float(Numj)/float(Num)
      if pj > leaf:
        leaf = j     
    
    return leaf


def cal_gain(all, left, right):
  H_all = information_entropy(all[:, 7])
  Hl = information_entropy(left[:, 7])
  Hr = information_entropy(right[:, 7])
  nl = float(len(left))
  nr = float(len(right))
  R = (nl/(nl+nr))* Hl + (nr/(nl+nr))* Hr
  #print(H_all, R)
  return H_all - R


def information_entropy(label):
    H = 0
    clas = np.unique(label)
    Num = label.shape[0]
    for j in clas:
      Numj = np.count_nonzero(label == j)
      pj = float(Numj)/float(Num)
      H -= pj* np.log(pj)  
    return H


def sort(data, attribute):
  column = data[:, attribute]
  idx = np.argsort(column)
  return data[idx]

    
def same_label(data):
  label = data[:, 7]
  label1 = label[0]
  for i in range(label.shape[0]):
    if label[i] != label1:
      return False
      break
  return True

########## Step 3: Evaluation ##########

N = 10

def ten_fold_cross_validation(data):
  total_len = data.shape[0]
  one_fold = int(total_len/10)

  total_error = 0

  for i in range(N):        
    test_data = data[one_fold*i : one_fold*(i+1)]
    train_data = np.concatenate((data[0: one_fold*i], data[one_fold*(i+1):]), axis = 0)
    (root_node, depth) = decision_tree_learning(train_data, 0) #### training function
    error = evaluate(root_node, test_data)
    total_error = total_error + error

  error_rate = float(total_error)/(one_fold*N)
  return 1-error_rate


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
    for i in split_data:
      if int(i[7]) != leaf:
        count = count + 1
    error = count

  return error
  

def error(leaf, split_data): ## not used!

  count = 0
  for i in split_data:
    if int(i[7]) == leaf:
      count = count + 1
  return count


def extract_data(data, atri_index, split_value):
  right_data = [[0,0,0,0,0,0,0,0]]
  left_data =[[0,0,0,0,0,0,0,0]]

  for i in range(np.array(data).shape[0]):
    if data[i, atri_index] > split_value:
      right_data = np.concatenate((right_data, [data[i, :]]), axis = 0)
    else:
      left_data = np.concatenate((left_data, [data[i, :]]), axis = 0) 
  return left_data[1:], right_data[1:]

##############################################
#step 4 pruning
'''
def pruing(root_node, validate_data):
  
def depth_first_tree(root):
  if root

def leaf(node):
  if  == ln
''' 

############################ TEST ###############################
dt = decision_tree_learning(data, 0)
cr = ten_fold_cross_validation(data)
print('----', cr)

a = np.array([[-59, -49, -53, -59, -62, -81, -83,   1.0],
 [-60, -54, -54, -60, -64, -88, -88,   4.0]])

#print(find_split(a))
