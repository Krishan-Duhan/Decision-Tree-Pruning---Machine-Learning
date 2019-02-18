import random
import sys
import pandas
import math
import copy

count = 0                                                                   # global variable to keep node counts

class Node():
    def __init__(self):
      self.left = None
      self.right = None
      self.val = None
      self.attrib = None
      self.num_pos = None                                                   # number of + instances
      self.num_neg = None                                                   # number of - instances
      self.out = None                                                       # output label
      self.ntype = None                                                     # node can be leaf or intermediate or root
      self.nodenum = None                                                   # node number, used in pruning.

    def assignval(self, ntype, val = None, attrib = None, num_pos = None, num_neg = None):
        self.val = val
        self.attrib = attrib
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.ntype = ntype

# Function to calculate entropy for a given 1-D set of data which contains 0's & 1's
def entropy(sample):
    total = sample.shape[0]                                                 # total sample values
    pos = sample.sum().sum()                                                # positive sample values;double sum() to convert to value 
    neg = total - pos
    if(pos == 0 or neg == 0):
        return 0
    pos_prop = pos/total                                                    # proportion of positive samples
    neg_prop = neg/total                                                    # proportion of negative samples
    entropy = -(pos_prop*math.log(pos_prop,2) + neg_prop*math.log(neg_prop,2))
    return entropy

# Function to find information gain for a given attribute.
def infogain(attrib_class):                                                     # argument is dataframe of 2 columns : Attribute & Class
    total = attrib_class.shape[0]
    pos = attrib_class[attrib_class[attrib_class.columns[0]] == 1].shape[0]     # No of rows where column0 has val = 1
    neg = total - pos

    entropyS = entropy(attrib_class[['Class']])                             # entropy of whole Sample with all values of given attribute
    # Entropy of sample for negative values of attribute
    entropy_attrib_0 = entropy(attrib_class[attrib_class[attrib_class.columns[0]] == 0][['Class']])  
    # Entropy of sample for positive values of attribute
    entropy_attrib_1 = entropy(attrib_class[attrib_class[attrib_class.columns[0]] == 1][['Class']])

    pos_prop = pos/total                                                    # proportion of positive samples
    neg_prop = neg/total                                                    # proportion of negative samples
    infogain = entropyS - pos_prop*entropy_attrib_1 - neg_prop*entropy_attrib_0
    return infogain

# Function to calculate variance impurity
def var_impurity(sample):
    total = sample.shape[0]
    pos = sample.sum().sum()
    neg = total - pos
    if(total == 0):
        return 0
    vi = (pos/total)*(neg/total)
    return vi

# Function to calculate the variance impurity gain
def vargain(attrib_class):
    total = attrib_class.shape[0]
    pos = attrib_class[attrib_class[attrib_class.columns[0]] == 1].shape[0]     # No of rows where column0 has val = 1
    neg = total - pos

    viS = var_impurity(attrib_class[['Class']])
    vi_attrib_0 = var_impurity(attrib_class[attrib_class[attrib_class.columns[0]] == 0][['Class']])
    vi_attrib_1 = var_impurity(attrib_class[attrib_class[attrib_class.columns[0]] == 1][['Class']])

    pos_prop = pos/total                                                    # proportion of positive samples
    neg_prop = neg/total
    vi_gain = viS - pos_prop*vi_attrib_0 - neg_prop*vi_attrib_1
    return vi_gain

# Function to return best attribute on which a node can split; based on info gain or variance impurity gain heuristic
def getSplitAttrib(examples, heuristic):
    max_gain = -10
    if(heuristic == "infogain"):
        for col in examples.columns.drop('Class'):
            gain = infogain(examples[[col, 'Class']])
            if gain > max_gain:
                max_gain = gain
                splitAttrib = col
        # print(max_gain)
    elif(heuristic == "var_impurity"):
        for col in examples.columns.drop('Class'):
            gain = vargain(examples[[col, 'Class']])
            if gain > max_gain:
                max_gain = gain
                splitAttrib = col
    return splitAttrib

# Function to predict output for a given test-example(row) based on a given tree
def getPrediction(example, root):
    if(root.out is not None):
        return root.out
    elif(example[root.left.attrib][example.index.tolist()[0]] == 1):
        return getPrediction(example, root.right)
    else:
        return getPrediction(example, root.left)

# Function to find accuracy of the given tree on given dataset
def findAccuracy(dset, tree):
    correct = 0
    for x in range(dset.shape[0]):
        prediction = getPrediction(dset.iloc[x:x+1, :].drop(['Class'], axis = 1), tree.root)
        if(prediction == dset['Class'][x]):
            correct += 1
    return (correct/dset.shape[0])

# Function to traverse & print the tree
def traverse_tree(root, depth):
    if(root.left is None and root.right is None):
        for i in range(depth):
            print("| ", end = '')
        depth += 1
        print("{} = {} : {}".format(root.attrib, root.val, root.out))
        return
    elif(root.left is not None and root.right is None):
        for i in range(depth):
            print("| ", end = '')
        depth += 1
        print("{} = {} : ".format(root.attrib, root.val))
        traverse_tree(root.left, depth)
    elif(root.left is None and root.right is not None):
        for i in range(depth):
            print("| ", end = '')
        depth += 1
        print("{} = {} : ".format(root.attrib, root.val))
        traverse_tree(root.right, depth)
    else:
        for i in range(depth):
            print("| ", end = '')
        depth += 1
        if(root.attrib is not None):
            print("{} = {} : ".format(root.attrib, root.val))
        traverse_tree(root.left, depth)
        traverse_tree(root.right, depth)

class Tree():
    def __init__(self):
        self.root = Node()
        self.root.assignval('r')
        self.root.nodenum = 0

    # Function to build a Decision Tree based on Information gain heuristic
    def ID3(self, examples, node):
        total = examples.shape[0]           # total no of samples in a column = no of rows
        pos = examples['Class'].sum()       # total no of positive samples
        neg = total - pos                   # total no of negative samples
        
        if(total == pos or total == neg or examples.shape[1] == 1):
            node.ntype = 'l'
            if(pos >= neg):
                node.out = 1
            else:
                node.out = 0
            return
        else:
            splitattrib = getSplitAttrib(examples, "infogain")
            # print(splitattrib)

            node.left = Node()
            # From my test file: print(ip[ip['XD'] == 0]['Class'].sum())
            pos_splitattrib_0 = examples[examples[splitattrib]==0]['Class'].sum()           # No of +ve instances when spliattrib = 0
            neg_splitattrib_0 = total - pos_splitattrib_0
            node.left.assignval('i', 0, splitattrib, pos_splitattrib_0, neg_splitattrib_0)
            
            node.right = Node()
            pos_splitattrib_1 = examples[examples[splitattrib]==1]['Class'].sum()           # No of +ve instances when spliattrib = 1
            neg_splitattrib_1 = total - pos_splitattrib_1
            node.right.assignval('i', 1, splitattrib, pos_splitattrib_1, neg_splitattrib_1)
            self.ID3(examples[examples[splitattrib] == 0].drop([splitattrib],axis = 1), node.left)
            self.ID3(examples[examples[splitattrib] == 1].drop([splitattrib],axis = 1), node.right)
    
    # Function to build a Decision Tree based on variance impurity heuristic
    def DT_varImp(self, examples, node):
        total = examples.shape[0]           # total no of samples in a column = no of rows
        pos = examples['Class'].sum()       # total no of positive samples
        neg = total - pos                   # total no of negative samples

        if(total == pos or total == neg or examples.shape[1] == 1):
            node.ntype = 'l'
            if(pos >= neg):
                node.out = 1
            else:
                node.out = 0
            return
        else:
            splitattrib = getSplitAttrib(examples, "var_impurity")
            # print(splitattrib)

            node.left = Node()
            pos_splitattrib_0 = examples[examples[splitattrib]==0]['Class'].sum()           # No of +ve instances when spliattrib = 0
            neg_splitattrib_0 = total - pos_splitattrib_0
            node.left.assignval('i', 0, splitattrib, pos_splitattrib_0, neg_splitattrib_0)

            node.right = Node()
            pos_splitattrib_1 = examples[examples[splitattrib]==1]['Class'].sum()           # No of +ve instances when spliattrib = 1
            neg_splitattrib_1 = total - pos_splitattrib_1
            node.right.assignval('i', 1, splitattrib, pos_splitattrib_1, neg_splitattrib_1)
            self.DT_varImp(examples[examples[splitattrib] == 0].drop([splitattrib],axis = 1), node.left)
            self.DT_varImp(examples[examples[splitattrib] == 1].drop([splitattrib],axis = 1), node.right)

# Function to number the non-leaf nodes in a given tree
def num_nodes(root):
    global count
    if(root.ntype == 'r'):
        root.nodenum = 0
        count += 1
        num_nodes(root.left)
        num_nodes(root.right)
    elif(root.ntype == 'i'):
        root.nodenum = count
        count += 1
        num_nodes(root.left)
        num_nodes(root.right)
    else:
        root.nodenum = None

# Function to find an internal node with given node-number in a given tree
def find_node(root, num):
    if((root.ntype == 'l')):
        return None
    if(root.nodenum == num):
        return root
    else:
        ret = None
        ret = find_node(root.left, num)
        if(ret is None):
            ret = find_node(root.right, num)
        return ret

# Function for post-pruning of a infogain tree. iReturns pruned tree
def post_prune(K, root):
    global count
    M = random.randint(1, K+1)
    for j in range(1, M+1):
        count = 0
        num_nodes(root)
        N = count
        if(N == 1):
            #print("got N = 1, break!")
            break
        p = random.randint(1, N) 
        curr_node = Node()
        curr_node = find_node(root, p)
        if(curr_node is None):
            #print("didnt find the node for p = {}".format(p))
            continue
        else:
            curr_node.ntype = 'l'
            curr_node.left = None
            curr_node.right = None
            #print("Made a leaf node at {} = {}".format(curr_node.attrib, curr_node.val))
            if(curr_node.num_pos > curr_node.num_neg):
                curr_node.out = 1
            else:
                curr_node.out = 0

pruned_tree_IG = Tree()
pruned_tree_VI = Tree()
Dbest = Tree()

def main():
    global count
    global pruned_tree_IG
    global pruned_tree_VI
    global Dbest

    if(len(sys.argv) != 7):
        print("Incorrect arguments passed!")
        sys.exit()
    
    L = int(sys.argv[1])
    K = int(sys.argv[2])
    train = pandas.read_csv(sys.argv[3])
    validate = pandas.read_csv(sys.argv[4])
    test = pandas.read_csv(sys.argv[5])
    to_print = sys.argv[6]
    
    if(to_print == "no"):
        original = sys.stdout
        sys.stdout = open('report.txt', 'a')

    tree1 = Tree()
    tree1.ID3(train, tree1.root)
    #print("DECISION TREE USING INFOGAIN, BEFORE PRUNING:")
    #traverse_tree(tree1.root, -1)
    #print("Accuracy of decision tree (using infogain) on test set before pruning = {}".format(findAccuracy(test, tree1)))
    
    tree2 = Tree()
    tree2.DT_varImp(train, tree2.root)
    print("\n\n")
    #print("DECISION TREE USING VARIANCE IMPURITY, BEFORE PRUNING:")
    #traverse_tree(tree2.root, -1)
    #print("Accuracy of decision tree (using var impurity) on test set before pruning = {}".format(findAccuracy(test, tree2)))

    # Pruning with infogain:
    Dbest = copy.deepcopy(tree1)
    Dbest_accuracy = findAccuracy(validate, Dbest)
    for i in range(1, L+1):
        #pruned_tree_IG = Tree()
        pruned_tree_IG = copy.deepcopy(tree1)
        post_prune(K, pruned_tree_IG.root)
        pruned_Accuracy_IG = findAccuracy(validate, pruned_tree_IG)
        if(pruned_Accuracy_IG > Dbest_accuracy):
            Dbest = copy.deepcopy(pruned_tree_IG)
            Dbest_accuracy = pruned_Accuracy_IG
    print("\n\n")
    #print("DECISION TREE USING INFOGAIN, AFTER PRUNING:")
    #traverse_tree(pruned_tree_IG.root, -1)
    print("Accuracy of decision tree (using infogain) on validation set after pruning is: {}".format(pruned_Accuracy_IG))
    print("Accuracy of decision tree (using infogain) on test set after pruning is: {}".format(findAccuracy(test, pruned_tree_IG)))
    
    # Pruning with variance impurity:
    Dbest = copy.deepcopy(tree2)
    Dbest_accuracy = findAccuracy(validate, Dbest)
    for i in range(1, L+1):
        #pruned_tree_IG = Tree()
        pruned_tree_VI = copy.deepcopy(Dbest)
        post_prune(K, pruned_tree_VI.root)
        pruned_Accuracy_VI = findAccuracy(validate, pruned_tree_VI)
        if(pruned_Accuracy_VI > Dbest_accuracy):
            Dbest = copy.deepcopy(pruned_tree_VI)
            Dbest_accuracy = pruned_Accuracy_VI
    print("\n\n")
    #print("DECISION TREE USING VARIANCE IMPURITY, AFTER PRUNING:")
    #traverse_tree(pruned_tree_IG.root, -1)
    print("Accuracy of decision tree (using variance impurity) on validation set after pruning is: {}".format(pruned_Accuracy_VI))
    print("Accuracy of decision tree (using variance impurity) on test set after pruning is: {}".format(findAccuracy(test, pruned_tree_VI)))

    if(to_print == "no"):
        sys.stdout = original

if __name__ == "__main__":main()
