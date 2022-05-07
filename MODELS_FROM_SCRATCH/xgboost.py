from regressiontree import Node
import numpy as np

class XGBoostRegressionTree():
    def __init__(self, min_samples_split=2, max_depth=2,l2=1):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.l2=l2
    def build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = X.shape
        best_split = {}
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
            if best_split["var_red"]>1e-7:

                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
             
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["var_red"])
        leaf_value = np.sum(Y)/(len(Y)+self.l2)
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        best_split = {}
        max_var_red = -np.inf
        
        for feature_index in range(num_features):
            unique=(np.unique(dataset[:, feature_index]))
            possible_thresholds = unique
            
            for  threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    #  information gain
                    curr_var_red = self.variance_reduction(y, left_y, right_y)
                    if curr_var_red>max_var_red:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["var_red"] = curr_var_red
                        max_var_red = curr_var_red
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def score(self,Y):
        return np.sum(Y)**2/(len(Y)+self.l2)
    
    def variance_reduction(self, parent, l_child, r_child):
        return 1/2*(self.score(l_child) +self.score(r_child) -self.score(parent)  )
   
    def fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    def make_prediction(self, x, tree):
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
    
    def predict(self, X):        
        preditions = np.array([self.make_prediction(x, self.root) for x in X])
        return preditions
class XGBoostRegressor():
    def __init__(self,n_trees=100,lr=0.1,max_depth=2, min_samples_split=2, l2=1):
        self.n_trees=n_trees
        self.lr=lr
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.l2=l2
        self.trees=[]

    def fit(self,X,y):
        self.y=y
        self.X=X
        F0 = np.mean(y)
        for m in range(self.n_trees):
            tree = XGBoostRegressionTree( max_depth=self.max_depth,l2=self.l2, min_samples_split=self.min_samples_split)
            r=np.sum([y,-F0.reshape(-1,1)],axis=0)#negative gradient
            tree.fit(X,r)
            self.trees.append(tree)
            h=tree.predict(X)
            F=F0+self.lr*h
            F0=F
        return

    def predict(self,x):
        
        pred=np.ones(x.shape[0])*np.mean(self.y)
        for i,tree in enumerate(self.trees):
            pred+= self.lr*tree.predict(x).reshape(-1)
        return pred
   
if __name__=="__main__":
     mdl=XGBoostRegressor(n_trees=100)