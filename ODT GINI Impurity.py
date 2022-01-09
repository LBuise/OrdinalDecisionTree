# Initial implementation from https://github.com/Eligijus112/decision-tree-python/blob/main/DecisionTree.py
# Test Dataset from http://lib.stat.cmu.edu/datasets/sleep

# Luuk Derks & Luco Buise
 
import pandas as pd 
import numpy as np 
from collections import Counter
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn import model_selection 

class Node: 
    """
    Class for creating the nodes for a decision tree 
    """
    def __init__(
        self, 
        Y: list,
        X: pd.DataFrame,
        min_samples_split=None,
        max_depth=None,
        depth=None,
        node_type=None,
        rule=None
    ):
        # Saving the data to the node 
        self.Y = Y 
        self.X = X

        # Saving the hyper parameters
        self.min_samples_split = min_samples_split if min_samples_split else 20
        self.max_depth = max_depth if max_depth else 5

        # Default current depth of node 
        self.depth = depth if depth else 0

        # Extracting all the features
        self.features = list(self.X.columns)

        # Type of node 
        self.node_type = node_type if node_type else 'root'

        # Rule for spliting 
        self.rule = rule if rule else ""

        # Calculating the counts of Y in the node 
        self.counts = Counter(Y)

        # Getting the GINI impurity based on the Y distribution
        self.gini_impurity = self.get_GINI()

        # Sorting the counts and saving the final prediction of the node 
        counts_sorted = list(sorted(self.counts.items(), key=lambda item: item[1]))

        # Getting the last item
        yhat = None
        if len(counts_sorted) > 0:
            yhat = counts_sorted[-1][0]

        # Saving to object attribute. This node will predict the class with the most frequent class
        self.yhat = yhat 

        # Saving the number of observations in the node 
        self.n = len(Y)

        # Initiating the left and right nodes as empty nodes
        self.left = None 
        self.right = None 

        # Default values for splits
        self.best_feature = None 
        self.best_value = None 

    @staticmethod
    def GINI_impurity(counts: list()) -> float:
        """
        Given the observations calculate the GINI impurity
        """

        # Getting the total observations
        n = sum(counts)
        
        # If n is 0 then we return the lowest possible gini impurity
        if n == 0:
            return 0.0

        # Getting the probability for each of the classes
        probabilities = []
        for count in counts:
            probabilities.append(count/n)
            
        # Calculating GINI 
        gini = 1
        for prob in probabilities:
            gini -= prob ** 2

        return gini

    def get_GINI(self):
        """
        Function to calculate the GINI impurity of a node 
        """
        counts = list(self.counts.values())

        # Getting the GINI impurity
        return self.GINI_impurity(counts)

    def best_split(self) -> tuple:
        """
        Given the X features and y targets calculates the best split 
        for an ordinal decision tree
        """
        # Creating a dataset for splitting
        df = self.X.copy()
        df['Y'] = self.Y

        # Getting the GINI impurity for the base input 
        GINI_base = self.get_GINI()

        # Finding which split yields the best GINI gain 
        max_gain = 0

        # Default best feature and split
        best_feature = None
        best_value = None

        for feature in self.features:
            # Droping missing values
            Xdf = df.dropna().sort_values(feature)

            # Getting the unique values and sorting them
            x_ranking = np.sort (np.unique (np.array(Xdf[feature])))
                        
            for value in x_ranking:
                # Spliting the dataset 
                left_counts = Counter(Xdf[Xdf[feature] <= value]['Y'])
                right_counts = Counter(Xdf[Xdf[feature] > value]['Y'])
            
                # Getting the Y distributions from the dicts
                y_left, y_right = list(left_counts.values()), list(right_counts.values())

                # Getting the left and right gini impurities
                gini_left = self.GINI_impurity(y_left)
                gini_right = self.GINI_impurity(y_right)

                # Getting the obs count from the left and the right data splits
                n_left = sum(y_left)
                n_right = sum(y_right)

                # Calculating the weights for each of the nodes
                w_left = n_left / (n_left + n_right)
                w_right = n_right / (n_left + n_right)

                # Calculating the weighted GINI impurity
                wGINI = w_left * gini_left + w_right * gini_right

                # Calculating the GINI gain 
                GINIgain = GINI_base - wGINI

                # Checking if this is the best split so far 
                if GINIgain > max_gain:
                    best_feature = feature
                    best_value = value 

                    # Setting the best gain to the current one 
                    max_gain = GINIgain

        return (best_feature, best_value)

    def grow_tree(self):
        """
        Recursive method to create the decision tree
        """
        # Making a df from the data 
        df = self.X.copy()
        df['Y'] = self.Y
        
        # If the max_depth and the minimum samples to split have not been reached yet we split further
        if (self.depth < self.max_depth) and (self.n >= self.min_samples_split):
            
            # Getting the best split 
            best_feature, best_value = self.best_split()

            if best_feature is not None:
                # Saving the best split to the current node 
                self.best_feature = best_feature
                self.best_value = best_value

                # Getting the left and right nodes
                left_df, right_df = df[df[best_feature]<=best_value].copy(), df[df[best_feature]>best_value].copy()

                # Creating the left and right nodes
                left = Node(
                    left_df['Y'].values.tolist(), 
                    left_df[self.features], 
                    depth=self.depth + 1, 
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split, 
                    node_type='left_node',
                    rule=f"{best_feature} <= {round(best_value, 3)}"
                    )

                self.left = left 
                self.left.grow_tree()

                right = Node(
                    right_df['Y'].values.tolist(), 
                    right_df[self.features], 
                    depth=self.depth + 1, 
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split,
                    node_type='right_node',
                    rule=f"{best_feature} > {round(best_value, 3)}"
                    )

                self.right = right
                self.right.grow_tree()

    def print_info(self, width=4):
        """
        Method to print the infromation about the tree
        """
        # Defining the number of spaces 
        const = int(self.depth * width ** 1.5)
        spaces = "-" * const
        
        if self.node_type == 'root':
            print("Root")
        else:
            print(f"|{spaces} Split rule: {self.rule}")
        print(f"{' ' * const}   | GINI impurity of the node: {round(self.gini_impurity, 2)}")
        print(f"{' ' * const}   | Class distribution in the node: {dict(self.counts)}")
        print(f"{' ' * const}   | Predicted class: {self.yhat}")   

    def print_tree(self):
        """
        Prints the whole tree from the current node to the bottom
        """
        self.print_info() 
        
        if self.left is not None: 
            self.left.print_tree()
        
        if self.right is not None:
            self.right.print_tree()

    def predict(self, X:pd.DataFrame):
        """
        Batch prediction method
        """
        predictions = []

        for _, x in X.iterrows():
            values = {}
            for feature in self.features:
                values.update({feature: x[feature]})
        
            predictions.append(self.predict_obs(values))
        
        return predictions

    def predict_obs(self, values: dict) -> int:
        """
        Method to predict the class given a set of features
        """
        cur_node = self
        while cur_node.depth < cur_node.max_depth:
            
            # Traversing the nodes all the way to the bottom
            best_feature = cur_node.best_feature
            best_value = cur_node.best_value
            
            if cur_node.n < cur_node.min_samples_split:
                break 

            if (values == None or best_feature == None):
                break
            
            if (values.get(best_feature) <= best_value):
                if self.left is not None:
                    cur_node = cur_node.left
            else:
                if self.right is not None:
                    cur_node = cur_node.right
         
        return cur_node.yhat
        
if __name__ == '__main__':
    # Reading training data
    d = pd.read_csv("data/sleep.csv")[['Predation', 'SleepExp', 'Danger']].dropna()

    # Constructing the X and Y matrices
    X = d[['Predation', 'SleepExp']]
    y = d['Danger'].values.tolist()
    
    # Initiating the Node
    root = Node(y, X, max_depth=5, min_samples_split=5)

    # Growing the tree
    root.grow_tree()

    # Printing the tree information 
    root.print_tree()
    
    
    # 10-fold cross validation to find the mean accuracy and MSE
    kf = model_selection.KFold(n_splits=10)
    
    accuracies = []
    errors = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        
        root = Node(y_train, X_train, max_depth=5, min_samples_split=5)
        
        root.grow_tree()
        
        y_pred = root.predict(X_test)
        
        accuracies.append(accuracy_score(y_test, y_pred))
        errors.append(mean_squared_error(y_test, y_pred))
        
    print("Mean accuracy 10-fold cross validation GINI impurity:", np.mean(accuracies))
    print("Mean error 10-fold cross validation GINI impurity:", np.mean(errors))
