# ID5-an-Incremental-ID3

In this project, I have implemented the algorithm described in this paper: https://www.sciencedirect.com/science/article/pii/B9780934613644500177</br>

**This algorithm is coded from scratch with python, without using any libraries.**

Instead of implementing the ID5 for a binary classification task, which was explained in the original papers, I implemented this algorithm for a multi-class classification task on the poker dataset.</br>
You can fined the dataset that I used in the following link: https://archive.ics.uci.edu/ml/datasets/Poker+Hand </br>

These are the main steps taken in this project:
1. Building a decision tree based on the ID5 algorithm.
2. Sorting the features based on their importance. The ones that appear on lower levels are less important. Since we wanted to use the decision tree for sorting features, I limited the algorithm to utilize each feature just once. 
3. Finding noisy data: I tried to recognize noise data by removing a portion of the least important features. As a matter of fact, when we remove a set of features and realize that the accuracy on test data increases, this means that the tree was overfitting the train data. In this situation, we can find the noise data by recognizing samples which were classified correctly before pruning the tree and classified incorrectly after pruning the tree.
4. cleaning the data by removing the least important features and noise data.
5. Building an MLP and reporting evaluation metrics.
