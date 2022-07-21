# ID5-an-Incremental-ID3

In this project, I have implemented the algorithm described in this paper: https://www.sciencedirect.com/science/article/pii/B9780934613644500177</br>

** This algorithm is coded from scratch with python, without using any libraries. **

Instead of implementing the ID5 for a binary classification task, which was explained in the original papers, I implemented this algorithm for a multi-class classification task on the poker dataset.</br>
You can fined the dataset that I used in the following link: https://archive.ics.uci.edu/ml/datasets/Poker+Hand </br>

After building a decision tree based on ID5, I utilized the tree to sort the features based on their importance. then i tried to recognize noise data by removing a portion of the less important features( the ones that appeared in the lowest level of the tree). As a matter of fact, when we remove a set of features and realize that the accuracy increases, this means that our tree was overfitting the train data. in this situation, we can find the samples which were classified correctly before pruning the tree and classified incorrectly after pruning the tree. those samples are our noise data!
