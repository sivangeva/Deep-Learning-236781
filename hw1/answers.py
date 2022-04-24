r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1. False. In sample error is the error rate you get on the same data set you used to build your predictor (i.e the training set). 
The test set will give us the out-sample error- the error we get from the model while using new data the model haven't seen before.

2. False. Choosing the right ratio between training set size to testing set size can improve the models perfoemance and lewer the generalization error. 
For too small training set the model might not be trained sufficiently and for too big training set there could be not enough 
testing samples to reach sufficient loss rate. In addition, we can adversely affect the models performance by choosing unequally distributed 
training and testing sets. Hence, to minimize the out-sample error we should choose randomly samples for each set.

3. True. The validation set is splited from the training set to tune the hyperparameters for the model and the model itself. 
While tuning the model and it's hyperparameters, the test-set should not be used because we need samples to check the models out-sample error.   

4. True. Performing k-fold cross validation means we split our training set to k groups and treat each group at a time as a testing 
set while all the residual training samples are used to tune the model's hyperparameters. Each fold performance can be used as a proxy 
for the model's generalization error because the model didn't use the fold's validation set to train the model.    


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**
The ideal pattern to see in a residual plot is horizontal line that intercept the $y - \hat{y}$ axis at zero. This means 
a linear regression model describe precisely the dataset or the transformed form of it. If we compare the fitness of linear 
regression model between the top 5 correlated dataset to the transformed features of the dataset we see that the transformed dataset 
has kess outliners and more concentrated around the ideal pattern in the residual plot, thus more fitted to linear regression model.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**
1. Adding non-linear features to the data and then using linear regression model to fit and predict samples doesn't change 
the model from being a linear regression model. The non-linear transformation we do on tha data gives as a transformed dataset 
we can treat as new sampled dataset that we fit to new hyperparameters of linear regression model.
2. Any non-linear function can be fitted on the original dataset, but some non-linear functions may produce overfitting of the model.
3. Adding non-linear features may change tha the number of hyperparameters we need to fit for the model. This means that the weight 
matrix may give a decision boundary from different order in space, but it can still be a hyperplane because linear-classification
can be used for unconditional superior number of features.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q3 = r"""
**Your answer:**
1. Using np.logspace to define the range of $\lambda$ is more suitable for this kind of hyperparameter because it allows us to 
test more broadly distributed range of values. It is an advantage for CV because it is computationally lighter than using np.linespace 
which we will need larger set of $\lambda$  values till we reach the full range we recieve with np.logspace.
2. The number of times the model was fitted is computed as described next:
$ num. \lambda * num. degrees * num. K-folds = 3*20*3 = 180  times $.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
