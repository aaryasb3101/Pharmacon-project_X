deep leanring course 

Day 1- 4 July 2026 

(C1W1L02)
To implement a neural network- you need only input x and will predict output y - all things in middle will be figured out by itself.(hidden units- each take all no. of x)
neural networks accurately map functions given that the number of training examples given are adequate 

(C1W1L03)- supervised learning with a neural network 

most common example is NN using user and ad info to predict if user will click on ad on website. 
cleverly select what should be x and y 
audio- 1D dimensional sequence - RNN
structured vs unstructured data 


(C1W2L01)- Binary classification 
logistic regression is an algo for binary classification 
notation: compact: X= col x1,x2….xm
where m = no of training examples
n= rows 
Python notation: Xshape= (nx,m)
Y= y1, y2, ..ym
Python notation: Yshape= (1,m)

(C1W2L02 and C1W2L03 )- Logistic Regression and COst function 

y-hat= probabilty that y=1 given x input features 
parameters: 1. w= real no with nx dimensions and 
2. b= real no 

y hat must be between 0 and 1- hence standard linear regression doesnt work.- thus use sigmond function for logistic 
sigmond (x) = 1/(1+e^-z)
if keep w and b separate 
MSE not used as squared error loss- we try to minimise loss- 
used in project: in my approach - in MLP output at end, we use sigmond layer so that raw score becomes the probabilty for each DDI- sigmond would make it more interpretable



(C1W2L04)- Gradient Descent
cross function measures how well parameters w and b are working on training set . - find w and b that minimise J(w,b)- cross function. 
J is convex function
Initialise w and b to either 0 or anything else 
Gradient descent minimises J by taking repeated steps in steepest downhill-
used in project: we will use this while training at the end - finds weights for gcn, smiles and cross attention matrices and transformers. 

(C1W2L05)- Derivatives 
i. Gradient/ slop is not uniform - depends on local postion- in our application 
ii. We apply this one parameter at a time- descision of where to push the classifier weights 
application to our project: every component has its own local slope wrt iii. final output- hence we push the classifier weights first then take that backwards. - need to focus on this part 


(C1W2L07 & 8)- Computation graph and derivatives 
i. back propogation- to go one step back- we compute derivative of final output variable J  wrt v. if (j=3v)
use chain rule to see how much each variable (intermediate) affects final output. 
ii. forward propagation: computes final output by moving left to right through the graph - using input values then calculating all intermediate. — plugging in input values and calculating each intermediate node in order.

iii. **NOTATION:** codes for BP: final output var to be optimised- for eg. J- convention in coding: dvar= derivative of final output wrt var. (dj/da= da- example) 

connection to project: 
i. Each part of the model- SMILES Transformer, GCN, multi-r elational GNN, and attention fusion layer are all one giant computation graph
ii. chain rule will tell us why attention layer is needed(interpretabilty)it measures sensitivity: slope value is what attention weighs. like dvar measures sensitivity of J to var. 
iii.  attention-based fusion layer is trained the same way: loss flows back through th weights, then further back into the GCN/GNN/Transformer.

(C1W2L09)- Logistic Regression Gradient Descent
i. using simple computation graph for logistic regression with two input features (x1, x2), weights (w1, w2), and bias (b) to compute z = w1x1 + w2x2 + b, then a = σ(z) (sigmoid), then the loss L(a,y)- mindmap attached 
using logistic we modify w 1, w2, b to reduce loss 
ii. Forward pass: Data flows left to right through the graph — inputs -> linear combination (z) -> activation (a) -> loss.
Backward pass : Gradients are computed right to left using the chain rule — first da (derivative of loss w.r. t. a), then dz = da · (derivative of a w.r.t. z), then dw1, dw2, db are found from dz.

(C1W2L10)- Gradient Descent on m Examples 
adds on to previous video:
i. average gradients to implement on m training examples: for loops are inefficient using matrix is better- vectorization.


 (C1W2L11 & 12)- Vectorization and example 
 i. compute using z= np.dot(w,x)+b - to find transpose directly  
 stack all vectors into matrix X.
 replace for loops with matrix and vector operations- run faster on gpus.
 faster due to  parallelism hardware instructions.

ii.example:
a. Example given: computing np.exp() (or log, abs, maximum, elementwise power, 1/v) on every element of a vector we do it in one line with numpy instead of looping element by element.

b. Applied to logistic regression: in the gradient descent, the inner loop over features (n features, updating dw_1, dw_2, ... dw_n) can be eliminated by initializing dw = np.zeros((n_x, 1)) and using vectorized addition instead of looping over eac.
c. we still have one loop left- m training examples
 
 iii. connection to project; 
 a. we are able to scale transformer/gnns largely in shortrer training time using vectorisation. - also in attention layer: vectorized weighted-sum (softmax)- they batch molecules 
 
 (C1W2L13)- Vectorizing Logistic Regression:
 i. sigmond- element wise: applies activation to entire row vector at once
 ii. backward pass: finds error for all examples at once
 iii. gradien: full parameter gradients in two matrix ops. 

 use instead of looping to find z and a for each separately.
 4 lines of numpy- one fill forward and back propogation. 


C1W2L14- Computation 
dZ = A - Y
db= (1/m) * np.sum(dZ)


(C1W2L15)- Broadcasting in Python 
i. Broadcasting -  NumPy performs arithmetic between arrays of different shapes without  loops — smaller is made to match the larger one for compatibilty.

take the sum of columns matrix A- and divide it by (cal.reshape(1,4)- it is a 1x4 matrix)
call reshape when ur not sure what dimensions for matrix is.

General principle: when applying +, -, *, / between an (m,n) / (m,1) / (1,n) array, or (1,1), copies 1,n m times to make compatible 

ii. used to normalise data, scaling. avoids for loops.
iii. connection to proj: 
a. embedding fusion: get tensor aligned so that attention layer works 
b. even when computign attention weights 
c. normalisation of esm/bias terms in gcn 
d. if (n,) shaped output- can show error. 



**(C1W2L16)- A Note on Python/Numpy Vectors** 
i. rank 1 arrays - neither row nor column vectors- transpose and normal give same output 
ii. np.random.randn(5,1)- always use column or row vector 
assert a.shape == (5,1)- use this to confirm the dimension of vector. 

**connection to project: assert shapes at each embedding stage- before attention layer.**



(C1W2L17)- Quick Tour of Jupyter/iPython Notebooks -

(C1W2L18)- Explanation of Logistic Regression's Cost Function 
i. MLE: pick parameters that make the labels most likely 
ii. mathematical jsutification of crosss entropy loss


 (C1W3L01)- Neural Network Overview 
i.  Simple strcuture:  simple neural network has an input layer, 1+hidden layers, and an output layer. Hidden just means 
you don't observe those values in the training data you only see inputs (X) and outputs (y).
Notation : Superscript [l] denotes the layer number (e.g., a^[1], W^[1]) - its different  from superscript (i), 
which denotes the i-th training example. Don't confuse the two.

The input layer's activations are written a^[0] = X.
Each hidden layer gives activations calleda^[1], a^[2], etc., and the final layers activation is called 
a^[L] = ŷ - this is the prediction. 
Layer counting: A "2-layer neural network" means 1 hidden layer + 1 output layer , input not counted 
Parameters per layer: Each layer has its own weight matrix W^[l] and bias vector b^[l. 

Pharmacon -stacked/parallel sub-networks (SMILES Transformer, GCN, multi-relational GNN, attention fusion layer).
 The layer notation convention here (W^[l], a^[l]) used.  Dimensions of W^[1] is  no fo  hidden units × no of input features.
 each hidden will be computed in two steps- z, then a. 

a neural network is essentially logistic regression stacked and repeated across layers/units.

