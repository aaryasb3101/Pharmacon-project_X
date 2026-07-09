deep leanring course 

Day 1- 4 July 2026 

**(C1W1L02)**
To implement a neural network- you need only input x and will predict output y - all things in middle will be figured out by itself.(hidden units- each take all no. of x)
neural networks accurately map functions given that the number of training examples given are adequate 

**(C1W1L03)- supervised learning with a neural network**

most common example is NN using user and ad info to predict if user will click on ad on website. 
cleverly select what should be x and y 
audio- 1D dimensional sequence - RNN
structured vs unstructured data 


**(C1W2L01)- Binary classification** 
logistic regression is an algo for binary classification 
notation: compact: X= col x1,x2….xm
where m = no of training examples
n= rows 
Python notation: Xshape= (nx,m)
Y= y1, y2, ..ym
Python notation: Yshape= (1,m)

**(C1W2L02 and C1W2L03 )- Logistic Regression and COst function** 

y-hat= probabilty that y=1 given x input features 
parameters: 1. w= real no with nx dimensions and 
2. b= real no 

y hat must be between 0 and 1- hence standard linear regression doesnt work.- thus use sigmond function for logistic 
sigmond (x) = 1/(1+e^-z)
if keep w and b separate 
MSE not used as squared error loss- we try to minimise loss- 
used in project: in my approach - in MLP output at end, we use sigmond layer so that raw score becomes the probabilty for each DDI- sigmond would make it more interpretable



**(C1W2L04)- Gradient Descent**
cross function measures how well parameters w and b are working on training set . - find w and b that minimise J(w,b)- cross function. 
J is convex function
Initialise w and b to either 0 or anything else 
Gradient descent minimises J by taking repeated steps in steepest downhill-
used in project: we will use this while training at the end - finds weights for gcn, smiles and cross attention matrices and transformers. 

**(C1W2L05)- Derivatives** 
i. Gradient/ slop is not uniform - depends on local postion- in our application 
ii. We apply this one parameter at a time- descision of where to push the classifier weights 
application to our project: every component has its own local slope wrt iii. final output- hence we push the classifier weights first then take that backwards. - need to focus on this part 


**(C1W2L07 & 8)- Computation graph and derivatives** 
i. back propogation- to go one step back- we compute derivative of final output variable J  wrt v. if (j=3v)
use chain rule to see how much each variable (intermediate) affects final output. 
ii. forward propagation: computes final output by moving left to right through the graph - using input values then calculating all intermediate. — plugging in input values and calculating each intermediate node in order.

iii. **NOTATION:** codes for BP: final output var to be optimised- for eg. J- convention in coding: dvar= derivative of final output wrt var. (dj/da= da- example) 

connection to project: 
i. Each part of the model- SMILES Transformer, GCN, multi-r elational GNN, and attention fusion layer are all one giant computation graph
ii. chain rule will tell us why attention layer is needed(interpretabilty)it measures sensitivity: slope value is what attention weighs. like dvar measures sensitivity of J to var. 
iii.  attention-based fusion layer is trained the same way: loss flows back through th weights, then further back into the GCN/GNN/Transformer.

**(C1W2L09)- Logistic Regression Gradient Descent**
i. using simple computation graph for logistic regression with two input features (x1, x2), weights (w1, w2), and bias (b) to compute z = w1x1 + w2x2 + b, then a = σ(z) (sigmoid), then the loss L(a,y)- mindmap attached 
using logistic we modify w 1, w2, b to reduce loss 
ii. Forward pass: Data flows left to right through the graph — inputs -> linear combination (z) -> activation (a) -> loss.
Backward pass : Gradients are computed right to left using the chain rule — first da (derivative of loss w.r. t. a), then dz = da · (derivative of a w.r.t. z), then dw1, dw2, db are found from dz.

**(C1W2L10)- Gradient Descent on m Examples** 
adds on to previous video:
i. average gradients to implement on m training examples: for loops are inefficient using matrix is better- vectorization.


 **(C1W2L11 & 12)- Vectorization and example** 
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
 
 **(C1W2L13)- Vectorizing Logistic Regression:**
 i. sigmond- element wise: applies activation to entire row vector at once
 ii. backward pass: finds error for all examples at once
 iii. gradien: full parameter gradients in two matrix ops. 

 use instead of looping to find z and a for each separately.
 4 lines of numpy- one fill forward and back propogation. 


**C1W2L14- Computation**
dZ = A - Y
db= (1/m) * np.sum(dZ)


**(C1W2L15)- Broadcasting in Python** 
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



**(C1W2L17)- Quick Tour of Jupyter/iPython Notebooks -**

**(C1W2L18)- Explanation of Logistic Regression's Cost Function** 
i. MLE: pick parameters that make the labels most likely 
ii. mathematical jsutification of crosss entropy loss


 **(C1W3L01)- Neural Network Overview & (C1W3L02)- Neural Network representations:**
i.  Simple strcuture:  simple neural network has an input layer, 1+hidden layers, and an output layer. Hidden just means 
you don't observe those values in the training data you only see inputs (X) and outputs (y).
Notation : Superscript [l] denotes the layer number (e.g., a^[1], W^[1]) - its different  from superscript (i), 
which denotes the i-th training example. Don't confuse the two.

the input layer's activations are written a^[0] = X.
Each hidden layer gives activations calleda^[1], a^[2], etc., and the final layers activation is called 
a^[L] = ŷ - this is the prediction. 
Layer counting: A "2-layer neural network" means 1 hidden layer + 1 output layer , input not counted 
Parameters per layer: Each layer has its own weight matrix W^[l] and bias vector b^[l. 

Pharmacon -stacked/parallel sub-networks (SMILES Transformer, GCN, multi-relational GNN, attention fusion layer).
 The layer notation convention here (W^[l], a^[l]) used.  Dimensions of W^[1] is  no fo  hidden units × no of input features.
 each hidden will be computed in two steps- z, then a. 

 debugging- if hidden layer with 4 units and 3 inputs- 4x3 and b is 4x1. debug like this.

a neural network is essentially logistic regression stacked and repeated across layers/units.


**(C1W3L03)- Computing Neural Network Output:**
output of one layer directly feeds as input to the next. ina  full forward pass for a 2- layer network.

 **(C1W3L04)- Vectorizing Across Multiple Examples:**
 
i. Forward prop becomes: same equations as before, just matrices instead of vectors.
ii. each column of Z and A = one training example; each row  = one hidden unit.
iii.b still adds using NumPy broadcasting- even tho the dimensions look unmatched.


**(C1W3L05): Explanation For Vectorized Implementation** 
matrix multiplication- it preserves one column in one column out struture- 
the horizontal axis of X (and of Z, A) represents different training examples, the vertical axis represents different features (or hidden units)


connection to pharma: 
processin of  many drug pairs at once instead of one at a time. Every time we give a batch of drugs in a ny component we stack them this way instead of looping. - we can trust this batching by what learnt in the explanation. 


**(C1W3L06 & 7)- Activation Functions & Explanation**
activation functions: give **non linear activations**- otherwise stacking is useless as it is still equivalent to single later linear. 
NN compute interesting functions using non linear- otherwise hidden layers useless. 
i. shift sigmond to get tanh- except for output layer (for binary)
tanh works since it (-1 to 1)- ie zero centered- learn faster. 
ii. slope is very small is when z is very large or very small- this is disadvantage: use relu.(rectified linear unit)(default choice for hidden layer:)- neural network learns faster. 
iii.leaky relu 

**C1W3L08 — derivatives of activation functions**
i. we need slope for backprop- tells how to adjust weights- 
ii. 
a. sigmoid: derivative is a(1-a)
b. tanh: derivative is 1-a²
c. relu: derivative is 0 for z<0 and 1 for z>0- if zero pick either 0 or 1
d. leaky relu: derivative is 0.01 for z<0 and 1 for z>0- doesnt consider 0.

pharma: gcn, transformer, or gnn- network needs slopes to learn - these slopes flow backwards and weights updated during training. relu in gcn (common)- we use this concept when we call .backward().

**(C1W3L09)- Gradient Descent For Neural Networks  and (C1W3L10)- Backpropagation Intuition**
i. gradient descent repeatedly updates these to minimize the cost function J.
ii. the element-wise multiply (*) with g′(Z(1)) in dZ(1) is the chain rule
it's how the error signal from the output layer gets distributed back through the activation function of the earlier layer-  dimensions must match at every step.

forward pass to get a prediction, backprop to get gradients, then update weights. the chain rule step

pharma: when pharmacon makes a wrong prediction,- the chain rule step will be the same idea used to backprop through the fusion layer into both the chemical and biological branches(into the gnn, gcn, and transformer), so both sides learn from the same interaction-prediction error and adjust weights. 


**(C1W3L11)- Random INitialisation**
i. initialse paramaters randomly not to zero. 
ii. if all weights in a layer are initialized to zero (or the same value), every hidden unit computes the exact same function and  this is called the symmetry problem- identical weights, identical activations, identical gradients, no matter how long you train. having multiple hidden units is now pointless.
iii. initialize w randomly ( np.random.randn(shape) * 0.01- ex- put any very small value), while b =0 is safe as w already unsymmetric.
iv. use any time u use a new network. 


 **(C1W4L01)- Deep L-Layer N eural Network & Forward Propagation in a Deep Network (C1W4L02) & Getting Matrix Dimensions Right (C1W4L03)**


i. deep neural network just means one with more hidden layers stacked between input and output
**debugging**
ii. getting the matrix dimensions right at each layer is critical for debugging
iii. for a network with L layers, w(l) must have shape (n(l), n(l-1)) — rows = units in current layer, columns = units in previous layer.
iv. b(l) has shape (n(l), 1) - one bias per unit in that layer.
z(l) and a(l) both have shape (n(l), 1)
dw(l) and db(l) (used in backprop) must match the shape of w(l) and b(l)exactly resp,- debugging.

**(C1W4L04)- Why Deep Representations?** 
Find simple thihngs like edges and building them up to identify more complex. 
i.  early layers detect simple things (edges in images, or low-level patterns), later layers combine these into more complex things (parts,  then whole objects/ oncepts)
ii. face recognisation/ audio
iii. circuit theory: some functions need exponentially more  hidden units to compute with a shallow network vs.  a deep one — depth more efficient than width

**(C1W4L05)- Building Blocks of a Deep Neural Network** 
training loop:
for the whole network,  chain  boxes(layers) together : forward pass computes y hat, then backward pass starts from da for the last layer and works back to get all the gradients (dw(l), db(l) for every layer.

**C1W4L06)- Forward and Backward Propagation** 
i. parameters:w,b- learned by models during training 
also need to tell learning rate - alpha 
ii. hyperparameters- set by us before traiing-  learning rate, number of iterations, number of hidden layers L, number of hidden units per layer, choice of activation function.
iii. hyperparameters control the parameters- determines final values of W and B
empirical- try and test and adjust to find best hyperparam

pharma: 
earning rate, number of GCN layers, embedding size, and attention heads are hyperparameters
weights inside the SMILES transformer, GCN, and GNN- parameters
fusion layer will have trial and eror


**What does this have to do with the brain? (C1W4L08):**
historical, doesnt really relate as much, stick to math and optimisation. 

