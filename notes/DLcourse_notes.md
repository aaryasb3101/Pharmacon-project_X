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
Gradient/ slop is not uniform - depends on local postion- in our application 
We apply this one parameter at a time- descision of where to push the classifier weights 
application to our project: every component has its own local slope wrt final output- hence we push the classifier weights first then take that backwards. - need to focus on this part 


(C1W2L07)- Computation graph 
 