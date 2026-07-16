1. Data Loading: I named columns myself and verified data
2. tensorflow- building and training NN(keras in this) 
3. df.values- converted to numpy array 
4. stratify- forces 9 percent ratio to stay the same even in the 30 percent split
5. scale to mean 0 and std 1 
6. calculate a weight inversely prop to frequency- pulsars get high weights 
(rare)- convert ro label: weight
7. SGD/Momentum use a higher  LR (0.01) than RMSprop/Adam (0.001) - standard, adaptive optimisers need smaller learning rates
8. optimiser loop: 
loop starts for each optimiser- builds a fresh model from scartch so that weights dont carry forward 
sequatential is a stack of layers: 
first hidden layer- 16 neurons, relu activation, he initialisation, 8 input features expected 
second- 8 neurons 
third: output layer- 0/1. predicts probabilty

also have binary cross entropy to make sure accurate during training- not the alone accuracy measurer- might be misleading. 
9. validation_data=(X_dev, y_dev) - after each epoch, evaluates on dev data too (without training on it), so we get the validation loss curve
10. store trianing history and trained model
11. history is a dictionary created by keras- contains lists per epoch 
12. flatten 2da array to 1d - convert probabilities to 0 or 1
13. print precision, recall, F1, support and compare with y_dev(true labels)
14.Compute precision and recall at every possible threshold, not just 0.5 - this is why we use y_pred_proba which is actual probabilities and not y_pred (0/1 labels).
15. find avg precision to get area under PR curve
16. test on lower LR- 0.01 to 0.002- to check if peaks are lower
17. test on higher batch size of momentum- 128 to 32 
18. PR curves are one longer than thresholds- and we find the best precision with recall greater than 90 percent 
19. run model on untouched test set- threshold as selected is 0.549 and not 0.5
20.  everything before this used dev data for decisions







qs
why do i need labels? 
why fix randomness 
whats dev 
why standard scaler? 
what is a pulsar 
how inverse shown? 
what are adaptive optimisers
binary cross why misleading
epoch?
ap_score:.3f
mark boolean array 