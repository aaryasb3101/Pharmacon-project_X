 In Decagon, the decoder is a simple per-relation bilinear/DistMult scoring function directly on two node embeddings. 

 VS PHARMACON:: 
 
That role is played by cross-attention + MLP classifier instead — A->B and B->A cross-attention produce FINAL_VECTOR_A/FINAL_VECTOR_B, which then feed an MLP classifier outputting the interaction probability.
MLP classfier si out decoder- architecturak upgrade.


