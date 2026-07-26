##1. C5W3L01 - Basic Models
Introduces sequence-to-sequence (seq2seq) architecture: an encoder RNN reads the input sentence and compresses it into a vector representation; a decoder RNN generates the output sentence one word at a time from that vector.
Example: French > English machine translation.
Related idea: image captioning - replace the encoder RNN with a pretrained CNN (e.g., AlexNet-style) that encodes the image, then feed that encoding into an RNN decoder to generate a caption instead of a full sentence.
Sets up the core question tackled by the rest of the playlist: how do you actually pick the best output sentence, rather than just sampling words?

##2. C5W3L02 - Picking the Most Likely Sentence
Frames machine translation as a conditional language model: instead of modeling P(sentence) from scratch (like a normal language model starting from all-zeros), the decoder is conditioned on the encoder's representation of the input sentence - i.e., it models P(English sentence | French sentence).
Key distinction from language generation: you don't want to randomly sample from the output distribution (that gives variable, sometimes poor-quality translations). You want the single most likely output sentence given the input.
This is a search problem: find the sequence of words y that maximizes P(y | x) - not decodable greedily word-by-word, because the best next word at each step doesn't guarantee the best overall sentence.
Motivates why greedy decoding is insufficient and sets up beam search as the algorithm to approximate the maximization.

##3. C5W3L03 - Beam Search
Beam search is the workhorse decoding algorithm for seq2seq models (used in translation, captioning, etc.).
Unlike greedy search (keeps only the single best next word), beam search keeps track of B candidates at each step, where B is the beam width (e.g., B = 3).
Process: at each decoding step, expand all current candidate partial sentences by every possible next word, compute probabilities, then keep only the top B most likely partial sentences so far.
If B = 1, beam search reduces to greedy search.
Larger B > better results but more computation/memory; smaller B > faster but more approximate (a classic quality/speed trade-off).

##4. C5W3L04 - Refinements to Beam Search
Addresses two practical problems with naive beam search:
Numerical underflow: multiplying many probabilities (all < 1) makes the joint probability vanishingly small. Fix: work with sum of log-probabilities instead of the product of probabilities.
Length bias: raw probability (or log-probability) naturally penalizes longer sentences more (more terms multiplied/summed, all negative in log space), so the algorithm favors unnaturally short outputs.
Fix: length normalization - divide the summed log-probability by a normalization term, often T_y^(alpha) (length of output raised to a power alpha, commonly alpha = 0.7), giving a heuristic that "softens" the length penalty.
Practical guidance on choosing beam width B: larger B explores more possibilities and gives better results with diminishing returns, at higher computational cost. Production systems often use B in the 10-100 range; research systems for maximum performance may use much larger B (1000+).

##5. C5W3L05 - Error Analysis in Beam Search
Since a full seq2seq translation system has two main components - the RNN model (encoder/decoder) and the beam search algorithm - error analysis helps decide which one to blame/improve when a translation is bad.
Method: compare the probability the model assigns to the human (reference) translation, P(y*|x), versus the probability it assigns to the model's own output, P(y^|x).
If P(y*|x) > P(y^|x): beam search chose a lower-probability sentence than the reference sentence existed - meaning beam search failed to find the best sequence > beam search is at fault (fix: increase beam width).
If P(y*|x) <=> P(y^|x): the model actually gave the human translation a lower (or equal) score than its own (worse) output - meaning the RNN model itself is at fault (fix: more training data, regularization, different architecture, etc.).
Running this analysis over many mistranslated examples lets you tally what fraction of errors are attributable to the RNN vs. to beam search, guiding where to invest further engineering effort.

##6. C5W3L06 - Bleu Score (Optional)
Addresses the evaluation problem: for tasks like translation, there can be multiple equally good reference translations, so you can't just measure "exact match accuracy."
Introduces BLEU score (Bilingual Evaluation Understudy) as an automatic metric that correlates reasonably well with human judgment.
Core idea: measure precision of n-grams - how many n-grams (unigrams, bigrams, etc.) in the machine-generated translation also appear in one or more of the human reference translations.
Uses modified/clipped precision: caps the count of each n-gram's credit at the maximum number of times it appears in any single reference (prevents rewarding degenerate outputs that just repeat a good word).
Combines precision scores across multiple n-gram lengths (typically 1- to 4-grams) and applies a brevity penalty to discourage overly short outputs from gaming the precision metric.
BLEU gave the MT (and broader text-generation) research community a single-number, automatic way to benchmark systems without needing constant human evaluation - a major accelerant for progress in the field, though it's an imperfect proxy for true translation quality.