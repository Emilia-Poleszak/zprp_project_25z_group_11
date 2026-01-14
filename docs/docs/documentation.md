## Weekly work progress report:
- **Week 1:**
  - Completed initial project setup and environment configuration.
  - Researched and gathered requirements for the project.
  - Assigned tasks.
  - Researched Linear Recurrent Unit (LRU) architecture and its implementation.
  - Started working on the literature review and analysis of scientific papers related to RNNs and LRU.
  - Started working on data generators
- **Week 2:**
  - Continued literature review and analysis of scientific papers.
  - Finalized data generators for training and testing.
- **Week 3:**
  - Implementing Adding experiment
  - First results of Adding experiment
  - Implementing saving generated data to .txt file
- **Week 4:**
  - Implementing Reber experiment
  - First results of Reber experiment
  - Optimizing Reber experiment implementation
  - 
- **Week 5:**
  - First stand-up. 
  - Break from working on the project in order to prepare for a test.
- **Week 6:**
  - Implementing tasks from first stand-up
- **Week 7:**
  - Working on implementing LRU model in all experiment
- **Week 8:**
  - Working on implementing LRU model in all experiment
- **Week 9:**
  - Implementing Multiplication experiment
  - First results from Multiplication experiment
- **Week 10:**
  - Christmas break.
- **Week 11:**
  - Add code documentation in Multiplication experiment
- **Week 12:**
  - Second stand-up.
  - Implementing tasks from second stand-up.
  - Add project documentation
  - Optimizing Adding experiment implementation
  - Optimizing Multiplication experiment implementation
- **Week 13:**
  - Write project report
  - Add tests
  - Merge all project
  - Update documentation and README

Comments:
- Implementing experiments got postponed by one week 
- 

## Scientific papers and articles analysis
1. https://www.researchgate.net/publication/13853244_Long_Short-Term_Memory

**Comment:** 

Section 1: Introduction of LSTM
Section 2: Review of previous works on recurrent nets with time-varying inputs
Section 3: Explains constant error backdrop – problem that LSTM is supposed to solve
Section 4: Description of LSTM architecture (the project will use newer implementation (proposed by PyTorch))
Section 5 describes experiments demonstrating the quality of a novel long time lag algorithm. The task, architecture, training and results are described.
Section 6 includes discussion on limitations of LSTM. The project will include comparison of how LRU performs against those limitations.

**Details:**

No code and pre-trained models are available.
Metrics included in evaluations of experiments 1, 4, 5 are: 
percentage of success trials, number of sequance presentations until success, MSE, “# wrong predictions” (numer of incorrectly precessed sequences”

Data used: 
-	experiment 1: 256 training and test strings, generated randomly,
-	experiment 4: pair of components (a real value randomly chosen from [-1; 1] and one of values {-1, 0, 1},
-	experiment 5: same as 4, but first component is from interval [0; 1].

Found by: Agnieszka Jegier

2. https://www.researchgate.net/publication/284476210_Unitary_Evolution_Recurrent_Neural_Networks#pf9

**Comment:**

Paper explores another solution to the vanishing gradient problem, when 
trying to learn long-term dependencies. 
Performs experiments to test unitary recurrent matrices and compares to LSTM.
Proposes algorithm for such problems, including adding problem that is analysed in the project.

**Details:**

Code: https://github.com/stwisdom/urnn 
Data used: synthetic + MNIST
Metrics: gradient norms, hidden state norms, distance to last state, 
accuracy (for classification), MSE, cross entropy.

Found by: Agnieszka Jegier

3. https://arxiv.org/pdf/2310.02367

**Comment:**

Paper explains LRURec - Linear Recurrent Units for Sequential Recommendation. 
It uses linear recursion, which makes this model more efficient. 

**Details:**

Code: https://github.com/yueqirex/LRURec

Metrics:
- Recall@k
- NDCG@k

Found by: Emilia Poleszak

4. https://arxiv.org/pdf/2303.06349

**Comment:**

Paper explains LRU architecture and shows how it solves RRN’s problems
with vanishing and exploding gradient.	

**Details:**

Code (unofficial implementations): 
https://github.com/esraaelelimy/LRU?tab=readme-ov-file

https://github.com/NicolasZucchet/minimal-LRU?tab=readme-ov-file

Found by: Emilia Poleszak

5. https://www.researchgate.net/profile/Y-Bengio/publication/2839938_Gradient_Flow_in_Recurrent_Nets_the_Difficulty_of_Learning_Long-Term_Dependencies/links/546cd26e0cf2193b94c577c2/Gradient-Flow-in-Recurrent-Nets-the-Difficulty-of-Learning-Long-Term-Dependencies.pdf

**Comment:**
	
This paper describes LSTM disadvantages in learning long-term dependencies. 
Main issues are caused by vanishing gradients and short-term gradient value domination. 
This is a good groundwork for network architecture development.	

- No official models/implementations found, 
however there is a project that tackles the same problem: 
https://github.com/sinha96/LSTM?utm
- Theoretical analysis

Found by: Anna Czarkowska

6. https://arxiv.org/abs/1504.00941

**Comment:**	

This article includes a IRNN, RNN with Rectified Linear Units and LSTM
comparison in different ML approaches (including Adding Problem).	

**Details:**

- No implementation found
- Metrics: Mean Squared Error (MSE), base model comparison
- DistBelief infrastructure

Found by: Anna Czarkowska

7. https://www.sciencedirect.com/science/article/pii/S0950705121009199?casa_token=ZEqVHR6ThqwAAAAA:H21pEWBwPvP68cs6Ll6gsm8ilaOIRl_v0UrQYKg1Y_IVYwN90ARSYmuZ8NCUZu5s9XJpNaFG
	
**Comment:**

This article describes Reber grammar and other tests for different RNN types 
with hidden representations. The goal was to provide interprability in 
difficult to understand RNN processing.

**Details:**

-	No implementation found;
-	Metrics: successful sequence prediction percentage, 
effective learning and testing time
-	Computing resources: MacBook Pro with Python scientific stack

Found by: Anna Czarkowska