[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/?usp=sharing)

# Analysis code from "Quantifying agreement between humans and machines (or any model) using performance-concordance to overcome internal noise in humans"

In order to make a fair comparison between humans and machines (or any model), we must take into account the fact that humans are noisy decision makers - given exactly the same stimulus and task, they don't always make the same decisions. Consequently, trial-by-trial analysis of human decisions will yield low error-consistency scores even when comparing a person to themselves! Thus, to the extent that internal-noise is independent of the stimulus, trial-by-trial analyses will underestimate the extent to which decision makers are prone to making errors on the same stimlui, which we call stimulus-level agreement. 

## What is performance concordance?
Performance-concordance is an analysis method for measuring agreement between two decision making systems (e.g., humans vs. deep neural network models), assessing whether those systems have a tendency to make errors on the same inputs, while aggregating across stimulus repetitions or observers to cancel out internal noise and isolate stimulus-level agreement. This method enables a fine-grained analysis, assessing agreement at the level of individual sitmuli, while overcoming stimulus-independent noise that can dominate human responses and mask high-levels of agreement. Re-analysis of existing datasets show that human-human agreement is greater than previously estimated, widening the gap between deep neural network models and human observers. The paper is available on [arXiv]().


