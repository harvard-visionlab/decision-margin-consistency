
[Paper](https://openreview.net/forum?id=y2FPllMQVg) | [Bibtex](#bibtex)

[George A. Alvarez](https://visionlab.harvard.edu/george/)\* $^{1,2,4}$, [Talia Konkle](https://konklab.fas.harvard.edu/)\* $^{1,2,3}$
(*equal contribution)

$^1$ Harvard University, $^2$ Kempner Institute for the Study of Natural and Artificial Intelligience, $^3$ Center For Brain Science, $^4$ Vision-Sciences Laboratory
<br>

# Decision-margin consistency: a principled metric for human and machine performance alignment

Understanding the alignment between human and machine perceptual decision- making is a fundamental challenge. While most current vision deep neural networks are deterministic and produce consistent outputs for the same input, human percep- tual decisions are notoriously noisy [1]. This noise can originate from perceptual encoding, decision processes, or even attentional fluctuations, leading to different responses for the same stimulus across trials. Thus, any meaningful comparison between human-to-human or human-to-machine decisions must take this internal noise into account to avoid underestimating alignment. In this paper, we introduce the decision-margin consistency metric, which draws on signal detection theory, by incorporating both the variability in decision difficulty across items and the noise in human responses. By focusing on decision-margin distances—-continuous measures of signal strength underlying binary outcomes—-our method can be applied to both model and human systems to capture the nuanced agreement in item-level difficulty. Applying this metric to existing visual categorization datasets reveals a dramatic increase in human-human agreement relative to the standard error consistency metric. Further, human-to-machine agreement showed only a modest increase, highlighting an even larger representational gap between these sys- tems on these challenging perceptual decisions. Broadly, this work underscores the importance of accounting for internal noise when comparing human and machine error patterns, and offers a new principled metric for measuring representational alignment for biological and artificial systems.

## Contents

This repository contains the data + anlayses for our [decision-margin-consistency paper](https://openreview.net/pdf?id=y2FPllMQVg). 

<a name="bibtex"></a>
## Citation

If you find our work or any of our materials useful, please cite our paper:
```
@inproceedings{konkle2023cognitive,
  title={Cognitive Steering in Deep Neural Networks via Long-Range Modulatory Feedback Connections},
  author={Konkle, Talia and Alvarez, George A},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
  url={https://openreview.net/forum?id=FCIj5KMn2m}
}

@inproceedings{alvarez2024decision,
  title={Decision-margin consistency: a principled metric for human and machine performance alignment},
  author={Alvarez, George A and Konkle, Talia},
  booktitle={UniReps: 2nd Edition of the Workshop on Unifying Representations in Neural Models},
  year={2024},
  url={https://openreview.net/forum?id=y2FPllMQVg}
}

```
