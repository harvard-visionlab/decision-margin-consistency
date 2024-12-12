
[Paper](https://openreview.net/forum?id=y2FPllMQVg) | [Bibtex](#bibtex)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iMycNR5rTfEtRheJfd0-HjAz3TKRD8xx?usp=sharing)

[George A. Alvarez](https://visionlab.harvard.edu/george/)\* $^{1,2,4}$, [Talia Konkle](https://konklab.fas.harvard.edu/)\* $^{1,2,3}$
(*equal contribution)

$^1$ [Harvard University](https://www.harvard.edu/), $^2$ [Kempner Institute for the Study of Natural and Artificial Intelligence](https://kempnerinstitute.harvard.edu/), $^3$ [Center For Brain Science](https://cbs.fas.harvard.edu/), $^4$ [Vision-Sciences Laboratory](https://visionlab.harvard.edu/)
<br>

# Decision-margin consistency: a principled metric for human and machine performance alignment

Understanding the alignment between human and machine perceptual decision- making is a fundamental challenge. While most current vision deep neural networks are deterministic and produce consistent outputs for the same input, human percep- tual decisions are notoriously noisy [1]. This noise can originate from perceptual encoding, decision processes, or even attentional fluctuations, leading to different responses for the same stimulus across trials. Thus, any meaningful comparison between human-to-human or human-to-machine decisions must take this internal noise into account to avoid underestimating alignment. In this paper, we introduce the decision-margin consistency metric, which draws on signal detection theory, by incorporating both the variability in decision difficulty across items and the noise in human responses. By focusing on decision-margin distances—-continuous measures of signal strength underlying binary outcomes—-our method can be applied to both model and human systems to capture the nuanced agreement in item-level difficulty. Applying this metric to existing visual categorization datasets reveals a dramatic increase in human-human agreement relative to the standard error consistency metric. Further, human-to-machine agreement showed only a modest increase, highlighting an even larger representational gap between these sys- tems on these challenging perceptual decisions. Broadly, this work underscores the importance of accounting for internal noise when comparing human and machine error patterns, and offers a new principled metric for measuring representational alignment for biological and artificial systems.

## Contents

This repository contains the data + anlayses for our [decision-margin-consistency paper](https://openreview.net/pdf?id=y2FPllMQVg) paper.

## Usage

The decision margin is a measure of how far a representation is from the decision boundary - easier decisions will be further from the decision boundary. Decision-margin consistency is the pearson correlation between the decision-margins for two different decision makers across the same set of items (see [paper](https://openreview.net/forum?id=y2FPllMQVg) for details). 

For model vs. model comparisons using decision-margin consistency, see the [collab notebook](https://colab.research.google.com/drive/1iMycNR5rTfEtRheJfd0-HjAz3TKRD8xx?usp=sharing) or the [demo notebook](https://github.com/harvard-visionlab/decision-margin-consistency/blob/main/notebooks/demo_model_vs_model.ipynb) in this repo. 

For analyses involving humans, percent correct for an item (averaged across trial repetitions and/or subjects), can be taken as an index of the decision-margin distance (see [paper](https://openreview.net/forum?id=y2FPllMQVg) for details). See the [edge image analysis](https://github.com/harvard-visionlab/decision-margin-consistency/blob/main/notebooks/edge_data_analysis.ipynb) notebook for analyses reported in the paper.

To run these notebooks locally, you can clone this repo and install the project (dependencies are not automatically installed, so you'll need to install torch, torchvision, scipy, numpy, pandas, and natsort).
```
git clone https://github.com/harvard-visionlab/decision-margin-consistency.git
cd decision-margin-consistency
python -m pip install .
```

Or you install from github:
```
pip install git+https://github.com/harvard-visionlab/decision-margin-consistency.git
```

Then you can import/use the decision-margin-consistency validation and consistency functions.
```
from decision_margin_consistency import validation, compute_consistency

df1 = validation(model1, dataset, meta=dict(subj='model1', condition='testing'))
df2 = validation(model2, dataset, meta=dict(subj='model2', condition='testing'))
con = compute_consistency(df1, df2)
```


<a name="bibtex"></a>
## Citation

If you find our work or any of our materials useful, please cite our paper:
```
@inproceedings{alvarez2024decision,
  title={Decision-margin consistency: a principled metric for human and machine performance alignment},
  author={Alvarez, George A and Konkle, Talia},
  booktitle={UniReps: 2nd Edition of the Workshop on Unifying Representations in Neural Models},
  year={2024},
  url={https://openreview.net/forum?id=y2FPllMQVg}
}

```
