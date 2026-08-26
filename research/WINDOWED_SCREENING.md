# Clean random-window screening

## Setup

Training and evaluation both use variable 24--96 token windows. Every document
receives a new window, gap, and frontier state on each epoch. The three matched
10.05M-parameter models train for 30 epochs, about 2,490 optimizer updates per
model and approximately twice the previous dynamic budget. There is no fixed
128-token canvas cue.

| Slice | Model | Joint length | Edit | Length MAE | NFE | Processed tokens |
|---|---|---:|---:|---:|---:|---:|
| IID one gap | Tree | 0.214 | 0.206 | 3.18 | 1.21 | 66 |
| IID one gap | Sequential | 0.214 | 0.211 | 3.47 | 1.02 | 58 |
| IID one gap | Learned length + masks | 0.210 | 0.211 | 3.46 | 1.03 | 58 |
| IID one gap | Oracle length + masks | 1.000 | 0.246 | 0.00 | 2.08 | 125 |
| Two-gap composition | Tree | 0.045 | 0.219 | 3.03 | 1.34 | 69 |
| Two-gap composition | Sequential | 0.052 | 0.243 | 3.23 | 1.15 | 60 |
| Two-gap composition | Learned length + masks | 0.052 | 0.242 | 3.21 | 1.10 | 59 |
| Length 9--16 | Tree | 0.000 | 0.006 | 11.35 | 1.42 | 61 |
| Length 9--16 | Sequential | 0.000 | 0.001 | 12.16 | 1.32 | 55 |
| Length 9--16 | Learned length + masks | 0.000 | 0.004 | 11.75 | 1.18 | 53 |

All non-oracle models converge to nearly immediate termination. On the IID test,
the true empty-gap rate is 21.0%, while greedy predicted empty rates are 89.3%
for tree, 99.7% for sequential, and 98.7% for learned length. Their roughly 21%
length accuracy is therefore mode selection, not successful reconstruction.

This behavior is statistically expected. Corruption draws zero length with
probability 0.2 and each non-empty length 1--8 with probability 0.1, independently
of text context. The entropy of that length distribution is approximately 2.164
nats. The masked model's final training length NLL is 2.177, essentially the
Bayes limit. More optimization cannot recover a random label absent from the
prompt.

The earlier capped-document experiments contained accidental information about
the missing length through absolute canvas size. Removing it converts the task
into an intentionally multimodal conditional generation problem. Greedy exact
recovery of the one sampled original is not a valid primary metric for that
problem. It rewards choosing the mode and penalizes legitimate alternative
lengths and text.

## Consequence for the research question

The clean experiment does not show that gap-tree generation fails as a
generative model. It shows that the current evaluation asks an unidentifiable
question. The next evaluation must measure:

- teacher-forced conditional likelihood or a valid bound under each topology;
- calibration of sampled length distributions rather than greedy point accuracy;
- diversity and semantic acceptability of sampled completions;
- constraint following on tasks where a requested length or structural unit is
  actually present in the prompt.

Length 9--16 cannot be a fair exact-label extrapolation test when the model is
never told which random length was selected. A meaningful long-form test should
condition on an explicit length bucket, remove a natural structural unit such as
a whole sentence, or score the generated distribution without assuming a unique
target length.

## Stochastic calibration result

Temperature-1 sampling on 128 IID prompts with 32 Monte Carlo samples per prompt
changes the conclusion from "all models predict zero" to a sharper distinction:

| Model | TV to prior | JS (nats) | Brier | P(empty) | P(overflow) | Capped mean |
|---|---:|---:|---:|---:|---:|---:|
| Balanced tree GT-DLM | 0.260 | 0.065 | 0.916 | 0.384 | 0.002 | 2.22 |
| Sequential blank filler | 0.388 | 0.098 | 1.011 | 0.537 | 0.008 | 1.45 |
| Learned length + masks | 0.038 | 0.001 | 0.867 | 0.214 | 0.000 | 3.55 |
| Analytic unweighted-frontier optimum | 0.353 | 0.077 | 0.968 | 0.522 | 0.000 | 1.61 |

The global length head has learned the intended multimodal distribution. The
local processes have not: both overproduce short spans even when sampled rather
than thresholded. This exposes a training-objective issue. The sequential data
pipeline samples one of the `L+1` trajectory states uniformly and applies an
unweighted mean loss, so a length-`L` example contributes only one local action
instead of the sum of its `L+1` log-probability terms. The tree pipeline likewise
samples one frontier depth and averages over active gaps. Neither loss is an
unbiased estimator of full derivation likelihood.

This diagnosis has a parameter-free check. Under the current sequential
sampler, a length `L` contributes state `t` with weight `q(L)/(L+1)`. The
resulting optimal hazard analytically implies `P(empty)=0.522`, TV `0.353`, and
mean length `1.61`, close to the learned sequential values `0.537`, `0.388`, and
`1.45`. The short-span bias is therefore predicted by the objective itself.

The next controlled intervention is therefore trajectory-corrected weighting:
sum local action losses over a complete trajectory, or multiply sampled-state
losses by their inverse sampling probabilities while retaining explicit action
counts. Sequential filling provides the exact-likelihood sanity check; the tree
then requires either a fixed canonical derivation or marginalization/ELBO over
latent trees. Full sampled histograms are in
`artifacts/text_windowed/LENGTH_SAMPLING.md` and `length_sampling.json`.

The 50--100M scale-up remains paused until that objective-level control is run.

That control has now been run. It reduces sequential TV from 0.388 to 0.066,
while the tree retains a child-topology factorization error. See
`research/TRAJECTORY_CORRECTION.md` for the matched experiment and next ablation.
