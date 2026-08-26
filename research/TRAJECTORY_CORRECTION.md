# Trajectory-corrected objective

## Intervention

The clean random-window experiment sampled one frontier state uniformly from
each target trajectory but optimized its unweighted mean local loss. This
reweights a sequential length-`L` example by `1/(L+1)` and biases the learned
STOP hazard toward short spans.

The corrected run keeps the corpus, dynamic corruption stream, 24--96-token
windows, 10.06M-parameter backbones, 30 epochs, batch size 32, learning rate,
initialization, and update count fixed. At a sampled frontier it:

1. sums loss over all valid local actions rather than averaging them;
2. multiplies by the inverse frontier-sampling probability;
3. averages the resulting full-trajectory estimate over source examples.

For sequential filling this is an unbiased estimator of exact trajectory NLL.
For the tree it estimates the likelihood of the fixed midpoint derivation.

## Stochastic calibration

All numbers use temperature-1 sampling on the same 128 IID prompts with 32
samples per prompt. TV is total-variation distance from the corruption prior.

| Model | Objective | TV | P(empty) | P(overflow) | Capped mean | Entropy |
|---|---|---:|---:|---:|---:|---:|
| Sequential | unweighted frontier | 0.388 | 0.537 | 0.008 | 1.45 | 1.578 |
| Sequential | trajectory-corrected | **0.066** | **0.188** | 0.005 | **3.66** | **2.179** |
| Tree | unweighted frontier | 0.260 | 0.384 | 0.002 | 2.22 | 1.817 |
| Tree | trajectory-corrected | 0.244 | **0.219** | 0.005 | **3.58** | 1.995 |
| Learned global length | categorical NLL | **0.038** | 0.214 | 0.000 | 3.55 | 2.151 |

The sequential result is the causal control: changing only the objective removes
the large short-span bias and approaches the categorical length head. Its final
sampled-state estimate of full STOP NLL is 2.210 nats, close to the 2.164-nat
entropy of the target length prior. Greedy decoding still chooses the zero mode,
as it should; distributional calibration, rather than greedy exact recovery, is
the relevant success criterion.

## Why the tree retains shape error

The corrected tree fixes root stopping and mean length, but its sampled length
histogram remains distorted:

| Length | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | overflow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Target prior | .200 | .100 | .100 | .100 | .100 | .100 | .100 | .100 | .100 | .000 |
| Corrected tree | .219 | .020 | .057 | .133 | .152 | .184 | .151 | .061 | .018 | .005 |

The current model predicts left- and right-child existence with two independent
Bernoulli losses. Under the midpoint derivation and conditional on a non-empty
root, the true topology distribution is:

- no children `(0,0)`: 0.125 (length 1);
- left only `(1,0)`: 0.125 (length 2);
- right only `(0,1)`: 0;
- both `(1,1)`: 0.750 (lengths 3--8).

Independent marginals instead imply `(0,0)=0.031`, `(1,0)=0.219`,
`(0,1)=0.094`, and `(1,1)=0.656`. Thus the head cannot simultaneously match
the canonical topology probabilities even at its population optimum. The very
low sampled probability of length 1 is the predicted signature of this
factorization error.

## Decision

The experiment validates local STOP generation when trained with a proper
trajectory objective: the sequential process now models unknown length without
a global length classifier. It also isolates the remaining parallel-tree issue
to child-topology parameterization rather than stopping or data exposure.

The next minimal intervention is a joint four-class child head for
`none/left/right/both`, trained with categorical NLL under the same corrected
trajectory objective. It should be compared against the independent-child
checkpoint without changing the backbone or sampling protocol. Scale-up remains
paused until that ablation is complete.

This ablation is now complete: TV improves from 0.244 to 0.165 and length-1
probability from 0.020 to 0.099. The next bottleneck is correlation across
separate gaps on the same frontier; see `research/JOINT_TOPOLOGY.md`.
