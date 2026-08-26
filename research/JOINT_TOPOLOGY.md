# Joint child-topology ablation

## Question

After correcting trajectory weighting, the tree still underproduced length 1.
The independent child head cannot represent the midpoint root distribution:
`none=.125`, `left=.125`, `right=0`, `both=.750` conditional on emission.
This ablation replaces its two Bernoulli outputs with one categorical
`none/left/right/both` head.

The corruption stream, random windows, midpoint derivations, 10.06M backbone,
initialization seed, optimizer, 30 epochs, and update budget are unchanged. The
joint head adds only 1,282 parameters, about 0.013% of the model.

## Result

Temperature-1 results use the same 128 IID prompts and 32 samples per prompt.

| Tree variant | TV to prior | JS (nats) | Brier | P(empty) | P(overflow) | Mean | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Independent children, unweighted objective | 0.260 | 0.065 | 0.916 | 0.384 | 0.002 | 2.22 | 1.817 |
| Independent children, corrected objective | 0.244 | 0.051 | **0.903** | 0.219 | **0.005** | 3.58 | 1.995 |
| **Joint topology, corrected objective** | **0.165** | **0.028** | 0.925 | 0.164 | 0.014 | 3.83 | **2.139** |
| Corrected sequential filler | 0.066 | 0.005 | 0.909 | 0.188 | 0.005 | 3.66 | 2.179 |
| Learned categorical length | 0.038 | 0.001 | 0.867 | 0.214 | 0.000 | 3.55 | 2.151 |

The marginal length histogram provides the cleanest mechanism check:

| Length | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | overflow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Target | .200 | .100 | .100 | .100 | .100 | .100 | .100 | .100 | .100 | .000 |
| Independent corrected | .219 | .020 | .057 | .133 | .152 | .184 | .151 | .061 | .018 | .005 |
| Joint corrected | .164 | **.099** | .047 | .091 | .123 | .201 | .127 | .097 | .037 | .014 |

The joint head restores length-1 probability from 0.020 to 0.099, almost exactly
the 0.100 target, and reduces overall TV by 32%. This confirms the predicted
within-node correlation failure. It is a real improvement, but the tree remains
less calibrated than sequential filling.

## Remaining limitation

At depths greater than zero, a frontier can contain multiple gaps. Their target
topologies are coupled by the same canonical tree and total span, but the model
samples one categorical decision independently at every gap position. A joint
four-way head captures correlation between the two children of one pivot; it
does not capture correlation across separate frontier gaps. Teacher forcing also
trains only canonical frontiers, while stochastic inference can enter
non-canonical states.

This explains the remaining redistribution toward length 5 and the increased
overflow rate. The next research control should distinguish the two mechanisms:

1. measure topology calibration on teacher-forced canonical frontiers versus
   free-running frontiers to quantify exposure error;
2. add a shared per-original-gap latent or a small iterative topology-denoising
   step so simultaneous frontier decisions are correlated;
3. retain sequential filling as the exact-likelihood reference and report the
   extra NFE of any coupling mechanism.

The joint head should replace independent child bits in subsequent tree
experiments, but the 50--100M scale-up remains premature.

## Follow-up audit

Canonical versus free-running measurement narrows the residual further. The
teacher/free predictive topology shift is only 0.005--0.023 TV across depths,
and no forbidden right-only event occurs in 7,905 sampled emit decisions.
Canonical depth-1 topology remains locally imperfect (TV 0.095), but its two
simultaneous gaps also carry 0.549 nats of total correlation that independent
per-gap sampling cannot express. Exposure error is therefore not the dominant
mechanism in this run. See `research/FRONTIER_DEPENDENCE.md` and
`artifacts/text_joint_topology/TOPOLOGY_EXPOSURE.md`.
