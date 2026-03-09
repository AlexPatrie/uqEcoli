# **7\. The "Parking Lot" (Out of Scope)** {#7.-the-"parking-lot"-(out-of-scope)}


## RFC006:

In the longer term, we see opportunities for UQ to contribute in other thematic areas relevant to SMS. Whereas for the **MS-08.4.2**\-satisfying work (above) is intended as a concrete work plan for March and April 2026, in this RFC our discussion with respect to these other topics is intended to be more speculative and serve as a record of thinking to inform future plans. That is, ideas from these sections are NOT being proposed as near term execution priorities.  

* Experimental data integration.  
  * [RFC-005: ParCa Refactor](https://docs.google.com/document/u/0/d/1skMt0BuJ_H2KqJjkBOy_nYl1SQOi1r8DaJ8olxZEcr0/edit)  
  * **08.4.1 – Milestone 8 (extensibility) Develop a data integration layer for multi-omics and phenotypic data**,  
* Strain design  
  * **09.2.1 – Milestone 9 (violacein) Simulate the effects of single perturbations on production for a set of 30+ perturbations**  
  * **09.2.2 – Milestone 9 (violacein**) **Use the model to predict the effects of perturbation combinations**  
  * **10.2.1 – Milestone 10 (extensibility) Enable gene knockout, up/down regulation, and pathway modification simulations**,  
* Development of ML surrogate models  
  * **09.4.1 – Milestone 9 (extensibility) Demonstrate the integration of an ML surrogate model to enhance simulation accuracy**.

Future Theme A: Opportunities for contributions of UQ to experimental data ingestion and reconciliation

NOT PROPOSED AS A NEAR TERM EXECUTION PRIORITY, BUT A RECORD OF IDEAS FOR FUTURE CONSIDERATION

* It is likely that the model will be often non-performant when arbitrary new experimental data are ingested and used for parameter estimation  
  * The ParCa refactor ([RFC-005: ParCa Refactor](https://docs.google.com/document/u/0/d/1skMt0BuJ_H2KqJjkBOy_nYl1SQOi1r8DaJ8olxZEcr0/edit)) for intelligibility will enable better debugging  
  * But, there will still be a need for robust tools and practices for identifying and handling problematic data points or parameters. Tools for parameter sensitivity analysis will be critical for this purpose.  
* An initial step could systematically perturb input data and capture the  variability in the ParCa outputs, i.e., perform a sensitivity analysis of the deterministic function `(experimental_data -> sim_data)`. This could help identify robustness problems within the sequential parameter optimisation procedures. A candidate numerical method would be the Morris method.   
* A second step could then further propagate the resulting distribution of `sim_data` parameters through the simulator, capturing effects on the model’s viability. This would be a direct application of the tooling developed under Theme A, simply using a different input distribution.  
* More generally, parameter estimation based on data can be improved such that reference parameters are used as Bayesian priors subject to update based on new experimental data. This type of framework could avoid over-weighting novel and potentially noisy experimental data and calculate parameters that appropriately balance the model’s structural sensitivities, reference (curated, trusted) values and the experimental inputs. 


### Future Theme B: Opportunities for contributions of UQ to strain design

NOT PROPOSED AS A NEAR TERM EXECUTION PRIORITY, BUT A RECORD OF IDEAS FOR FUTURE CONSIDERATION

* For strain design in bioproduction, we will seek to identify combinations of genetic perturbations (including gene knockouts, knockdowns, and overexpressions) that result in increases in a target flux (in SMS, violacein exchange), while maintaining cell viability and growth.  
* We may wish to calculate flux control coefficients (FCC’s), i.e. the partial derivative of violacein flux to enzyme expression, for many genes in the model. This would involve implementing gene expression perturbations (additive or multiplicative) up and down and capturing the resulting violacein and growth fluxes.   
  * This effort would benefit from robust UQ to calculate first-order sensitivity indices, which are very similar to the classical FCC’s, across the ensemble of cell replicates and cell cycle stages.  
  * We may wish to attempt this using different model parameters, e.g. the objective function weighting in the metabolism module

### 

### Future Theme C: Opportunities for contributions of UQ to ML surrogacy

NOT PROPOSED AS A NEAR TERM EXECUTION PRIORITY, BUT A RECORD OF IDEAS FOR FUTURE CONSIDERATION

* UQ can help identify potentially beneficial usage sites for surrogate models:  
  * Highly uncertain parameters or process components, which need further investigation. In such cases, surrogates may help in testing baseline hypotheses, or may be useful directly as phenomenological models until more reliable mechanistic models become available.  
  * Processes whose outputs display very low relative variance. These could be temporarily replaced by cheaper surrogates while executing large workflows that serve other purposes. However, care needs to be taken to ensure that such a replacement doesn’t significantly distort overall results, and that it actually pays off in terms of overall compute cost of the coupled system.  
* On the other hand, the numerical methods underlying many UQ approaches internally learn surrogate functions from simulation samples, usually based on some variation of regularised regression, which are then used to estimate sensitivity indices. However, using such surrogate functions for anything other than estimating sensitivity indices may lead to misleading outcomes.

