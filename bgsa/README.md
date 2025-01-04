Testing uncertainty quantification of sensitivity indices and / or Shapley values (basically attribution methods) through adaptive training of BNNs.

First few experiments will focus on generating scores for a deterministic model, of a Borehole function. Ideas to explore (probably in order):

- First, how many training samples do we need to have a good approximation to BHF(and / or match the sensitivity values?) ? Probably a lot.

- Verify that first and total order indices bound the SHAP values correctly.

- What happens to the rankings when the function is misspecified? Can we construct a worst case example?

- Explore BNN, check uncertainty in Sobol index and / or SHAP values.
