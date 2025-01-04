Testing uncertainty quantification of sensitivity indices and / or Shapley values (basically attribution methods) through adaptive training of BNNs.

First few experiments will focus on generating scores for a deterministic model,for example, a  Borehole function or the G-function. Ideas to explore (probably in order):

- verify we can scale inputs to unit hypercube or not. Does it matter that NN will need scaled inputs? (should matter when imposing and sampling from prior)

- First, how many training samples do we need to have a good approximation to BHF(and / or match the sensitivity values?) ? Probably a lot.

- Verify that first and total order indices bound the SHAP values correctly.

- What happens to the rankings when the function is misspecified? Can we construct a worst case example?

- Explore BNN, check uncertainty in Sobol index and / or SHAP values.
