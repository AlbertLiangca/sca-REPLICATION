Sparse Component Analysis is a method for identifying latent factors in population neural activity without prior knowledge of the number of factors, and allowing these factors to overlap in time. It was originally outlined and demonstrated by Joshua I. Glaser et al. in "Identifying Interpretable Latent Factors with Sparse Component Analysis", and this is my attempt to replicate the function and its results.

## Post-Mortem:

1) Model Structure: create a model object instead of a function; all of the component properties of the model can be called by "self.[property]", instead of having to be stored as a dict output of the function. Additionally, stuff like fitting the model to new data is easier to do. A function call signature also makes it unclear what the model is doing; is calling SCA([data]) returning the fitted data, the matrices that fit the data, the losses, or all three??
2) Similar things should have similar call signatures: in my code, when I fit SCA, I use one call signature (SCA([data])), and then when I want to get the latents. By contrast, when I fit wPCA, I write out the fitting using either torch.linalg.svd or sklearn.decomposition.TruncatedSVD:

```
svd = TruncatedSVD()
svd.fit([data])
U = svd.components_.T
V = svd.components_
```

In the original code, the author wrapped this function in an object, so that interfacing with both models is identical.

3) Use NumPy: When I initially began the project, I thought that since the autoencoder uses PyTorch, it would be smart to just code the entire thing using Torch tensors. NumPy, however, is easier to do array splicing and manipulation with and interfaces with other libraries better, so in the end it is smarter to relegate torch tensors only to the realm of being passed into a torch model.

4) No magic numbers: every parameter or hyperparameter that could be tweaked or adjusted should be adjustable by interfacing with the model. Stuff like the lambda sparsity ratio shouldn't be hard-coded.
