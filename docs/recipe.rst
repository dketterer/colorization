.. _recipe:

The Training and Debug Recipe
=============================

This is taken from Andrej Karpathy: http://karpathy.github.io/2019/04/25/recipe/


Baseline
--------

1. Fix random seeds (Done)
2. Simplify (Done)
3. Add significant digits to your eval (TODO)
4. Verify loss @ init (TODO)
5. Init well (TODO?)
6. Human baseline (Unnecessary)
7. Input-independent baseline (TODO)
8. Overfit one batch (TODO)
9. Verify decreasing training loss (TODO)
10. Visualize just before the net (Done / TODO)
11. Visualize prediction dynamics (Done / TODO)
12. Use backprop to char dependencies (Maybe)
13. Generalize a speical case (General advice)

Overfit
-------

Don't be a hero!

1. Picking the model (Done)
2. Adam 3e-4 is safe (TODO)
3. Complexify only one at a time (Advice)
4. Do not trust learning rate decay defaults (TODO)


Regularize
----------


1. Get more data (Done)
2. Data augment (Maybe)
3. Creative augmentation (Maybe)
4. Pretrain (TODO)
5. Stick with supervised learning (Advice)
6. Smaller input dimensionality (Advice)
7. Smaller model size (TODO)
8. Decrease the batch size (Done /TODO)
9. Dropout (Maybe)
10. Weight decay (TODO)
11. Early stopping (TODO)
12. Try a larger model (TODO)


Tune
----

1. Random over grid search
2. Hyper-parameter optimization

Squeeze out the juice
---------------------

1. Ensembles
2. Leave it training