# NFL DraftKings Example Code

This code is a complete and streamlined worfklow for NFL DraftKings. Fun project and no longer working on it, so sharing for anyone interested.

It includes a points prediction model with slightly better accuracy (R^2) than any publicly available points predictions (either free or commercial). It also includes standalone lineup optimization that enables more flexibility than any paid tool. 
The lineup optimization is based off of a nice paper by Hunter, Vielma, Zaman (https://arxiv.org/abs/1604.01455). I use Gurobi (commercial, free license available to students), but there are free solvers compatible with the code.

The only data files the example code relies on which are outside the data scraping pieces are the DraftKings lineup input files and contest results/prizes files that are downloadable from the DraftKings website. Some other players use scripting here to streamline further,
but I haven't included scripts as it violates the TOS.

I've left out a few key pieces/innovations in case I want to ever revisit, but what I shared is streamlined and already breakeven. With some effort you can improve the points prediction model significantly with more historical data (e.g., https://github.com/maksimhorowitz/nflscrapR) and more sophisticated machine learning models (e.g., DNN). Also room to improve the mathematical optimization formulation, both w.r.t. constraints and nonlinear optimization formulations.
