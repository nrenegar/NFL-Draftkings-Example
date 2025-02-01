# NFL DraftKings Example Code

This code is a complete and streamlined worfklow for NFL DraftKings. I'm more involved in financial markets now, so making this available for people to have fun with.

It includes a points prediction model with slightly better accuracy (R^2) than any publicly available points predictions (either free or commercial). It also includes standalone lineup optimization that enables more flexibility than any paid tool. 
The lineup optimization is based off of a nice paper by Hunter, Vielma, Zaman (https://arxiv.org/abs/1604.01455). I use Gurobi (commercial, free license available to students), but there are free solvers compatible with the code.

The only data files the example code relies on which are outside the data scraping pieces are the DraftKings lineup input files and contest results/prizes files that are downloadable from the DraftKings website. Some other players use scripting here to streamline further,
but I haven't included scripts as it violates the TOS.

I've left out a few key pieces/innovations in case I want to ever revisit, but this is borderline profitable already. Think about how to improve the points prediction model with more historical data (e.g., https://github.com/maksimhorowitz/nflscrapR) and more sophisticated machine learning (e.g., DNN) models. It's not impossible to get much, much better than public points projection models.
Also think about how to improve the mathematical optimization formulation, both w.r.t. constraints and nonlinear optimization formulations. Do both and you will make money.

Good luck!
