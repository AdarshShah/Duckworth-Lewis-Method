# Duckworth-Lewis-Method
[Data Analytics] Predicting the cricket scores using DLS

The data in the spreadsheet contains data on ODI matches from 1999 to 2011.

1. Using the first innings data alone in the above spreadsheet, find the best fit 'run production functions' in terms of wickets-in-hand w and overs-to-go u. Assume the model Z(u,w) = Z0(w)[1 - exp{-b(w)u}]. Use the sum of squared errors loss function, summed across overs and wickets. You will find one Z0(w) and b(w) for each w. In your report, you should provide a plot of the ten functions, report the (20) parameters associated with the (10) production functions, and the error per point. Your function should be named:

Z0, b = DuckworthLewis20Params()

2. Now assume the model Z(u,w) = Z0(w)[1 - exp{-Lu/Z0(w)}] and use the sum of squared errors loss function, summed over overs and wickets. Note that this new regression forces all slopes to be equal at u = 0.  In your report, you should provide a plot of the ten functions, report the (11) parameters associated with the (10) production functions, and the error per point. This function should be named:

Z0, u = DuckworthLewis11Params()
