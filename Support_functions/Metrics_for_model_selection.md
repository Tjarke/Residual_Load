What do we want to achive?
a model that is capable of accurately predicting the residual load

what are good metrics for time series analysis?

we perform the mean absolute error for each day and then we use a mean double squared for all the days we are testing - in order to penalize very hard any deviations
 - this means we need to make sure we get a mae per day so we need to compute it per calender day.

 - The absolute sum of the error! this shows how much was the real dismatch in the prediction and we can calculate the price
