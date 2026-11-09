# Short presentation notes

## Slide 1: Forecasting US inflation

We ask whether machine learning can improve US inflation forecasts and explain what drives them. We compare models using data from 1978 to 2026 and forecast from one month to one year ahead.

## Slide 2: What this presentation covers

Why forecasts matter: interest-rate decisions affect prices with a delay, so central banks need an early view of inflation. What we test: whether machine learning improves on traditional models and the simple assumption that inflation stays unchanged. Our data: US prices, energy costs, interest rates, money, jobs, stocks and GDP from 1978–2026. Our method: train on earlier years and test forecasts on later years, including the pandemic and inflation surge. Model comparison: identify the best forecasts at 1, 3, 6 and 12 months, and check whether they predict rises and falls. Added information: test whether electricity prices, stock returns and GDP improve accuracy. Forecast explanations: show which indicators push predictions up or down, using two historical examples. Live demo and conclusions: change economic inputs, compare forecasts and discuss what the models cannot tell us.

## Slide 3: Why inflation is difficult to forecast

Interest-rate changes take time to influence borrowing, spending and prices. Decisions today therefore depend on future inflation. The pandemic, energy shocks and rapid rate increases changed familiar economic relationships. A model trained on calm years may struggle during these shifts. Our starting comparison is “future inflation equals today’s inflation.” This is the random-walk forecast. Complex models are useful only if they improve on this simple benchmark when tested on new data. Machine learning can capture changing patterns across many indicators. We also examine why a forecast changes, so the result can support an economic discussion rather than just provide a number.

## Slide 4: How the course literature shapes our project

The inflation papers compare traditional economic models with machine learning. Singh and Bhoi discuss the value of these methods for inflation forecasting. Li and colleagues study XGBoost. We build on this work by comparing model families on the same US data and test periods. Electricity prices capture household and business costs that may stay high after oil prices fall. Stock returns provide information about financial conditions. GDP adds a broader measure of economic activity. These ideas connect the four topics in the course reading list. Test whether these extra indicators improve inflation forecasts. Compare the explanations from linear models, tree models and a combined tree–neural model. Assess whether deep learning helps when decades of monthly data still provide relatively few observations.

## Slide 5: Four questions guide the analysis

Compare simple economic models, linear models, tree models and neural networks at 1, 3, 6 and 12 months. Judge accuracy on later data that the model did not use for training. Check predicted rises and falls in inflation, not just whether forecast lines resemble the actual series. Compare performance during the pandemic, the inflation surge and the later decline. Add electricity prices, stock returns and GDP to the core data. Measure whether these additions reduce forecast error, rather than assuming more data always help. Identify the indicators that push a prediction up or down. Compare their importance across forecast horizons and demonstrate the results in an interactive tool.

## Slide 6: How the forecasting methods differ

Future inflation = today’s inflation + predicted change. Each model learns the change from current economic indicators. Predicting zero change gives the random-walk benchmark. Combine indicators using fixed weights. Ridge and Elastic Net limit large weights, helping the model handle overlapping information without fitting every fluctuation in the training data. XGBoost combines many small decision trees. It can respond differently when indicators move into a new range. SHAP breaks its forecast into contributions from individual inputs, measured in inflation percentage points. Sequence models use the previous 12 months to learn how indicators evolve. Our hybrid combines XGBoost signals with a neural network that learns recent trends. We test whether combining them improves accuracy.

## Slide 7: The data and how we test the models

The raw dataset covers 585 months from 1978–2026. Prices and supply costs: consumer prices, producer prices, crude oil and electricity. Financial conditions: interest rates, money supply and stock returns. Economic activity: GDP, industrial production, unemployment and consumer sentiment. Expectations: households’ views of inflation over the next year. Test forecasts on 2019–2026, retraining each year using earlier observations. Exclude training examples whose future inflation outcome would not yet be known at the test date. Delay GDP inputs by three months to allow for publication. Use growth rates and changes to make series more comparable. Important limit: some series contain later revisions, so the exercise does not fully recreate information available in real time.

## Slide 8: Inflation relationships change across periods

Inflation was high around 1980, much calmer for many years, and rose sharply again in 2021–2022. The link between unemployment and inflation differs across these periods. One fixed relationship cannot explain every episode. Producer prices also show a curved relationship with future consumer inflation, rather than one constant slope. Why it matters: flexible models may capture these changes, but they still need to prove their value on new data.

## Slide 9: Several indicators carry overlapping information

A value close to 1 means two series tend to rise and fall together. A value near 0 means little overall linear relationship. Future inflation moves closely with current inflation, expectations and producer prices. Electricity prices also carry a clear signal. Many inputs overlap. Giving each an unrestricted weight can make a linear model unstable. Ridge and Elastic Net help control this problem. These correlations describe patterns, not cause and effect.

## Slide 10: Three-month forecasts beat the no-change benchmark

Elastic Net: 0.777 percentage points. XGBoost: 0.783. The no-change forecast: 1.004. We measure error using RMSE, which gives larger mistakes more weight. Lower is better. Elastic Net slightly leads XGBoost. The basic Phillips-curve model is close to the no-change benchmark. Neural networks improve on the benchmark but trail the leaders. Greater complexity does not guarantee greater accuracy.

## Slide 11: Do the models anticipate changes in inflation?

The upper chart compares the predicted change in inflation with the change that actually occurred. The no-change forecast stays at zero. XGBoost improves on that benchmark in four of the five periods shown. Its error is about 28% lower during the 2021 surge and 65% lower during the 2022–2023 rate-hike period. Early 2022 is the exception: the models miss the renewed rise around the Ukraine energy shock. Predicting some turns does not mean predicting every shock.

## Slide 12: Different forecast horizons have different winners

1 month: XGBoost + neural network. 3 months: Elastic Net. 6 months: an autoregressive model, which uses past inflation. 12 months: the Transformer sequence model. Model rankings change with the forecast horizon and the economic period. The chart shows how performance differs between calm years, supply shocks and falling inflation. Model choice should follow the forecasting task.

## Slide 13: Why combine XGBoost with a neural network?

48 years of monthly data still give only 585 raw observations, with 460 training months in the main three-month test. That is a small sample for a complex neural network. Trees can provide useful patterns for it to build on. XGBoost identifies patterns across indicators. The neural network adds recent trends in those signals. The hybrid wins the one-month test and beats standalone neural models at three months. It does not win at every horizon.

## Slide 14: Electricity, stocks and GDP add useful information

Electricity prices adjust differently from crude oil and can remain high after oil prices fall. Stocks reflect financial conditions, while GDP captures activity beyond the industrial sector. These signals widen the model’s view of the economy. Adding the indicators lowers Elastic Net’s error from 0.825 to 0.777 percentage points, an improvement of about 6%. The gain depends on the model. Extra inputs do not improve every specification or every step of the comparison.

## Slide 15: What the models use to explain inflation

Last year’s price changes matter because annual inflation compares prices with a year earlier. Electricity, producer prices and money growth also contribute. The models broadly agree that several signals matter together. Elastic Net uses fixed weights. XGBoost and the hybrid can capture relationships that change across conditions. A small unemployment weight means limited added information in these models. It does not show that employment has no economic effect.

## Slide 16: Forecast contributions can change with conditions

Each dot on the left represents one month. A positive SHAP value raises the forecast, while a negative value lowers it. The largest contributions come from earlier price changes. Electricity and real interest rates also appear among the leading inputs. The right-hand curves show how contributions vary with earlier inflation and electricity prices. These changing responses help explain what the model learned. They do not establish the causal effect of changing an input or a policy.

## Slide 17: Two historical examples: a rise and a fall

Inflation starts at 1.37%. XGBoost predicts 2.72%, while actual inflation reaches 4.13%. Changes in the year-earlier comparison push the forecast upward. The model gets the direction right but misses much of the increase. Inflation starts at 6.40%. The forecast is 4.77%, close to the actual 4.92%. Earlier price changes and several economic indicators pull the forecast down. This example captures both the direction and size of the decline more closely.

## Slide 18: Testing on future months prevents misleading results

The left chart shows the weight the Transformer places on different months in its input window. The weights vary across periods. They help describe how the model uses history, but do not prove which months cause inflation. Randomly mixing months can let information from later observations help predict earlier ones. In this comparison, XGBoost’s error looks about 47% lower with a random split. Testing in time order gives a more realistic measure of forecasting performance.

## Slide 19: The one-year forecast uses a broader set of signals

The Transformer leads the 12-month test. Recent inflation momentum, producer prices and money growth are prominent inputs. The contribution from older price comparisons falls from 14.6% at three months to 2.9% at twelve months. Over a full year, older price movements leave the annual inflation calculation. Other economic signals become relatively more important. The model draws on different months within its input window. These weights explain its predictions, not economic causation.

## Slide 20: The useful signals change with the forecast horizon

Energy and producer prices are useful near-term signals. The supply group accounts for about 29% of XGBoost’s explanation at one month. Past prices also matter mechanically because annual inflation uses a year-earlier comparison. The money and financial-conditions group has its largest share at six months. Expectations rise to about 17% at twelve months. These patterns help organise forecast monitoring. They are model-based associations, not a timetable of proven policy effects.

## Slide 21: Live demo: compare and explain a forecast

Compare the trained models for forecasts 1, 3, 6 and 12 months ahead. Choose a historical month to see how forecasts compare with the inflation outcome. Change inputs such as electricity prices or interest rates and observe the model response. Use the contribution chart to see which inputs push XGBoost’s prediction up or down. Select December 2022 and the three-month horizon. Compare XGBoost’s 4.77% forecast with actual inflation of 4.92% and the no-change forecast of 6.40%. Change electricity-price inflation and inspect how the prediction and contribution bars move. Treat the slider result as a model scenario. It does not measure what a real policy intervention would cause.

## Slide 22: What we conclude, and what remains uncertain

Elastic Net and XGBoost reduce three-month forecast error by about 22% compared with assuming no change. Electricity prices, stock returns and GDP improve the linear model’s accuracy when added together. No model wins at every horizon. The hybrid leads at one month, while the Transformer leads at twelve. Forecast explanations reveal useful signals and how their importance changes with the horizon. Unexpected events can still produce large errors, as the early-2022 episode shows. Some data include later revisions. A true real-time test would use the exact data releases available then. Monthly data leave a limited training sample, especially for deep learning. Prediction and explanation do not establish cause and effect. Further work could use real-time data and report forecast uncertainty.

## Slide 23: Project resources and selected references

FRED supplies the public economic data used in the analysis. The project notebook documents the model comparisons, charts and forecast explanations. Run run_pipeline.py to rebuild the analysis and app.py to start the interactive predictor. The presentation notes include the reference list and explanations for discussion. Singh & Bhoi (2022): machine learning for inflation forecasting. Li et al. (2022): inflation forecasting with XGBoost. Kuskaya & Bilgili (2025): electricity-price forecasting and validation. Yoon (2021): machine learning for GDP growth. Lundberg & Lee (2017): SHAP, a method for explaining model predictions.
