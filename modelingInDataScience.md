### Modeling Features in Data Science

#### Full Stack Path

Hypothesis Formulation, Gather Data, Explore Data, 
Label Data, Analyze Labels, Generate Features, 
Analyze Features, Prototype Models, Label Hold-out Data, 
Evaluate Model, Model Review, Deploy Model, 
Choose Final Parameters, A/B Test Model, 
Monitor Model, Repeat.

#### Model Framework

* Label- data before, during and after you build a model

* Extract- features in one place

* Reuse- your model building code

* Release- softly and log everything

* Validate- and review every model

* Monitor- after deploying

* Retrain- when needed

#### Model Building

Plot learning curves.

Features > Model Builder > Model > Model Predictor > Predictions.

Features > Feature Extraction > Model Predictor.

Model Builder:

* Features- feature sampling, feature scaling, feature selection

* Model Builder- test/train splits, cross validation

* Model- generate plots, email results, export model

Write some feature extraction code, to provide a matrix.
Load into model builder with a configuration file.
That build a model which is written out into a standard format.
Pass model to model predictor code.

#### Model Review

Another Perspective, Transparency and Reproducibility, 
Awareness of Tokyo, Seattle, SF, Austin.

* Context- 
  * What should this model enable us to do 
(highlighting, filtering, sorting, etc.)? 
  * What products/ interfaces/ workflows will 
  initially use this model?
  * What will it allow to be done which 
  could not be done before?
  * How are we going to use this? 
  * What are we trying to predict?
  * What kind of features did we use?
  * How did we use the features? 
  * How did we choose the best model?
  * What was the performance of that model?
  * When is the recommendation for when the model should be released.

* Data
  * What queries and filters were used?
  * From what time range did your data originate?
  * Did you sample your dataset?
  * Is the data from an employer or a job seeker?
  * Or is it data we labeled ourselves?
  * How did we label that data?
  * Share the labeling tasks. Provide the assumptions.
  
* Response variable
  * How was the response variable labeled or collected?
  * What the model outputs (predictions) represent
  and how they should be scaled or thresholded?
  * How should the predictions be used? 
  * Is it to provide a recommendation, a probability, 
  or normalized counts?

* Features
  * How were your features generated?
  * Which features were most important?
  
* Model selection and performance
  * Performance reports on train/ test sets
  * Overall cross validation (CV) search strategy
  and scoring function
  * Other performance tests (e.g. newer
  hold out sets, stress testing)
  * Did we train others to label our data?
  * Expected model performance

* Transparency and recommendations
  * Properties files for Model Builder
  * Link to branch of Model Builder code
  * Examples of Model Predictions
  * Possible directions for future improvements
  * A couple sentences on why you think the model is
  ready for production
  
  #### Is something going wrong?
  
  Look at class distributions.
  
  Look at a histogram of the 
  regression values being predicted 
  from day to day. If that's shifting 
  outside of the normal, something could be wrong.
  
  Within classification models, 
  look at the ratios of class predictions 
  over time.
  
  Use feature monitoring tool to determine feature stability.
  Or choose less sensitive features.

#### Sources

##### [Data to Deployment- Full Stack Data Science](http://engineering.indeedblog.com/talks/data-to-deployment/)

Notes from Ben Link.

##### [Managing Experiments and Behavior Dynamically with Proctor](http://engineering.indeedblog.com/talks/managing-experiments-behavior-dynamically-proctor/)

* A/B Testing is a bake-off between two versions of 
a web page. The one with the best conversion rate wins.

#### Machine Learning Notes

Read a blog,  [Friendly Machine Learning](http://engineering.indeedblog.com/blog/2017/06/friendly-machine-learning/)
about [John Langford's](https://github.com/JohnLangford) 
[Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit/wiki).
This paper, [Normalized online learning](https://arxiv.org/pdf/1305.6646.pdf)
is pretty amazing.

Last note: use GitHub wikis more. :)

#### Going down the rabbit hole... I may choose to look down a few worm holes.


