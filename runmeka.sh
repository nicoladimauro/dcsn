python3 runmeka.py Arts1500 -f 5 -c 26 -mc meka.classifiers.multilabel.BR -o ./exp/meka/nbbr/ > ./exp/meka/nbbr/arts.log
python3 runmeka.py birds -c 19 -f 5 -mc meka.classifiers.multilabel.BR -o ./exp/meka/nbbr/ > ./exp/meka/nbbr/birds.log
python3 runmeka.py Business1500 -c 30 -f 5 -mc meka.classifiers.multilabel.BR -o ./exp/meka/nbbr/ > ./exp/meka/nbbr/business.log
python3 runmeka.py CAL500 -c 174 -f 5 -mc meka.classifiers.multilabel.BR -o ./exp/meka/nbbr/ > ./exp/meka/nbbr/cal.log
python3 runmeka.py emotions -c 6 -f 5 -mc meka.classifiers.multilabel.BR -o ./exp/meka/nbbr/ > ./exp/meka/nbbr/emotions.log
python3 runmeka.py flags -c 7 -f 5 -mc meka.classifiers.multilabel.BR -o ./exp/meka/nbbr/ > ./exp/meka/nbbr/flags.log
python3 runmeka.py Health1500 -c 32 -f 5 -mc meka.classifiers.multilabel.BR -o ./exp/meka/nbbr/ > ./exp/meka/nbbr/health.log
python3 runmeka.py human3106 -c 14 -f 5 -mc meka.classifiers.multilabel.BR -o ./exp/meka/nbbr/ > ./exp/meka/nbbr/human.log
python3 runmeka.py plant978 -c 12 -f 5 -mc meka.classifiers.multilabel.BR -o ./exp/meka/nbbr/ > ./exp/meka/nbbr/plant.log
python3 runmeka.py scene -c 6 -f 5 -mc meka.classifiers.multilabel.BR -o ./exp/meka/nbbr/ > ./exp/meka/nbbr/scene.log
python3 runmeka.py yeast -c 14 -f 5 -mc meka.classifiers.multilabel.BR -o ./exp/meka/nbbr/ > ./exp/meka/nbbr/yeast.log

python3 runmeka.py Arts1500 -c 26 -f 5 -mc meka.classifiers.multilabel.CC -o ./exp/meka/nbcc/ > ./exp/meka/nbcc/arts.log
python3 runmeka.py birds -c 19 -f 5 -mc meka.classifiers.multilabel.CC -o ./exp/meka/nbcc/ > ./exp/meka/nbcc/birds.log
python3 runmeka.py Business1500 -c 30 -f 5 -mc meka.classifiers.multilabel.CC -o ./exp/meka/nbcc/ > ./exp/meka/nbcc/business.log
python3 runmeka.py CAL500 -c 174 -f 5 -mc meka.classifiers.multilabel.CC -o ./exp/meka/nbcc/ > ./exp/meka/nbcc/cal.log
python3 runmeka.py emotions -c 6 -f 5 -mc meka.classifiers.multilabel.CC -o ./exp/meka/nbcc/ > ./exp/meka/nbcc/emotions.log
python3 runmeka.py flags -c 7 -f 5 -mc meka.classifiers.multilabel.CC -o ./exp/meka/nbcc/ > ./exp/meka/nbcc/flags.log
python3 runmeka.py Health1500 -c 32 -f 5 -mc meka.classifiers.multilabel.CC -o ./exp/meka/nbcc/ > ./exp/meka/nbcc/health.log
python3 runmeka.py human3106 -c 14 -f 5 -mc meka.classifiers.multilabel.CC -o ./exp/meka/nbcc/ > ./exp/meka/nbcc/human.log
python3 runmeka.py plant978 -c 12 -f 5 -mc meka.classifiers.multilabel.CC -o ./exp/meka/nbcc/ > ./exp/meka/nbcc/plant.log
python3 runmeka.py scene -c 6 -f 5 -mc meka.classifiers.multilabel.CC -o ./exp/meka/nbcc/ > ./exp/meka/nbcc/scene.log
python3 runmeka.py yeast -c 14 -f 5 -mc meka.classifiers.multilabel.CC -o ./exp/meka/nbcc/ > ./exp/meka/nbcc/yeast.log

python3 runmeka.py Arts1500 -f 5 -c 26 -mc meka.classifiers.multilabel.BR -wc weka.classifiers.functions.SMO -o ./exp/meka/SMO/br/ > ./exp/meka/SMO/br/arts.log
python3 runmeka.py birds -c 19 -f 5 -mc meka.classifiers.multilabel.BR -wc weka.classifiers.functions.SMO -o ./exp/meka/SMO/br/ > ./exp/meka/SMO/br/birds.log
python3 runmeka.py Business1500 -c 30 -f 5 -mc meka.classifiers.multilabel.BR -wc weka.classifiers.functions.SMO -o ./exp/meka/SMO/br/ > ./exp/meka/SMO/br/business.log
python3 runmeka.py CAL500 -c 174 -f 5 -mc meka.classifiers.multilabel.BR  -wc weka.classifiers.functions.SMO -o ./exp/meka/SMO/br/ > ./exp/meka/SMO/br/cal.log
python3 runmeka.py emotions -c 6 -f 5 -mc meka.classifiers.multilabel.BR -wc weka.classifiers.functions.SMO -o ./exp/meka/SMO/br/ > ./exp/meka/SMO/br/emotions.log
python3 runmeka.py flags -c 7 -f 5 -mc meka.classifiers.multilabel.BR -wc weka.classifiers.functions.SMO -o ./exp/meka/SMO/br/ > ./exp/meka/SMO/br/flags.log
python3 runmeka.py Health1500 -c 32 -f 5 -mc meka.classifiers.multilabel.BR -wc weka.classifiers.functions.SMO -o ./exp/meka/SMO/br/ > ./exp/meka/SMO/br/health.log
python3 runmeka.py human3106 -c 14 -f 5 -mc meka.classifiers.multilabel.BR -wc weka.classifiers.functions.SMO -o ./exp/meka/SMO/br/ > ./exp/meka/SMO/br/human.log
python3 runmeka.py plant978 -c 12 -f 5 -mc meka.classifiers.multilabel.BR -wc weka.classifiers.functions.SMO -o ./exp/meka/SMO/br/ > ./exp/meka/SMO/br/plant.log
python3 runmeka.py scene -c 6 -f 5 -mc meka.classifiers.multilabel.BR -wc weka.classifiers.functions.SMO -o ./exp/meka/SMO/br/ > ./exp/meka/SMO/br/scene.log
python3 runmeka.py yeast -c 14 -f 5 -mc meka.classifiers.multilabel.BR -wc weka.classifiers.functions.SMO -o ./exp/meka/SMO/br/ > ./exp/meka/SMO/br/yeast.log

python3 runmeka.py Arts1500 -c 26 -f 5 -mc meka.classifiers.multilabel.CC -wc weka.classifiers.functions.SMO -o ./exp/meka/SMO/cc/ > ./exp/meka/SMO/cc/arts.log
python3 runmeka.py birds -c 19 -f 5 -mc meka.classifiers.multilabel.CC -wc weka.classifiers.functions.SMO -o ./exp/meka/SMO/cc/ > ./exp/meka/SMO/cc/birds.log
python3 runmeka.py Business1500 -c 30 -f 5 -mc meka.classifiers.multilabel.CC -wc weka.classifiers.functions.SMO -o ./exp/meka/SMO/cc/ > ./exp/meka/SMO/cc/business.log
python3 runmeka.py CAL500 -c 174 -f 5 -mc meka.classifiers.multilabel.CC -wc weka.classifiers.functions.SMO -o ./exp/meka/SMO/cc/ > ./exp/meka/SMO/cc/cal.log
python3 runmeka.py emotions -c 6 -f 5 -mc meka.classifiers.multilabel.CC -wc weka.classifiers.functions.SMO -o ./exp/meka/SMO/cc/ > ./exp/meka/SMO/cc/emotions.log
python3 runmeka.py flags -c 7 -f 5 -mc meka.classifiers.multilabel.CC -wc weka.classifiers.functions.SMO -o ./exp/meka/SMO/cc/ > ./exp/meka/SMO/cc/flags.log
python3 runmeka.py Health1500 -c 32 -f 5 -mc meka.classifiers.multilabel.CC -wc weka.classifiers.functions.SMO -o ./exp/meka/SMO/cc/ > ./exp/meka/SMO/cc/health.log
python3 runmeka.py human3106 -c 14 -f 5 -mc meka.classifiers.multilabel.CC -wc weka.classifiers.functions.SMO -o ./exp/meka/SMO/cc/ > ./exp/meka/SMO/cc/human.log
python3 runmeka.py plant978 -c 12 -f 5 -mc meka.classifiers.multilabel.CC -wc weka.classifiers.functions.SMO -o ./exp/meka/SMO/cc/ > ./exp/meka/SMO/cc/plant.log
python3 runmeka.py scene -c 6 -f 5 -mc meka.classifiers.multilabel.CC -wc weka.classifiers.functions.SMO -o ./exp/meka/SMO/cc/ > ./exp/meka/SMO/cc/scene.log
python3 runmeka.py yeast -c 14 -f 5 -mc meka.classifiers.multilabel.CC -wc weka.classifiers.functions.SMO -o ./exp/meka/SMO/cc/ > ./exp/meka/SMO/cc/yeast.log

python3 runmeka.py Arts1500 -f 5 -c 26 -mc meka.classifiers.multilabel.BR -wc weka.classifiers.trees.J48 -o ./exp/meka/J48/br/ > ./exp/meka/J48/br/arts.log
python3 runmeka.py birds -c 19 -f 5 -mc meka.classifiers.multilabel.BR -wc weka.classifiers.trees.J48 -o ./exp/meka/J48/br/ > ./exp/meka/J48/br/birds.log
python3 runmeka.py Business1500 -c 30 -f 5 -mc meka.classifiers.multilabel.BR -wc weka.classifiers.trees.J48 -o ./exp/meka/J48/br/ > ./exp/meka/J48/br/business.log
python3 runmeka.py CAL500 -c 174 -f 5 -mc meka.classifiers.multilabel.BR  -wc weka.classifiers.trees.J48 -o ./exp/meka/J48/br/ > ./exp/meka/J48/br/cal.log
python3 runmeka.py emotions -c 6 -f 5 -mc meka.classifiers.multilabel.BR -wc weka.classifiers.trees.J48 -o ./exp/meka/J48/br/ > ./exp/meka/J48/br/emotions.log
python3 runmeka.py flags -c 7 -f 5 -mc meka.classifiers.multilabel.BR -wc weka.classifiers.trees.J48 -o ./exp/meka/J48/br/ > ./exp/meka/J48/br/flags.log
python3 runmeka.py Health1500 -c 32 -f 5 -mc meka.classifiers.multilabel.BR -wc weka.classifiers.trees.J48 -o ./exp/meka/J48/br/ > ./exp/meka/J48/br/health.log
python3 runmeka.py human3106 -c 14 -f 5 -mc meka.classifiers.multilabel.BR -wc weka.classifiers.trees.J48 -o ./exp/meka/J48/br/ > ./exp/meka/J48/br/human.log
python3 runmeka.py plant978 -c 12 -f 5 -mc meka.classifiers.multilabel.BR -wc weka.classifiers.trees.J48 -o ./exp/meka/J48/br/ > ./exp/meka/J48/br/plant.log
python3 runmeka.py scene -c 6 -f 5 -mc meka.classifiers.multilabel.BR -wc weka.classifiers.trees.J48 -o ./exp/meka/J48/br/ > ./exp/meka/J48/br/scene.log
python3 runmeka.py yeast -c 14 -f 5 -mc meka.classifiers.multilabel.BR -wc weka.classifiers.trees.J48 -o ./exp/meka/J48/br/ > ./exp/meka/J48/br/yeast.log

python3 runmeka.py Arts1500 -c 26 -f 5 -mc meka.classifiers.multilabel.CC -wc weka.classifiers.trees.J48 -o ./exp/meka/J48/cc/ > ./exp/meka/J48/cc/arts.log
python3 runmeka.py birds -c 19 -f 5 -mc meka.classifiers.multilabel.CC -wc weka.classifiers.trees.J48 -o ./exp/meka/J48/cc/ > ./exp/meka/J48/cc/birds.log
python3 runmeka.py Business1500 -c 30 -f 5 -mc meka.classifiers.multilabel.CC -wc weka.classifiers.trees.J48 -o ./exp/meka/J48/cc/ > ./exp/meka/J48/cc/business.log
python3 runmeka.py CAL500 -c 174 -f 5 -mc meka.classifiers.multilabel.CC -wc weka.classifiers.trees.J48 -o ./exp/meka/J48/cc/ > ./exp/meka/J48/cc/cal.log
python3 runmeka.py emotions -c 6 -f 5 -mc meka.classifiers.multilabel.CC -wc weka.classifiers.trees.J48 -o ./exp/meka/J48/cc/ > ./exp/meka/J48/cc/emotions.log
python3 runmeka.py flags -c 7 -f 5 -mc meka.classifiers.multilabel.CC -wc weka.classifiers.trees.J48 -o ./exp/meka/J48/cc/ > ./exp/meka/J48/cc/flags.log
python3 runmeka.py Health1500 -c 32 -f 5 -mc meka.classifiers.multilabel.CC -wc weka.classifiers.trees.J48 -o ./exp/meka/J48/cc/ > ./exp/meka/J48/cc/health.log
python3 runmeka.py human3106 -c 14 -f 5 -mc meka.classifiers.multilabel.CC -wc weka.classifiers.trees.J48 -o ./exp/meka/J48/cc/ > ./exp/meka/J48/cc/human.log
python3 runmeka.py plant978 -c 12 -f 5 -mc meka.classifiers.multilabel.CC -wc weka.classifiers.trees.J48 -o ./exp/meka/J48/cc/ > ./exp/meka/J48/cc/plant.log
python3 runmeka.py scene -c 6 -f 5 -mc meka.classifiers.multilabel.CC -wc weka.classifiers.trees.J48 -o ./exp/meka/J48/cc/ > ./exp/meka/J48/cc/scene.log
python3 runmeka.py yeast -c 14 -f 5 -mc meka.classifiers.multilabel.CC -wc weka.classifiers.trees.J48 -o ./exp/meka/J48/cc/ > ./exp/meka/J48/cc/yeast.log

python3 runmeka.py Arts1500 -f 5 -c 26 -mc meka.classifiers.multilabel.BR -wc "weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5"  -o ./exp/meka/TAN/br/ 
python3 runmeka.py birds -c 19 -f 5 -mc  meka.classifiers.multilabel.BR -wc "weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5" -o ./exp/meka/TAN/br/ 
python3 runmeka.py Business1500 -c 30 -f 5 -mc meka.classifiers.multilabel.BR -wc "weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5" -o ./exp/meka/TAN/br/ 
python3 runmeka.py CAL500 -c 174 -f 5 -mc meka.classifiers.multilabel.BR -wc "weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5" -o ./exp/meka/TAN/br/ 
python3 runmeka.py emotions -c 6 -f 5 -mc meka.classifiers.multilabel.BR -wc "weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5" -o ./exp/meka/TAN/br/ 
python3 runmeka.py flags -c 7 -f 5 -mc meka.classifiers.multilabel.BR -wc "weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5" -o ./exp/meka/TAN/br/ 
python3 runmeka.py Health1500 -c 32 -f 5 -mc meka.classifiers.multilabel.BR -wc "weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5" -o ./exp/meka/TAN/br/ 
python3 runmeka.py human3106 -c 14 -f 5 -mc meka.classifiers.multilabel.BR -wc "weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5" -o ./exp/meka/TAN/br/ 
python3 runmeka.py plant978 -c 12 -f 5 -mc meka.classifiers.multilabel.BR -wc "weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5" -o ./exp/meka/TAN/br/ 
python3 runmeka.py scene -c 6 -f 5 -mc meka.classifiers.multilabel.BR -wc "weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5" -o ./exp/meka/TAN/br/ 
python3 runmeka.py yeast -c 14 -f 5 -mc meka.classifiers.multilabel.BR -wc "weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5" -o ./exp/meka/TAN/br/ 

python3 runmeka.py Arts1500 -f 5 -c 26 -mc meka.classifiers.multilabel.CC -wc "weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5" -o ./exp/meka/TAN/cc/ 
python3 runmeka.py birds -c 19 -f 5 -mc  meka.classifiers.multilabel.CC -wc "weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5" -o ./exp/meka/TAN/cc/
python3 runmeka.py Business1500 -c 30 -f 5 -mc meka.classifiers.multilabel.CC -wc "weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5" -o ./exp/meka/TAN/cc/
python3 runmeka.py CAL500 -c 174 -f 5 -mc meka.classifiers.multilabel.CC -wc "weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5" -o ./exp/meka/TAN/cc/
python3 runmeka.py emotions -c 6 -f 5 -mc meka.classifiers.multilabel.CC -wc "weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5" -o ./exp/meka/TAN/cc/
python3 runmeka.py flags -c 7 -f 5 -mc meka.classifiers.multilabel.CC -wc "weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5" -o ./exp/meka/TAN/cc/
python3 runmeka.py Health1500 -c 32 -f 5 -mc meka.classifiers.multilabel.CC -wc "weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5" -o ./exp/meka/TAN/cc/
python3 runmeka.py human3106 -c 14 -f 5 -mc meka.classifiers.multilabel.CC -wc "weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5" -o ./exp/meka/TAN/cc/
python3 runmeka.py plant978 -c 12 -f 5 -mc meka.classifiers.multilabel.CC -wc "weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5" -o ./exp/meka/TAN/cc/
python3 runmeka.py scene -c 6 -f 5 -mc meka.classifiers.multilabel.CC -wc "weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5" -o ./exp/meka/TAN/cc/
python3 runmeka.py yeast -c 14 -f 5 -mc meka.classifiers.multilabel.CC -wc "weka.classifiers.bayes.BayesNet -- -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5" -o ./exp/meka/TAN/cc/



