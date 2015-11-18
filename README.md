# dcsn
dCSN is a package for learning Cutset Networks (CNets), a recently introduced tractable Probabilistic Graphical Model.

## Usage

    python dcsn.py nltcs -k 1 -a 0.4

learn a single csn on the nltcs dataset with alpha smoothing parameter
alpha set to 0.4.

    python dcsn.py nltcs -k 5 10 15 20 -a 0.1 0.2 0.3 0.4 -d 10 50 100 -s 2 4 6 

do a grid search for k in {5,10,15,20}, a in {0.1,0.2,0.3,0.4}, d in
{10,50,100} and s in {2,4,6}. Since k is greather than 1 than a
bagging approach is used. Furthermore, by specifying the -r option a
random forest approach may be used.

    usage: dcsn.py [-h] [--seed [SEED]] [-o [OUTPUT]] [-r] [-k K [K ...]]
                   [-d D [D ...]] [-s S [S ...]] [-a ALPHA [ALPHA ...]]
                   [--al] [--an] [-v [VERBOSE]]
                   dataset

    positional arguments:
      dataset               Specify a dataset name from data/ (es. nltcs)

    optional arguments:
      -h, --help            show this help message and exit
      --seed [SEED]         Seed for the random generator
      -o [OUTPUT], --output [OUTPUT]
                            Output dir path
      -r, --random          Random Forest. If set a Random Forest approach is
                            used.
      -k K [K ...]          Number of components to use. If greater than 1, then a
                            bagging approach is used.
      -d D [D ...]          Min number of instances in a slice to split.
      -s S [S ...]          Min number of features in a slice to split.
      -a ALPHA [ALPHA ...], --alpha ALPHA [ALPHA ...]
                            Smoothing factor for leaf probability estimation
      --al                  Use and nodes as leaves (i.e., CL forests).
      --an                  Use and nodes as inner nodes and leaves (i.e., CL
                            forests).
      -v [VERBOSE], --verbose [VERBOSE]
                            Verbosity level
