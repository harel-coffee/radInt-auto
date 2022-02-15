
# Evaluation of the dependence of radiomic features on the machine learning model

This is the code for the paper. Regretabbly, it is not very much polished,
but it should work nonetheless.


# Setting up

Install all requirements by

$ pip install -r requirements.txt

It *might* be possible that pymrmre needs to  be installed last.
I installed it afterwards and needed to install other packages
before pymrmre. Also, there are more packages than necessary,
becaues it comes from another project.

We also need the scikit-feature package from github,
which is already downloaded to the skfeature directory.
It doesnt need to be installed.


# Execution

Executing ```./eval.sh``` will start the experiment as well as
thet evaluation.


# Experiment

The experiment is then started with ```./startExperiment.py```.
The path for the results is given by the TrackingPath parameter
in the parameters.py file.
Also, it uses 24 cores for running, this can be changed
at the very bottom of the startExperiment file.

Experiments already executed will not execute a second time.



# Evaluation

Evaluation can be found in ```./evaluate.py```. Evaluation code is unfortunately
rather messy, the reason is that i used code from other projects in a very
hacky way. It could be that evaluation needs a few packages more than the
requirements.txt contains.

Evaluation needs access to the whole mlruns folders,
because it needs to recompute some of the results,
which were not computed during the experiment.
For convenience the feathered results can be found in ./results in this repo.




# License

Note: Data and scikit.features package have their own license. Please refer to the respective publications and to the scikit.features package

Other code is licensed with the MIT license:

Copyright (c) 2022 Aydin Demircioglu Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
