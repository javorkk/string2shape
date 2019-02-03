# string2shape
This is the implementation of the Eurographics 2019 paper: String-Based Synthesis of Structured Shapes

It contains CUDA/C++ implementation of the main procedural modeling application that is compiled into a python package.

The code builds and runs on Windows with Visual Studio 2015 or Linux with cmake, CUDA 8.0, and python 2.7.

To build the code, ensure that cudart.dll and a python executable are visible in the global environment (e.g. in PATH) and load the .sln file in Visual Studio.

The obj2string project is set up to use OpenMP as parallelization backend.
To switch to CUDA or TBB right-click on the project in Visual Studio, go to CUDA C++ -> Host -> Preprocessor Definitions and set THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA (or ..._TBB)
The same can be done with cmake via USE_OPENMP

After succesfull build, the code can be tested by running the "tests" project (see tests/main.cu). or in a python 2.7 environment by running the following:

Generate new shape variants using: augment_dataset.py

Be advised that this can take a while and will generate large amount of .obj files and consume a lot of disk space.

Generate a set of strings from the shapes using: create_obj_dataset.py

To create a training and validation set from the generated data use: preprocess.py

To train an autoencoder use: train.py

To sample from the latent space of the autoencoder use: sample.py or sample_graph.py

To train a sequence to sequence model for string embedding use: train_seq2seq.py

To embed a string in a shape use: graph_embedding.
