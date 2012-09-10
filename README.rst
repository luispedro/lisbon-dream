=================
Lisbon Dream Team
=================

This is the Lisbon DREAM challenge team repository. It contains code for our
solution and the writeup.

Instructions
------------

Start by running the following command::

    ./getdata.sh

this will download the starting data for you.

Running the Code
----------------

If you have ``pip`` installed, you should be able to use the following commands
to install all the dependencies::

    pip install numpy
    pip install milk
    pip install scipy
    pip install jug
    pip install scikit-learn

Now, you can run the code::

    cd sources
    jug execute

If you have multiple processors, just run multiple ``jug execute`` jobs; e.g.,
for 4 processors::

    jug execute & 
    jug execute & 
    jug execute & 
    jug execute & 

    wait

Results will appear in a ``results.txt`` file.

