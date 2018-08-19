We implemented Libor market model based on the referece book by Brigo and Mercurio.
The input feeds are from Table 7.1 and Table 7.4 in the reference book, and the output recovers volatilities Table 7.5 (those that can be unquely determined)
main.py:  the driver to run the test
LFM.py:   the class to implement the framework of LFM, and the cascade calibration algorithm (CCA)
LMM_ATM.pdf:  background and implemenation specification
