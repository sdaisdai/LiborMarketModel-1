We implemented Libor market model based on the referece book by Brigo and Mercurio (chapter 6 and 7):  Interest Rate Models - Theory and Practice (2006).

The descriptions of the files:

1) The input feeds are from Table 7.1 and Table 7.4 of the reference book, and the output recovers instantaneous volatilities in Table 7.5 , but only for those that can be unquely determined.

2) main.py:  the driver to run the test

3) LFM.py:   the class to implement the framework of LFM, and the cascade calibration algorithm (CCA)

4) LMM_ATM.pdf:  background and implemenation specification
