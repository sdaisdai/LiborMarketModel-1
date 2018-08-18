import pandas as pd
from LFM import LFM
import unittest

class MyTestCase(unittest.TestCase):
    def test1(self):
        baseTime = 0
        T0 = 1
        initRate = 0.0469
        file_swaption_vol = 'table7_4.xlsx'
        annual_rate_capletvol = 'Table7_1.xlsx'

        swaption_vol = pd.ExcelFile(file_swaption_vol).parse(index_col=0, header=[0])
        tarVol_swaption = swaption_vol.copy()

        capMarketInfo = pd.ExcelFile(annual_rate_capletvol).parse(index_col=0, header=[0])
        theta_values = [0.0147, 0.0643, 0.1032, 0.1502, 0.1969,0.2239, 0.2771, 0.2950, 0.3630, 0.3810,
                        0.4217, 0.4836, 0.5204, 0.5418, 0.5791,0.6496, 0.6679, 0.7126, 0.7659]
        opt = 'rank2_theta'
        handlingOpt = 'False'
        outputfileName = 'output_' + opt + handlingOpt + '.xlsx'
        lfm_obj = LFM(baseTime, T0, initRate, capMarketInfo, tarVol_swaption, opt, handlingOpt, outputfileName)
        sigma = lfm_obj.CCA(theta_values)
        lfm_obj.mySave(sigma)


if __name__ == '__main__':
    unittest.main()